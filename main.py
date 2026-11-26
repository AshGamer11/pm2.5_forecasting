"""End-to-end PM2.5 forecasting: python main.py [--data data/city_hour.csv --city Delhi] [--epochs 40]"""
import argparse, json, random
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import *
from src.data import load_dataset
from src.preprocess import prepare, to_ugm3
from src.models import MODELS
from src.train import fit, predict
from src.evaluate import metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None); ap.add_argument("--city", default="Delhi")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    a = ap.parse_args()

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    RESULTS_DIR.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df, source = load_dataset(a.data, a.city)
    print(f"Data source: {source} | {len(df)} hourly rows | device: {device}")
    d = prepare(df)
    print("Missing values handled (per column):\n", d["missing_report"].to_string())
    print(f"Features ({len(d['features'])}): {d['features']}")
    print(f"Windows -> train {len(d['X_train'])}, val {len(d['X_val'])}, test {len(d['X_test'])}\n")

    nf = d["X_train"].shape[2]
    y_true = to_ugm3(d["y_test"], d["t_mean"], d["t_std"])           # (N, 3) real units
    spike_thr = np.percentile(y_true[:, 0], 90)                      # top-10% = "spike" hours

    rows, preds, hists = [], {}, {}

    # Baseline: persistence (PM2.5 stays at its last observed value)
    # Persistence baseline: PM2.5 stays at the last observed value.
    # (PM2.5 feature scaler stats == target stats on the train split, so inversion is exact.)
    last_pm = to_ugm3(d["X_test"][:, -1, d["features"].index(TARGET)], d["t_mean"], d["t_std"])
    pers = np.repeat(last_pm[:, None], len(HORIZONS), axis=1)
    preds["Persistence"] = pers
    for k, h in enumerate(HORIZONS):
        m = metrics(y_true[:, k], pers[:, k]); rows.append({"Model": "Persistence", "Horizon": f"{h}h", **m})

    for name, cls in MODELS.items():
        print(f"Training {name} ...")
        model = cls(nf, len(HORIZONS))
        model, hist = fit(model, d, device, a.epochs, name=name)
        hists[name] = hist
        p = to_ugm3(predict(model, d["X_test"], device), d["t_mean"], d["t_std"])
        preds[name] = p
        for k, h in enumerate(HORIZONS):
            m = metrics(y_true[:, k], p[:, k])
            sp = y_true[:, k] >= np.percentile(y_true[:, k], 90)
            m_sp = metrics(y_true[sp, k], p[sp, k])
            rows.append({"Model": name, "Horizon": f"{h}h", **m, "Spike_MAE": m_sp["MAE"]})
        torch.save(model.state_dict(), RESULTS_DIR / f"{name.replace(' ', '_')}.pt")

    # Spike MAE for persistence too
    for k, h in enumerate(HORIZONS):
        sp = y_true[:, k] >= np.percentile(y_true[:, k], 90)
        for r in rows:
            if r["Model"] == "Persistence" and r["Horizon"] == f"{h}h":
                r["Spike_MAE"] = metrics(y_true[sp, k], pers[sp, k])["MAE"]

    res = pd.DataFrame(rows).round(3)
    res.to_csv(RESULTS_DIR / "metrics.csv", index=False)
    print("\n=== TEST RESULTS (µg/m³) ===\n", res.to_string(index=False))

    # ---------- Plots ----------
    colors = {"Persistence": "gray", "LSTM": "tab:blue", "GRU": "tab:green", "1D CNN": "tab:orange"}
    # 1. Error vs horizon
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for j, met in enumerate(["MAE", "RMSE", "R2"]):
        for name in colors:
            s = res[res.Model == name]
            ax[j].plot(s.Horizon, s[met], marker="o", label=name, color=colors[name])
        ax[j].set_title(f"{met} vs forecast horizon"); ax[j].set_xlabel("Horizon"); ax[j].grid(alpha=.3)
    ax[0].legend(); plt.tight_layout(); plt.savefig(RESULTS_DIR / "horizon_comparison.png", dpi=150); plt.close()

    # 2. Predictions vs actual (last 7 days of test, 1h and 6h)
    n = 24 * 7
    fig, ax = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    for r, k in enumerate([0, 2]):
        ax[r].plot(y_true[-n:, k], "k", lw=1.5, label="Actual")
        for name in ["LSTM", "GRU", "1D CNN"]:
            ax[r].plot(preds[name][-n:, k], lw=1, label=name, color=colors[name])
        ax[r].set_title(f"PM2.5 — {HORIZONS[k]}h ahead (last 7 days of test set)"); ax[r].set_ylabel("µg/m³"); ax[r].grid(alpha=.3)
    ax[0].legend(ncol=4); plt.tight_layout(); plt.savefig(RESULTS_DIR / "predictions.png", dpi=150); plt.close()

    # 3. Training curves
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.5))
    for j, (name, h) in enumerate(hists.items()):
        ax[j].plot(h["train"], label="train"); ax[j].plot(h["val"], label="val"); ax[j].set_title(name + " loss"); ax[j].legend(); ax[j].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(RESULTS_DIR / "training_curves.png", dpi=150); plt.close()

    # 4. Scatter 6h
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for j, name in enumerate(["LSTM", "GRU", "1D CNN"]):
        ax[j].scatter(y_true[:, 2], preds[name][:, 2], s=3, alpha=.3, color=colors[name])
        lim = [0, max(y_true[:, 2].max(), preds[name][:, 2].max())]; ax[j].plot(lim, lim, "k--", lw=1)
        ax[j].set_title(f"{name}: predicted vs actual (6h)"); ax[j].set_xlabel("Actual"); ax[j].set_ylabel("Predicted")
    plt.tight_layout(); plt.savefig(RESULTS_DIR / "scatter_6h.png", dpi=150); plt.close()

    # run info
    json.dump({"source": source, "spike_threshold_ugm3_1h": float(spike_thr)}, open(RESULTS_DIR / "run_info.json", "w"))
    print(f"\nSaved metrics + plots to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
