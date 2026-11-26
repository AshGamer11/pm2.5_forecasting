# PM2.5 Forecasting (1h / 3h / 6h) with LSTM, GRU and 1D CNN

## Run
```
pip install -r requirements.txt
python main.py                      # synthetic Delhi-like data (works out of the box)
python main.py --city Delhi         # uses data/city_hour.csv if present (Kaggle "Air Quality Data in India")
python main.py --data my_openaq.csv # any hourly CSV with datetime + PM2.5, PM10, NO2, CO, O3 (+ weather) columns
python main.py --epochs 10          # faster run
```
Outputs go to `results/`: metrics.csv, horizon_comparison.png, predictions.png, training_curves.png, scatter_6h.png, saved models.

## Pipeline
1. Load hourly data (real CSV or synthetic).
2. Missing values: short gaps (<=6h) time-interpolated, long gaps ffill/bfill/median, plus a "PM2.5 was missing" flag feature.
3. Features: pollutants, weather, cyclic hour/day-of-year encodings. log1p on skewed pollutants (handles spikes).
4. 24-hour sliding windows -> predict PM2.5 at +1h, +3h, +6h (one multi-output model).
5. Chronological 70/15/15 split, scalers fit on train only (no leakage).
6. Models: LSTM, GRU, 1D CNN + persistence baseline. Huber loss (spike-robust), early stopping.
7. Evaluate MAE, RMSE, R² per horizon, plus MAE on the top-10% (spike) hours.

## Note
Synthetic data is only for demonstration. For the final presentation, download a real dataset (OpenAQ or Kaggle India) and say so.
