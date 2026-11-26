# Presentation outline (8-10 slides)
1. Problem: why forecast PM2.5 (early warning), regression not good/bad classification
2. Data: pollutants + weather, hourly; source (state clearly if synthetic or real)
3. Preprocessing: missing values (interpolation + flag), log transform, scaling, time encodings
4. Windowing: 24h history -> 1h/3h/6h targets (diagram)
5. Models: LSTM (long-term memory), GRU (lighter), 1D CNN (local patterns)
6. Training: Huber loss for spikes, early stopping, chronological split
7. Results table (results/metrics.csv) + horizon_comparison.png
8. Predictions plot (predictions.png) + spike-hour MAE
9. Findings: error grows with horizon, R² drops; compare to persistence baseline
10. Limitations + future work (attention/Transformer, more stations, real weather forecasts)

# Likely questions
- Why log1p? Pollution is right-skewed; spikes would dominate MSE.
- Why chronological split? Random split leaks future info.
- Why Huber loss? Less sensitive to extreme spikes than MSE.
- Why does error grow with horizon? Less of the future is determined by recent history.
