# Energy Demand Forecasting: LSTM vs. GRU vs. Transformer

Multi-horizon (1–24h ahead) electricity demand forecasting, benchmarking a naive baseline, linear regression, and three deep sequence architectures (LSTM, GRU, Transformer) head-to-head — on accuracy *and* inference latency. Built to mirror the forecasting problem real grid operators and demand-forecasting teams (utilities, Uptake, C3.ai) actually solve: predict load hours to days ahead, where under-forecasting risks blackouts and over-forecasting wastes generation capacity.

## Key finding

**Linear regression beat all three deep architectures.** With well-engineered calendar and lag features, a simple linear model outperformed LSTM, GRU, and Transformer on this seasonal, well-structured data — a real, defensible finding, not a failure. Among the neural models, the simplest (GRU) generalized best, and the Transformer's extra flexibility didn't pay off on ~4,300 training sequences. At inference time, though, the Transformer was fastest — attention processes all 168 input hours in one parallel operation, while LSTM/GRU must step through them sequentially.

See [Results](#results) below for the full numbers and the reasoning behind them.

## Project structure

```
energy-demand-forecasting/
├── README.md
├── requirements.txt
├── data/
│   └── generate_synthetic_data.py   # synthetic PJM-schema data generator
├── src/
│   ├── data_pipeline.py             # loading, feature engineering, sequencing, splitting
│   ├── baselines.py                 # naive persistence + linear regression
│   ├── lstm_model.py                # RNNForecaster (LSTM/GRU via cell_type switch)
│   ├── train_lstm.py                # trains the LSTM
│   ├── train_gru.py                 # trains the GRU (reuses train_lstm.py's loop)
│   ├── transformer_model.py         # TransformerForecaster
│   ├── train_transformer.py         # trains the Transformer
│   ├── evaluate.py                  # compares all 5 models, saves plots
│   └── latency_benchmark.py         # inference latency / throughput benchmark
└── results/                         # created automatically: checkpoints, plots, logs
```

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**requirements.txt:**
```
numpy
pandas
scikit-learn
torch
matplotlib
holidays
```

## How to run

Run in order from the project root, except where noted:

```bash
python data/generate_synthetic_data.py   # from project root

cd src
python data_pipeline.py        # sanity-check the pipeline
python baselines.py            # naive persistence + linear regression
python lstm_model.py           # shape check
python train_lstm.py           # trains LSTM, saves results/lstm_best.pt
python train_gru.py            # trains GRU,  saves results/gru_best.pt
python transformer_model.py    # shape check
python train_transformer.py    # trains Transformer, saves results/transformer_best.pt
python evaluate.py             # full 5-model comparison + plots
python latency_benchmark.py    # inference latency / throughput
```

## About the data

Kaggle wasn't reachable in the environment this was built in, so `generate_synthetic_data.py` produces a synthetic hourly series with the **exact schema** of the real dataset (Kaggle: `robikscube/hourly-energy-consumption`, file `PJME_hourly.csv` — columns `Datetime`, `PJME_MW`). The synthetic series includes daily/weekly/annual seasonality, a mild multi-year trend, US holiday effects, and autocorrelated (non-iid) noise, so it isn't trivially easy.

**To use the real data instead:** download `PJME_hourly.csv` from Kaggle and place it at `data/PJME_hourly.csv`. Everything downstream works unchanged — no code edits needed. Note the results below are on synthetic data; real load data has weather shocks and irregularities a linear model can't capture as easily, so the accuracy gap may narrow with the real file.

## Methodology notes

- **Chronological train/val/test split (70/15/15), never shuffled.** Time series: the future can't be used to predict the past, so a random split would leak information and inflate reported accuracy.
- **Scaling fit on training data only**, applied to val/test — standard leakage-safe practice.
- **Direct multi-step forecasting**: each model predicts all 24 horizon hours in one forward pass, rather than feeding predictions back in autoregressively, avoiding compounding one-step errors.
- **Linear baseline uses future-known calendar features** (hour/day-of-week of the timestamp being forecast, which is deterministic and knowable in advance) rather than only the last-observed timestep. An earlier version without this lost badly to naive persistence (MAE 3154 vs. 1073) — worth knowing about since it's an easy mistake to repeat.
- **`RNNForecaster` takes `cell_type` as a parameter** rather than being two separate classes, so the GRU variant required zero new model code — only a new training entry point.

## Results

**Accuracy — full 24h-ahead forecast:**

| Model | MAE (MW) | RMSE (MW) | MAPE (%) |
|---|---|---|---|
| Naive persistence | 1074.3 | 1506.1 | 3.35 |
| **Linear regression** | **585.7** | **765.4** | **1.83** |
| LSTM | 730.9 | 996.5 | 2.31 |
| GRU | 708.7 | 963.0 | 2.23 |
| Transformer | 752.6 | 1009.6 | 2.36 |

**Error by forecast horizon (MAE, MW):**

| Model | h=1 | h=12 | h=24 |
|---|---|---|---|
| Naive persistence | 1106.3 | 1045.9 | 1074.5 |
| Linear regression | 300.7 | 629.4 | 646.1 |
| LSTM | 685.7 | 699.4 | 833.2 |
| GRU | 647.8 | 679.8 | 842.8 |
| Transformer | 626.9 | 748.7 | 849.1 |

Linear regression's advantage is largest at 1 hour ahead (300.7 vs. ~650–690 MW for the neural models) — its lag features (`lag_24`, `lag_168`, `rolling_mean_24`) almost directly encode the short-horizon answer, a relationship the neural models had to learn implicitly instead of being handed explicitly.

**Inference latency** (batch=1 / batch=64, CPU):

| Model | Params | Single (ms) | Batch64 (ms) | Throughput (/s) |
|---|---|---|---|---|
| LSTM | 55,608 | 2.86 | 20.33 | 3148 |
| GRU | 42,424 | 18.61 | 31.27 | 2047 |
| Transformer | 21,264 | 0.81 | 15.52 | 4123 |

## Limitations

- **Results are on synthetic data** (see [About the data](#about-the-data)) — treat findings as directional until validated on the real Kaggle file.
- **The Transformer has fewer parameters than the LSTM/GRU** (21K vs. 56K/42K) — it was sized down to keep CPU training time reasonable. Part of its latency advantage in the table above reflects having fewer parameters, not purely its parallelizable attention mechanism. A parameter-matched rerun would be needed to isolate the architectural effect alone.
- All models were trained on CPU only; training times and the Transformer's architecture (1 layer, d_model=48) were scoped accordingly. A GPU would allow a larger, more standard Transformer configuration.

## Tech stack

Python · PyTorch · scikit-learn · pandas · NumPy · matplotlib · holidays

---

*Built by Praveena Kamanuru.*
