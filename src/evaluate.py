"""
Loads the trained LSTM, GRU, and Transformer, runs all five models
(persistence, linear, LSTM, GRU, transformer) on the held-out test set, and
reports MAE/RMSE/MAPE - both overall and broken down by forecast horizon
(1h ahead vs 24h ahead), since a model's error usually grows with horizon
and that growth rate is the real finding.
"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from data_pipeline import load_data, engineer_features, chronological_split, fit_scaler, make_sequences
from baselines import naive_persistence_predict, train_linear_baseline, predict_linear_baseline
from lstm_model import RNNForecaster
from transformer_model import TransformerForecaster

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results")


def mae(pred, true):
    return np.mean(np.abs(pred - true))


def rmse(pred, true):
    return np.sqrt(np.mean((pred - true) ** 2))


def mape(pred, true):
    return np.mean(np.abs((pred - true) / true)) * 100


def load_rnn(ckpt_path):
    """Loads either the LSTM or GRU checkpoint - cell_type in the checkpoint
    decides which, same RNNForecaster class either way (see lstm_model.py)."""
    ckpt = torch.load(ckpt_path, weights_only=False)
    model = RNNForecaster(
        n_features=ckpt["n_features"], hidden_size=ckpt["hidden_size"],
        num_layers=ckpt["num_layers"], horizon=ckpt["horizon"], cell_type=ckpt["cell_type"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["y_mean"], ckpt["y_std"]


def load_transformer(ckpt_path):
    ckpt = torch.load(ckpt_path, weights_only=False)
    model = TransformerForecaster(
        n_features=ckpt["n_features"], d_model=ckpt["d_model"], nhead=ckpt["nhead"],
        num_layers=ckpt["num_layers"], dim_feedforward=ckpt["dim_feedforward"],
        horizon=ckpt["horizon"], max_len=ckpt["max_len"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["y_mean"], ckpt["y_std"]


def predict_torch_model(model, X, y_mean, y_std):
    with torch.no_grad():
        pred_n = model(torch.from_numpy(X)).numpy()
    return pred_n * y_std + y_mean  # back to raw MW


def evaluate_all(horizon=24):
    df = load_data()
    df = engineer_features(df)
    train_df, val_df, test_df = chronological_split(df)
    scaler = fit_scaler(train_df)

    X_train, y_train, ts_train = make_sequences(train_df, scaler, horizon=horizon)
    X_test, y_test, ts_test = make_sequences(test_df, scaler, horizon=horizon)
    year_range = range(df.index.year.min(), df.index.year.max() + 1)

    persistence_preds = naive_persistence_predict(df, ts_test, horizon=horizon)

    linear_model = train_linear_baseline(X_train, y_train, ts_train, horizon, year_range)
    linear_preds = predict_linear_baseline(linear_model, X_test, ts_test, horizon, year_range)

    lstm_model, y_mean, y_std = load_rnn(os.path.join(RESULTS_DIR, "lstm_best.pt"))
    lstm_preds = predict_torch_model(lstm_model, X_test, y_mean, y_std)

    gru_model, y_mean, y_std = load_rnn(os.path.join(RESULTS_DIR, "gru_best.pt"))
    gru_preds = predict_torch_model(gru_model, X_test, y_mean, y_std)

    tf_model, y_mean, y_std = load_transformer(os.path.join(RESULTS_DIR, "transformer_best.pt"))
    tf_preds = predict_torch_model(tf_model, X_test, y_mean, y_std)

    results = {
        "Naive persistence": persistence_preds,
        "Linear regression": linear_preds,
        "LSTM": lstm_preds,
        "GRU": gru_preds,
        "Transformer": tf_preds,
    }

    print(f"{'Model':<20}{'MAE (MW)':>12}{'RMSE (MW)':>12}{'MAPE (%)':>12}")
    print("-" * 56)
    for name, preds in results.items():
        print(f"{name:<20}{mae(preds, y_test):>12.1f}{rmse(preds, y_test):>12.1f}{mape(preds, y_test):>12.2f}")

    # Error growth by horizon step (1h ahead vs 24h ahead) - the key diagnostic
    print(f"\n{'Model':<20}{'h=1 MAE':>12}{'h=12 MAE':>12}{'h=24 MAE':>12}")
    print("-" * 56)
    for name, preds in results.items():
        h1 = mae(preds[:, 0], y_test[:, 0])
        h12 = mae(preds[:, 11], y_test[:, 11])
        h24 = mae(preds[:, 23], y_test[:, 23])
        print(f"{name:<20}{h1:>12.1f}{h12:>12.1f}{h24:>12.1f}")

    # Plot 1: error vs horizon step, all models overlaid
    plt.figure(figsize=(8, 4.5))
    for name, preds in results.items():
        per_step_mae = [mae(preds[:, h], y_test[:, h]) for h in range(horizon)]
        plt.plot(range(1, horizon + 1), per_step_mae, marker="o", markersize=3, label=name)
    plt.xlabel("Hours ahead")
    plt.ylabel("MAE (MW)")
    plt.title("Forecast error growth by horizon")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "error_by_horizon.png"), dpi=120)

    # Plot 2: one sample forecast window, actual vs all predictions
    sample_i = len(X_test) // 2
    plt.figure(figsize=(9, 4.5))
    plt.plot(range(horizon), y_test[sample_i], "k-", linewidth=2, label="Actual")
    for name, preds in results.items():
        plt.plot(range(horizon), preds[sample_i], "--", label=name)
    plt.xlabel("Hours ahead")
    plt.ylabel("Load (MW)")
    plt.title(f"Sample 24h forecast starting {ts_test[sample_i]}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "sample_forecast.png"), dpi=120)

    print(f"\nSaved plots to {RESULTS_DIR}/error_by_horizon.png and {RESULTS_DIR}/sample_forecast.png")


if __name__ == "__main__":
    evaluate_all()