"""
Trains the TransformerForecaster. Structurally the same training loop as
train_lstm.py (early stopping on val loss, normalized targets, same data
pipeline) - kept as its own file rather than merged with train_lstm.py so
each script stays self-contained and independently runnable.
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

from data_pipeline import load_data, engineer_features, chronological_split, fit_scaler, make_sequences
from transformer_model import TransformerForecaster

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def train(d_model=48, nhead=4, num_layers=1, dim_feedforward=96, horizon=24,
          epochs=40, batch_size=64, lr=1e-3, patience=6, seed=42):
    torch.manual_seed(seed)

    df = load_data()
    df = engineer_features(df)
    train_df, val_df, test_df = chronological_split(df)
    scaler = fit_scaler(train_df)

    X_train, y_train, _ = make_sequences(train_df, scaler, horizon=horizon)
    X_val, y_val, _ = make_sequences(val_df, scaler, horizon=horizon)

    y_mean, y_std = y_train.mean(), y_train.std()
    y_train_n = (y_train - y_mean) / y_std
    y_val_n = (y_val - y_mean) / y_std

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train_n))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val_n))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TransformerForecaster(
        n_features=X_train.shape[-1], d_model=d_model, nhead=nhead,
        num_layers=num_layers, dim_feedforward=dim_feedforward,
        horizon=horizon, max_len=X_train.shape[1],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": []}
    ckpt_path = os.path.join(RESULTS_DIR, "transformer_best.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_losses.append(criterion(pred, yb).item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"epoch {epoch:2d}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save({
                "model_state": model.state_dict(),
                "y_mean": y_mean, "y_std": y_std,
                "d_model": d_model, "nhead": nhead, "num_layers": num_layers,
                "dim_feedforward": dim_feedforward, "horizon": horizon,
                "n_features": X_train.shape[-1], "max_len": X_train.shape[1],
            }, ckpt_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    plt.figure(figsize=(7, 4))
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("MSE loss (normalized target)")
    plt.title("Transformer training curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "transformer_training_curve.png"), dpi=120)
    print(f"Saved best checkpoint to {ckpt_path}")
    return ckpt_path


if __name__ == "__main__":
    train()