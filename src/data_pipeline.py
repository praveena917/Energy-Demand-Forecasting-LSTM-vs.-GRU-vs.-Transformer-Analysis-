"""
Data pipeline for the energy demand forecasting project.

Handles: loading, calendar/lag feature engineering, sliding-window sequence
construction for supervised multi-step forecasting, and a chronological
(never shuffled) train/val/test split with leakage-safe scaling.
"""
import os
import numpy as np
import pandas as pd
import holidays
from sklearn.preprocessing import StandardScaler

TARGET_COL = "PJME_MW"
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_PATH = os.path.join(_PROJECT_ROOT, "data", "PJME_hourly.csv")


def load_data(path=DEFAULT_DATA_PATH):
    df = pd.read_csv(path, parse_dates=["Datetime"])
    df = df.sort_values("Datetime").drop_duplicates("Datetime").set_index("Datetime")
    # Real PJM data has occasional missing hours (DST, sensor gaps) - forward
    # fill after reindexing to a complete hourly grid, exactly as you'd need
    # to handle with the real dataset.
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="h")
    df = df.reindex(full_idx)
    df[TARGET_COL] = df[TARGET_COL].ffill()
    df.index.name = "Datetime"
    return df


def engineer_features(df):
    df = df.copy()
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    us_holidays = holidays.US(years=range(df.index.year.min(), df.index.year.max() + 1))
    df["is_holiday"] = df.index.to_series().apply(lambda d: d.date() in us_holidays).astype(int)

    # Cyclical encodings so the model sees hour 23 and hour 0 as adjacent,
    # not maximally far apart (a raw integer 0-23 implies a false discontinuity)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

    # Autoregressive signal: same hour yesterday / same hour last week
    df["lag_24"] = df[TARGET_COL].shift(24)
    df["lag_168"] = df[TARGET_COL].shift(168)
    df["rolling_mean_24"] = df[TARGET_COL].shift(1).rolling(24).mean()

    df = df.dropna()
    return df


FEATURE_COLS = [
    TARGET_COL, "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "is_weekend", "is_holiday", "lag_24", "lag_168", "rolling_mean_24",
]


def chronological_split(df, train_frac=0.70, val_frac=0.15):
    """Never shuffle. Time series: the future can't be used to predict the past,
    so a random split would leak information and inflate reported accuracy."""
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def fit_scaler(train_df, cols=FEATURE_COLS):
    scaler = StandardScaler()
    scaler.fit(train_df[cols])
    return scaler


def make_sequences(df, scaler, input_window=168, horizon=24, stride=7, cols=FEATURE_COLS):
    """
    Turns a scaled feature dataframe into (X, y) arrays for supervised learning.

    X[i] = past `input_window` hours of all features (shape: input_window x n_features)
    y[i] = next `horizon` hours of the TARGET only, in ORIGINAL (unscaled) units,
           so evaluation metrics (MAE/RMSE) are directly interpretable in MW.

    stride controls how many hours to skip between consecutive training examples.
    IMPORTANT: don't use a stride that divides evenly into 24 (e.g. 24, 12, 6) -
    that makes every example start its forecast at the same hour(s) of day, which
    starves any model of hour-of-day variation across training examples. stride=7
    (coprime with 24) cycles through all 24 start hours while still subsampling
    for speed. Use stride=1 for maximum data once you have more compute (e.g. a
    Colab GPU) - it doesn't have this problem since it hits every hour anyway.
    """
    scaled = scaler.transform(df[cols])
    target_idx = cols.index(TARGET_COL)
    target_raw = df[TARGET_COL].values

    X, y, timestamps = [], [], []
    last_start = len(df) - input_window - horizon
    for start in range(0, last_start + 1, stride):
        end_in = start + input_window
        end_out = end_in + horizon
        X.append(scaled[start:end_in])
        y.append(target_raw[end_in:end_out])
        timestamps.append(df.index[end_in])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), np.array(timestamps)


if __name__ == "__main__":
    df = load_data()
    df = engineer_features(df)
    print(f"After feature engineering: {df.shape}")
    print(df[FEATURE_COLS].head())

    train_df, val_df, test_df = chronological_split(df)
    print(f"\nSplit sizes -> train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")
    print(f"train: {train_df.index.min()} to {train_df.index.max()}")
    print(f"val:   {val_df.index.min()} to {val_df.index.max()}")
    print(f"test:  {test_df.index.min()} to {test_df.index.max()}")

    scaler = fit_scaler(train_df)
    X_train, y_train, _ = make_sequences(train_df, scaler)
    print(f"\nX_train: {X_train.shape}, y_train: {y_train.shape}")