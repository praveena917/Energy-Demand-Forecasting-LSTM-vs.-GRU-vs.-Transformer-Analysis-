"""
Two baselines every forecasting project should include before touching deep
learning, because a complex model that doesn't beat these isn't worth the
compute cost:

1. Naive persistence: forecast = same hour, one week ago. This exploits daily
   AND weekly seasonality with zero training. Notoriously hard for a fancy
   model to beat cleanly on smooth seasonal data - if your LSTM only edges
   this out narrowly, that IS a legitimate finding, not a failure.
2. Linear regression on engineered features: multi-output, one prediction
   per horizon step. Uses future-known calendar features (hour/day-of-week of
   the timestamp being forecast, which is deterministic and known in advance)
   plus the most recently observed lag/rolling features. Tests how much an
   LSTM's extra complexity is actually buying you over a linear model with
   good features.
"""
import numpy as np
import pandas as pd
import holidays
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor

from data_pipeline import TARGET_COL, FEATURE_COLS


def naive_persistence_predict(df, timestamps, horizon=24):
    """For each forecast start time, predict = load at (t - 168h + h) for h in [1..horizon],
    i.e. repeat the pattern from exactly one week earlier."""
    target = df[TARGET_COL]
    preds = []
    for t in timestamps:
        base_time = t - np.timedelta64(168, "h")
        window = target.loc[base_time: base_time + np.timedelta64(horizon - 1, "h")]
        preds.append(window.values)
    return np.array(preds)


_LAG_CONTEXT_COLS = [TARGET_COL, "lag_24", "lag_168", "rolling_mean_24"]
_LAG_CONTEXT_IDX = [FEATURE_COLS.index(c) for c in _LAG_CONTEXT_COLS]


def _future_calendar_features(start_timestamps, horizon, year_range):
    """Calendar features are known in advance for any future timestamp (a linear
    model forecasting tomorrow 3pm's load is allowed to know it IS 3pm tomorrow).
    Building these explicitly is what makes this a fair comparison against the
    LSTM, which implicitly gets this information as part of its recurrent state.

    NOTE: the first version of this baseline skipped this and only used the
    LAST OBSERVED timestep's features for all 24 outputs. It lost badly to
    naive persistence (MAE 3154 vs 1073) - not because linear models are bad,
    but because the model had no way to know which hour-of-day it was even
    forecasting. Worth knowing about since it's an easy mistake to repeat.
    """
    us_holidays = holidays.US(years=year_range)
    n = len(start_timestamps)
    feats = np.zeros((n, horizon, 6), dtype=np.float32)
    for i, t0 in enumerate(start_timestamps):
        future_idx = pd.date_range(t0, periods=horizon, freq="h")
        hour = future_idx.hour.values
        dow = future_idx.dayofweek.values
        feats[i, :, 0] = np.sin(2 * np.pi * hour / 24)
        feats[i, :, 1] = np.cos(2 * np.pi * hour / 24)
        feats[i, :, 2] = np.sin(2 * np.pi * dow / 7)
        feats[i, :, 3] = np.cos(2 * np.pi * dow / 7)
        feats[i, :, 4] = (dow >= 5).astype(float)
        feats[i, :, 5] = [d.date() in us_holidays for d in future_idx]
    return feats.reshape(n, horizon * 6)


def _build_linear_inputs(X, timestamps, horizon, year_range):
    last_step_context = X[:, -1, _LAG_CONTEXT_IDX]          # (n, 4) - most recent known lags
    future_cal = _future_calendar_features(timestamps, horizon, year_range)  # (n, horizon*6)
    return np.concatenate([last_step_context, future_cal], axis=1)


def train_linear_baseline(X_train, y_train, ts_train, horizon, year_range):
    X_in = _build_linear_inputs(X_train, ts_train, horizon, year_range)
    model = MultiOutputRegressor(LinearRegression())
    model.fit(X_in, y_train)
    return model


def predict_linear_baseline(model, X, timestamps, horizon, year_range):
    X_in = _build_linear_inputs(X, timestamps, horizon, year_range)
    return model.predict(X_in)


if __name__ == "__main__":
    from data_pipeline import load_data, engineer_features, chronological_split, fit_scaler, make_sequences

    df = load_data()
    df = engineer_features(df)
    train_df, val_df, test_df = chronological_split(df)
    scaler = fit_scaler(train_df)

    X_train, y_train, ts_train = make_sequences(train_df, scaler)
    X_test, y_test, ts_test = make_sequences(test_df, scaler)
    year_range = range(df.index.year.min(), df.index.year.max() + 1)

    linear_model = train_linear_baseline(X_train, y_train, ts_train, horizon=24, year_range=year_range)
    linear_preds = predict_linear_baseline(linear_model, X_test, ts_test, horizon=24, year_range=year_range)
    persistence_preds = naive_persistence_predict(df, ts_test)

    linear_mae = np.mean(np.abs(linear_preds - y_test))
    persistence_mae = np.mean(np.abs(persistence_preds - y_test))
    print(f"Test MAE - naive persistence: {persistence_mae:.1f} MW")
    print(f"Test MAE - linear regression: {linear_mae:.1f} MW")