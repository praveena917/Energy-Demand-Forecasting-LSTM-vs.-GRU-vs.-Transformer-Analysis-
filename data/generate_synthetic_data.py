"""
Generates a synthetic hourly electricity demand series that mimics the real
PJM Hourly Energy Consumption dataset (Kaggle: robikscube/hourly-energy-consumption,
file PJME_hourly.csv).

Why synthetic: Kaggle isn't reachable in the environment I tested this in. The
output schema (Datetime, PJME_MW) matches the real file exactly, so this file
is a drop-in placeholder. To use real data instead:
  1. Download PJME_hourly.csv from Kaggle
  2. Put it at data/PJME_hourly.csv
  3. Everything downstream (src/data_pipeline.py onward) works unchanged.

The synthetic series includes the structure real load data actually has:
  - daily cycle (low overnight, peaks morning + evening)
  - weekly cycle (lower on weekends)
  - annual cycle (winter heating + summer cooling -> dual seasonal peak)
  - slow multi-year trend
  - major US holidays depressing demand
  - autocorrelated random noise (not iid) so it isn't trivially easy
"""
import numpy as np
import pandas as pd
import holidays

RNG_SEED = 42


def generate(start="2015-01-01", end="2019-12-31 23:00", freq="h", out_path="data/PJME_hourly.csv"):
    rng = np.random.default_rng(RNG_SEED)

    idx = pd.date_range(start=start, end=end, freq=freq)
    n = len(idx)
    hour = idx.hour.values
    dow = idx.dayofweek.values          # 0 = Monday
    doy = idx.dayofyear.values
    year = idx.year.values

    base = 32000  # MW, roughly realistic PJM-region scale

    # Daily shape: two humps (morning ramp + evening peak), trough overnight
    daily = (
        3500 * np.exp(-((hour - 8) ** 2) / (2 * 3.0 ** 2))
        + 5000 * np.exp(-((hour - 19) ** 2) / (2 * 3.5 ** 2))
        - 2500 * np.exp(-((hour - 4) ** 2) / (2 * 2.5 ** 2))
    )

    # Weekly shape: weekends run ~12% lower (less commercial/industrial load)
    weekend_dip = np.where(dow >= 5, -0.12, 0.0) * base

    # Annual shape: heating in winter, cooling in summer (dual peak)
    annual = 4000 * np.cos(2 * np.pi * (doy - 15) / 365.0) + 3000 * np.cos(4 * np.pi * (doy - 200) / 365.0)

    # Slow year-over-year trend (mild load growth)
    trend = (year - year.min()) * 250

    # Holiday effect: major US holidays depress demand like a weekend
    us_holidays = holidays.US(years=range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1))
    is_holiday = np.array([d.date() in us_holidays for d in idx])
    holiday_dip = np.where(is_holiday, -0.15, 0.0) * base

    # Autocorrelated noise: an AR(1) process so noise isn't iid (more realistic)
    noise = np.zeros(n)
    eps = rng.normal(0, 350, n)
    phi = 0.85
    for t in range(1, n):
        noise[t] = phi * noise[t - 1] + eps[t]

    load = base + daily + weekend_dip + annual + trend + holiday_dip + noise
    load = np.clip(load, 15000, None)  # demand can't go below a realistic floor

    df = pd.DataFrame({"Datetime": idx, "PJME_MW": load.round(1)})
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df):,} hourly rows ({idx.min().date()} to {idx.max().date()}) to {out_path}")
    return df


if __name__ == "__main__":
    generate()