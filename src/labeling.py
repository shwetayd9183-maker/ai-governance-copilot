# src/labeling.py

import numpy as np
from config import CROP_CONFIG, SEVERITY_THRESHOLDS


def compute_crop_aware_drop(df):
    """
    Compute future drop percentage using crop-specific horizons.
    Uses smoothed future 3-day average minimum.
    """

    df = df.sort_values(['Crop', 'District', 'Date']).copy()
    df['future_avg_min'] = np.nan

    for crop, config in CROP_CONFIG.items():

        horizon = config["horizon"]
        avg_window = 3

        mask = df['Crop'] == crop
        df_crop = df.loc[mask]

        if df_crop.empty:
            continue

        future_avg = (
            df_crop.groupby('District')['Modal_Price']
            .apply(
                lambda x: x.shift(-1)
                          .rolling(window=avg_window, min_periods=avg_window)
                          .mean()
                          .rolling(window=horizon - avg_window + 1, min_periods=1)
                          .min()
            )
        )

        df.loc[mask, 'future_avg_min'] = future_avg.values

    df['drop_pct'] = (
        (df['Modal_Price'] - df['future_avg_min'])
        / df['Modal_Price']
    )

    return df


def classify_severity(drop):
    """
    Convert drop percentage into 3-class severity label.
    """

    if np.isnan(drop):
        return np.nan
    elif drop < SEVERITY_THRESHOLDS["moderate"]:
        return 0  # Stable
    elif drop < SEVERITY_THRESHOLDS["severe"]:
        return 1  # Moderate
    else:
        return 2  # Severe


def apply_severity_labels(df):
    """
    Apply severity classification to dataframe.
    """

    df = compute_crop_aware_drop(df)
    df['severity_class'] = df['drop_pct'].apply(classify_severity)

    df = df.dropna(subset=['severity_class'])

    return df