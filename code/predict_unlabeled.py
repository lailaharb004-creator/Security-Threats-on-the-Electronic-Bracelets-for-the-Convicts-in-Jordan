import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd


ROLLING_WINDOW = 5

UNLABELED_FILE = "data/gps_data_spoofed_3000_no_label.csv"

MODEL_PATH = "outputs/models/human_ensemble_model.pkl"
PREPROCESSING_PATH = "outputs/models/human_preprocessing.pkl"

OUTPUT_DIR = "outputs"


def haversine_m(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1.astype(float))
    lon1 = np.radians(lon1.astype(float))
    lat2 = np.radians(lat2.astype(float))
    lon2 = np.radians(lon2.astype(float))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return 6371000 * c


def bearing_deg(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1.astype(float))
    lat2 = np.radians(lat2.astype(float))
    dlon = np.radians(lon2.astype(float) - lon1.astype(float))

    x = np.sin(dlon) * np.cos(lat2)
    y = (
        np.cos(lat1) * np.sin(lat2)
        - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    )

    brng = np.degrees(np.arctan2(x, y))
    return (brng + 360) % 360


def circular_diff_deg(a, b):
    diff = np.abs(a - b) % 360
    return np.minimum(diff, 360 - diff)


def load_unlabeled_data(csv_file):
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"File not found: {csv_file}")

    df = pd.read_csv(csv_file)

    required_cols = [
        "session_id", "gps_date", "gps_time", "latitude", "longitude",
        "velocity", "course", "satellites_in_view", "satellites_used", "hdop"
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["timestamp"] = pd.to_datetime(
        df["gps_date"].astype(str) + " " + df["gps_time"].astype(str),
        errors="coerce"
    )

    if df["timestamp"].notna().sum() > 0:
        df = df.sort_values(["session_id", "timestamp"]).reset_index(drop=True)
    else:
        df = df.sort_values(["session_id"]).reset_index(drop=True)

    return df


def create_features(df):
    df = df.copy()

    numeric_cols = [
        "latitude", "longitude", "velocity", "course",
        "satellites_in_view", "satellites_used", "hdop"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["sat_count"] = df["satellites_in_view"]
    df["sat_locks"] = df["satellites_used"]
    df["sat_ratio"] = df["sat_locks"] / (df["sat_count"] + 1e-6)
    df["sat_discrepancy"] = (df["sat_count"] - df["sat_locks"]).abs()

    if "time_delta_sec" in df.columns:
        df["time_delta"] = pd.to_numeric(df["time_delta_sec"], errors="coerce")
    else:
        df["time_delta"] = df.groupby("session_id")["timestamp"].diff().dt.total_seconds()

    median_dt = df["time_delta"].median()
    if pd.isna(median_dt) or median_dt <= 0:
        median_dt = 1.0

    df["time_delta"] = df["time_delta"].fillna(median_dt).clip(lower=0.2, upper=10)

    df["prev_lat"] = df.groupby("session_id")["latitude"].shift(1)
    df["prev_lon"] = df.groupby("session_id")["longitude"].shift(1)

    first_rows = df["prev_lat"].isna() | df["prev_lon"].isna()
    df.loc[first_rows, "prev_lat"] = df.loc[first_rows, "latitude"]
    df.loc[first_rows, "prev_lon"] = df.loc[first_rows, "longitude"]

    df["distance_m"] = haversine_m(
        df["prev_lat"], df["prev_lon"],
        df["latitude"], df["longitude"]
    )

    df["coord_speed"] = df["distance_m"] / df["time_delta"]
    df["speed_residual"] = (df["velocity"] - df["coord_speed"]).abs()

    df["velocity_diff"] = df.groupby("session_id")["velocity"].diff().abs().fillna(0)
    df["acceleration"] = df["velocity_diff"] / df["time_delta"]

    df["bearing_from_coords"] = bearing_deg(
        df["prev_lat"], df["prev_lon"],
        df["latitude"], df["longitude"]
    )

    df.loc[df["distance_m"] < 0.7, "bearing_from_coords"] = np.nan

    df["course_filled"] = df["course"]
    df["course_filled"] = df["course_filled"].fillna(df["bearing_from_coords"])
    df["course_filled"] = (
        df.groupby("session_id")["course_filled"]
        .ffill()
        .bfill()
        .fillna(0)
    )

    df["prev_course"] = (
        df.groupby("session_id")["course_filled"]
        .shift(1)
        .fillna(df["course_filled"])
    )

    df["course_change"] = circular_diff_deg(
        df["course_filled"],
        df["prev_course"]
    )

    df["course_bearing_diff"] = circular_diff_deg(
        df["course_filled"].fillna(0),
        df["bearing_from_coords"].fillna(df["course_filled"])
    )

    df.loc[df["distance_m"] < 0.7, "course_bearing_diff"] = 0

    df["hdop_diff"] = df.groupby("session_id")["hdop"].diff().abs().fillna(0)

    df["is_stationary"] = (df["velocity"] < 0.25).astype(int)
    df["is_fast_human"] = (df["velocity"] > 2.8).astype(int)

    rolling_base_cols = [
        "velocity", "coord_speed", "speed_residual", "sat_ratio",
        "sat_discrepancy", "hdop", "course_change", "course_bearing_diff"
    ]

    for col in rolling_base_cols:
        roll = df.groupby("session_id")[col].rolling(ROLLING_WINDOW, min_periods=1)

        df[f"{col}_mean_{ROLLING_WINDOW}"] = (
            roll.mean().reset_index(level=0, drop=True)
        )

        df[f"{col}_std_{ROLLING_WINDOW}"] = (
            roll.std().reset_index(level=0, drop=True).fillna(0)
        )

    return df


def main():
    print("=" * 80)
    print("PREDICTING UNLABELED GPS DATA")
    print("=" * 80)

    model = joblib.load(MODEL_PATH)
    preprocessing = joblib.load(PREPROCESSING_PATH)

    imputer = preprocessing["imputer"]
    scaler = preprocessing["scaler"]
    normalizer = preprocessing["normalizer"]
    feature_names = preprocessing["feature_names"]

    df = load_unlabeled_data(UNLABELED_FILE)
    df = create_features(df)

    missing_features = [f for f in feature_names if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing generated features: {missing_features}")

    X = df[feature_names].copy()

    X_imp = imputer.transform(X)
    X_scaled = scaler.transform(X_imp)
    X_norm = normalizer.transform(X_scaled)

    pred = model.predict(X_norm)
    prob = model.predict_proba(X_norm)

    df["prediction_numeric"] = pred
    df["prediction"] = np.where(pred == 1, "spoofed", "normal")
    df["confidence"] = [prob[i, p] * 100 for i, p in enumerate(pred)]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(
        OUTPUT_DIR,
        f"unlabeled_predictions_{timestamp}.csv"
    )

    df.to_csv(output_path, index=False)

    print("[OK] Prediction finished")
    print(f"Saved file: {output_path}")

    print("\nPrediction counts:")
    print(df["prediction"].value_counts())


if __name__ == "__main__":
    main()