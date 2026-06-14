"""
Human GPS Spoofing Detection Code
Updated from the old AV-GPS detector to work with the Part 2 human GPS dataset.

Expected input columns:
    session_id, gps_date, gps_time, latitude, longitude, velocity, course,
    satellites_in_view, satellites_used, hdop, label

Important:
    Do NOT use the generator/debug columns as ML features:
    attack_round_id, attack_type, attack_phase, is_generated_spoof,
    replay_source_start_index

Output:
    human_detection_outputs/plots/
    human_detection_outputs/models/
    human_detection_outputs/human_gps_predictions_*.csv
    human_detection_outputs/human_model_report_*.txt
"""

import os
import warnings
from datetime import datetime

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import PowerTransformer, StandardScaler

warnings.filterwarnings("ignore")

# ================================================================
# CONFIGURATION
# ================================================================
DATASET_FILE = "data/gps_data_spoofed_3.csv"   # generated dataset with labels 0/1
OUTPUT_DIR = "outputs"
RANDOM_STATE = 42
TEST_SIZE = 0.25
ROLLING_WINDOW = 5
# ================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "plots"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "models"), exist_ok=True)

sns.set_theme(style="whitegrid")


# -----------------------------
# Geometry helper functions
# -----------------------------
def haversine_m(lat1, lon1, lat2, lon2):
    """Distance in meters between two latitude/longitude points."""
    lat1 = np.radians(lat1.astype(float))
    lon1 = np.radians(lon1.astype(float))
    lat2 = np.radians(lat2.astype(float))
    lon2 = np.radians(lon2.astype(float))

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return 6371000 * c


def bearing_deg(lat1, lon1, lat2, lon2):
    """Initial bearing in degrees from point 1 to point 2."""
    lat1 = np.radians(lat1.astype(float))
    lat2 = np.radians(lat2.astype(float))
    dlon = np.radians(lon2.astype(float) - lon1.astype(float))

    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    brng = np.degrees(np.arctan2(x, y))
    return (brng + 360) % 360


def circular_diff_deg(a, b):
    """Smallest absolute difference between two angles in degrees."""
    diff = np.abs(a - b) % 360
    return np.minimum(diff, 360 - diff)


class HumanGPSDetector:
    def __init__(self):
        self.models = {
            "random_forest": RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features="sqrt",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                class_weight="balanced",
                bootstrap=True,
            ),
            "neural_network": MLPClassifier(
                hidden_layer_sizes=(50,),
                activation="relu",
                solver="adam",
                max_iter=80,
                random_state=RANDOM_STATE,
                early_stopping=True,
            ),
            "extra_trees": ExtraTreesClassifier(
                n_estimators=150,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features="sqrt",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                class_weight="balanced",
            ),
        }

        self.ensemble_model = None
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.normalizer = PowerTransformer(method="yeo-johnson")
        self.feature_names = []
        self.performance = {}

    # -----------------------------
    # Data loading and validation
    # -----------------------------
    def load_data(self, csv_file):
        print("=" * 80)
        print("HUMAN GPS SPOOFING DETECTOR")
        print("=" * 80)
        print(f"Loading dataset: {csv_file}")

        if not os.path.exists(csv_file):
            raise FileNotFoundError(
                f"Dataset not found: {csv_file}\n"
                f"Put the CSV file in the same folder as this script or change DATASET_FILE."
            )

        df = pd.read_csv(csv_file)
        print(f"[OK] Loaded {len(df)} rows and {len(df.columns)} columns")
        print(f"Columns: {list(df.columns)}")

        required_cols = [
            "session_id", "gps_date", "gps_time", "latitude", "longitude",
            "velocity", "course", "satellites_in_view", "satellites_used", "hdop", "label"
        ]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Label handling
        if df["label"].dtype == object:
            label_map = {"normal": 0, "legitimate": 0, "fake": 1, "spoofed": 1, "attack": 1}
            df["label_numeric"] = df["label"].str.lower().map(label_map)
        else:
            df["label_numeric"] = df["label"].astype(int)

        if df["label_numeric"].isnull().any():
            raise ValueError("Some label values could not be converted to 0/1.")

        df["label_text"] = df["label_numeric"].map({0: "normal", 1: "spoofed"})

        normal_count = int((df["label_numeric"] == 0).sum())
        spoof_count = int((df["label_numeric"] == 1).sum())
        print("\nLabel distribution:")
        print(f"  Normal:  {normal_count} ({normal_count / len(df) * 100:.2f}%)")
        print(f"  Spoofed: {spoof_count} ({spoof_count / len(df) * 100:.2f}%)")

        if spoof_count == 0 or normal_count == 0:
            raise ValueError(
                "This dataset contains only one class. For supervised training you need both normal and spoofed rows. "
                "Use the generated file human_gps_with_realistic_spoofing.csv, not the original all-normal file."
            )

        # Sort by time within session if possible
        df["timestamp"] = pd.to_datetime(
            df["gps_date"].astype(str) + " " + df["gps_time"].astype(str),
            errors="coerce"
        )
        if df["timestamp"].notna().sum() > 0:
            df = df.sort_values(["session_id", "timestamp"]).reset_index(drop=True)
        else:
            df = df.sort_values(["session_id"]).reset_index(drop=True)

        return df

    # -----------------------------
    # Human-specific feature engineering
    # -----------------------------
    def create_features(self, df):
        print("\nCreating human-movement GPS features...")
        df = df.copy()

        # Convert numeric columns safely
        numeric_cols = [
            "latitude", "longitude", "velocity", "course",
            "satellites_in_view", "satellites_used", "hdop"
        ]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Basic satellite/signal features
        df["sat_count"] = df["satellites_in_view"]
        df["sat_locks"] = df["satellites_used"]
        df["sat_ratio"] = df["sat_locks"] / (df["sat_count"] + 1e-6)
        df["sat_discrepancy"] = (df["sat_count"] - df["sat_locks"]).abs()

        # Compute time delta per session
        if "time_delta_sec" in df.columns:
            df["time_delta"] = pd.to_numeric(df["time_delta_sec"], errors="coerce")
        else:
            df["time_delta"] = df.groupby("session_id")["timestamp"].diff().dt.total_seconds()

        median_dt = df["time_delta"].median()
        if pd.isna(median_dt) or median_dt <= 0:
            median_dt = 1.0
        df["time_delta"] = df["time_delta"].fillna(median_dt).clip(lower=0.2, upper=10)

        # Previous coordinates per session
        df["prev_lat"] = df.groupby("session_id")["latitude"].shift(1)
        df["prev_lon"] = df.groupby("session_id")["longitude"].shift(1)

        first_rows = df["prev_lat"].isna() | df["prev_lon"].isna()
        df.loc[first_rows, "prev_lat"] = df.loc[first_rows, "latitude"]
        df.loc[first_rows, "prev_lon"] = df.loc[first_rows, "longitude"]

        # Movement features
        df["distance_m"] = haversine_m(df["prev_lat"], df["prev_lon"], df["latitude"], df["longitude"])
        df["coord_speed"] = df["distance_m"] / df["time_delta"]
        df["speed_residual"] = (df["velocity"] - df["coord_speed"]).abs()
        df["velocity_diff"] = df.groupby("session_id")["velocity"].diff().abs().fillna(0)
        df["acceleration"] = df["velocity_diff"] / df["time_delta"]

        # Course/bearing features
        df["bearing_from_coords"] = bearing_deg(df["prev_lat"], df["prev_lon"], df["latitude"], df["longitude"])

        # If movement is too small, bearing is unreliable. Do not punish stationary points.
        df.loc[df["distance_m"] < 0.7, "bearing_from_coords"] = np.nan

        df["course_filled"] = df["course"]
        df["course_filled"] = df["course_filled"].fillna(df["bearing_from_coords"])
        df["course_filled"] = df.groupby("session_id")["course_filled"].ffill().bfill().fillna(0)

        df["prev_course"] = df.groupby("session_id")["course_filled"].shift(1).fillna(df["course_filled"])
        df["course_change"] = circular_diff_deg(df["course_filled"], df["prev_course"])

        df["course_bearing_diff"] = circular_diff_deg(
            df["course_filled"].fillna(0),
            df["bearing_from_coords"].fillna(df["course_filled"])
        )
        df.loc[df["distance_m"] < 0.7, "course_bearing_diff"] = 0

        # HDOP changes
        df["hdop_diff"] = df.groupby("session_id")["hdop"].diff().abs().fillna(0)

        # Human motion flags
        df["is_stationary"] = (df["velocity"] < 0.25).astype(int)
        df["is_fast_human"] = (df["velocity"] > 2.8).astype(int)

        # Rolling features per session. This captures attack-window behavior.
        rolling_base_cols = [
            "velocity", "coord_speed", "speed_residual", "sat_ratio",
            "sat_discrepancy", "hdop", "course_change", "course_bearing_diff"
        ]
        for col in rolling_base_cols:
            roll = df.groupby("session_id")[col].rolling(ROLLING_WINDOW, min_periods=1)
            df[f"{col}_mean_{ROLLING_WINDOW}"] = roll.mean().reset_index(level=0, drop=True)
            df[f"{col}_std_{ROLLING_WINDOW}"] = roll.std().reset_index(level=0, drop=True).fillna(0)

        # Final feature list. Do NOT include raw label/debug attack columns.
        self.feature_names = [
            "sat_count", "sat_locks", "sat_ratio", "sat_discrepancy",
            "hdop", "hdop_diff",
            "velocity", "velocity_diff", "acceleration",
            "distance_m", "coord_speed", "speed_residual",
            "course_filled", "course_change", "course_bearing_diff",
            "is_stationary", "is_fast_human",
        ]

        for col in df.columns:
            if col.endswith(f"_mean_{ROLLING_WINDOW}") or col.endswith(f"_std_{ROLLING_WINDOW}"):
                self.feature_names.append(col)

        # Safety: remove duplicates while preserving order
        self.feature_names = list(dict.fromkeys(self.feature_names))

        print(f"[OK] Created {len(self.feature_names)} features")
        print("Main features:")
        for f in self.feature_names[:12]:
            print(f"  - {f}")
        if len(self.feature_names) > 12:
            print(f"  ... plus {len(self.feature_names) - 12} rolling/context features")

        return df

    # -----------------------------
    # Data quality and class separation
    # -----------------------------
    def analyze_data_quality(self, df):
        print("\nData quality and class separation:")
        missing = df[self.feature_names].isna().sum().sum()
        print(f"  Missing feature values before imputation: {missing}")

        important = [
            "velocity", "coord_speed", "speed_residual", "distance_m",
            "sat_count", "sat_locks", "sat_ratio", "sat_discrepancy",
            "hdop", "course_change", "course_bearing_diff"
        ]

        summary_rows = []
        for col in important:
            normal_mean = df.loc[df["label_numeric"] == 0, col].mean()
            spoof_mean = df.loc[df["label_numeric"] == 1, col].mean()
            summary_rows.append({
                "feature": col,
                "normal_mean": normal_mean,
                "spoofed_mean": spoof_mean,
                "difference": abs(normal_mean - spoof_mean),
            })

        summary = pd.DataFrame(summary_rows).sort_values("difference", ascending=False)
        print("\nTop class-separation features by mean difference:")
        print(summary.head(10).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

        summary.to_csv(os.path.join(OUTPUT_DIR, "feature_class_separation.csv"), index=False)

    # -----------------------------
    # Preprocessing
    # -----------------------------
    def fit_preprocess(self, X):
        X_imp = self.imputer.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_imp)
        X_norm = self.normalizer.fit_transform(X_scaled)
        return X_norm

    def transform_preprocess(self, X):
        X_imp = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imp)
        X_norm = self.normalizer.transform(X_scaled)
        return X_norm

    # -----------------------------
    # Training and evaluation
    # -----------------------------
    def train_and_evaluate(self, df):
        print("\n" + "=" * 80)
        print("TRAINING HUMAN GPS DETECTION MODELS")
        print("=" * 80)

        X = df[self.feature_names].copy()
        y = df["label_numeric"].copy()

        # Random stratified split: comparable to the old AV-GPS code.
        train_idx, test_idx = train_test_split(
            np.arange(len(df)),
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )

        X_train_raw = X.iloc[train_idx]
        X_test_raw = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        X_train = self.fit_preprocess(X_train_raw)
        X_test = self.transform_preprocess(X_test_raw)

        print("\nRandom stratified split:")
        print(f"  Training samples: {len(y_train)}")
        print(f"  Testing samples:  {len(y_test)}")
        print(f"  Train spoofed: {(y_train == 1).sum()} | Test spoofed: {(y_test == 1).sum()}")

        model_results = []
        for name, model in self.models.items():
            print(f"\nTraining {name.replace('_', ' ').title()}...")
            model.fit(X_train, y_train)
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)

            train_acc = accuracy_score(y_train, train_pred) * 100
            test_acc = accuracy_score(y_test, test_pred) * 100

            model_results.append({
                "Model": name,
                "Train_Acc": train_acc,
                "Test_Acc": test_acc,
            })
            print(f"  Train Accuracy: {train_acc:.2f}%")
            print(f"  Test Accuracy:  {test_acc:.2f}%")

        print("\nCreating soft-voting ensemble...")
        self.ensemble_model = VotingClassifier(
            estimators=[(name, model) for name, model in self.models.items()],
            voting="soft",
            weights=[1, 1, 1],
            n_jobs=1,
        )
        self.ensemble_model.fit(X_train, y_train)

        train_ens = self.ensemble_model.predict(X_train)
        test_ens = self.ensemble_model.predict(X_test)
        test_probs = self.ensemble_model.predict_proba(X_test)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, test_ens, average="binary", zero_division=0
        )

        self.performance = {
            "train_accuracy": accuracy_score(y_train, train_ens),
            "test_accuracy": accuracy_score(y_test, test_ens),
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }

        print("\nENSEMBLE RESULTS - Random Stratified Test:")
        print(f"  Training Accuracy: {self.performance['train_accuracy'] * 100:.2f}%")
        print(f"  Testing Accuracy:  {self.performance['test_accuracy'] * 100:.2f}%")
        print(f"  Precision:         {precision * 100:.2f}%")
        print(f"  Recall:            {recall * 100:.2f}%")
        print(f"  F1-score:          {f1 * 100:.2f}%")

        cm = confusion_matrix(y_test, test_ens)
        self.plot_confusion_matrix(cm, "random_test_set")

        print("\nClassification report:")
        print(classification_report(y_test, test_ens, target_names=["Normal", "Spoofed"], zero_division=0))

        self.plot_model_performance(model_results)
        self.plot_feature_importance(df)

        # Also evaluate a time-based split to reduce temporal leakage.
        self.evaluate_time_based_split(df)

        return train_idx, test_idx, test_ens, test_probs, model_results

    def evaluate_time_based_split(self, df):
        print("\n" + "-" * 80)
        print("TIME-BASED EVALUATION (more realistic for GPS streams)")
        print("-" * 80)

        split_point = int(len(df) * (1 - TEST_SIZE))
        train_df = df.iloc[:split_point].copy()
        test_df = df.iloc[split_point:].copy()

        y_train = train_df["label_numeric"]
        y_test = test_df["label_numeric"]

        # Need both classes in train and test.
        if y_train.nunique() < 2 or y_test.nunique() < 2:
            print("[WARNING] Time-based split skipped because train/test does not contain both classes.")
            return

        X_train = self.fit_preprocess(train_df[self.feature_names])
        X_test = self.transform_preprocess(test_df[self.feature_names])

        time_model = VotingClassifier(
            estimators=[
                ("rf", RandomForestClassifier(
                    n_estimators=100, max_depth=10, min_samples_split=5, min_samples_leaf=2,
                    max_features="sqrt", random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced"
                )),
                ("mlp", MLPClassifier(
                    hidden_layer_sizes=(50,), activation="relu", solver="adam",
                    max_iter=80, random_state=RANDOM_STATE, early_stopping=True
                )),
                ("extra", ExtraTreesClassifier(
                    n_estimators=150, max_depth=12, min_samples_split=5, min_samples_leaf=2,
                    max_features="sqrt", random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced"
                )),
            ],
            voting="soft",
            weights=[1, 1, 1],
            n_jobs=1,
        )
        time_model.fit(X_train, y_train)
        pred = time_model.predict(X_test)

        acc = accuracy_score(y_test, pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, pred, average="binary", zero_division=0
        )

        print(f"  Train rows: {len(train_df)} | Test rows: {len(test_df)}")
        print(f"  Train spoofed: {(y_train == 1).sum()} | Test spoofed: {(y_test == 1).sum()}")
        print(f"  Time-based Accuracy: {acc * 100:.2f}%")
        print(f"  Time-based Precision: {precision * 100:.2f}%")
        print(f"  Time-based Recall: {recall * 100:.2f}%")
        print(f"  Time-based F1-score: {f1 * 100:.2f}%")

        cm = confusion_matrix(y_test, pred)
        self.plot_confusion_matrix(cm, "time_based_test_set")

    # -----------------------------
    # Plots and reporting
    # -----------------------------
    def plot_confusion_matrix(self, cm, suffix):
        plt.figure(figsize=(7, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Normal", "Spoofed"],
            yticklabels=["Normal", "Spoofed"],
        )
        plt.title(f"Confusion Matrix - {suffix.replace('_', ' ').title()}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        path = os.path.join(OUTPUT_DIR, "plots", f"confusion_matrix_{suffix}.png")
        plt.tight_layout()
        plt.savefig(path, dpi=300)
        plt.close()
        print(f"  Saved plot: {path}")

    def plot_model_performance(self, model_results):
        res = pd.DataFrame(model_results).sort_values("Test_Acc", ascending=False)
        res.to_csv(os.path.join(OUTPUT_DIR, "model_performance.csv"), index=False)

        plt.figure(figsize=(10, 5))
        x = np.arange(len(res))
        width = 0.32
        plt.bar(x - width / 2, res["Train_Acc"], width, label="Train")
        plt.bar(x + width / 2, res["Test_Acc"], width, label="Test")
        plt.xticks(x, [m.replace("_", " ").title() for m in res["Model"]], rotation=10)
        plt.ylabel("Accuracy (%)")
        plt.title("Human GPS Detection Model Performance")
        plt.legend()
        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, "plots", "model_performance.png")
        plt.savefig(path, dpi=300)
        plt.close()
        print(f"  Saved plot: {path}")

    def plot_feature_importance(self, df):
        rf = self.models["random_forest"]
        if not hasattr(rf, "feature_importances_"):
            return

        imp = pd.DataFrame({
            "feature": self.feature_names,
            "importance": rf.feature_importances_,
            "importance_percent": rf.feature_importances_ * 100,
        }).sort_values("importance", ascending=False)
        imp.to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False)

        top = imp.head(15)
        plt.figure(figsize=(10, 7))
        plt.barh(top["feature"], top["importance_percent"])
        plt.gca().invert_yaxis()
        plt.xlabel("Importance (%)")
        plt.title("Top Human GPS Spoofing Detection Features")
        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, "plots", "feature_importance.png")
        plt.savefig(path, dpi=300)
        plt.close()
        print(f"  Saved plot: {path}")

    def plot_stream_behavior(self, df):
        # Helpful visualizations for the report.
        plot_cols = ["label_numeric", "velocity", "hdop", "sat_locks", "distance_m", "speed_residual"]
        for col in plot_cols:
            if col not in df.columns:
                continue
            plt.figure(figsize=(12, 4))
            plt.plot(df.index, df[col], linewidth=0.8)
            plt.title(f"{col} over time")
            plt.xlabel("Record index")
            plt.ylabel(col)
            plt.tight_layout()
            path = os.path.join(OUTPUT_DIR, "plots", f"timeline_{col}.png")
            plt.savefig(path, dpi=300)
            plt.close()

        plt.figure(figsize=(7, 7))
        normal = df[df["label_numeric"] == 0]
        spoof = df[df["label_numeric"] == 1]
        plt.scatter(normal["longitude"], normal["latitude"], s=4, alpha=0.5, label="Normal")
        plt.scatter(spoof["longitude"], spoof["latitude"], s=4, alpha=0.5, label="Spoofed")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title("Trajectory: Normal vs Spoofed")
        plt.legend()
        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, "plots", "trajectory_normal_vs_spoofed.png")
        plt.savefig(path, dpi=300)
        plt.close()
        print(f"  Saved timeline and trajectory plots in: {os.path.join(OUTPUT_DIR, 'plots')}")

    # -----------------------------
    # Predict all rows and save outputs
    # -----------------------------
    def predict_all_and_save(self, df):
        print("\nMaking predictions on the full dataset for output file...")
        X_all = self.transform_preprocess(df[self.feature_names])
        pred = self.ensemble_model.predict(X_all)
        prob = self.ensemble_model.predict_proba(X_all)

        out = df.copy()
        out["prediction_numeric"] = pred
        out["prediction"] = np.where(pred == 1, "spoofed", "normal")
        out["confidence"] = [prob[i, p] * 100 for i, p in enumerate(pred)]
        out["is_correct"] = out["prediction_numeric"] == out["label_numeric"]

        # Per-attack-type performance if generator columns exist.
        if "attack_type" in out.columns:
            per_type = (
                out.groupby("attack_type")
                .agg(
                    rows=("label_numeric", "size"),
                    actual_spoofed=("label_numeric", "sum"),
                    predicted_spoofed=("prediction_numeric", "sum"),
                    accuracy=("is_correct", "mean"),
                    avg_confidence=("confidence", "mean"),
                )
                .reset_index()
            )
            per_type["accuracy"] *= 100
            per_type.to_csv(os.path.join(OUTPUT_DIR, "performance_by_attack_type.csv"), index=False)
            print("\nPerformance by attack_type saved.")
            print(per_type.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pred_path = os.path.join(OUTPUT_DIR, f"human_gps_predictions_{timestamp}.csv")
        out.to_csv(pred_path, index=False)
        print(f"[OK] Saved predictions: {pred_path}")

        # Save model + preprocessing
        model_path = os.path.join(OUTPUT_DIR, "models", f"human_ensemble_model.pkl")
        prep_path = os.path.join(OUTPUT_DIR, "models", f"human_preprocessing.pkl")
        joblib.dump(self.ensemble_model, model_path)
        joblib.dump({
            "imputer": self.imputer,
            "scaler": self.scaler,
            "normalizer": self.normalizer,
            "feature_names": self.feature_names,
        }, prep_path)
        print(f"[OK] Saved model: {model_path}")
        print(f"[OK] Saved preprocessing: {prep_path}")

        # Text report
        report_path = os.path.join(OUTPUT_DIR, f"human_model_report_{timestamp}.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("HUMAN GPS SPOOFING DETECTION REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Dataset: {DATASET_FILE}\n")
            f.write(f"Total records: {len(df)}\n")
            f.write(f"Normal records: {(df['label_numeric'] == 0).sum()}\n")
            f.write(f"Spoofed records: {(df['label_numeric'] == 1).sum()}\n\n")
            f.write("IMPORTANT NOTE:\n")
            f.write("Generator/debug columns were not used as ML features.\n")
            f.write("Used features:\n")
            for name in self.feature_names:
                f.write(f"  - {name}\n")
            f.write("\nRandom stratified ensemble performance:\n")
            for k, v in self.performance.items():
                f.write(f"  {k}: {v * 100:.2f}%\n")
        print(f"[OK] Saved report: {report_path}")

        return out

    def run(self, csv_file):
        df = self.load_data(csv_file)
        df = self.create_features(df)
        self.analyze_data_quality(df)
        self.plot_stream_behavior(df)
        self.train_and_evaluate(df)
        self.predict_all_and_save(df)
        print("\n" + "=" * 80)
        print("DONE")
        print("=" * 80)
        print(f"Outputs saved in: {OUTPUT_DIR}")
        print(f"Final random-test accuracy: {self.performance['test_accuracy'] * 100:.2f}%")


def main():
    detector = HumanGPSDetector()
    detector.run(DATASET_FILE)


if __name__ == "__main__":
    main()
