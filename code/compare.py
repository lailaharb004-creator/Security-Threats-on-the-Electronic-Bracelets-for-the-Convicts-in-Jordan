import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

true_file = "data/gps_data_spoofed_3000.csv"
pred_file = "outputs/unlabeled_predictions_20260612_061608.csv"

true_df = pd.read_csv(true_file)
pred_df = pd.read_csv(pred_file)

# إذا اللابل أرقام 0 و 1
true_df["label_numeric"] = true_df["label"].astype(int)

# نفس ترتيب كود predict_unlabeled.py
true_df["timestamp"] = pd.to_datetime(
    true_df["gps_date"].astype(str) + " " + true_df["gps_time"].astype(str),
    errors="coerce"
)

true_df = true_df.sort_values(["session_id", "timestamp"]).reset_index(drop=True)
pred_df = pred_df.reset_index(drop=True)

y_true = true_df["label_numeric"]
y_pred = pred_df["prediction_numeric"].astype(int)

accuracy = accuracy_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=["Normal", "Spoofed"]
))