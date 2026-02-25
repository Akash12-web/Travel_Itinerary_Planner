# Train Isolation Forest for Safety Transport Anomaly Detection

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load dataset
path = "safety_transport_1.csv"
df = pd.read_csv(path)

# Select features for anomaly detection
features = [
    "scheduled_travel_minutes",
    "safety_delay_minutes",
    "night_travel",
    "adjusted_reliability_score"
]

X = df[features]

# Build pipeline: scaling + Isolation Forest
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("isoforest", IsolationForest(
        n_estimators=200,
        contamination=0.18,
        random_state=42
    ))
])

# Train model
pipeline.fit(X)

# Predict anomalies
df["iforest_prediction"] = pipeline.predict(X)   # -1 = anomaly, 1 = normal
df["iforest_anomaly"] = (df["iforest_prediction"] == -1).astype(int)

# Compare with ground truth
comparison = pd.crosstab(
    df["anomaly_label"],
    df["iforest_anomaly"],
    rownames=["Actual"],
    colnames=["Predicted"]
)

print(comparison)

# -------------------------------
# ✅ SAVE TRAINED MODEL (.pkl)
# -------------------------------
model_path = "safety_transport_iforest_model.pkl"
joblib.dump(pipeline, model_path)

print(f"Model saved at: {model_path}")

# -------------------------------
# ✅ SAVE CSV RESULTS
# -------------------------------
csv_path = "safety_transport_iforest_results.csv"
df.to_csv(csv_path, index=False)

print(f"Results CSV saved at: {csv_path}")