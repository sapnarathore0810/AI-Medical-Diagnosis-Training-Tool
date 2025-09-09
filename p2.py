import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib


# ---------------------------
# Helper functions for input
# ---------------------------
def get_input_float(label):
    return float(st.text_input(label, "0"))

def get_input_int(label):
    return int(st.text_input(label, "0"))

# ---------------------------
# Ensure models folder exists
# ---------------------------
os.makedirs("models", exist_ok=True)

# ---------------------------
# Load dataset
# ---------------------------
df = pd.read_csv("bp.csv")

# Drop Patient_Number column
if "Patient_Number" in df.columns:
    df = df.drop(columns=["Patient_Number"])

# Handle missing values
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].mean())

# Features & target
X = df.drop(columns=["Blood_Pressure_Abnormality"])
y = df["Blood_Pressure_Abnormality"]

# One-hot encode categorical columns
X = pd.get_dummies(X, drop_first=True)

# Save feature names for later
feature_cols = X.columns

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# --- Save model, scaler, and feature columns ---
joblib.dump(model, "models/bp_model.pkl")
joblib.dump(scaler, "models/bp_scaler.pkl")
joblib.dump(feature_cols, "models/bp_features.pkl")

print("✅ BP model, scaler, and feature columns saved!")