import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import joblib

# Ensure models folder exists
os.makedirs("models", exist_ok=True)

# Load dataset
data = pd.read_csv("diabetes.csv")

# Encode categorical
label_cols = ["gender", "smoking_history"]
encoders = {}
for col in label_cols:
    enc = LabelEncoder()
    data[col] = enc.fit_transform(data[col].astype(str).str.lower())
    encoders[col] = enc

# Features & target
X = data.drop("diabetes", axis=1)
y = data["diabetes"]

# Handle NaN
imputer = SimpleImputer(strategy="mean")
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42, class_weight="balanced")
model.fit(X_train, y_train)

# Save model + encoders + imputer + features
joblib.dump(model, "models/diabetes_model.pkl")
joblib.dump(encoders, "models/diabetes_encoders.pkl")
joblib.dump(imputer, "models/diabetes_imputer.pkl")
joblib.dump(X.columns, "models/diabetes_features.pkl")

print("✅ Diabetes model saved!")

joblib.dump(X.columns, "models/diabetes_features.pkl")

print("✅ Diabetes model saved!")
