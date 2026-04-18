import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("data/employee_data.csv")

# DROP ID
df = df.drop("Employee_ID", axis=1)

# -----------------------------
# ENCODING
# -----------------------------
dept_encoder = LabelEncoder()
df["Department"] = dept_encoder.fit_transform(df["Department"])

target_encoder = LabelEncoder()
df["Performance"] = target_encoder.fit_transform(df["Performance"])

# -----------------------------
# BALANCE CHECK (DEBUG INFO)
# -----------------------------
print("\nClass Distribution:")
print(df["Performance"].value_counts())

# -----------------------------
# SPLIT DATA
# -----------------------------
X = df.drop("Performance", axis=1)
y = df["Performance"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y   # IMPORTANT FIX
)

# -----------------------------
# MODEL
# -----------------------------
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# EVALUATION
# -----------------------------
y_pred = model.predict(X_test)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# -----------------------------
# SAVE MODEL
# -----------------------------
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/performance_model.pkl")
joblib.dump(dept_encoder, "models/department_encoder.pkl")
joblib.dump(target_encoder, "models/target_encoder.pkl")

print("Model retrained and saved successfully")