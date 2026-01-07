
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    confusion_matrix
)

import joblib

# ===============================
# 1. LOAD DATASET
# ===============================
df = pd.read_csv("cardio_train.csv", sep=";")
print("Initial Shape:", df.shape)

# ===============================
# 2. DATA CLEANING
# ===============================

# Missing values
df = df.fillna(df.mean(numeric_only=True))
df = df.fillna(df.mode().iloc[0])

# Remove invalid BP values
df = df[(df["ap_hi"] > 50) & (df["ap_hi"] < 250)]
df = df[(df["ap_lo"] > 30) & (df["ap_lo"] < 200)]

# ===============================
# 3. FEATURE ENGINEERING
# ===============================

df["age"] = (df["age"] / 365).astype(int)
df["bmi"] = df["weight"] / ((df["height"] / 100) ** 2)

# ===============================
# 4. FEATURE SELECTION
# ===============================
features = [
    "age", "height", "weight",
    "ap_hi", "ap_lo",
    "cholesterol", "gluc",
    "smoke", "alco", "active",
    "bmi"
]

X = df[features]
y = df["cardio"]

# ===============================
# 5. TRAIN-TEST SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ===============================
# 6. PIPELINE + HYPERPARAMETER TUNING
# ===============================
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(random_state=42))
])

param_grid = {
    "model__n_estimators": [150, 200],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_split": [2, 5]
}

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

grid.fit(X_train, y_train)

model = grid.best_estimator_

print("Best Parameters:", grid.best_params_)

# ===============================
# 7. EVALUATION
# ===============================
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nROC-AUC:", roc_auc_score(y_test, y_prob))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ===============================
# 8. SAVE MODEL
# ===============================
joblib.dump(model, "model.pkl")

print("\n✅ Model saved as model.pkl") 

