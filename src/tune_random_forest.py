import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV
)

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. LOAD DATA
# ============================================================

DATA_PATH = "data/driver_features.csv"

df = pd.read_csv(DATA_PATH)

print("\n========================================")
print("RANDOM FOREST - HYPERPARAMETER TUNING")
print("========================================")

print("\nDataset shape:")
print(df.shape)

print("\nClass distribution:")
print(df["label"].value_counts())


# ============================================================
# 2. FEATURES
# ============================================================

features = [
    "ear",
    "mar",
    "head_horizontal",
    "head_vertical"
]

X = df[features]
y = df["label"]


# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# 4. BASE RANDOM FOREST
# ============================================================

baseline_model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

print("\nTraining baseline Random Forest...")

baseline_model.fit(
    X_train,
    y_train
)

baseline_pred = baseline_model.predict(
    X_test
)


# ============================================================
# 5. BASELINE METRICS
# ============================================================

baseline_accuracy = accuracy_score(
    y_test,
    baseline_pred
)

baseline_balanced_accuracy = balanced_accuracy_score(
    y_test,
    baseline_pred
)

baseline_f1 = f1_score(
    y_test,
    baseline_pred,
    average="macro"
)


print("\n========================================")
print("BASELINE RANDOM FOREST")
print("========================================")

print(
    f"\nAccuracy: "
    f"{baseline_accuracy:.4f}"
)

print(
    f"Balanced Accuracy: "
    f"{baseline_balanced_accuracy:.4f}"
)

print(
    f"Macro F1: "
    f"{baseline_f1:.4f}"
)


# ============================================================
# 6. HYPERPARAMETER SEARCH SPACE
# ============================================================

parameter_grid = {

    "n_estimators": [
        100,
        200,
        300,
        500
    ],

    "max_depth": [
        None,
        5,
        10,
        15,
        20,
        30
    ],

    "min_samples_split": [
        2,
        5,
        10
    ],

    "min_samples_leaf": [
        1,
        2,
        4
    ],

    "max_features": [
        "sqrt",
        "log2",
        None
    ]
}


# ============================================================
# 7. BASE ESTIMATOR
# ============================================================

rf = RandomForestClassifier(
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)


# ============================================================
# 8. RANDOMIZED SEARCH
# ============================================================

search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=parameter_grid,
    n_iter=20,
    scoring="f1_macro",
    cv=5,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

print("\n========================================")
print("STARTING HYPERPARAMETER SEARCH")
print("========================================")

search.fit(
    X_train,
    y_train
)


# ============================================================
# 9. BEST PARAMETERS
# ============================================================

print("\n========================================")
print("BEST PARAMETERS")
print("========================================")

print(
    search.best_params_
)

print(
    f"\nBest CV Macro F1: "
    f"{search.best_score_:.4f}"
)


# ============================================================
# 10. BEST MODEL
# ============================================================

best_model = search.best_estimator_

best_pred = best_model.predict(
    X_test
)


# ============================================================
# 11. FINAL METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    best_pred
)

balanced_accuracy = balanced_accuracy_score(
    y_test,
    best_pred
)

precision = precision_score(
    y_test,
    best_pred,
    average="macro",
    zero_division=0
)

recall = recall_score(
    y_test,
    best_pred,
    average="macro",
    zero_division=0
)

f1 = f1_score(
    y_test,
    best_pred,
    average="macro",
    zero_division=0
)


# ============================================================
# 12. RESULTS
# ============================================================

print("\n========================================")
print("TUNED RANDOM FOREST PERFORMANCE")
print("========================================")

print(
    f"\nAccuracy: "
    f"{accuracy:.4f}"
)

print(
    f"Balanced Accuracy: "
    f"{balanced_accuracy:.4f}"
)

print(
    f"Macro Precision: "
    f"{precision:.4f}"
)

print(
    f"Macro Recall: "
    f"{recall:.4f}"
)

print(
    f"Macro F1: "
    f"{f1:.4f}"
)


# ============================================================
# 13. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        best_pred,
        digits=4,
        zero_division=0
    )
)


# ============================================================
# 14. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    best_pred,
    labels=[
        "ALERT",
        "DROWSY",
        "DISTRACTED"
    ]
)

print("\nConfusion Matrix:")

print(cm)


# ============================================================
# 15. BASELINE VS TUNED
# ============================================================

print("\n========================================")
print("BASELINE VS TUNED")
print("========================================")

print(
    f"\nBaseline Accuracy: "
    f"{baseline_accuracy:.4f}"
)

print(
    f"Tuned Accuracy:    "
    f"{accuracy:.4f}"
)

print(
    f"\nBaseline Balanced Accuracy: "
    f"{baseline_balanced_accuracy:.4f}"
)

print(
    f"Tuned Balanced Accuracy:    "
    f"{balanced_accuracy:.4f}"
)

print(
    f"\nBaseline Macro F1: "
    f"{baseline_f1:.4f}"
)

print(
    f"Tuned Macro F1:    "
    f"{f1:.4f}"
)


# ============================================================
# 16. SAVE TUNED MODEL
# ============================================================

import joblib

MODEL_PATH = (
    "models/"
    "driver_safety_random_forest_tuned.pkl"
)

joblib.dump(
    best_model,
    MODEL_PATH
)

print("\n========================================")
print("TUNED MODEL SAVED")
print("========================================")

print(
    f"\nSaved to: {MODEL_PATH}"
)