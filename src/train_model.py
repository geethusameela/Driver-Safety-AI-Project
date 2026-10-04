import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# DRIVER SAFETY AI - MODEL TRAINING
# ============================================================

print("========================================")
print("DRIVER SAFETY AI - MODEL TRAINING")
print("========================================")


# ============================================================
# PATHS
# ============================================================

DATA_PATH = "data/driver_features.csv"
MODEL_PATH = "models/driver_safety_random_forest.pkl"
CONFUSION_MATRIX_PATH = "results/confusion_matrix.png"


# ============================================================
# CREATE REQUIRED DIRECTORIES
# ============================================================

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(DATA_PATH)

print("\nDataset shape:")
print(df.shape)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\nClass distribution:")

print(
    df["label"].value_counts()
)


# ============================================================
# FEATURES
# ============================================================

features = [
    "ear",
    "mar",
    "head_horizontal",
    "head_vertical"
]

target = "label"


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = features + [target]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    print("\nERROR: Missing columns:")

    for column in missing_columns:
        print("-", column)

    raise ValueError(
        "Required dataset columns are missing."
    )


# ============================================================
# REMOVE MISSING VALUES
# ============================================================

df = df.dropna(
    subset=required_columns
).copy()


# ============================================================
# INPUT AND TARGET
# ============================================================

X = df[features]

y = df[target]


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print(
    "\nTraining samples:",
    len(X_train)
)

print(
    "Testing samples :",
    len(X_test)
)


# ============================================================
# RANDOM FOREST MODEL
# ============================================================

print("\nTraining Random Forest...")


model = RandomForestClassifier(

    n_estimators=200,

    class_weight="balanced",

    random_state=42,

    n_jobs=-1
)


# ============================================================
# TRAIN MODEL
# ============================================================

model.fit(
    X_train,
    y_train
)


print("Training completed.")


# ============================================================
# PREDICTIONS
# ============================================================

y_pred = model.predict(
    X_test
)


# ============================================================
# PERFORMANCE METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

balanced_accuracy = balanced_accuracy_score(
    y_test,
    y_pred
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(

    y_test,
    y_pred,

    digits=4
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(

    y_test,
    y_pred,

    labels=[
        "ALERT",
        "DROWSY",
        "DISTRACTED"
    ]
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n========================================")
print("MODEL PERFORMANCE")
print("========================================")

print(
    f"\nAccuracy          : {accuracy:.4f}"
)

print(
    f"Balanced Accuracy : {balanced_accuracy:.4f}"
)


print("\nClassification Report:")

print(report)


print("Confusion Matrix:")

print(cm)


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

disp = ConfusionMatrixDisplay(

    confusion_matrix=cm,

    display_labels=[
        "ALERT",
        "DROWSY",
        "DISTRACTED"
    ]
)


fig, ax = plt.subplots(
    figsize=(7, 6)
)


disp.plot(
    ax=ax,
    cmap="Blues",
    values_format="d"
)


plt.title(
    "Driver Safety AI - Random Forest Confusion Matrix"
)


plt.tight_layout()


plt.savefig(
    CONFUSION_MATRIX_PATH,
    dpi=300,
    bbox_inches="tight"
)


plt.close(fig)


print(
    f"\nConfusion matrix saved to:"
)

print(
    CONFUSION_MATRIX_PATH
)


# ============================================================
# SAVE RANDOM FOREST MODEL
# ============================================================

joblib.dump(

    model,

    MODEL_PATH

)


# ============================================================
# VERIFY MODEL FILE
# ============================================================

if os.path.exists(MODEL_PATH):

    model_size = os.path.getsize(
        MODEL_PATH
    )

    print("\n========================================")
    print("MODEL SAVED SUCCESSFULLY")
    print("========================================")

    print(
        f"Model path : {MODEL_PATH}"
    )

    print(
        f"Model size : {model_size:,} bytes"
    )

else:

    print("\nERROR: Model file was not created.")


# ============================================================
# MODEL INFORMATION
# ============================================================

print("\n========================================")
print("MODEL INFORMATION")
print("========================================")

print(
    "Algorithm       : Random Forest"
)

print(
    "Number of trees : 200"
)

print(
    "Features used   :"
)

for feature in features:

    print(
        f"  - {feature}"
    )


print(
    "\nClasses:"
)

for class_name in model.classes_:

    print(
        f"  - {class_name}"
    )


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n========================================")
print("TRAINING COMPLETE")
print("========================================")

print(
    "\nThe baseline Random Forest model is ready"
)

print(
    "for real-time ML integration."
)

print(
    f"\nSaved model:"
)

print(
    MODEL_PATH
)

print(
    "\nYou can now run:"
)

print(
    "python src/main.py"
)