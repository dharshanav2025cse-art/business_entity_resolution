
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    fbeta_score,
    average_precision_score,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
from joblib import dump


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_FILE = (
    PROJECT_ROOT
    / "dataset"
    / "features"
    / "train_features.tsv.gz"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_FILE = (
    MODEL_DIR
    / "random_forest_model.joblib"
)

RESULT_FILE = (
    MODEL_DIR
    / "model2_results.txt"
)


# ============================================================
# SETTINGS
# ============================================================

NEGATIVES_PER_POSITIVE = 10

RANDOM_STATE = 42

VALIDATION_SIZE = 0.20


# ============================================================
# START
# ============================================================

print("=" * 70)
print("BUSINESS ENTITY RESOLUTION")
print("MODEL 2 - RANDOM FOREST")
print("=" * 70)


# ============================================================
# LOAD FEATURES
# ============================================================

print("\nLoading feature dataset...")

df = pd.read_csv(
    FEATURE_FILE,
    sep="\t"
)


print(
    f"Total feature rows: {len(df):,}"
)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

positive_df = df[
    df["label"] == 1
].copy()


negative_df = df[
    df["label"] == 0
].copy()


print(
    f"Positive pairs: {len(positive_df):,}"
)

print(
    f"Negative pairs: {len(negative_df):,}"
)


# ============================================================
# SAMPLE NEGATIVES
# ============================================================

negative_sample_size = min(
    len(negative_df),
    len(positive_df)
    * NEGATIVES_PER_POSITIVE
)


print(
    "\nSampling negative examples..."
)

print(
    f"Negative examples requested: "
    f"{negative_sample_size:,}"
)


negative_sample = negative_df.sample(
    n=negative_sample_size,
    random_state=RANDOM_STATE
)


# ============================================================
# BUILD MODEL DATASET
# ============================================================

model_df = pd.concat(
    [
        positive_df,
        negative_sample
    ],
    ignore_index=True
)


model_df = model_df.sample(
    frac=1.0,
    random_state=RANDOM_STATE
).reset_index(
    drop=True
)


print(
    f"\nModel dataset size: "
    f"{len(model_df):,}"
)


print(
    f"Positive examples: "
    f"{(model_df['label'] == 1).sum():,}"
)


print(
    f"Negative examples: "
    f"{(model_df['label'] == 0).sum():,}"
)


# ============================================================
# FEATURES
# ============================================================

FEATURE_COLUMNS = [

    "name_exact",
    "name_token_exact",
    "name_jaccard",
    "name_length_difference",
    "name_length_ratio",
    "name_tfidf",

    "address_exact",
    "address_token_exact",
    "address_jaccard",
    "address_length_difference",
    "address_length_ratio",
    "address_tfidf",

    "country_exact",

    "source1_name_missing",
    "candidate_name_missing",

    "source1_address_missing",
    "candidate_address_missing",

    "source1_country_missing",
    "candidate_country_missing"
]


# ============================================================
# VERIFY FEATURES
# ============================================================

missing_features = [
    feature
    for feature in FEATURE_COLUMNS
    if feature not in model_df.columns
]


if missing_features:

    raise ValueError(
        "Missing feature columns: "
        + ", ".join(missing_features)
    )


# ============================================================
# X / Y
# ============================================================

X = model_df[
    FEATURE_COLUMNS
].astype(
    np.float32
)


y = model_df[
    "label"
].astype(
    np.int8
)


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

print(
    "\nCreating train/validation split..."
)


X_train, X_valid, y_train, y_valid = (
    train_test_split(
        X,
        y,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
)


print(
    f"Training rows: "
    f"{len(X_train):,}"
)


print(
    f"Validation rows: "
    f"{len(X_valid):,}"
)


# ============================================================
# RANDOM FOREST
# ============================================================

print(
    "\nTraining Random Forest..."
)

print(
    "This may take a few minutes."
)


model = RandomForestClassifier(

    n_estimators=200,

    max_depth=16,

    min_samples_leaf=2,

    max_features="sqrt",

    class_weight="balanced",

    n_jobs=-1,

    random_state=RANDOM_STATE
)


model.fit(
    X_train,
    y_train
)


print(
    "Training complete."
)


# ============================================================
# PREDICTIONS
# ============================================================

print(
    "\nGenerating validation predictions..."
)


validation_probability = (
    model.predict_proba(
        X_valid
    )[:, 1]
)


# ============================================================
# DEFAULT THRESHOLD
# ============================================================

threshold = 0.50


validation_prediction = (
    validation_probability
    >= threshold
).astype(
    np.int8
)


# ============================================================
# METRICS
# ============================================================

precision = precision_score(
    y_valid,
    validation_prediction,
    zero_division=0
)


recall = recall_score(
    y_valid,
    validation_prediction,
    zero_division=0
)


f05 = fbeta_score(
    y_valid,
    validation_prediction,
    beta=0.5,
    zero_division=0
)


pr_auc = average_precision_score(
    y_valid,
    validation_probability
)


tn, fp, fn, tp = confusion_matrix(
    y_valid,
    validation_prediction
).ravel()


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("MODEL 2 RESULTS")
print("=" * 70)


print(
    f"\nThreshold : {threshold:.2f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F0.5      : {f05:.4f}"
)

print(
    f"PR-AUC    : {pr_auc:.4f}"
)


print("\nConfusion Matrix:")

print(
    f"True Negatives  : {tn:,}"
)

print(
    f"False Positives : {fp:,}"
)

print(
    f"False Negatives : {fn:,}"
)

print(
    f"True Positives   : {tp:,}"
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print(
    "\nFeature importance:"
)


importance_table = pd.DataFrame(
    {
        "feature": FEATURE_COLUMNS,
        "importance": model.feature_importances_
    }
).sort_values(
    "importance",
    ascending=False
)


print(
    importance_table.to_string(
        index=False
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

print(
    "\nSaving model..."
)


dump(
    model,
    MODEL_FILE
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_text = f"""
BUSINESS ENTITY RESOLUTION
MODEL 2 - RANDOM FOREST
========================================

Original feature rows:
{len(df):,}

Positive pairs:
{len(positive_df):,}

Negative pairs:
{len(negative_df):,}

Negative examples used:
{len(negative_sample):,}

Model dataset:
{len(model_df):,}

Training rows:
{len(X_train):,}

Validation rows:
{len(X_valid):,}

Random Forest settings:
n_estimators = 200
max_depth = 16
min_samples_leaf = 2
max_features = sqrt
class_weight = balanced

Threshold:
{threshold:.2f}

Precision:
{precision:.6f}

Recall:
{recall:.6f}

F0.5:
{f05:.6f}

PR-AUC:
{pr_auc:.6f}

Confusion Matrix:
True Negatives  = {tn:,}
False Positives = {fp:,}
False Negatives = {fn:,}
True Positives  = {tp:,}

FEATURE IMPORTANCE
========================================

{importance_table.to_string(index=False)}
"""


with open(
    RESULT_FILE,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        results_text.strip()
    )


# ============================================================
# DONE
# ============================================================

print(
    f"\nModel saved to:"
)

print(
    MODEL_FILE
)


print(
    f"\nResults saved to:"
)

print(
    RESULT_FILE
)


print("\nModel 2 complete.")

