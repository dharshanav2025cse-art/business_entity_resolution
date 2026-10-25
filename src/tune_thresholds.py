from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import confusion_matrix, fbeta_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_FILE = PROJECT_ROOT / "dataset" / "features" / "train_features.tsv.gz"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_CONFIG = MODEL_DIR / "selected_model_config.json"
OUTPUT_RESULTS = MODEL_DIR / "threshold_tuning_results.json"

NEGATIVES_PER_POSITIVE = 10
RANDOM_STATE = 42
VALIDATION_SIZE = 0.20
THRESHOLD_GRID = np.linspace(0.05, 0.95, 181)

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
    "candidate_country_missing",
]

MODEL_FILES = {
    "Logistic Regression": MODEL_DIR / "logistic_regression_model.joblib",
    "Random Forest": MODEL_DIR / "random_forest_model.joblib",
    "Gradient Boosting": MODEL_DIR / "gradient_boosting_model.joblib",
}


def load_model_dataset() -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    df = pd.read_csv(FEATURE_FILE, sep="\t")

    positive_df = df[df["label"] == 1].copy()
    negative_df = df[df["label"] == 0].copy()

    negative_sample_size = min(len(negative_df), len(positive_df) * NEGATIVES_PER_POSITIVE)
    negative_sample = negative_df.sample(n=negative_sample_size, random_state=RANDOM_STATE)

    model_df = pd.concat([positive_df, negative_sample], ignore_index=True)
    model_df = model_df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    X = model_df[FEATURE_COLUMNS].astype(np.float32)
    y = model_df["label"].astype(np.int8)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    return model_df, X_train, X_valid, y_train, y_valid


def score_thresholds(y_true: pd.Series, probabilities: np.ndarray) -> tuple[float, float, float, float, int, int, int, int]:
    best = {
        "threshold": 0.5,
        "precision": -1.0,
        "recall": -1.0,
        "f05": -1.0,
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "tn": 0,
    }

    for threshold in THRESHOLD_GRID:
        predictions = (probabilities >= threshold).astype(np.int8)
        precision = precision_score(y_true, predictions, zero_division=0)
        recall = recall_score(y_true, predictions, zero_division=0)
        f05 = fbeta_score(y_true, predictions, beta=0.5, zero_division=0)

        tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()

        candidate = {
            "threshold": float(threshold),
            "precision": float(precision),
            "recall": float(recall),
            "f05": float(f05),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        }

        if candidate["f05"] > best["f05"]:
            best = candidate
        elif abs(candidate["f05"] - best["f05"]) < 1e-12 and candidate["precision"] > best["precision"]:
            best = candidate

    return (
        best["threshold"],
        best["precision"],
        best["recall"],
        best["f05"],
        best["tn"],
        best["fp"],
        best["fn"],
        best["tp"],
    )


def main() -> None:
    _, _, X_valid, _, y_valid = load_model_dataset()

    results = []
    for model_name, model_path in MODEL_FILES.items():
        model = load(model_path)
        probabilities = model.predict_proba(X_valid)[:, 1]
        threshold, precision, recall, f05, tn, fp, fn, tp = score_thresholds(y_valid, probabilities)

        result = {
            "model": model_name,
            "path": str(model_path),
            "best_threshold": float(threshold),
            "precision": float(precision),
            "recall": float(recall),
            "f05": float(f05),
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            },
        }
        results.append(result)

    selected = max(results, key=lambda item: (item["f05"], item["precision"]))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_RESULTS.write_text(json.dumps(results, indent=2), encoding="utf-8")

    config = {
        "selected_model": selected["model"],
        "selected_model_path": selected["path"],
        "selected_threshold": float(selected["best_threshold"]),
        "selection_metric": "validation_f05",
        "selected_f05": float(selected["f05"]),
        "all_results": results,
    }
    OUTPUT_CONFIG.write_text(json.dumps(config, indent=2), encoding="utf-8")

    print("=" * 80)
    print("THRESHOLD TUNING: EXISTING SAVED MODELS")
    print("=" * 80)
    for item in results:
        print(
            f"{item['model']:>22} | threshold={item['best_threshold']:.2f} | "
            f"precision={item['precision']:.6f} | recall={item['recall']:.6f} | "
            f"F0.5={item['f05']:.6f}"
        )

    print("=" * 80)
    print("SELECTED MODEL")
    print(f"Model: {selected['model']}")
    print(f"Threshold: {selected['best_threshold']:.2f}")
    print(f"Validation F0.5: {selected['f05']:.6f}")
    print(f"Results file: {OUTPUT_RESULTS}")
    print(f"Config file: {OUTPUT_CONFIG}")
    print("=" * 80)


if __name__ == "__main__":
    main()
