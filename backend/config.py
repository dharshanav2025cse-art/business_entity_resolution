from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "dataset"
NORMALIZED_TRAIN_DIR = DATASET_ROOT / "normalized" / "train"
NORMALIZED_TEST_DIR = DATASET_ROOT / "normalized" / "test"
TRAIN_DIR = DATASET_ROOT / "train"
TEST_DIR = DATASET_ROOT / "test"
CANDIDATES_DIR = DATASET_ROOT / "candidates"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"
MAX_DATASET_ROWS = 500
TEST_FEATURES_FILE = DATASET_ROOT / "features" / "test_features.tsv.gz"
MATCHING_RESULTS_FILE = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_PAIRS_FILE = OUTPUT_DIR / "candidate_pairs.tsv"

MODEL_PATHS = {
    "Logistic Regression": MODELS_DIR / "logistic_regression_model.joblib",
    "Random Forest": MODELS_DIR / "random_forest_model.joblib",
    "Gradient Boosting": MODELS_DIR / "gradient_boosting_model.joblib",
}

RESULT_FILES = {
    "Logistic Regression": MODELS_DIR / "model1_results.txt",
    "Random Forest": MODELS_DIR / "model2_results.txt",
    "Gradient Boosting": MODELS_DIR / "model3_results.txt",
}

SELECTED_MODEL_NAME = "Gradient Boosting"
SELECTED_THRESHOLD = 0.94
