
import pandas as pd
import re
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_DIR = BASE_DIR / "dataset" / "train"
TEST_DIR = BASE_DIR / "dataset" / "test"

CLEANED_DIR = BASE_DIR / "dataset" / "cleaned"
CLEANED_TRAIN_DIR = CLEANED_DIR / "train"
CLEANED_TEST_DIR = CLEANED_DIR / "test"

# Create output folders
CLEANED_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
CLEANED_TEST_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# ADDRESS ABBREVIATIONS
# ============================================================

ADDRESS_MAP = {
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "blvd": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "hwy": "highway",
    "pkwy": "parkway",
    "ct": "court",
    "cir": "circle",
    "trl": "trail",
    "pl": "place",
    "sq": "square"
}


# ============================================================
# BUSINESS NAME CLEANING
# ============================================================

def clean_name(value):

    # Missing value
    if pd.isna(value):
        return ""

    text = str(value).strip().lower()

    # Convert & to "and"
    text = text.replace("&", " and ")

    # Remove punctuation
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Remove repeated whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# ADDRESS CLEANING
# ============================================================

def clean_address(value):

    # Missing value
    if pd.isna(value):
        return ""

    text = str(value).strip().lower()

    # Convert & to "and"
    text = text.replace("&", " and ")

    # Remove punctuation
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Remove repeated whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Expand common address abbreviations
    words = text.split()

    words = [
        ADDRESS_MAP.get(word, word)
        for word in words
    ]

    return " ".join(words)


# ============================================================
# COUNTRY CLEANING
# ============================================================

def clean_country(value):

    if pd.isna(value):
        return ""

    text = str(value).strip().lower()

    # Remove unnecessary whitespace
    text = re.sub(r"\s+", " ", text)

    return text


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

REQUIRED_COLUMNS = [
    "entity_id",
    "business_name",
    "business_address",
    "country"
]


def validate_columns(df, filename):

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"{filename} is missing columns: "
            f"{missing_columns}"
        )


# ============================================================
# CLEAN ONE DATASET
# ============================================================

def clean_dataset(input_file, output_file):

    print("=" * 70)
    print(f"Reading: {input_file}")
    print("=" * 70)

    # --------------------------------------------------------
    # Read TSV
    # --------------------------------------------------------

    df = pd.read_csv(
        input_file,
        sep="\t"
    )

    print(f"Original rows: {len(df)}")
    print(f"Original columns: {len(df.columns)}")

    # --------------------------------------------------------
    # Validate structure
    # --------------------------------------------------------

    validate_columns(
        df,
        input_file.name
    )

    # --------------------------------------------------------
    # Clean entity ID
    # --------------------------------------------------------

    df["entity_id"] = (
        df["entity_id"]
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # Check duplicate IDs
    # --------------------------------------------------------

    duplicate_ids = df["entity_id"].duplicated().sum()

    if duplicate_ids > 0:

        print(
            f"WARNING: {duplicate_ids} duplicate "
            f"entity IDs found!"
        )

    else:

        print("Duplicate entity IDs: 0")

    # --------------------------------------------------------
    # Missing-value indicators
    # --------------------------------------------------------

    df["name_missing"] = (
        df["business_name"]
        .isna()
        .astype(int)
    )

    df["address_missing"] = (
        df["business_address"]
        .isna()
        .astype(int)
    )

    df["country_missing"] = (
        df["country"]
        .isna()
        .astype(int)
    )

    # --------------------------------------------------------
    # Clean business name
    # --------------------------------------------------------

    df["business_name_clean"] = (
        df["business_name"]
        .apply(clean_name)
    )

    # --------------------------------------------------------
    # Clean address
    # --------------------------------------------------------

    df["business_address_clean"] = (
        df["business_address"]
        .apply(clean_address)
    )

    # --------------------------------------------------------
    # Clean country
    # --------------------------------------------------------

    df["country_clean"] = (
        df["country"]
        .apply(clean_country)
    )

    # --------------------------------------------------------
    # Additional useful basic features
    # --------------------------------------------------------

    df["name_length"] = (
        df["business_name_clean"]
        .str.len()
    )

    df["address_length"] = (
        df["business_address_clean"]
        .str.len()
    )

    df["name_token_count"] = (
        df["business_name_clean"]
        .str.split()
        .str.len()
    )

    df["address_token_count"] = (
        df["business_address_clean"]
        .str.split()
        .str.len()
    )

    # --------------------------------------------------------
    # Save cleaned dataset
    # --------------------------------------------------------

    df.to_csv(
        output_file,
        sep="\t",
        index=False
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print(f"Cleaned rows: {len(df)}")
    print(
        f"Missing names: "
        f"{df['name_missing'].sum()}"
    )

    print(
        f"Missing addresses: "
        f"{df['address_missing'].sum()}"
    )

    print(
        f"Missing countries: "
        f"{df['country_missing'].sum()}"
    )

    print(
        f"Saved to: {output_file}"
    )

    print()


# ============================================================
# TRAINING DATA
# ============================================================

print()
print("#" * 70)
print("CLEANING TRAINING DATA")
print("#" * 70)
print()

TRAIN_FILES = [
    "train_source1.tsv",
    "train_source2.tsv",
    "train_source3.tsv"
]

for filename in TRAIN_FILES:

    clean_dataset(
        TRAIN_DIR / filename,
        CLEANED_TRAIN_DIR / filename
    )


# ============================================================
# GROUND TRUTH
# ============================================================

print("=" * 70)
print("COPYING TRAINING GROUND TRUTH")
print("=" * 70)

ground_truth = (
    TRAIN_DIR /
    "train_ground_truth.tsv"
)

ground_truth_output = (
    CLEANED_TRAIN_DIR /
    "train_ground_truth.tsv"
)

ground_truth_df = pd.read_csv(
    ground_truth,
    sep="\t"
)

ground_truth_df.to_csv(
    ground_truth_output,
    sep="\t",
    index=False
)

print(
    f"Ground truth rows: "
    f"{len(ground_truth_df)}"
)

print(
    f"Saved to: "
    f"{ground_truth_output}"
)

print()


# ============================================================
# TEST DATA
# ============================================================

print()
print("#" * 70)
print("CLEANING TEST DATA")
print("#" * 70)
print()

TEST_FILES = [
    "test_source1.tsv",
    "test_source2.tsv",
    "test_source3.tsv"
]

for filename in TEST_FILES:

    clean_dataset(
        TEST_DIR / filename,
        CLEANED_TEST_DIR / filename
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("#" * 70)
print("DATA CLEANING COMPLETED")
print("#" * 70)

print()
print("Cleaned dataset location:")
print(CLEANED_DIR)

print()
print("Training files:")
for filename in TRAIN_FILES:
    print(
        CLEANED_TRAIN_DIR / filename
    )

print()
print("Ground truth:")
print(ground_truth_output)

print()
print("Test files:")
for filename in TEST_FILES:
    print(
        CLEANED_TEST_DIR / filename
    )

print()
print("Original datasets were NOT modified.")
print("Cleaned copies were created separately.")
print()


