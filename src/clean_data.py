import pandas as pd
import re
from pathlib import Path


# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_DIR = BASE_DIR / "dataset" / "train"
TEST_DIR = BASE_DIR / "dataset" / "test"

CLEANED_DIR = BASE_DIR / "dataset" / "cleaned"
CLEANED_TRAIN_DIR = CLEANED_DIR / "train"
CLEANED_TEST_DIR = CLEANED_DIR / "test"


# Create cleaned folders
CLEANED_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
CLEANED_TEST_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# NAME CLEANING
# --------------------------------------------------

def clean_name(value):

    if pd.isna(value):
        return ""

    text = str(value).lower()

    # & → and
    text = text.replace("&", " and ")

    # Remove punctuation
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# --------------------------------------------------
# ADDRESS CLEANING
# --------------------------------------------------

ADDRESS_MAP = {
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "blvd": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "hwy": "highway",
    "pkwy": "parkway"
}


def clean_address(value):

    if pd.isna(value):
        return ""

    text = str(value).lower()

    # & → and
    text = text.replace("&", " and ")

    # Remove punctuation
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Expand common address abbreviations
    words = text.split()

    words = [
        ADDRESS_MAP.get(word, word)
        for word in words
    ]

    return " ".join(words)


# --------------------------------------------------
# COUNTRY CLEANING
# --------------------------------------------------

def clean_country(value):

    if pd.isna(value):
        return ""

    return str(value).strip().lower()


# --------------------------------------------------
# CLEAN ONE DATASET
# --------------------------------------------------

def clean_dataset(input_file, output_file):

    print(f"Reading: {input_file}")

    df = pd.read_csv(
        input_file,
        sep="\t"
    )

    print(f"Rows: {len(df)}")

    # Missing-value indicators
    df["name_missing"] = df["business_name"].isna().astype(int)
    df["address_missing"] = df["business_address"].isna().astype(int)
    df["country_missing"] = df["country"].isna().astype(int)

    # Create cleaned versions
    df["business_name_clean"] = (
        df["business_name"]
        .apply(clean_name)
    )

    df["business_address_clean"] = (
        df["business_address"]
        .apply(clean_address)
    )

    df["country_clean"] = (
        df["country"]
        .apply(clean_country)
    )

    # Save
    df.to_csv(
        output_file,
        sep="\t",
        index=False
    )

    print(f"Saved: {output_file}")
    print()


# --------------------------------------------------
# TRAIN DATA
# --------------------------------------------------

train_files = [
    "train_source1.tsv",
    "train_source2.tsv",
    "train_source3.tsv"
]

for filename in train_files:

    clean_dataset(
        TRAIN_DIR / filename,
        CLEANED_TRAIN_DIR / filename
    )


# Ground truth is NOT cleaned
# because IDs and matching information
# must remain exactly as provided.

ground_truth = TRAIN_DIR / "train_ground_truth.tsv"
ground_truth_output = CLEANED_TRAIN_DIR / "train_ground_truth.tsv"

pd.read_csv(
    ground_truth,
    sep="\t"
).to_csv(
    ground_truth_output,
    sep="\t",
    index=False
)

print("Copied train_ground_truth.tsv")


# --------------------------------------------------
# TEST DATA
# --------------------------------------------------

test_files = [
    "test_source1.tsv",
    "test_source2.tsv",
    "test_source3.tsv"
]

for filename in test_files:

    clean_dataset(
        TEST_DIR / filename,
        CLEANED_TEST_DIR / filename
    )


print("===================================")
print("DATA CLEANING COMPLETED")
print("===================================")