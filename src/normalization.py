
import pandas as pd
import re
import unicodedata
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CLEANED_TRAIN_DIR = (
    BASE_DIR / "dataset" / "cleaned" / "train"
)

CLEANED_TEST_DIR = (
    BASE_DIR / "dataset" / "cleaned" / "test"
)

NORMALIZED_DIR = (
    BASE_DIR / "dataset" / "normalized"
)

NORMALIZED_TRAIN_DIR = (
    NORMALIZED_DIR / "train"
)

NORMALIZED_TEST_DIR = (
    NORMALIZED_DIR / "test"
)

NORMALIZED_TRAIN_DIR.mkdir(
    parents=True,
    exist_ok=True
)

NORMALIZED_TEST_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# COUNTRY NORMALIZATION
# ============================================================

COUNTRY_MAP = {

    # India
    "india": "india",
    "in": "india",
    "ind": "india",
    "bharat": "india",
    "republic of india": "india",

    # USA
    "usa": "usa",
    "us": "usa",
    "u.s.": "usa",
    "u.s.a.": "usa",
    "united states": "usa",
    "united states of america": "usa",

    # France
    "france": "france",
    "fr": "france",
    "fra": "france",
    "french republic": "france",
    "république française": "france"
}


def normalize_country(value):

    if pd.isna(value):
        return ""

    text = str(value).strip().lower()

    # Remove repeated whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return COUNTRY_MAP.get(
        text,
        text
    )


# ============================================================
# ACCENT NORMALIZATION
# ============================================================

def remove_accents(text):

    text = unicodedata.normalize(
        "NFKD",
        text
    )

    text = "".join(
        character
        for character in text
        if not unicodedata.combining(character)
    )

    return text


# ============================================================
# BUSINESS NAME NORMALIZATION
# ============================================================

def normalize_business_name(value):

    if pd.isna(value):
        return ""

    text = str(value).lower().strip()

    # Remove accents
    text = remove_accents(text)

    # & → and
    text = text.replace(
        "&",
        " and "
    )

    # Remove punctuation
    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    # Standardize common business terms
    replacements = {

        "private limited": "pvt ltd",
        "private ltd": "pvt ltd",
        "pvt limited": "pvt ltd",
        "pvt ltd": "pvt ltd",

        "corporation": "corp",
        "incorporated": "inc",
        "company": "co",

        "international": "intl",
        "technology": "tech",
        "technologies": "tech",

        "services": "svc",
        "solutions": "soln"
    }

    # Replace multi-word expressions first
    for old, new in replacements.items():

        text = re.sub(
            rf"\b{re.escape(old)}\b",
            new,
            text
        )

    # Normalize repeated whitespace again
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# NAME TOKEN NORMALIZATION
# ============================================================

def normalize_name_tokens(value):

    if not value:
        return ""

    tokens = value.split()

    # Remove extremely common legal suffixes
    legal_terms = {
        "pvt",
        "ltd",
        "llc",
        "inc",
        "corp",
        "co",
        "sa",
        "sas",
        "sarl"
    }

    tokens = [
        token
        for token in tokens
        if token not in legal_terms
    ]

    # Sort tokens so that word-order changes
    # have less effect during comparison.
    tokens = sorted(tokens)

    return " ".join(tokens)


# ============================================================
# ADDRESS NORMALIZATION
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


def normalize_address(value):

    if pd.isna(value):
        return ""

    text = str(value).lower().strip()

    # Remove accents
    text = remove_accents(text)

    # & → and
    text = text.replace(
        "&",
        " and "
    )

    # Remove punctuation
    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    # Expand address abbreviations
    tokens = text.split()

    tokens = [
        ADDRESS_MAP.get(
            token,
            token
        )
        for token in tokens
    ]

    return " ".join(tokens)


# ============================================================
# ADDRESS TOKEN NORMALIZATION
# ============================================================

def normalize_address_tokens(value):

    if not value:
        return ""

    tokens = value.split()

    # Remove common non-identifying address words
    removable_words = {
        "near",
        "opposite",
        "beside",
        "behind",
        "next",
        "to"
    }

    tokens = [
        token
        for token in tokens
        if token not in removable_words
    ]

    # Sort tokens for order-invariant comparison
    tokens = sorted(tokens)

    return " ".join(tokens)


# ============================================================
# PROCESS ONE FILE
# ============================================================

def normalize_dataset(
    input_file,
    output_file
):

    print("=" * 70)
    print(
        f"Processing: {input_file.name}"
    )
    print("=" * 70)

    df = pd.read_csv(
        input_file,
        sep="\t"
    )

    # --------------------------------------------------------
    # Country
    # --------------------------------------------------------

    df["country_normalized"] = (
        df["country"]
        .apply(normalize_country)
    )

    # --------------------------------------------------------
    # Business name
    # --------------------------------------------------------

    df["business_name_normalized"] = (
        df["business_name"]
        .apply(normalize_business_name)
    )

    # --------------------------------------------------------
    # Order-independent name representation
    # --------------------------------------------------------

    df["business_name_token_normalized"] = (
        df["business_name_normalized"]
        .apply(normalize_name_tokens)
    )

    # --------------------------------------------------------
    # Address
    # --------------------------------------------------------

    df["business_address_normalized"] = (
        df["business_address"]
        .apply(normalize_address)
    )

    # --------------------------------------------------------
    # Order-independent address representation
    # --------------------------------------------------------

    df["business_address_token_normalized"] = (
        df["business_address_normalized"]
        .apply(normalize_address_tokens)
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    df.to_csv(
        output_file,
        sep="\t",
        index=False
    )

    print(
        f"Rows processed: {len(df):,}"
    )

    print(
        f"Saved: {output_file}"
    )

    print()


# ============================================================
# TRAINING DATA
# ============================================================

TRAIN_FILES = [
    "train_source1.tsv",
    "train_source2.tsv",
    "train_source3.tsv"
]

print()
print("#" * 70)
print("NORMALIZING TRAINING DATA")
print("#" * 70)
print()

for filename in TRAIN_FILES:

    normalize_dataset(
        CLEANED_TRAIN_DIR / filename,
        NORMALIZED_TRAIN_DIR / filename
    )


# ============================================================
# COPY GROUND TRUTH
# ============================================================

ground_truth = (
    CLEANED_TRAIN_DIR /
    "train_ground_truth.tsv"
)

ground_truth_output = (
    NORMALIZED_TRAIN_DIR /
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
    "Ground truth copied successfully."
)

print()


# ============================================================
# TEST DATA
# ============================================================

TEST_FILES = [
    "test_source1.tsv",
    "test_source2.tsv",
    "test_source3.tsv"
]

print()
print("#" * 70)
print("NORMALIZING TEST DATA")
print("#" * 70)
print()

for filename in TEST_FILES:

    normalize_dataset(
        CLEANED_TEST_DIR / filename,
        NORMALIZED_TEST_DIR / filename
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("#" * 70)
print("NORMALIZATION COMPLETED")
print("#" * 70)

print()
print(
    "Normalized training data:"
)

for filename in TRAIN_FILES:

    print(
        NORMALIZED_TRAIN_DIR / filename
    )

print()

print(
    "Normalized test data:"
)

for filename in TEST_FILES:

    print(
        NORMALIZED_TEST_DIR / filename
    )

print()

print(
    "Original and cleaned datasets remain unchanged."
)

print()
print(
    "Next stage: Candidate Generation / Blocking"
)

