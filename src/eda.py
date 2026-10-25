
import pandas as pd
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


# ============================================================
# LOAD DATA
# ============================================================

source1 = pd.read_csv(
    CLEANED_TRAIN_DIR / "train_source1.tsv",
    sep="\t"
)

source2 = pd.read_csv(
    CLEANED_TRAIN_DIR / "train_source2.tsv",
    sep="\t"
)

source3 = pd.read_csv(
    CLEANED_TRAIN_DIR / "train_source3.tsv",
    sep="\t"
)

ground_truth = pd.read_csv(
    CLEANED_TRAIN_DIR / "train_ground_truth.tsv",
    sep="\t"
)


# ============================================================
# BASIC INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("BUSINESS ENTITY RESOLUTION - EDA")
print("=" * 70)


print("\nDATASET SIZES")
print("-" * 70)

print(f"Source 1 : {source1.shape[0]:,} rows")
print(f"Source 2 : {source2.shape[0]:,} rows")
print(f"Source 3 : {source3.shape[0]:,} rows")
print(f"Ground Truth : {ground_truth.shape[0]:,} rows")


# ============================================================
# COLUMNS
# ============================================================

print("\nCOLUMNS")
print("-" * 70)

print("Source 1:")
print(source1.columns.tolist())

print("\nSource 2:")
print(source2.columns.tolist())

print("\nSource 3:")
print(source3.columns.tolist())

print("\nGround Truth:")
print(ground_truth.columns.tolist())


# ============================================================
# SAMPLE RECORDS
# ============================================================

print("\nSOURCE 1 SAMPLE")
print("-" * 70)

print(
    source1[
        [
            "entity_id",
            "business_name",
            "business_address",
            "country"
        ]
    ].head(5).to_string(index=False)
)


print("\nSOURCE 2 SAMPLE")
print("-" * 70)

print(
    source2[
        [
            "entity_id",
            "business_name",
            "business_address",
            "country"
        ]
    ].head(5).to_string(index=False)
)


print("\nSOURCE 3 SAMPLE")
print("-" * 70)

print(
    source3[
        [
            "entity_id",
            "business_name",
            "business_address",
            "country"
        ]
    ].head(5).to_string(index=False)
)


# ============================================================
# MISSING VALUES
# ============================================================

def missing_report(df, name):

    print(f"\n{name} - MISSING VALUES")
    print("-" * 70)

    columns = [
        "entity_id",
        "business_name",
        "business_address",
        "country"
    ]

    for column in columns:

        missing = df[column].isna().sum()

        percentage = (
            missing / len(df) * 100
        )

        print(
            f"{column:25s} "
            f"{missing:6,} "
            f"({percentage:.2f}%)"
        )


missing_report(source1, "SOURCE 1")
missing_report(source2, "SOURCE 2")
missing_report(source3, "SOURCE 3")


# ============================================================
# DUPLICATE IDs
# ============================================================

print("\nDUPLICATE ENTITY IDs")
print("-" * 70)

print(
    "Source 1:",
    source1["entity_id"].duplicated().sum()
)

print(
    "Source 2:",
    source2["entity_id"].duplicated().sum()
)

print(
    "Source 3:",
    source3["entity_id"].duplicated().sum()
)


# ============================================================
# COUNTRY DISTRIBUTION
# ============================================================

def country_report(df, name):

    print(f"\n{name} - COUNTRY DISTRIBUTION")
    print("-" * 70)

    print(
        df["country_clean"]
        .value_counts(dropna=False)
        .to_string()
    )


country_report(source1, "SOURCE 1")
country_report(source2, "SOURCE 2")
country_report(source3, "SOURCE 3")


# ============================================================
# NAME LENGTH
# ============================================================

def text_statistics(df, name):

    print(f"\n{name} - TEXT STATISTICS")
    print("-" * 70)

    print(
        "Average name length:",
        round(df["name_length"].mean(), 2)
    )

    print(
        "Minimum name length:",
        df["name_length"].min()
    )

    print(
        "Maximum name length:",
        df["name_length"].max()
    )

    print(
        "Average address length:",
        round(df["address_length"].mean(), 2)
    )

    print(
        "Average name token count:",
        round(df["name_token_count"].mean(), 2)
    )

    print(
        "Average address token count:",
        round(df["address_token_count"].mean(), 2)
    )


text_statistics(source1, "SOURCE 1")
text_statistics(source2, "SOURCE 2")
text_statistics(source3, "SOURCE 3")


# ============================================================
# GROUND TRUTH ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("GROUND TRUTH ANALYSIS")
print("=" * 70)


def count_matches(value):

    if pd.isna(value):
        return 0

    value = str(value).strip()

    if value == "":
        return 0

    return len(
        [
            x
            for x in value.split(",")
            if x.strip()
        ]
    )


ground_truth["match_count"] = (
    ground_truth["matched_entity_ids"]
    .apply(count_matches)
)


# ============================================================
# MATCH DISTRIBUTION
# ============================================================

print("\nMATCH COUNT DISTRIBUTION")
print("-" * 70)

print(
    ground_truth["match_count"]
    .value_counts()
    .sort_index()
    .to_string()
)


# ============================================================
# ZERO MATCH
# ============================================================

zero_matches = (
    ground_truth["match_count"] == 0
).sum()

print("\nSOURCE 1 ENTITIES WITH ZERO MATCHES")
print("-" * 70)

print(zero_matches)


# ============================================================
# ONE MATCH
# ============================================================

one_match = (
    ground_truth["match_count"] == 1
).sum()

print("\nSOURCE 1 ENTITIES WITH EXACTLY ONE MATCH")
print("-" * 70)

print(one_match)


# ============================================================
# MULTIPLE MATCHES
# ============================================================

multiple_matches = (
    ground_truth["match_count"] > 1
).sum()

print("\nSOURCE 1 ENTITIES WITH MULTIPLE MATCHES")
print("-" * 70)

print(multiple_matches)


# ============================================================
# MATCH STATISTICS
# ============================================================

print("\nMATCH STATISTICS")
print("-" * 70)

print(
    "Average matches per Source 1:",
    round(
        ground_truth["match_count"].mean(),
        3
    )
)

print(
    "Maximum matches for one Source 1:",
    ground_truth["match_count"].max()
)


# ============================================================
# EXAMPLES OF ZERO MATCHES
# ============================================================

print("\nEXAMPLES - ZERO MATCHES")
print("-" * 70)

zero_ids = ground_truth.loc[
    ground_truth["match_count"] == 0,
    "source1_entity_id"
].head(5)

print(
    source1[
        source1["entity_id"].isin(zero_ids)
    ][
        [
            "entity_id",
            "business_name",
            "business_address",
            "country"
        ]
    ].to_string(index=False)
)


# ============================================================
# EXAMPLES OF MULTIPLE MATCHES
# ============================================================

print("\nEXAMPLES - MULTIPLE MATCHES")
print("-" * 70)

multiple_rows = ground_truth[
    ground_truth["match_count"] > 1
].head(5)

print(
    multiple_rows[
        [
            "source1_entity_id",
            "matched_entity_ids",
            "match_count"
        ]
    ].to_string(index=False)
)


# ============================================================
# TEST DATA SIZES
# ============================================================

print("\n" + "=" * 70)
print("TEST DATA SIZES")
print("=" * 70)

test_source1 = pd.read_csv(
    CLEANED_TEST_DIR / "test_source1.tsv",
    sep="\t"
)

test_source2 = pd.read_csv(
    CLEANED_TEST_DIR / "test_source2.tsv",
    sep="\t"
)

test_source3 = pd.read_csv(
    CLEANED_TEST_DIR / "test_source3.tsv",
    sep="\t"
)

print(
    f"Test Source 1 : {len(test_source1):,}"
)

print(
    f"Test Source 2 : {len(test_source2):,}"
)

print(
    f"Test Source 3 : {len(test_source3):,}"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("EDA COMPLETED")
print("=" * 70)

print("\nNext stage:")
print("1. Detailed normalization")
print("2. Candidate generation / blocking")
print("3. Feature engineering")
print("4. Model training")




