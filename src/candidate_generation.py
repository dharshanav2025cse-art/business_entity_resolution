

import os
import re
import unicodedata
import pandas as pd
import numpy as np

from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NORMALIZED_TRAIN_DIR = os.path.join(
    BASE_DIR, "dataset", "normalized", "train"
)

NORMALIZED_TEST_DIR = os.path.join(
    BASE_DIR, "dataset", "normalized", "test"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR, "dataset", "candidates"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

# Number of nearest neighbours retrieved by TF-IDF blocking.
# We use a reasonably large number because blocking recall is
# more important than aggressive reduction at this stage.
TFIDF_NEIGHBORS = 20

# Minimum token length used for token blocking.
MIN_TOKEN_LENGTH = 3


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_text(value):
    """Convert missing/non-string values into an empty string."""
    if pd.isna(value):
        return ""
    return str(value).strip()


def get_tokens(text):
    """Return useful tokens from a normalized text field."""
    text = safe_text(text).lower()

    if not text:
        return []

    tokens = re.findall(r"[a-z0-9]+", text)

    return [
        token
        for token in tokens
        if len(token) >= MIN_TOKEN_LENGTH
    ]


def add_candidate(candidate_dict, s1_id, s2_id):
    """Add a candidate pair to the dictionary."""
    if not s2_id:
        return

    candidate_dict[s1_id].add(s2_id)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("STEP 5: CANDIDATE GENERATION / BLOCKING")
print("=" * 70)

print("\nLoading normalized training data...")

s1 = pd.read_csv(
    os.path.join(
        NORMALIZED_TRAIN_DIR,
        "train_source1.tsv"
    ),
    sep="\t",
    dtype=str
)

s2 = pd.read_csv(
    os.path.join(
        NORMALIZED_TRAIN_DIR,
        "train_source2.tsv"
    ),
    sep="\t",
    dtype=str
)

s3 = pd.read_csv(
    os.path.join(
        NORMALIZED_TRAIN_DIR,
        "train_source3.tsv"
    ),
    sep="\t",
    dtype=str
)

ground_truth = pd.read_csv(
    os.path.join(
        NORMALIZED_TRAIN_DIR,
        "train_ground_truth.tsv"
    ),
    sep="\t",
    dtype=str
)


# Replace NaN with empty strings
s1 = s1.fillna("")
s2 = s2.fillna("")
s3 = s3.fillna("")
ground_truth = ground_truth.fillna("")


print(f"S1 records : {len(s1):,}")
print(f"S2 records : {len(s2):,}")
print(f"S3 records : {len(s3):,}")


# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

required_columns = [
    "entity_id",
    "business_name_clean",
    "business_address_clean",
    "country_clean",
    "business_name_token_normalized",
    "business_address_token_normalized"
]

for dataframe_name, dataframe in [
    ("S1", s1),
    ("S2", s2),
    ("S3", s3)
]:

    missing = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing:
        raise ValueError(
            f"{dataframe_name} is missing columns: {missing}"
        )


# ============================================================
# GENERIC BLOCKING FUNCTION
# ============================================================

def generate_blocking_candidates(source1, source_other):
    """
    Generate candidates between Source 1 and another source.

    Returns:
        candidate_dict:
            {
                s1_entity_id: set(candidate_entity_ids)
            }

        strategy_counts:
            number of candidate pairs produced by each strategy
    """

    candidate_dict = defaultdict(set)

    strategy_counts = {
        "exact_name": 0,
        "exact_token_name": 0,
        "name_token": 0,
        "address_token": 0,
        "country_name_initial": 0,
        "tfidf_name": 0
    }

    # --------------------------------------------------------
    # BUILD INDEXES
    # --------------------------------------------------------

    print("\nBuilding blocking indexes...")

    exact_name_index = defaultdict(set)
    token_name_index = defaultdict(set)
    name_token_index = defaultdict(set)
    address_token_index = defaultdict(set)
    country_initial_index = defaultdict(set)

    for _, row in source_other.iterrows():

        entity_id = safe_text(row["entity_id"])

        name = safe_text(row["business_name_clean"])
        token_name = safe_text(
            row["business_name_token_normalized"]
        )
        address_token = safe_text(
            row["business_address_token_normalized"]
        )
        country = safe_text(row["country_clean"])

        # ----------------------------------------------------
        # Exact normalized name
        # ----------------------------------------------------

        if name:
            exact_name_index[name].add(entity_id)

        # ----------------------------------------------------
        # Exact token-normalized name
        # ----------------------------------------------------

        if token_name:
            token_name_index[token_name].add(entity_id)

        # ----------------------------------------------------
        # Individual name tokens
        # ----------------------------------------------------

        for token in get_tokens(token_name):
            name_token_index[token].add(entity_id)

        # ----------------------------------------------------
        # Individual address tokens
        # ----------------------------------------------------

        for token in get_tokens(address_token):
            address_token_index[token].add(entity_id)

        # ----------------------------------------------------
        # Country + first character of name
        # ----------------------------------------------------

        if country and name:
            key = (
                country,
                name[0]
            )

            country_initial_index[key].add(entity_id)


    # ========================================================
    # 1. EXACT NORMALIZED NAME
    # ========================================================

    print("\nBlocking strategy 1: Exact normalized name...")

    for _, row in source1.iterrows():

        s1_id = safe_text(row["entity_id"])
        name = safe_text(row["business_name_clean"])

        if not name:
            continue

        matches = exact_name_index.get(name, set())

        for candidate_id in matches:
            add_candidate(
                candidate_dict,
                s1_id,
                candidate_id
            )
            strategy_counts["exact_name"] += 1


    # ========================================================
    # 2. EXACT TOKEN-NORMALIZED NAME
    # ========================================================

    print("Blocking strategy 2: Exact token-normalized name...")

    for _, row in source1.iterrows():

        s1_id = safe_text(row["entity_id"])

        token_name = safe_text(
            row["business_name_token_normalized"]
        )

        if not token_name:
            continue

        matches = token_name_index.get(
            token_name,
            set()
        )

        for candidate_id in matches:
            add_candidate(
                candidate_dict,
                s1_id,
                candidate_id
            )
            strategy_counts["exact_token_name"] += 1


    # ========================================================
    # 3. SHARED NAME TOKEN
    # ========================================================

    print("Blocking strategy 3: Shared business-name token...")

    for _, row in source1.iterrows():

        s1_id = safe_text(row["entity_id"])

        token_name = safe_text(
            row["business_name_token_normalized"]
        )

        tokens = get_tokens(token_name)

        for token in tokens:

            matches = name_token_index.get(
                token,
                set()
            )

            for candidate_id in matches:

                add_candidate(
                    candidate_dict,
                    s1_id,
                    candidate_id
                )

                strategy_counts["name_token"] += 1


    # ========================================================
    # 4. SHARED ADDRESS TOKEN
    # ========================================================

    print("Blocking strategy 4: Shared address token...")

    for _, row in source1.iterrows():

        s1_id = safe_text(row["entity_id"])

        address_token = safe_text(
            row["business_address_token_normalized"]
        )

        tokens = get_tokens(address_token)

        for token in tokens:

            matches = address_token_index.get(
                token,
                set()
            )

            for candidate_id in matches:

                add_candidate(
                    candidate_dict,
                    s1_id,
                    candidate_id
                )

                strategy_counts["address_token"] += 1


    # ========================================================
    # 5. COUNTRY + NAME INITIAL
    # ========================================================

    print("Blocking strategy 5: Country + name initial...")

    for _, row in source1.iterrows():

        s1_id = safe_text(row["entity_id"])

        country = safe_text(
            row["country_clean"]
        )

        name = safe_text(
            row["business_name_clean"]
        )

        if not country or not name:
            continue

        key = (
            country,
            name[0]
        )

        matches = country_initial_index.get(
            key,
            set()
        )

        for candidate_id in matches:

            add_candidate(
                candidate_dict,
                s1_id,
                candidate_id
            )

            strategy_counts[
                "country_name_initial"
            ] += 1


    # ========================================================
    # 6. TF-IDF CHARACTER N-GRAM BLOCKING
    # ========================================================

    print(
        "Blocking strategy 6: "
        "TF-IDF character n-gram nearest neighbours..."
    )

    source_other_names = (
        source_other["business_name_clean"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    source1_names = (
        source1["business_name_clean"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    # Only build TF-IDF if there are usable names.
    if source_other_names:

        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            min_df=1,
            sublinear_tf=True
        )

        other_matrix = vectorizer.fit_transform(
            source_other_names
        )

        s1_matrix = vectorizer.transform(
            source1_names
        )

        n_neighbors = min(
            TFIDF_NEIGHBORS,
            len(source_other)
        )

        if n_neighbors > 0:

            nn = NearestNeighbors(
                n_neighbors=n_neighbors,
                metric="cosine",
                algorithm="brute"
            )

            nn.fit(other_matrix)

            distances, indices = nn.kneighbors(
                s1_matrix
            )

            other_ids = (
                source_other["entity_id"]
                .astype(str)
                .tolist()
            )

            for s1_index, neighbour_indices in enumerate(indices):

                s1_id = safe_text(
                    source1.iloc[s1_index]["entity_id"]
                )

                # If S1 name is missing, skip TF-IDF.
                if not source1_names[s1_index].strip():
                    continue

                for neighbour_index in neighbour_indices:

                    candidate_id = other_ids[
                        neighbour_index
                    ]

                    add_candidate(
                        candidate_dict,
                        s1_id,
                        candidate_id
                    )

                    strategy_counts[
                        "tfidf_name"
                    ] += 1


    return candidate_dict, strategy_counts


# ============================================================
# GENERATE S1 -> S2 CANDIDATES
# ============================================================

print("\n" + "=" * 70)
print("GENERATING S1 -> S2 CANDIDATES")
print("=" * 70)

candidates_s2, counts_s2 = generate_blocking_candidates(
    s1,
    s2
)


# ============================================================
# GENERATE S1 -> S3 CANDIDATES
# ============================================================

print("\n" + "=" * 70)
print("GENERATING S1 -> S3 CANDIDATES")
print("=" * 70)

candidates_s3, counts_s3 = generate_blocking_candidates(
    s1,
    s3
)


# ============================================================
# CONVERT TO LONG-FORM PAIRS
# ============================================================

def candidates_to_dataframe(
    candidate_dict,
    candidate_source_name
):

    rows = []

    for s1_id, candidate_ids in candidate_dict.items():

        for candidate_id in sorted(candidate_ids):

            rows.append({
                "source1_entity_id": s1_id,
                "candidate_entity_id": candidate_id,
                "candidate_source": candidate_source_name
            })

    return pd.DataFrame(rows)


candidate_pairs_s2 = candidates_to_dataframe(
    candidates_s2,
    "source2"
)

candidate_pairs_s3 = candidates_to_dataframe(
    candidates_s3,
    "source3"
)


candidate_pairs_all = pd.concat(
    [
        candidate_pairs_s2,
        candidate_pairs_s3
    ],
    ignore_index=True
)


# ============================================================
# SAVE LONG-FORM CANDIDATES
# ============================================================

long_output_path = os.path.join(
    OUTPUT_DIR,
    "candidate_pairs_train_long.tsv"
)

candidate_pairs_all.to_csv(
    long_output_path,
    sep="\t",
    index=False
)


# ============================================================
# CREATE CHALLENGE-STYLE CANDIDATE FILE
# ============================================================

print("\nCreating challenge-style candidate file...")

candidate_rows = []

for _, row in s1.iterrows():

    s1_id = safe_text(row["entity_id"])

    s2_ids = sorted(
        candidates_s2.get(s1_id, set())
    )

    s3_ids = sorted(
        candidates_s3.get(s1_id, set())
    )

    candidate_rows.append({
        "source1_entity_id": s1_id,
        "source2_candidate_ids": ",".join(s2_ids),
        "source3_candidate_ids": ",".join(s3_ids)
    })


candidate_pairs = pd.DataFrame(
    candidate_rows
)

candidate_output_path = os.path.join(
    OUTPUT_DIR,
    "candidate_pairs_train.tsv"
)

candidate_pairs.to_csv(
    candidate_output_path,
    sep="\t",
    index=False
)


#```python
# ============================================================
# GROUND TRUTH PARSING
# ============================================================

print("\nEvaluating blocking recall...")

print(
    "\nGround truth columns:",
    list(ground_truth.columns)
)

# The challenge ground truth contains:
#
# source1_entity_id
# matched_entity_ids
#
# matched_entity_ids contains BOTH S2 and S3 IDs.
#
# Example:
# S1-00001    S2-00047,S2-00193,S3-00812
#
# Therefore, we separate the IDs using their prefixes.

required_gt_columns = [
    "source1_entity_id",
    "matched_entity_ids"
]

for column in required_gt_columns:

    if column not in ground_truth.columns:

        raise ValueError(
            f"Ground truth is missing required column: {column}"
        )


# ============================================================
# PARSE MATCHED ENTITY IDS
# ============================================================

def parse_id_list(value):

    value = safe_text(value)

    if not value:
        return set()

    parts = re.split(
        r"[,;|]",
        value
    )

    return {
        part.strip()
        for part in parts
        if part.strip()
    }


def separate_source_ids(ids):

    s2_ids = set()
    s3_ids = set()

    for entity_id in ids:

        entity_id = entity_id.strip()

        if entity_id.startswith("S2-"):
            s2_ids.add(entity_id)

        elif entity_id.startswith("S3-"):
            s3_ids.add(entity_id)

    return s2_ids, s3_ids


# ============================================================
# CALCULATE BLOCKING RECALL
# ============================================================

total_true_s2 = 0
found_true_s2 = 0

total_true_s3 = 0
found_true_s3 = 0

s1_with_true_matches = 0
s1_with_all_matches_found = 0


for _, row in ground_truth.iterrows():

    s1_id = safe_text(
        row["source1_entity_id"]
    )

    # Get all true matched IDs
    all_true_ids = parse_id_list(
        row["matched_entity_ids"]
    )

    # Separate S2 and S3 using ID prefix
    true_s2, true_s3 = separate_source_ids(
        all_true_ids
    )

    # Candidates generated by our blocking stage
    predicted_s2 = candidates_s2.get(
        s1_id,
        set()
    )

    predicted_s3 = candidates_s3.get(
        s1_id,
        set()
    )

    # --------------------------------------------------------
    # S2 recall
    # --------------------------------------------------------

    total_true_s2 += len(true_s2)

    found_true_s2 += len(
        true_s2.intersection(
            predicted_s2
        )
    )

    # --------------------------------------------------------
    # S3 recall
    # --------------------------------------------------------

    total_true_s3 += len(true_s3)

    found_true_s3 += len(
        true_s3.intersection(
            predicted_s3
        )
    )

    # --------------------------------------------------------
    # Entity-level complete recall
    # --------------------------------------------------------

    if all_true_ids:

        s1_with_true_matches += 1

        found_ids = (
            predicted_s2.intersection(true_s2)
        ).union(
            predicted_s3.intersection(true_s3)
        )

        if found_ids == all_true_ids:

            s1_with_all_matches_found += 1


# ============================================================
# RECALL VALUES
# ============================================================

s2_recall = (
    found_true_s2 / total_true_s2
    if total_true_s2 > 0
    else 0
)

s3_recall = (
    found_true_s3 / total_true_s3
    if total_true_s3 > 0
    else 0
)

overall_true = (
    total_true_s2 +
    total_true_s3
)

overall_found = (
    found_true_s2 +
    found_true_s3
)

overall_recall = (
    overall_found / overall_true
    if overall_true > 0
    else 0
)

entity_recall = (
    s1_with_all_matches_found /
    s1_with_true_matches
    if s1_with_true_matches > 0
    else 0
)



# ============================================================
# SEARCH SPACE REDUCTION
# ============================================================

possible_s2_pairs = len(s1) * len(s2)
possible_s3_pairs = len(s1) * len(s3)

possible_total_pairs = (
    possible_s2_pairs +
    possible_s3_pairs
)

candidate_total_pairs = len(
    candidate_pairs_all
)

reduction_percentage = (
    1 -
    candidate_total_pairs /
    possible_total_pairs
) * 100


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("BLOCKING RESULTS")
print("=" * 70)

print("\nS2 blocking:")
print(
    f"  Candidate pairs: "
    f"{len(candidate_pairs_s2):,}"
)

print(
    f"  True matches: "
    f"{total_true_s2:,}"
)

print(
    f"  True matches found: "
    f"{found_true_s2:,}"
)

print(
    f"  Blocking recall: "
    f"{s2_recall:.4%}"
)


print("\nS3 blocking:")
print(
    f"  Candidate pairs: "
    f"{len(candidate_pairs_s3):,}"
)

print(
    f"  True matches: "
    f"{total_true_s3:,}"
)

print(
    f"  True matches found: "
    f"{found_true_s3:,}"
)

print(
    f"  Blocking recall: "
    f"{s3_recall:.4%}"
)


print("\nOverall:")
print(
    f"  Total possible comparisons: "
    f"{possible_total_pairs:,}"
)

print(
    f"  Total candidate comparisons: "
    f"{candidate_total_pairs:,}"
)

print(
    f"  Overall blocking recall: "
    f"{overall_recall:.4%}"
)

print(
    f"  Entity-level complete recall: "
    f"{entity_recall:.4%}"
)

print(
    f"  Search-space reduction: "
    f"{reduction_percentage:.2f}%"
)


# ============================================================
# STRATEGY STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("BLOCKING STRATEGY STATISTICS")
print("=" * 70)

print("\nS1 -> S2")

for strategy, count in counts_s2.items():

    print(
        f"  {strategy:<25} "
        f"{count:,}"
    )


print("\nS1 -> S3")

for strategy, count in counts_s3.items():

    print(
        f"  {strategy:<25} "
        f"{count:,}"
    )


# ============================================================
# CANDIDATE DISTRIBUTION
# ============================================================

s2_candidate_counts = (
    candidate_pairs["source2_candidate_ids"]
    .apply(
        lambda x:
        len(
            [
                v for v in x.split(",")
                if v.strip()
            ]
        )
    )
)

s3_candidate_counts = (
    candidate_pairs["source3_candidate_ids"]
    .apply(
        lambda x:
        len(
            [
                v for v in x.split(",")
                if v.strip()
            ]
        )
    )
)

print("\n" + "=" * 70)
print("CANDIDATE DISTRIBUTION")
print("=" * 70)

print("\nS2 candidates per S1:")
print(
    s2_candidate_counts.describe()
)

print("\nS3 candidates per S1:")
print(
    s3_candidate_counts.describe()
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_path = os.path.join(
    OUTPUT_DIR,
    "blocking_summary.txt"
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "BUSINESS ENTITY RESOLUTION - "
        "BLOCKING SUMMARY\n"
    )

    f.write("=" * 60 + "\n\n")

    f.write(
        f"S1 records: {len(s1):,}\n"
    )

    f.write(
        f"S2 records: {len(s2):,}\n"
    )

    f.write(
        f"S3 records: {len(s3):,}\n\n"
    )

    f.write(
        f"Possible comparisons: "
        f"{possible_total_pairs:,}\n"
    )

    f.write(
        f"Candidate comparisons: "
        f"{candidate_total_pairs:,}\n"
    )

    f.write(
        f"Search-space reduction: "
        f"{reduction_percentage:.2f}%\n\n"
    )

    f.write(
        f"S2 blocking recall: "
        f"{s2_recall:.4%}\n"
    )

    f.write(
        f"S3 blocking recall: "
        f"{s3_recall:.4%}\n"
    )

    f.write(
        f"Overall blocking recall: "
        f"{overall_recall:.4%}\n"
    )

    f.write(
        f"Entity-level complete recall: "
        f"{entity_recall:.4%}\n"
    )


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("CANDIDATE GENERATION COMPLETED")
print("=" * 70)

print(
    f"\nSaved:"
)

print(
    f"  {long_output_path}"
)

print(
    f"  {candidate_output_path}"
)

print(
    f"  {summary_path}"
)

print(
    "\nOriginal, cleaned and normalized datasets were not modified."
)

print(
    "\nNext stage: Feature Engineering"
)

print("=" * 70)



