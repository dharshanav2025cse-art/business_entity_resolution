import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NORMALIZED_TEST_DIR = os.path.join(BASE_DIR, "dataset", "normalized", "test")
OUTPUT_DIR = os.path.join(BASE_DIR, "dataset", "candidates")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TFIDF_NEIGHBORS = 20
MIN_TOKEN_LENGTH = 3


def safe_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def get_tokens(text):
    text = safe_text(text).lower()
    if not text:
        return []
    tokens = re.findall(r"[a-z0-9]+", text)
    return [token for token in tokens if len(token) >= MIN_TOKEN_LENGTH]


def add_candidate(candidate_dict, s1_id, s2_id):
    if not s2_id:
        return
    candidate_dict[s1_id].add(s2_id)


def generate_blocking_candidates(source1, source_other):
    candidate_dict = defaultdict(set)
    strategy_counts = {
        "exact_name": 0,
        "exact_token_name": 0,
        "name_token": 0,
        "address_token": 0,
        "country_name_initial": 0,
        "tfidf_name": 0,
    }

    exact_name_index = defaultdict(set)
    token_name_index = defaultdict(set)
    name_token_index = defaultdict(set)
    address_token_index = defaultdict(set)
    country_initial_index = defaultdict(set)

    for _, row in source_other.iterrows():
        entity_id = safe_text(row["entity_id"])
        name = safe_text(row["business_name_clean"])
        token_name = safe_text(row["business_name_token_normalized"])
        address_token = safe_text(row["business_address_token_normalized"])
        country = safe_text(row["country_clean"])

        if name:
            exact_name_index[name].add(entity_id)

        if token_name:
            token_name_index[token_name].add(entity_id)

        for token in get_tokens(token_name):
            name_token_index[token].add(entity_id)

        for token in get_tokens(address_token):
            address_token_index[token].add(entity_id)

        if country and name:
            country_initial_index[(country, name[0])].add(entity_id)

    print("\nBlocking strategy 1: Exact normalized name...")
    for _, row in source1.iterrows():
        s1_id = safe_text(row["entity_id"])
        name = safe_text(row["business_name_clean"])
        if not name:
            continue
        matches = exact_name_index.get(name, set())
        for candidate_id in matches:
            add_candidate(candidate_dict, s1_id, candidate_id)
            strategy_counts["exact_name"] += 1

    print("Blocking strategy 2: Exact token-normalized name...")
    for _, row in source1.iterrows():
        s1_id = safe_text(row["entity_id"])
        token_name = safe_text(row["business_name_token_normalized"])
        if not token_name:
            continue
        matches = token_name_index.get(token_name, set())
        for candidate_id in matches:
            add_candidate(candidate_dict, s1_id, candidate_id)
            strategy_counts["exact_token_name"] += 1

    print("Blocking strategy 3: Shared business-name token...")
    for _, row in source1.iterrows():
        s1_id = safe_text(row["entity_id"])
        token_name = safe_text(row["business_name_token_normalized"])
        tokens = get_tokens(token_name)
        for token in tokens:
            matches = name_token_index.get(token, set())
            for candidate_id in matches:
                add_candidate(candidate_dict, s1_id, candidate_id)
                strategy_counts["name_token"] += 1

    print("Blocking strategy 4: Shared address token...")
    for _, row in source1.iterrows():
        s1_id = safe_text(row["entity_id"])
        address_token = safe_text(row["business_address_token_normalized"])
        tokens = get_tokens(address_token)
        for token in tokens:
            matches = address_token_index.get(token, set())
            for candidate_id in matches:
                add_candidate(candidate_dict, s1_id, candidate_id)
                strategy_counts["address_token"] += 1

    print("Blocking strategy 5: Country + name initial...")
    for _, row in source1.iterrows():
        s1_id = safe_text(row["entity_id"])
        country = safe_text(row["country_clean"])
        name = safe_text(row["business_name_clean"])
        if not country or not name:
            continue
        key = (country, name[0])
        matches = country_initial_index.get(key, set())
        for candidate_id in matches:
            add_candidate(candidate_dict, s1_id, candidate_id)
            strategy_counts["country_name_initial"] += 1

    print("Blocking strategy 6: TF-IDF character n-gram nearest neighbours...")
    source_other_names = source_other["business_name_clean"].fillna("").astype(str).tolist()
    source1_names = source1["business_name_clean"].fillna("").astype(str).tolist()

    if source_other_names:
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            min_df=1,
            sublinear_tf=True,
        )
        other_matrix = vectorizer.fit_transform(source_other_names)
        s1_matrix = vectorizer.transform(source1_names)
        n_neighbors = min(TFIDF_NEIGHBORS, len(source_other))

        if n_neighbors > 0:
            nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", algorithm="brute")
            nn.fit(other_matrix)
            distances, indices = nn.kneighbors(s1_matrix)
            other_ids = source_other["entity_id"].astype(str).tolist()

            for s1_index, neighbour_indices in enumerate(indices):
                s1_id = safe_text(source1.iloc[s1_index]["entity_id"])
                if not source1_names[s1_index].strip():
                    continue
                for neighbour_index in neighbour_indices:
                    candidate_id = other_ids[neighbour_index]
                    add_candidate(candidate_dict, s1_id, candidate_id)
                    strategy_counts["tfidf_name"] += 1

    return candidate_dict, strategy_counts


def candidates_to_dataframe(candidate_dict, candidate_source_name):
    rows = []
    for s1_id, candidate_ids in candidate_dict.items():
        for candidate_id in sorted(candidate_ids):
            rows.append(
                {
                    "source1_entity_id": s1_id,
                    "candidate_entity_id": candidate_id,
                    "candidate_source": candidate_source_name,
                }
            )
    return pd.DataFrame(rows)


print("=" * 70)
print("STEP 2: TEST CANDIDATE GENERATION")
print("=" * 70)

s1 = pd.read_csv(
    os.path.join(NORMALIZED_TEST_DIR, "test_source1.tsv"),
    sep="\t",
    dtype=str,
).fillna("")
s2 = pd.read_csv(
    os.path.join(NORMALIZED_TEST_DIR, "test_source2.tsv"),
    sep="\t",
    dtype=str,
).fillna("")
s3 = pd.read_csv(
    os.path.join(NORMALIZED_TEST_DIR, "test_source3.tsv"),
    sep="\t",
    dtype=str,
).fillna("")

required_columns = [
    "entity_id",
    "business_name_clean",
    "business_address_clean",
    "country_clean",
    "business_name_token_normalized",
    "business_address_token_normalized",
]

for dataframe_name, dataframe in [("S1", s1), ("S2", s2), ("S3", s3)]:
    missing = [column for column in required_columns if column not in dataframe.columns]
    if missing:
        raise ValueError(f"{dataframe_name} is missing columns: {missing}")

print(f"S1 records : {len(s1):,}")
print(f"S2 records : {len(s2):,}")
print(f"S3 records : {len(s3):,}")

print("\n" + "=" * 70)
print("GENERATING S1 -> S2 CANDIDATES")
print("=" * 70)
candidates_s2, counts_s2 = generate_blocking_candidates(s1, s2)

print("\n" + "=" * 70)
print("GENERATING S1 -> S3 CANDIDATES")
print("=" * 70)
candidates_s3, counts_s3 = generate_blocking_candidates(s1, s3)

candidate_pairs_s2 = candidates_to_dataframe(candidates_s2, "source2")
candidate_pairs_s3 = candidates_to_dataframe(candidates_s3, "source3")
candidate_pairs_all = pd.concat([candidate_pairs_s2, candidate_pairs_s3], ignore_index=True)

long_output_path = os.path.join(OUTPUT_DIR, "candidate_pairs_test_long.tsv")
candidate_pairs_all.to_csv(long_output_path, sep="\t", index=False)

candidate_rows = []
for _, row in s1.iterrows():
    s1_id = safe_text(row["entity_id"])
    s2_ids = sorted(candidates_s2.get(s1_id, set()))
    s3_ids = sorted(candidates_s3.get(s1_id, set()))
    candidate_rows.append(
        {
            "source1_entity_id": s1_id,
            "source2_candidate_ids": ",".join(s2_ids),
            "source3_candidate_ids": ",".join(s3_ids),
        }
    )

candidate_pairs = pd.DataFrame(candidate_rows)
candidate_output_path = os.path.join(OUTPUT_DIR, "candidate_pairs_test.tsv")
candidate_pairs.to_csv(candidate_output_path, sep="\t", index=False)

s2_candidate_counts = candidate_pairs["source2_candidate_ids"].apply(lambda x: len([v for v in x.split(",") if v.strip()]))
s3_candidate_counts = candidate_pairs["source3_candidate_ids"].apply(lambda x: len([v for v in x.split(",") if v.strip()]))

max_s2 = int(s2_candidate_counts.max()) if len(s2_candidate_counts) else 0
min_s2 = int(s2_candidate_counts.min()) if len(s2_candidate_counts) else 0
max_s3 = int(s3_candidate_counts.max()) if len(s3_candidate_counts) else 0
min_s3 = int(s3_candidate_counts.min()) if len(s3_candidate_counts) else 0
avg_s2 = float(s2_candidate_counts.mean()) if len(s2_candidate_counts) else 0.0
avg_s3 = float(s3_candidate_counts.mean()) if len(s3_candidate_counts) else 0.0

total_candidate_pairs = len(candidate_pairs_all)
average_candidates_per_s1 = float(total_candidate_pairs / len(s1)) if len(s1) else 0.0
min_candidates_per_s1 = int(min(len(candidates_s2.get(s1_id, set())) + len(candidates_s3.get(s1_id, set())) for s1_id in s1["entity_id"].astype(str))) if len(s1) else 0
max_candidates_per_s1 = int(max(len(candidates_s2.get(s1_id, set())) + len(candidates_s3.get(s1_id, set())) for s1_id in s1["entity_id"].astype(str))) if len(s1) else 0

print("\n" + "=" * 70)
print("TEST CANDIDATE GENERATION SUMMARY")
print("=" * 70)
print(f"Source 1 test entities       : {len(s1):,}")
print(f"Source 2 test entities       : {len(s2):,}")
print(f"Source 3 test entities       : {len(s3):,}")
print(f"S2 candidate pairs           : {len(candidate_pairs_s2):,}")
print(f"S3 candidate pairs           : {len(candidate_pairs_s3):,}")
print(f"Total candidate pairs        : {total_candidate_pairs:,}")
print(f"Average candidates per S1    : {average_candidates_per_s1:.2f}")
print(f"Minimum candidates per S1    : {min_candidates_per_s1:,}")
print(f"Maximum candidates per S1    : {max_candidates_per_s1:,}")
print(f"S2 avg candidates per S1     : {avg_s2:.2f}")
print(f"S2 min candidates per S1     : {min_s2:,}")
print(f"S2 max candidates per S1     : {max_s2:,}")
print(f"S3 avg candidates per S1     : {avg_s3:.2f}")
print(f"S3 min candidates per S1     : {min_s3:,}")
print(f"S3 max candidates per S1     : {max_s3:,}")
print("=" * 70)

summary_path = os.path.join(OUTPUT_DIR, "candidate_pairs_test_summary.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("BUSINESS ENTITY RESOLUTION - TEST CANDIDATE GENERATION SUMMARY\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Source 1 test entities: {len(s1):,}\n")
    f.write(f"Source 2 test entities: {len(s2):,}\n")
    f.write(f"Source 3 test entities: {len(s3):,}\n\n")
    f.write(f"S2 candidate pairs: {len(candidate_pairs_s2):,}\n")
    f.write(f"S3 candidate pairs: {len(candidate_pairs_s3):,}\n")
    f.write(f"Total candidate pairs: {total_candidate_pairs:,}\n")
    f.write(f"Average candidates per Source 1: {average_candidates_per_s1:.2f}\n")
    f.write(f"Minimum candidates per Source 1: {min_candidates_per_s1:,}\n")
    f.write(f"Maximum candidates per Source 1: {max_candidates_per_s1:,}\n")

print(f"\nSaved: {long_output_path}")
print(f"Saved: {candidate_output_path}")
print(f"Saved: {summary_path}")
print("\nNo test ground truth was used; blocking recall is not calculated for the test set.")
print("\nCandidate generation completed.")
print("=" * 70)
