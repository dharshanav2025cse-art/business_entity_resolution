from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd
from joblib import load


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NORMALIZED_TEST = PROJECT_ROOT / "dataset" / "normalized" / "test"
CANDIDATE_FILE = (
    PROJECT_ROOT
    / "dataset"
    / "candidates"
    / "candidate_pairs_test_long.tsv"
)
OUTPUT_DIR = PROJECT_ROOT / "dataset" / "features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "test_features.tsv.gz"
MODEL_DIR = PROJECT_ROOT / "models"
NAME_TFIDF_VECTORIZER_FILE = MODEL_DIR / "name_tfidf_vectorizer.joblib"
ADDRESS_TFIDF_VECTORIZER_FILE = MODEL_DIR / "address_tfidf_vectorizer.joblib"

CHUNK_SIZE = 100_000

IDENTIFIER_COLUMNS = [
    "source1_entity_id",
    "candidate_entity_id",
    "candidate_source",
]

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

OUTPUT_COLUMNS = IDENTIFIER_COLUMNS + FEATURE_COLUMNS


def normalize_basic(text):
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def calculate_jaccard_series(left, right):
    result = np.zeros(len(left), dtype=np.float32)
    left_values = left.to_numpy()
    right_values = right.to_numpy()
    exact_mask = (left_values == right_values) & (left_values != "")
    result[exact_mask] = 1.0

    for idx in np.where(~exact_mask)[0]:
        left_text = left_values[idx]
        right_text = right_values[idx]
        if not left_text or not right_text:
            continue
        left_tokens = set(left_text.split())
        right_tokens = set(right_text.split())
        union = left_tokens | right_tokens
        if union:
            result[idx] = len(left_tokens & right_tokens) / len(union)

    return result


def get_length_features(left, right):
    left_length = left.str.len().to_numpy().astype(np.float32)
    right_length = right.str.len().to_numpy().astype(np.float32)
    maximum = np.maximum(left_length, right_length)
    ratio = np.divide(
        np.minimum(left_length, right_length),
        maximum,
        out=np.zeros(len(left), dtype=np.float32),
        where=maximum > 0,
    )
    difference = np.abs(left_length - right_length).astype(np.int16)
    return difference, ratio


def pairwise_tfidf_scores(
    chunk,
    candidate_mask,
    source1_positions,
    candidate_positions,
    source1_matrix,
    candidate_matrix,
):
    scores = np.zeros(len(chunk), dtype=np.float32)
    indices = np.where(candidate_mask.to_numpy())[0]
    if not len(indices):
        return scores

    source1_rows = (
        chunk.iloc[indices]["source1_entity_id"]
        .map(source1_positions)
        .to_numpy()
    )
    candidate_rows = (
        chunk.iloc[indices]["candidate_entity_id"]
        .map(candidate_positions)
        .to_numpy()
    )
    valid = pd.notna(source1_rows) & pd.notna(candidate_rows)
    valid_positions = np.where(valid)[0]
    if len(valid_positions):
        left_rows = source1_rows[valid_positions].astype(int)
        right_rows = candidate_rows[valid_positions].astype(int)
        similarity = (
            source1_matrix[left_rows]
            .multiply(candidate_matrix[right_rows])
            .sum(axis=1)
        )
        scores[indices[valid_positions]] = np.asarray(similarity).ravel().astype(
            np.float32
        )
    return scores


def main():
    print("=" * 70)
    print("TEST FEATURE ENGINEERING")
    print("=" * 70)
    print("Loading full normalized TEST datasets...")

    required_columns = [
        "entity_id",
        "business_name_normalized",
        "business_name_token_normalized",
        "business_address_normalized",
        "business_address_token_normalized",
        "country_normalized",
    ]
    sources = {}
    for source_number in (1, 2, 3):
        source = pd.read_csv(
            NORMALIZED_TEST / f"test_source{source_number}.tsv",
            sep="\t",
            dtype=str,
        ).fillna("")
        missing = [column for column in required_columns if column not in source]
        if missing:
            raise ValueError(
                f"Source {source_number} is missing columns: {missing}"
            )
        sources[source_number] = source[required_columns].copy()
        print(f"Source {source_number}: {len(source):,} records")

    s1 = sources[1].set_index("entity_id")
    s2 = sources[2].set_index("entity_id")
    s3 = sources[3].set_index("entity_id")

    s1_name_positions = {
        entity_id: position for position, entity_id in enumerate(s1.index)
    }
    s2_name_positions = {
        entity_id: position for position, entity_id in enumerate(s2.index)
    }
    s3_name_positions = {
        entity_id: position for position, entity_id in enumerate(s3.index)
    }
    s1_address_positions = s1_name_positions
    s2_address_positions = s2_name_positions
    s3_address_positions = s3_name_positions

    print("Loading persisted TRAIN TF-IDF vectorizers...")
    if not NAME_TFIDF_VECTORIZER_FILE.is_file():
        raise FileNotFoundError(
            f"Training name TF-IDF vectorizer not found: {NAME_TFIDF_VECTORIZER_FILE}"
        )
    if not ADDRESS_TFIDF_VECTORIZER_FILE.is_file():
        raise FileNotFoundError(
            f"Training address TF-IDF vectorizer not found: {ADDRESS_TFIDF_VECTORIZER_FILE}"
        )
    name_vectorizer = load(NAME_TFIDF_VECTORIZER_FILE)
    address_vectorizer = load(ADDRESS_TFIDF_VECTORIZER_FILE)

    expected_vectorizer_settings = {
        "analyzer": "char",
        "ngram_range": (2, 5),
        "min_df": 2,
        "max_features": 100_000,
    }
    for name, vectorizer in (
        ("name", name_vectorizer),
        ("address", address_vectorizer),
    ):
        actual_settings = vectorizer.get_params()
        for setting, expected_value in expected_vectorizer_settings.items():
            if actual_settings[setting] != expected_value:
                raise ValueError(
                    f"Unexpected {name} vectorizer setting {setting}: "
                    f"{actual_settings[setting]!r}"
                )

    name_matrices = {
        source_number: name_vectorizer.transform(
            source["business_name_normalized"]
        )
        for source_number, source in sources.items()
    }
    address_matrices = {
        source_number: address_vectorizer.transform(
            source["business_address_normalized"]
        )
        for source_number, source in sources.items()
    }

    if not CANDIDATE_FILE.exists():
        raise FileNotFoundError(f"Test candidate file not found: {CANDIDATE_FILE}")
    if OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()

    print(f"\nProcessing candidates in chunks of {CHUNK_SIZE:,}...")
    total_rows = 0
    s2_rows = 0
    s3_rows = 0
    first_write = True

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            CANDIDATE_FILE,
            sep="\t",
            dtype=str,
            chunksize=CHUNK_SIZE,
        ),
        start=1,
    ):
        chunk = chunk.fillna("")
        missing = [column for column in IDENTIFIER_COLUMNS if column not in chunk]
        if missing:
            raise ValueError(f"Candidate file is missing columns: {missing}")

        source2_mask = chunk["candidate_source"] == "source2"
        source3_mask = chunk["candidate_source"] == "source3"
        if not (source2_mask | source3_mask).all():
            invalid_sources = sorted(
                set(chunk.loc[~(source2_mask | source3_mask), "candidate_source"])
            )
            raise ValueError(f"Unexpected candidate_source values: {invalid_sources}")

        s1_data = s1.reindex(chunk["source1_entity_id"].to_numpy())
        s1_name = s1_data["business_name_normalized"].reset_index(drop=True)
        s1_name_token = s1_data[
            "business_name_token_normalized"
        ].reset_index(drop=True)
        s1_address = s1_data[
            "business_address_normalized"
        ].reset_index(drop=True)
        s1_address_token = s1_data[
            "business_address_token_normalized"
        ].reset_index(drop=True)
        s1_country = s1_data["country_normalized"].reset_index(drop=True)

        candidate_name = pd.Series("", index=chunk.index, dtype=object)
        candidate_name_token = pd.Series("", index=chunk.index, dtype=object)
        candidate_address = pd.Series("", index=chunk.index, dtype=object)
        candidate_address_token = pd.Series("", index=chunk.index, dtype=object)
        candidate_country = pd.Series("", index=chunk.index, dtype=object)

        for mask, source in ((source2_mask, s2), (source3_mask, s3)):
            source_ids = chunk.loc[mask, "candidate_entity_id"]
            if source_ids.empty:
                continue
            candidate_data = source.reindex(source_ids.to_numpy())
            candidate_name.loc[mask] = candidate_data[
                "business_name_normalized"
            ].to_numpy()
            candidate_name_token.loc[mask] = candidate_data[
                "business_name_token_normalized"
            ].to_numpy()
            candidate_address.loc[mask] = candidate_data[
                "business_address_normalized"
            ].to_numpy()
            candidate_address_token.loc[mask] = candidate_data[
                "business_address_token_normalized"
            ].to_numpy()
            candidate_country.loc[mask] = candidate_data[
                "country_normalized"
            ].to_numpy()

        candidate_name = candidate_name.fillna("").astype(str).reset_index(drop=True)
        candidate_name_token = candidate_name_token.fillna("").astype(str).reset_index(
            drop=True
        )
        candidate_address = candidate_address.fillna("").astype(str).reset_index(
            drop=True
        )
        candidate_address_token = candidate_address_token.fillna("").astype(
            str
        ).reset_index(drop=True)
        candidate_country = candidate_country.fillna("").astype(str).reset_index(
            drop=True
        )

        s1_name = s1_name.fillna("").astype(str)
        s1_name_token = s1_name_token.fillna("").astype(str)
        s1_address = s1_address.fillna("").astype(str)
        s1_address_token = s1_address_token.fillna("").astype(str)
        s1_country = s1_country.fillna("").astype(str)

        name_length_difference, name_length_ratio = get_length_features(
            s1_name, candidate_name
        )
        address_length_difference, address_length_ratio = get_length_features(
            s1_address, candidate_address
        )

        s2_name_tfidf = pairwise_tfidf_scores(
            chunk,
            source2_mask,
            s1_name_positions,
            s2_name_positions,
            name_matrices[1],
            name_matrices[2],
        )
        s3_name_tfidf = pairwise_tfidf_scores(
            chunk,
            source3_mask,
            s1_name_positions,
            s3_name_positions,
            name_matrices[1],
            name_matrices[3],
        )
        s2_address_tfidf = pairwise_tfidf_scores(
            chunk,
            source2_mask,
            s1_address_positions,
            s2_address_positions,
            address_matrices[1],
            address_matrices[2],
        )
        s3_address_tfidf = pairwise_tfidf_scores(
            chunk,
            source3_mask,
            s1_address_positions,
            s3_address_positions,
            address_matrices[1],
            address_matrices[3],
        )

        output = pd.DataFrame(
            {
                "source1_entity_id": chunk["source1_entity_id"].to_numpy(),
                "candidate_entity_id": chunk["candidate_entity_id"].to_numpy(),
                "candidate_source": chunk["candidate_source"].to_numpy(),
                "name_exact": (
                    (s1_name != "") & (candidate_name != "") & (s1_name == candidate_name)
                ).astype(np.int8),
                "name_token_exact": (
                    (s1_name_token != "")
                    & (candidate_name_token != "")
                    & (s1_name_token == candidate_name_token)
                ).astype(np.int8),
                "name_jaccard": calculate_jaccard_series(
                    s1_name_token, candidate_name_token
                ),
                "name_length_difference": name_length_difference,
                "name_length_ratio": name_length_ratio,
                "name_tfidf": s2_name_tfidf + s3_name_tfidf,
                "address_exact": (
                    (s1_address != "")
                    & (candidate_address != "")
                    & (s1_address == candidate_address)
                ).astype(np.int8),
                "address_token_exact": (
                    (s1_address_token != "")
                    & (candidate_address_token != "")
                    & (s1_address_token == candidate_address_token)
                ).astype(np.int8),
                "address_jaccard": calculate_jaccard_series(
                    s1_address_token, candidate_address_token
                ),
                "address_length_difference": address_length_difference,
                "address_length_ratio": address_length_ratio,
                "address_tfidf": s2_address_tfidf + s3_address_tfidf,
                "country_exact": (
                    (s1_country != "")
                    & (candidate_country != "")
                    & (s1_country == candidate_country)
                ).astype(np.int8),
                "source1_name_missing": (s1_name == "").astype(np.int8),
                "candidate_name_missing": (candidate_name == "").astype(np.int8),
                "source1_address_missing": (s1_address == "").astype(np.int8),
                "candidate_address_missing": (candidate_address == "").astype(np.int8),
                "source1_country_missing": (s1_country == "").astype(np.int8),
                "candidate_country_missing": (candidate_country == "").astype(np.int8),
            },
            columns=OUTPUT_COLUMNS,
        )

        output.to_csv(
            OUTPUT_FILE,
            sep="\t",
            index=False,
            mode="w" if first_write else "a",
            header=first_write,
            compression="gzip",
        )
        first_write = False
        chunk_rows = len(chunk)
        total_rows += chunk_rows
        s2_rows += int(source2_mask.sum())
        s3_rows += int(source3_mask.sum())
        print(
            f"Chunk {chunk_number}: {chunk_rows:,} rows; "
            f"total processed {total_rows:,}"
        )

    if first_write:
        pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(
            OUTPUT_FILE, sep="\t", index=False, compression="gzip"
        )

    print("\n" + "=" * 70)
    print("TEST FEATURE ENGINEERING COMPLETE")
    print("=" * 70)
    print(f"Total test candidate pairs processed: {total_rows:,}")
    print(f"Number of S2 candidates: {s2_rows:,}")
    print(f"Number of S3 candidates: {s3_rows:,}")
    print(f"Number of feature rows: {total_rows:,}")
    print(f"Feature columns ({len(OUTPUT_COLUMNS)}): {OUTPUT_COLUMNS}")
    print(f"Output file path: {OUTPUT_FILE}")
    print("No labels were created.")


if __name__ == "__main__":
    main()