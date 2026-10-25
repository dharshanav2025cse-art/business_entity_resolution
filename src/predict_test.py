from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_FILE = PROJECT_ROOT / "models" / "gradient_boosting_model.joblib"
FEATURE_FILE = PROJECT_ROOT / "dataset" / "features" / "test_features.tsv.gz"
TEST_DIR = PROJECT_ROOT / "dataset" / "normalized" / "test"
OUTPUT_DIR = PROJECT_ROOT / "output"
MATCHING_FILE = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_FILE = OUTPUT_DIR / "candidate_pairs.tsv"

THRESHOLD = 0.94
CHUNK_SIZE = 100_000

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

IDENTIFIER_COLUMNS = [
    "source1_entity_id",
    "candidate_entity_id",
    "candidate_source",
]


def read_test_ids(source_number):
    path = TEST_DIR / f"test_source{source_number}.tsv"
    frame = pd.read_csv(path, sep="\t", dtype=str, usecols=["entity_id"])
    entity_ids = frame["entity_id"].fillna("").astype(str).tolist()
    if any(not entity_id for entity_id in entity_ids):
        raise ValueError(f"Source {source_number} contains an empty entity ID")
    if len(entity_ids) != len(set(entity_ids)):
        raise ValueError(f"Source {source_number} contains duplicate entity IDs")
    return entity_ids


def read_output(path, expected_columns):
    frame = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )
    if list(frame.columns) != expected_columns:
        raise AssertionError(
            f"Unexpected columns in {path.name}: {list(frame.columns)}"
        )
    return frame


def main():
    print("=" * 70)
    print("FINAL TEST PREDICTION")
    print("=" * 70)

    if not MODEL_FILE.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_FILE}")
    if not FEATURE_FILE.is_file():
        raise FileNotFoundError(f"Test feature file not found: {FEATURE_FILE}")

    source1_ids = read_test_ids(1)
    source2_ids = set(read_test_ids(2))
    source3_ids = set(read_test_ids(3))
    if source2_ids & source3_ids:
        raise ValueError("Source 2 and Source 3 contain overlapping entity IDs")

    candidate_ids_by_s1 = {entity_id: set() for entity_id in source1_ids}
    matched_ids_by_s1 = {entity_id: set() for entity_id in source1_ids}

    model = load(MODEL_FILE)
    model_features = list(getattr(model, "feature_names_in_", []))
    if model_features != FEATURE_COLUMNS:
        raise ValueError(
            "Saved model feature order does not match the expected training features. "
            f"Model: {model_features}"
        )
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError(f"Saved model has no positive class 1: {classes}")
    positive_class_index = classes.index(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_candidate_pairs = 0
    s2_candidate_pairs = 0
    s3_candidate_pairs = 0
    s2_predicted_matches = 0
    s3_predicted_matches = 0
    duplicate_matches_removed = 0
    minimum_probability = float("inf")
    maximum_probability = float("-inf")

    print(f"Source 1 test entities: {len(source1_ids):,}")
    print(f"Source 2 test entities: {len(source2_ids):,}")
    print(f"Source 3 test entities: {len(source3_ids):,}")
    print(f"Loading test features from: {FEATURE_FILE}")

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            FEATURE_FILE,
            sep="\t",
            dtype={
                "source1_entity_id": str,
                "candidate_entity_id": str,
                "candidate_source": str,
            },
            chunksize=CHUNK_SIZE,
        ),
        start=1,
    ):
        missing = [
            column
            for column in IDENTIFIER_COLUMNS + FEATURE_COLUMNS
            if column not in chunk.columns
        ]
        if missing:
            raise ValueError(f"Test features are missing columns: {missing}")

        features = chunk[FEATURE_COLUMNS].astype(np.float32)
        probabilities = model.predict_proba(features)[:, positive_class_index]
        if len(probabilities):
            minimum_probability = min(minimum_probability, float(probabilities.min()))
            maximum_probability = max(maximum_probability, float(probabilities.max()))

        source1_chunk_ids = chunk["source1_entity_id"].to_numpy()
        candidate_chunk_ids = chunk["candidate_entity_id"].to_numpy()
        candidate_sources = chunk["candidate_source"].to_numpy()

        for source1_id, candidate_id, candidate_source, probability in zip(
            source1_chunk_ids,
            candidate_chunk_ids,
            candidate_sources,
            probabilities,
        ):
            if source1_id not in candidate_ids_by_s1:
                raise ValueError(
                    f"Candidate row references non-test Source 1 ID: {source1_id}"
                )
            if candidate_source == "source2":
                if candidate_id not in source2_ids:
                    raise ValueError(
                        f"Source 2 candidate ID is not in TEST Source 2: {candidate_id}"
                    )
                s2_candidate_pairs += 1
            elif candidate_source == "source3":
                if candidate_id not in source3_ids:
                    raise ValueError(
                        f"Source 3 candidate ID is not in TEST Source 3: {candidate_id}"
                    )
                s3_candidate_pairs += 1
            else:
                raise ValueError(
                    f"Unexpected candidate_source value: {candidate_source!r}"
                )

            total_candidate_pairs += 1
            candidate_ids_by_s1[source1_id].add(candidate_id)

            if probability >= THRESHOLD:
                accepted_ids = matched_ids_by_s1[source1_id]
                if candidate_id in accepted_ids:
                    duplicate_matches_removed += 1
                else:
                    accepted_ids.add(candidate_id)
                    if candidate_source == "source2":
                        s2_predicted_matches += 1
                    else:
                        s3_predicted_matches += 1

        print(
            f"Chunk {chunk_number}: {len(chunk):,} candidate pairs; "
            f"total processed {total_candidate_pairs:,}"
        )

    matching_rows = [
        {
            "source1_entity_id": source1_id,
            "matched_entity_ids": ",".join(sorted(matched_ids_by_s1[source1_id])),
        }
        for source1_id in source1_ids
    ]
    candidate_rows = [
        {
            "source1_entity_id": source1_id,
            "candidate_entity_ids": ",".join(
                sorted(candidate_ids_by_s1[source1_id])
            ),
        }
        for source1_id in source1_ids
    ]

    pd.DataFrame(
        matching_rows,
        columns=["source1_entity_id", "matched_entity_ids"],
    ).to_csv(MATCHING_FILE, sep="\t", index=False)
    pd.DataFrame(
        candidate_rows,
        columns=["source1_entity_id", "candidate_entity_ids"],
    ).to_csv(CANDIDATE_FILE, sep="\t", index=False)

    # Re-read both official-format outputs and verify all requested invariants.
    matching_output = read_output(
        MATCHING_FILE,
        ["source1_entity_id", "matched_entity_ids"],
    )
    candidate_output = read_output(
        CANDIDATE_FILE,
        ["source1_entity_id", "candidate_entity_ids"],
    )
    expected_source1 = set(source1_ids)

    if len(matching_output) != len(source1_ids):
        raise AssertionError("matching_results.tsv row count does not match TEST S1")
    if len(candidate_output) != len(source1_ids):
        raise AssertionError("candidate_pairs.tsv row count does not match TEST S1")
    if matching_output["source1_entity_id"].duplicated().any():
        raise AssertionError("Duplicate Source 1 rows in matching_results.tsv")
    if candidate_output["source1_entity_id"].duplicated().any():
        raise AssertionError("Duplicate Source 1 rows in candidate_pairs.tsv")
    if set(matching_output["source1_entity_id"]) != expected_source1:
        raise AssertionError("matching_results.tsv Source 1 IDs do not match TEST")
    if set(candidate_output["source1_entity_id"]) != expected_source1:
        raise AssertionError("candidate_pairs.tsv Source 1 IDs do not match TEST")

    candidate_output_by_s1 = candidate_output.set_index("source1_entity_id")[
        "candidate_entity_ids"
    ]
    matching_output_by_s1 = matching_output.set_index("source1_entity_id")[
        "matched_entity_ids"
    ]
    for source1_id in source1_ids:
        expected_candidates = candidate_ids_by_s1[source1_id]
        written_candidates = {
            candidate_id
            for candidate_id in candidate_output_by_s1[source1_id].split(",")
            if candidate_id
        }
        if written_candidates != expected_candidates:
            raise AssertionError(
                f"Candidate output differs from model input candidates for {source1_id}"
            )

        matched_list = [
            candidate_id
            for candidate_id in matching_output_by_s1[source1_id].split(",")
            if candidate_id
        ]
        matched_set = set(matched_list)
        if len(matched_list) != len(matched_set):
            raise AssertionError(f"Duplicate matched IDs for {source1_id}")
        if not matched_set.issubset(written_candidates):
            raise AssertionError(f"Non-candidate match found for {source1_id}")
        if not matched_set.issubset(source2_ids | source3_ids):
            raise AssertionError(f"Match ID is outside TEST Source 2/3 for {source1_id}")

    entities_with_matches = sum(bool(ids) for ids in matched_ids_by_s1.values())
    entities_without_matches = len(source1_ids) - entities_with_matches
    total_predicted_matches = s2_predicted_matches + s3_predicted_matches
    match_rate = (
        entities_with_matches / len(source1_ids) * 100 if source1_ids else 0.0
    )
    minimum_text = (
        f"{minimum_probability:.8f}"
        if np.isfinite(minimum_probability)
        else "N/A"
    )
    maximum_text = (
        f"{maximum_probability:.8f}"
        if np.isfinite(maximum_probability)
        else "N/A"
    )

    print("\n" + "=" * 70)
    print("FINAL TEST PREDICTION SUMMARY")
    print("=" * 70)
    print(f"Total Source 1 test entities: {len(source1_ids):,}")
    print(f"Entities with at least one predicted match: {entities_with_matches:,}")
    print(f"Entities with no predicted matches: {entities_without_matches:,}")
    print(f"Total predicted matches: {total_predicted_matches:,}")
    print(f"Source 2 predicted matches: {s2_predicted_matches:,}")
    print(f"Source 3 predicted matches: {s3_predicted_matches:,}")
    print(f"Total candidate pairs: {total_candidate_pairs:,}")
    print(f"Match rate (entities with matches): {match_rate:.2f}%")
    print(f"Number of duplicate matches removed: {duplicate_matches_removed:,}")
    print(f"Minimum probability: {minimum_text}")
    print(f"Maximum probability: {maximum_text}")
    print("\nVerification results:")
    print("[PASS] matching_results.tsv has exactly one row per TEST Source 1 entity")
    print("[PASS] candidate_pairs.tsv has exactly one row per TEST Source 1 entity")
    print("[PASS] Every matched ID is in that entity's candidate list")
    print("[PASS] Every matched ID belongs to TEST Source 2 or Source 3")
    print("[PASS] No duplicate matched IDs")
    print("[PASS] No duplicate Source 1 rows")
    print(f"\nCreated: {MATCHING_FILE}")
    print(f"Created: {CANDIDATE_FILE}")
    print("No test accuracy metrics were calculated.")


if __name__ == "__main__":
    main()