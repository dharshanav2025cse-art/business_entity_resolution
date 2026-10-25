from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from backend.config import (
    CANDIDATE_PAIRS_FILE,
    MATCHING_RESULTS_FILE,
    SELECTED_MODEL_NAME,
    SELECTED_THRESHOLD,
    TEST_FEATURES_FILE,
)
from backend.services.data_service import DataService
from backend.services.model_service import ModelService


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


class PipelineResultsService:
    def __init__(self, data_service: DataService, model_service: ModelService) -> None:
        self.data_service = data_service
        self.model_service = model_service
        self._candidate_ids_by_s1: dict[str, list[str]] | None = None
        self._matched_ids_by_s1: dict[str, list[str]] | None = None
        self._source_records: dict[str, dict[str, dict[str, str]]] | None = None
        self._selected_model = None

    @staticmethod
    def _split_ids(value: str) -> list[str]:
        return [entity_id.strip() for entity_id in str(value).split(",") if entity_id.strip()]

    def _load_output_rows(self) -> None:
        if self._candidate_ids_by_s1 is not None:
            return
        for path in (CANDIDATE_PAIRS_FILE, MATCHING_RESULTS_FILE):
            if not path.is_file():
                raise FileNotFoundError(f"Required ML output not found: {path}")

        candidates = pd.read_csv(
            CANDIDATE_PAIRS_FILE,
            sep="\t",
            dtype=str,
            keep_default_na=False,
        )
        matches = pd.read_csv(
            MATCHING_RESULTS_FILE,
            sep="\t",
            dtype=str,
            keep_default_na=False,
        )
        if list(candidates.columns) != ["source1_entity_id", "candidate_entity_ids"]:
            raise ValueError("candidate_pairs.tsv has an unexpected schema")
        if list(matches.columns) != ["source1_entity_id", "matched_entity_ids"]:
            raise ValueError("matching_results.tsv has an unexpected schema")
        if candidates["source1_entity_id"].duplicated().any():
            raise ValueError("candidate_pairs.tsv contains duplicate Source 1 rows")
        if matches["source1_entity_id"].duplicated().any():
            raise ValueError("matching_results.tsv contains duplicate Source 1 rows")

        self._candidate_ids_by_s1 = {
            row.source1_entity_id: self._split_ids(row.candidate_entity_ids)
            for row in candidates.itertuples(index=False)
        }
        self._matched_ids_by_s1 = {
            row.source1_entity_id: self._split_ids(row.matched_entity_ids)
            for row in matches.itertuples(index=False)
        }
        if set(self._candidate_ids_by_s1) != set(self._matched_ids_by_s1):
            raise ValueError("Candidate and matching outputs have different Source 1 IDs")

    def _load_source_records(self) -> None:
        if self._source_records is not None:
            return
        self._source_records = {}
        for source_name in ("source2", "source3"):
            source = self.data_service.get_test_source_dataframe(source_name)
            self._source_records[source_name] = (
                source.set_index("entity_id", drop=False).to_dict(orient="index")
            )

    def _source_for_candidate(self, candidate_id: str) -> str:
        self._load_source_records()
        if candidate_id in self._source_records["source2"]:
            return "source2"
        if candidate_id in self._source_records["source3"]:
            return "source3"
        raise ValueError(f"Candidate ID is not present in TEST Source 2 or 3: {candidate_id}")

    def get_candidates_for_entity(self, entity_id: str) -> list[dict[str, Any]]:
        self._load_output_rows()
        if entity_id not in self._candidate_ids_by_s1:
            raise KeyError(f"Entity {entity_id} not found")
        self._load_source_records()

        results = []
        for candidate_id in self._candidate_ids_by_s1[entity_id]:
            source_name = self._source_for_candidate(candidate_id)
            candidate = self._source_records[source_name].get(candidate_id)
            if candidate is None:
                raise ValueError(f"Candidate output references unknown ID: {candidate_id}")
            results.append(
                {
                    "candidate_entity_id": candidate_id,
                    "source": "Source 2" if source_name == "source2" else "Source 3",
                    "business_name": str(candidate.get("business_name", "")),
                    "business_address": str(candidate.get("business_address", "")),
                    "country": str(candidate.get("country", "")),
                }
            )
        return results

    def _get_feature_rows(self, entity_id: str, candidate_ids: set[str]) -> pd.DataFrame:
        if not TEST_FEATURES_FILE.is_file():
            raise FileNotFoundError(f"Required TEST feature file not found: {TEST_FEATURES_FILE}")
        columns = ["source1_entity_id", "candidate_entity_id", "candidate_source", *FEATURE_COLUMNS]
        matching_chunks = []
        for chunk in pd.read_csv(
            TEST_FEATURES_FILE,
            sep="\t",
            usecols=columns,
            dtype={
                "source1_entity_id": str,
                "candidate_entity_id": str,
                "candidate_source": str,
            },
            chunksize=100_000,
        ):
            mask = (chunk["source1_entity_id"] == entity_id) & chunk[
                "candidate_entity_id"
            ].isin(candidate_ids)
            if mask.any():
                matching_chunks.append(chunk.loc[mask])

        if not matching_chunks:
            return pd.DataFrame(columns=columns)
        return pd.concat(matching_chunks, ignore_index=True)

    def get_entity_matches(self, entity_id: str) -> list[dict[str, Any]]:
        self._load_output_rows()
        if entity_id not in self._candidate_ids_by_s1:
            raise KeyError(f"Entity {entity_id} not found")
        candidate_ids = self._candidate_ids_by_s1[entity_id]
        if not candidate_ids:
            return []

        feature_rows = self._get_feature_rows(entity_id, set(candidate_ids))
        if feature_rows["candidate_entity_id"].duplicated().any():
            raise ValueError(f"Duplicate TEST feature rows found for {entity_id}")
        if set(feature_rows["candidate_entity_id"]) != set(candidate_ids):
            raise ValueError(f"TEST feature rows do not match candidate output for {entity_id}")

        if self._selected_model is None:
            self._selected_model = self.model_service.load_model(SELECTED_MODEL_NAME)
        model_features = list(getattr(self._selected_model, "feature_names_in_", []))
        if model_features != FEATURE_COLUMNS:
            raise ValueError("Selected model feature order does not match TEST features")

        probabilities = self._selected_model.predict_proba(
            feature_rows[FEATURE_COLUMNS].astype(np.float32)
        )[:, list(self._selected_model.classes_).index(1)]
        matched_ids = set(self._matched_ids_by_s1[entity_id])
        self._load_source_records()

        results = []
        for row_index, row in feature_rows.reset_index(drop=True).iterrows():
            candidate_id = row["candidate_entity_id"]
            source_name = self._source_for_candidate(candidate_id)
            candidate = self._source_records[source_name][candidate_id]
            probability = float(probabilities[row_index])
            is_match = candidate_id in matched_ids
            if is_match != (probability >= SELECTED_THRESHOLD):
                raise ValueError(
                    f"Saved prediction differs from model threshold for {entity_id}/{candidate_id}"
                )
            results.append(
                {
                    "candidate_entity_id": candidate_id,
                    "source": "Source 2" if source_name == "source2" else "Source 3",
                    "business_name": str(candidate.get("business_name", "")),
                    "business_address": str(candidate.get("business_address", "")),
                    "country": str(candidate.get("country", "")),
                    "probability": probability,
                    "is_match": is_match,
                    "features": {
                        name: float(row[name]) for name in FEATURE_COLUMNS
                    },
                }
            )
        return results

    def get_submission_summary(self) -> dict[str, Any]:
        self._load_output_rows()
        self._load_source_records()
        test_source1_count = len(self.data_service.get_test_entities())
        if len(self._matched_ids_by_s1) != test_source1_count:
            raise ValueError("Submission output does not cover every TEST Source 1 entity")

        source2_ids = set(self._source_records["source2"])
        source3_ids = set(self._source_records["source3"])
        all_matched_ids = [
            entity_id
            for matched_ids in self._matched_ids_by_s1.values()
            for entity_id in matched_ids
        ]
        if any(entity_id not in source2_ids | source3_ids for entity_id in all_matched_ids):
            raise ValueError("matching_results.tsv contains an ID outside TEST Source 2/3")

        total_source1_entities = len(self._matched_ids_by_s1)
        entities_with_matches = sum(bool(ids) for ids in self._matched_ids_by_s1.values())
        total_matches = len(all_matched_ids)
        return {
            "status": "available",
            "available": True,
            "total_source1_entities": total_source1_entities,
            "entities_with_matches": entities_with_matches,
            "entities_without_matches": total_source1_entities - entities_with_matches,
            "total_matches": total_matches,
            "source2_matches": sum(entity_id in source2_ids for entity_id in all_matched_ids),
            "source3_matches": sum(entity_id in source3_ids for entity_id in all_matched_ids),
            "total_candidate_pairs": sum(
                len(candidate_ids) for candidate_ids in self._candidate_ids_by_s1.values()
            ),
            "official_validator_status": "NOT_AVAILABLE",
        }
