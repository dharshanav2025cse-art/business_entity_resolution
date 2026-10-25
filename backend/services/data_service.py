from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from backend.config import (
    CANDIDATES_DIR,
    DATASET_ROOT,
    NORMALIZED_TEST_DIR,
    TEST_DIR,
    TRAIN_DIR,
)


class DataService:
    def __init__(self) -> None:
        self._train_cache: dict[str, pd.DataFrame] = {}
        self._test_cache: dict[str, pd.DataFrame] = {}

    def _read_tsv(self, path: Path, limit: int | None = None) -> pd.DataFrame:
        df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False).fillna("")
        if limit is not None:
            return df.head(limit)
        return df

    def get_train_counts(self) -> dict[str, int]:
        files = {
            "source1_train_count": TRAIN_DIR / "train_source1.tsv",
            "source2_train_count": TRAIN_DIR / "train_source2.tsv",
            "source3_train_count": TRAIN_DIR / "train_source3.tsv",
        }
        return {
            key: len(self._read_tsv(path))
            for key, path in files.items()
        }

    def get_test_counts(self) -> dict[str, int | None]:
        files = {
            "source1_test_count": TEST_DIR / "test_source1.tsv",
            "source2_test_count": TEST_DIR / "test_source2.tsv",
            "source3_test_count": TEST_DIR / "test_source3.tsv",
        }
        return {
            key: len(self._read_tsv(path)) if path.exists() else None
            for key, path in files.items()
        }

    def get_dashboard_summary(self) -> dict[str, Any]:
        summary_map = {
            "blocking_recall": 0.0,
            "search_space_reduction": 0.0,
            "positive_matches": 0,
            "training_candidate_count": 0,
        }
        summary_path = CANDIDATES_DIR / "blocking_summary.txt"
        if summary_path.exists():
            text = summary_path.read_text(encoding="utf-8")
            for line in text.splitlines():
                if "Overall blocking recall:" in line:
                    summary_map["blocking_recall"] = float(line.split(":", 1)[1].strip().replace("%", "")) / 100
                elif "Search-space reduction:" in line:
                    summary_map["search_space_reduction"] = float(line.split(":", 1)[1].strip().replace("%", ""))
                elif "Candidate comparisons:" in line:
                    summary_map["training_candidate_count"] = int(
                        line.split(":", 1)[1].strip().replace(",", "")
                    )
        feature_summary_path = DATASET_ROOT / "features" / "feature_summary.txt"
        if feature_summary_path.exists():
            content = feature_summary_path.read_text(encoding="utf-8")
            match = re.search(r"Positive matches:\s*([0-9,]+)", content)
            if match:
                summary_map["positive_matches"] = int(match.group(1).replace(",", ""))

        return summary_map

    def get_test_entities(self) -> pd.DataFrame:
        if "source1" not in self._test_cache:
            self._test_cache["source1"] = self._read_tsv(
                TEST_DIR / "test_source1.tsv"
            )
        return self._test_cache["source1"]

    def search_entities(self, query: str, limit: int = 25) -> list[dict[str, str]]:
        if not query or not query.strip():
            return []
        df = self.get_test_entities()
        q = query.strip().lower()
        mask = (
            df["entity_id"].astype(str).str.lower().str.contains(q, na=False)
            | df["business_name"].astype(str).str.lower().str.contains(q, na=False)
            | df["business_address"].astype(str).str.lower().str.contains(q, na=False)
            | df["country"].astype(str).str.lower().str.contains(q, na=False)
        )
        results = df.loc[mask, ["entity_id", "business_name", "business_address", "country"]].head(limit)
        return results.to_dict(orient="records")

    def get_entity(self, entity_id: str) -> dict[str, str]:
        df = self.get_test_entities()
        match = df[df["entity_id"].astype(str) == str(entity_id).strip()]
        if match.empty:
            raise KeyError(f"Entity {entity_id} not found")
        return match.iloc[0].to_dict()

    def get_test_source_dataframe(self, source_name: str) -> pd.DataFrame:
        source_map = {
            "source2": NORMALIZED_TEST_DIR / "test_source2.tsv",
            "source3": NORMALIZED_TEST_DIR / "test_source3.tsv",
        }
        if source_name not in self._test_cache:
            raw_source_map = {
                "source2": TEST_DIR / "test_source2.tsv",
                "source3": TEST_DIR / "test_source3.tsv",
            }
            self._test_cache[source_name] = self._read_tsv(
                raw_source_map[source_name]
            )
        return self._test_cache[source_name]

    def get_candidate_pairs_for_entity(self, entity_id: str) -> list[dict[str, str]]:
        source1 = self.get_test_entities()
        if entity_id not in source1["entity_id"].tolist():
            raise KeyError(f"Entity {entity_id} not found")

        candidate_rows: list[dict[str, str]] = []
        for gauge in ("source2", "source3"):
            candidate_table = self.get_test_source_dataframe(gauge)
            for _, row in candidate_table.iterrows():
                candidate_rows.append({
                    "candidate_entity_id": row["entity_id"],
                    "source": "Source 2" if gauge == "source2" else "Source 3",
                    "business_name": row.get("business_name", ""),
                    "business_address": row.get("business_address", ""),
                    "country": row.get("country", ""),
                })
        return candidate_rows[:MAX_DATASET_ROWS]

    def get_submission_status(self) -> dict[str, Any]:
        output_dir = DATASET_ROOT.parent / "output"
        files = [output_dir / "matching_results.tsv", output_dir / "candidate_pairs.tsv"]
        if all(path.exists() for path in files):
            return {"status": "available", "available": True}
        return {
            "status": "not_available",
            "available": False,
            "message": "Final submission files not found in output/.",
            "missing_files": [path.name for path in files if not path.exists()],
        }
