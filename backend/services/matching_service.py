from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from backend.config import NORMALIZED_TEST_DIR, NORMALIZED_TRAIN_DIR, SELECTED_THRESHOLD


FEATURE_NAMES = [
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


class MatchingService:
    def __init__(self) -> None:
        self._train_frames = None
        self._test_frames = None
        self._name_vectorizer = None
        self._address_vectorizer = None
        self._s1_name_positions = None
        self._s2_name_positions = None
        self._s3_name_positions = None
        self._s1_address_positions = None
        self._s2_address_positions = None
        self._s3_address_positions = None
        self._s1_name_matrix = None
        self._s2_name_matrix = None
        self._s3_name_matrix = None
        self._s1_address_matrix = None
        self._s2_address_matrix = None
        self._s3_address_matrix = None
        self._ready = False

    def _load_training_data(self):
        if self._ready:
            return

        s1 = pd.read_csv(NORMALIZED_TRAIN_DIR / "train_source1.tsv", sep="\t", dtype=str).fillna("")
        s2 = pd.read_csv(NORMALIZED_TRAIN_DIR / "train_source2.tsv", sep="\t", dtype=str).fillna("")
        s3 = pd.read_csv(NORMALIZED_TRAIN_DIR / "train_source3.tsv", sep="\t", dtype=str).fillna("")

        self._train_frames = {"source1": s1, "source2": s2, "source3": s3}
        self._build_tfidf_models()
        self._ready = True

    def _load_test_data(self):
        if self._test_frames is not None:
            return self._test_frames
        self._test_frames = {
            "source1": pd.read_csv(NORMALIZED_TEST_DIR / "test_source1.tsv", sep="\t", dtype=str).fillna(""),
            "source2": pd.read_csv(NORMALIZED_TEST_DIR / "test_source2.tsv", sep="\t", dtype=str).fillna(""),
            "source3": pd.read_csv(NORMALIZED_TEST_DIR / "test_source3.tsv", sep="\t", dtype=str).fillna(""),
        }
        return self._test_frames

    def _build_tfidf_models(self):
        s1 = self._train_frames["source1"]
        s2 = self._train_frames["source2"]
        s3 = self._train_frames["source3"]

        name_vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2, max_features=100_000)
        address_vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2, max_features=100_000)

        all_names = pd.concat([s1["business_name_normalized"], s2["business_name_normalized"], s3["business_name_normalized"]], ignore_index=True)
        all_addresses = pd.concat([s1["business_address_normalized"], s2["business_address_normalized"], s3["business_address_normalized"]], ignore_index=True)

        name_vectorizer.fit(all_names)
        address_vectorizer.fit(all_addresses)

        self._name_vectorizer = name_vectorizer
        self._address_vectorizer = address_vectorizer
        self._s1_name_positions = {entity_id: idx for idx, entity_id in enumerate(s1["entity_id"])}
        self._s2_name_positions = {entity_id: idx for idx, entity_id in enumerate(s2["entity_id"])}
        self._s3_name_positions = {entity_id: idx for idx, entity_id in enumerate(s3["entity_id"])}
        self._s1_address_positions = {entity_id: idx for idx, entity_id in enumerate(s1["entity_id"])}
        self._s2_address_positions = {entity_id: idx for idx, entity_id in enumerate(s2["entity_id"])}
        self._s3_address_positions = {entity_id: idx for idx, entity_id in enumerate(s3["entity_id"])}

        self._s1_name_matrix = self._name_vectorizer.transform(s1["business_name_normalized"])
        self._s2_name_matrix = self._name_vectorizer.transform(s2["business_name_normalized"])
        self._s3_name_matrix = self._name_vectorizer.transform(s3["business_name_normalized"])
        self._s1_address_matrix = self._address_vectorizer.transform(s1["business_address_normalized"])
        self._s2_address_matrix = self._address_vectorizer.transform(s2["business_address_normalized"])
        self._s3_address_matrix = self._address_vectorizer.transform(s3["business_address_normalized"])

    def _normalize_country(self, value: Any) -> str:
        if pd.isna(value):
            return ""
        text = str(value).strip().lower()
        text = re.sub(r"\s+", " ", text)
        country_map = {
            "india": "india",
            "in": "india",
            "ind": "india",
            "bharat": "india",
            "republic of india": "india",
            "usa": "usa",
            "us": "usa",
            "u.s.": "usa",
            "u.s.a.": "usa",
            "united states": "usa",
            "united states of america": "usa",
            "france": "france",
            "fr": "france",
            "fra": "france",
            "french republic": "france",
            "république française": "france",
        }
        return country_map.get(text, text)

    def _remove_accents(self, text: str) -> str:
        text = unicodedata.normalize("NFKD", str(text))
        return "".join(c for c in text if not unicodedata.combining(c))

    def _normalize_basic(self, value: Any) -> str:
        if value is None or pd.isna(value):
            return ""
        text = self._remove_accents(str(value)).lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _normalize_business_name(self, value: Any) -> str:
        text = self._normalize_basic(value)
        if not text:
            return ""
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
            "solutions": "soln",
        }
        for old, new in replacements.items():
            text = re.sub(rf"\b{re.escape(old)}\b", new, text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _normalize_name_tokens(self, value: Any) -> str:
        text = self._normalize_business_name(value)
        if not text:
            return ""
        legal_terms = {"pvt", "ltd", "llc", "inc", "corp", "co", "sa", "sas", "sarl"}
        tokens = [token for token in text.split() if token not in legal_terms]
        tokens = sorted(tokens)
        return " ".join(tokens)

    def _normalize_address(self, value: Any) -> str:
        text = self._normalize_basic(value)
        if not text:
            return ""
        token_map = {
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
            "sq": "square",
        }
        tokens = [token_map.get(token, token) for token in text.split()]
        return " ".join(tokens)

    def _normalize_address_tokens(self, value: Any) -> str:
        text = self._normalize_address(value)
        if not text:
            return ""
        removable_words = {"near", "opposite", "beside", "behind", "next", "to"}
        tokens = [token for token in text.split() if token not in removable_words]
        tokens = sorted(tokens)
        return " ".join(tokens)

    def _jaccard_pair(self, left: str, right: str) -> float:
        if not left and not right:
            return 1.0
        if not left or not right:
            return 0.0
        left_set = set(left.split())
        right_set = set(right.split())
        if not left_set and not right_set:
            return 1.0
        union = left_set | right_set
        if not union:
            return 0.0
        return len(left_set & right_set) / len(union)

    def _get_tokens(self, text: str) -> list[str]:
        text = self._normalize_basic(text).lower()
        tokens = re.findall(r"[a-z0-9]+", text)
        return [token for token in tokens if len(token) >= 3]

    def _generate_candidate_ids_for_entity(self, source1_entity_id: str, source_df: pd.DataFrame) -> set[str]:
        s1_row = self._load_test_data()["source1"][self._load_test_data()["source1"]["entity_id"] == source1_entity_id].iloc[0]
        source1_name = self._normalize_business_name(s1_row["business_name"])
        source1_token_name = self._normalize_name_tokens(s1_row["business_name"])
        source1_country = self._normalize_country(s1_row["country"])
        source1_address_token = self._normalize_address_tokens(s1_row["business_address"])

        candidate_dict: set[str] = set()
        exact_name_index = defaultdict(set)
        token_name_index = defaultdict(set)
        name_token_index = defaultdict(set)
        address_token_index = defaultdict(set)
        country_initial_index = defaultdict(set)

        for _, row in source_df.iterrows():
            entity_id = str(row["entity_id"]).strip()
            name = self._normalize_business_name(row["business_name"])
            token_name = self._normalize_name_tokens(row["business_name"])
            address_token = self._normalize_address_tokens(row["business_address"])
            country = self._normalize_country(row["country"])
            if name:
                exact_name_index[name].add(entity_id)
            if token_name:
                token_name_index[token_name].add(entity_id)
            for token in self._get_tokens(token_name):
                name_token_index[token].add(entity_id)
            for token in self._get_tokens(address_token):
                address_token_index[token].add(entity_id)
            if country and name:
                country_initial_index[(country, name[0])].add(entity_id)

        if source1_name:
            for candidate_id in exact_name_index.get(source1_name, set()):
                candidate_dict.add(candidate_id)
        if source1_token_name:
            for candidate_id in token_name_index.get(source1_token_name, set()):
                candidate_dict.add(candidate_id)
        for token in self._get_tokens(source1_token_name):
            for candidate_id in name_token_index.get(token, set()):
                candidate_dict.add(candidate_id)
        for token in self._get_tokens(source1_address_token):
            for candidate_id in address_token_index.get(token, set()):
                candidate_dict.add(candidate_id)
        if source1_country and source1_name:
            for candidate_id in country_initial_index.get((source1_country, source1_name[0]), set()):
                candidate_dict.add(candidate_id)

        if candidate_dict:
            source_names = source_df["business_name"].fillna("").astype(str).tolist()
            vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=1, sublinear_tf=True)
            other_matrix = vectorizer.fit_transform(source_names)
            s1_names = [source1_name]
            s1_matrix = vectorizer.transform(s1_names)
            n_neighbors = min(20, len(source_df))
            if n_neighbors > 0:
                from sklearn.neighbors import NearestNeighbors
                nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", algorithm="brute")
                nn.fit(other_matrix)
                _, indices = nn.kneighbors(s1_matrix)
                candidate_ids = [str(source_df.iloc[index]["entity_id"]).strip() for index in indices[0]]
                candidate_dict.update(candidate_ids)
        return candidate_dict

    def get_candidates_for_entity(self, entity_id: str) -> list[dict[str, Any]]:
        test_frames = self._load_test_data()
        if entity_id not in test_frames["source1"]["entity_id"].tolist():
            raise KeyError(f"Entity {entity_id} not found")

        source1_row = test_frames["source1"][test_frames["source1"]["entity_id"] == entity_id].iloc[0]
        candidates: list[dict[str, Any]] = []
        for source_name in ["source2", "source3"]:
            source_df = test_frames[source_name]
            candidate_ids = self._generate_candidate_ids_for_entity(entity_id, source_df)
            for candidate_id in sorted(candidate_ids):
                row = source_df[source_df["entity_id"] == candidate_id]
                if row.empty:
                    continue
                candidate = row.iloc[0]
                candidate_record = {
                    "candidate_entity_id": str(candidate["entity_id"]),
                    "source": "Source 2" if source_name == "source2" else "Source 3",
                    "business_name": str(candidate.get("business_name", "")),
                    "business_address": str(candidate.get("business_address", "")),
                    "country": str(candidate.get("country", "")),
                }
                candidates.append(candidate_record)
        return sorted(candidates, key=lambda item: item["source"] + item["candidate_entity_id"])

    def _compute_pair_features(self, source1_row: pd.Series, candidate_row: pd.Series) -> tuple[dict[str, float | int], dict[str, float | int]]:
        self._load_training_data()

        s1_name = self._normalize_business_name(source1_row["business_name"])
        candidate_name = self._normalize_business_name(candidate_row["business_name"])
        s1_name_token = self._normalize_name_tokens(source1_row["business_name"])
        candidate_name_token = self._normalize_name_tokens(candidate_row["business_name"])
        s1_address = self._normalize_address(source1_row["business_address"])
        candidate_address = self._normalize_address(candidate_row["business_address"])
        s1_address_token = self._normalize_address_tokens(source1_row["business_address"])
        candidate_address_token = self._normalize_address_tokens(candidate_row["business_address"])
        s1_country = self._normalize_country(source1_row["country"])
        candidate_country = self._normalize_country(candidate_row["country"])

        name_exact = int(bool(s1_name and candidate_name and s1_name == candidate_name))
        name_token_exact = int(bool(s1_name_token and candidate_name_token and s1_name_token == candidate_name_token))
        name_jaccard = float(self._jaccard_pair(s1_name_token, candidate_name_token))
        name_length_difference = abs(len(s1_name) - len(candidate_name))
        s1_name_length = float(max(len(s1_name), 1))
        candidate_name_length = float(max(len(candidate_name), 1))
        name_length_ratio = min(s1_name_length, candidate_name_length) / max(s1_name_length, candidate_name_length)

        name_vector = self._name_vectorizer.transform([s1_name])[0]
        candidate_name_vector = self._name_vectorizer.transform([candidate_name])[0]
        name_tfidf = float((name_vector.multiply(candidate_name_vector)).sum())

        address_exact = int(bool(s1_address and candidate_address and s1_address == candidate_address))
        address_token_exact = int(bool(s1_address_token and candidate_address_token and s1_address_token == candidate_address_token))
        address_jaccard = float(self._jaccard_pair(s1_address_token, candidate_address_token))
        address_length_difference = abs(len(s1_address) - len(candidate_address))
        s1_address_length = float(max(len(s1_address), 1))
        candidate_address_length = float(max(len(candidate_address), 1))
        address_length_ratio = min(s1_address_length, candidate_address_length) / max(s1_address_length, candidate_address_length)

        address_vector = self._address_vectorizer.transform([s1_address])[0]
        candidate_address_vector = self._address_vectorizer.transform([candidate_address])[0]
        address_tfidf = float((address_vector.multiply(candidate_address_vector)).sum())

        country_exact = int(bool(s1_country and candidate_country and s1_country == candidate_country))
        source1_name_missing = int(not bool(s1_name))
        candidate_name_missing = int(not bool(candidate_name))
        source1_address_missing = int(not bool(s1_address))
        candidate_address_missing = int(not bool(candidate_address))
        source1_country_missing = int(not bool(s1_country))
        candidate_country_missing = int(not bool(candidate_country))

        feature_values = {
            "name_exact": name_exact,
            "name_token_exact": name_token_exact,
            "name_jaccard": name_jaccard,
            "name_length_difference": name_length_difference,
            "name_length_ratio": name_length_ratio,
            "name_tfidf": name_tfidf,
            "address_exact": address_exact,
            "address_token_exact": address_token_exact,
            "address_jaccard": address_jaccard,
            "address_length_difference": address_length_difference,
            "address_length_ratio": address_length_ratio,
            "address_tfidf": address_tfidf,
            "country_exact": country_exact,
            "source1_name_missing": source1_name_missing,
            "candidate_name_missing": candidate_name_missing,
            "source1_address_missing": source1_address_missing,
            "candidate_address_missing": candidate_address_missing,
            "source1_country_missing": source1_country_missing,
            "candidate_country_missing": candidate_country_missing,
        }

        vector = [
            feature_values[feature] for feature in FEATURE_NAMES
        ]
        return feature_values, {"vector": vector}

    def get_entity_matches(self, entity_id: str, model, selected_threshold: float = SELECTED_THRESHOLD) -> list[dict[str, Any]]:
        test_frames = self._load_test_data()
        source1_df = test_frames["source1"]
        if entity_id not in source1_df["entity_id"].tolist():
            raise KeyError(f"Entity {entity_id} not found")

        source1_row = source1_df[source1_df["entity_id"] == entity_id].iloc[0]
        results: list[dict[str, Any]] = []

        for source_name in ["source2", "source3"]:
            candidate_df = test_frames[source_name]
            candidate_ids = self._generate_candidate_ids_for_entity(entity_id, candidate_df)
            for candidate_id in sorted(candidate_ids):
                candidate_row = candidate_df[candidate_df["entity_id"] == candidate_id]
                if candidate_row.empty:
                    continue
                candidate_row = candidate_row.iloc[0]
                feature_values, _ = self._compute_pair_features(source1_row, candidate_row)
                vector = np.asarray([[feature_values[feature] for feature in FEATURE_NAMES]], dtype=np.float32)
                probability = float(model.predict_proba(vector)[0, 1])
                is_match = bool(probability >= selected_threshold)
                result = {
                    "candidate_entity_id": str(candidate_row["entity_id"]),
                    "source": "Source 2" if source_name == "source2" else "Source 3",
                    "business_name": str(candidate_row.get("business_name", "")),
                    "business_address": str(candidate_row.get("business_address", "")),
                    "country": str(candidate_row.get("country", "")),
                    "probability": round(probability, 4),
                    "is_match": is_match,
                    "features": {
                        key: round(float(value), 6) if isinstance(value, (int, float)) and key not in {"source1_name_missing", "candidate_name_missing", "source1_address_missing", "candidate_address_missing", "source1_country_missing", "candidate_country_missing"} else int(value)
                        for key, value in feature_values.items()
                    },
                }
                results.append(result)

        return sorted(results, key=lambda item: item["source"] + item["candidate_entity_id"])

    def get_entity_by_id(self, entity_id: str) -> dict[str, Any]:
        test_frames = self._load_test_data()
        rows = test_frames["source1"][test_frames["source1"]["entity_id"] == entity_id]
        if rows.empty:
            raise KeyError(f"Entity {entity_id} not found")
        return rows.iloc[0].to_dict()
