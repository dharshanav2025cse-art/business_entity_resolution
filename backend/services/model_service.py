from __future__ import annotations

import re
from pathlib import Path

import joblib
import pandas as pd

from backend.config import MODEL_PATHS, RESULT_FILES, SELECTED_MODEL_NAME, SELECTED_THRESHOLD


class ModelService:
    def __init__(self) -> None:
        self._models: dict[str, object] = {}
        self._metrics: list[dict] = []

    def load_model(self, model_name: str):
        model_path = MODEL_PATHS[model_name]
        return joblib.load(model_path)

    def get_selected_model_name(self) -> str:
        return SELECTED_MODEL_NAME

    def get_selected_threshold(self) -> float:
        return float(SELECTED_THRESHOLD)

    def get_model_metrics(self) -> list[dict]:
        if self._metrics:
            return self._metrics

        metrics: list[dict] = []
        for model_name, result_path in RESULT_FILES.items():
            text = result_path.read_text(encoding="utf-8")
            values = {
                "name": model_name,
                "precision": self._extract_metric(text, "Precision:"),
                "recall": self._extract_metric(text, "Recall:"),
                "f05": self._extract_metric(text, "F0.5:"),
                "pr_auc": self._extract_metric(text, "PR-AUC:"),
                "false_positives": self._extract_confusion_value(text, "False Positives ="),
                "false_negatives": self._extract_confusion_value(text, "False Negatives ="),
            }
            metrics.append(values)

        self._metrics = metrics
        return metrics

    def _extract_metric(self, text: str, label: str) -> float:
        match = re.search(rf"{re.escape(label)}\s*([0-9.]+)", text, re.IGNORECASE)
        if match:
            return float(match.group(1))
        return 0.0

    def _extract_confusion_value(self, text: str, label: str) -> int:
        match = re.search(rf"{re.escape(label)}\s*([0-9,]+)", text, re.IGNORECASE)
        if match:
            return int(match.group(1).replace(",", ""))
        return 0

    def get_selected_model_metrics(self) -> dict:
        metrics = self.get_model_metrics()
        selected = max(metrics, key=lambda item: item["f05"])
        selected["selected"] = True
        return selected

    def load_all_models(self) -> dict[str, object]:
        if self._models:
            return self._models
        self._models = {name: self.load_model(name) for name in MODEL_PATHS}
        return self._models
