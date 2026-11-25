"""Inference module for PathPilot Review NLP (ABSA).

Owner: Anayan (feature/nlp-model)
Returns aspect, sentiment, confidence, evidence text, source date, model version,
and explicit limitations. Never computes a route safety score.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ml.nlp.labels import (
    DEFAULT_LIMITATIONS,
    ID_TO_POLARITY,
    PATHPILOT_ASPECTS,
    SCHEMA_VERSION,
)
from ml.nlp.preprocess import anonymise_text

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


class ReviewABSAInference:
    """CPU-compatible RoBERTa ABSA inference wrapper with strict output contract."""

    def __init__(
        self,
        model_dir: Optional[Path] = None,
        model_version: str = "pathpilot-distilroberta-absa-v1.0.0",
        model_domain: str = "semeval2014_transfer_benchmark",
        max_length: int = 96,
    ) -> None:
        self.model_dir = Path(model_dir) if model_dir is not None else None
        self.model_version = model_version
        self.model_domain = model_domain
        self.max_length = max_length
        self.device = torch.device("cpu")
        self.tokenizer: Optional[Any] = None
        self.model: Optional[Any] = None

        if self.model_dir is not None and self.model_dir.exists():
            self.load(self.model_dir)

    def load(self, model_dir: Path) -> None:
        """Load saved tokenizer and classification weights onto CPU."""
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(model_dir), local_files_only=True
        ).to(self.device)
        self.model.eval()
        self.model_dir = Path(model_dir)

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    @torch.no_grad()
    def predict_pair(self, text: str, aspect: str) -> Dict[str, Any]:
        """Predict sentiment polarity and softmax confidence for one (text, aspect) pair."""
        if not self.is_loaded:
            raise RuntimeError(
                "Model artefacts are not loaded. Call load(model_dir) or use allow_mock=True explicitly."
            )
        cleaned_text, _ = anonymise_text(text)
        encoded = self.tokenizer(
            f"Aspect: {aspect}",
            cleaned_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        logits = self.model(**encoded).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        conf, pred_id = torch.max(probs, dim=-1)
        sentiment = ID_TO_POLARITY[int(pred_id.item())]
        return {
            "sentiment": sentiment,
            "confidence": round(float(conf.item()), 4),
            "evidence_text": cleaned_text,
            "probabilities": {
                ID_TO_POLARITY[i]: round(float(probs[i].item()), 4) for i in range(len(probs))
            },
        }

    def analyse_reviews(
        self,
        reviews: Sequence[Dict[str, Any]],
        location_id: Optional[str] = None,
        allow_mock: bool = False,
    ) -> Dict[str, Any]:
        """Run contract-compliant ABSA analysis over a list of review payloads."""
        generated_at = datetime.now(timezone.utc).isoformat()
        results: List[Dict[str, Any]] = []
        is_mock = False
        warnings = [
            "Reviews are subjective perceptions and may be sparse or dated; verify source_date before use."
        ]

        if not self.is_loaded:
            if not allow_mock:
                raise RuntimeError("Trained model artefacts not loaded and allow_mock=False.")
            is_mock = True
            warnings.insert(
                0,
                "is_mock=true: trained model weights unavailable; returning neutral contract placeholder.",
            )

        for idx, item in enumerate(reviews):
            raw_text = str(item.get("text", "")).strip()
            cleaned_text, _ = anonymise_text(raw_text)
            review_id = str(item.get("review_id", f"rev_{idx}"))
            loc_id = str(item.get("location_id") or location_id or "unknown")
            source_id = str(item.get("source_id", "unknown"))
            source_date = item.get("source_date")
            aspects = item.get("candidate_aspects") or [PATHPILOT_ASPECTS[7]]

            for aspect in aspects:
                if is_mock:
                    pred = {
                        "sentiment": "neutral",
                        "confidence": 0.0,
                        "evidence_text": cleaned_text,
                    }
                    limitations = [
                        "MOCK FALLBACK OUTPUT (is_mock=true); not produced by trained RoBERTa weights.",
                        *DEFAULT_LIMITATIONS,
                    ]
                else:
                    pred = self.predict_pair(cleaned_text, str(aspect))
                    limitations = list(DEFAULT_LIMITATIONS)

                results.append(
                    {
                        "review_id": review_id,
                        "location_id": loc_id,
                        "aspect": str(aspect),
                        "sentiment": pred["sentiment"],
                        "confidence": pred["confidence"],
                        "evidence_text": pred["evidence_text"],
                        "source_id": source_id,
                        "source_date": source_date,
                        "anonymised": True,
                        "model_version": "mock-fallback-v1.0.0" if is_mock else self.model_version,
                        "limitations": limitations,
                    }
                )

        return {
            "schema_version": SCHEMA_VERSION,
            "generated_at": generated_at,
            "location_id": location_id or "unknown",
            "model_version": "mock-fallback-v1.0.0" if is_mock else self.model_version,
            "model_domain": "mock_fallback" if is_mock else self.model_domain,
            "is_mock": is_mock,
            "results": results,
            "warnings": warnings,
        }
