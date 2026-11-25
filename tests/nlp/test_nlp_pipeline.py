"""Unit tests for PathPilot Review NLP preprocessing, leakage checks, and inference contract."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ml.nlp.inference import ReviewABSAInference
from ml.nlp.labels import PATHPILOT_ASPECTS, POLARITIES
from ml.nlp.preprocess import (
    anonymise_text,
    assert_no_split_leakage,
    deduplicate_records,
    grouped_train_val_split,
    parse_semeval2014_restaurant_xml,
    validate_annotation_df,
)


def _sample_annotation_df() -> pd.DataFrame:
    rows = []
    for idx in range(12):
        loc = f"kmc_ward_{idx // 2:03d}"
        sentiment = POLARITIES[idx % 3]
        aspect = PATHPILOT_ASPECTS[idx % len(PATHPILOT_ASPECTS)]
        rows.append(
            {
                "review_id": f"rev_{idx}",
                "location_id": loc,
                "group_id": loc,
                "text": f"Unique street observation number {idx} near metro station exit {idx * 7}.",
                "aspect": aspect,
                "sentiment": sentiment,
                "evidence_span_start": -1,
                "evidence_span_end": -1,
                "source_id": "consented_pilot",
                "collection_date": "2026-11-25",
                "anonymised": True,
                "adjudication_status": "double_agreed",
            }
        )
    return pd.DataFrame(rows)


def test_anonymise_text_redacts_pii() -> None:
    raw = "Contact Mr. Rahul Sen at +91 9876543210 or rahul.sen@example.com or @rahul_k via https://example.com"
    cleaned, changed = anonymise_text(raw)
    assert changed is True
    assert "9876543210" not in cleaned
    assert "rahul.sen@example.com" not in cleaned
    assert "@rahul_k" not in cleaned
    assert "https://example.com" not in cleaned
    assert "[REDACTED_NAME]" in cleaned
    assert "[REDACTED_PHONE]" in cleaned
    assert "[REDACTED_EMAIL]" in cleaned
    assert "[REDACTED_HANDLE]" in cleaned
    assert "[REDACTED_URL]" in cleaned


def test_validate_annotation_df_valid_and_invalid() -> None:
    df = _sample_annotation_df()
    assert validate_annotation_df(df) == []

    bad = df.copy()
    bad.loc[0, "aspect"] = "nonexistent_aspect"
    bad.loc[1, "anonymised"] = False
    errors = validate_annotation_df(bad)
    assert len(errors) >= 2


def test_deduplicate_and_grouped_split_zero_leakage() -> None:
    df = _sample_annotation_df()
    dup = df.iloc[[0]].copy()
    dup["review_id"] = "rev_dup"
    df_with_dup = pd.concat([df, dup], ignore_index=True)

    deduped, removed = deduplicate_records(df_with_dup)
    assert removed == 1
    assert len(deduped) == len(df)

    train_df, val_df = grouped_train_val_split(deduped, group_col="group_id", val_size=0.33, seed=42)
    assert_no_split_leakage({"train": train_df, "val": val_df}, group_col="group_id")


def test_assert_no_split_leakage_catches_near_duplicate() -> None:
    df = _sample_annotation_df()
    a = df.iloc[:6].copy()
    b = df.iloc[6:].copy()
    b.loc[b.index[0], "text"] = a.loc[a.index[0], "text"] + "!"
    with pytest.raises(AssertionError):
        assert_no_split_leakage({"train": a, "val": b}, group_col="group_id", near_dup_threshold=0.90)


def test_parse_semeval_trial_xml_if_present(tmp_path: Path) -> None:
    sample_xml = tmp_path / "sample.xml"
    sample_xml.write_text(
        """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<sentences>
  <sentence id="101">
    <text>Great ambience near the corner, though pricey.</text>
    <aspectCategories>
      <aspectCategory category="ambience" polarity="positive"/>
      <aspectCategory category="price" polarity="negative"/>
      <aspectCategory category="food" polarity="conflict"/>
    </aspectCategories>
  </sentence>
</sentences>
""",
        encoding="utf-8",
    )
    df, stats = parse_semeval2014_restaurant_xml(sample_xml, source_id="unit_test")
    assert stats.total_sentences == 1
    assert stats.excluded_conflict == 1
    assert len(df) == 2
    assert set(df["sentiment"]) == {"positive", "negative"}


def test_inference_mock_contract_and_no_safety_score() -> None:
    engine = ReviewABSAInference(model_dir=None)
    payload = engine.analyse_reviews(
        reviews=[
            {
                "review_id": "r1",
                "text": "Bright streetlights near Mr. Das shop at 9 PM.",
                "source_id": "consented_pilot",
                "source_date": "2026-11-25T21:00:00+05:30",
                "candidate_aspects": ["lighting"],
            }
        ],
        location_id="kmc_ward_063",
        allow_mock=True,
    )
    assert payload["is_mock"] is True
    assert payload["schema_version"] == "1.0.0"
    item = payload["results"][0]
    assert item["aspect"] == "lighting"
    assert item["sentiment"] in POLARITIES
    assert "safety_score" not in item
    assert "safety_score" not in payload
    assert "Mr. Das" not in item["evidence_text"]
    assert len(item["limitations"]) >= 1


def test_training_notebook_is_valid_nbformat() -> None:
    import nbformat

    nb_path = Path("ml/nlp/training/pathpilot_review_absa.ipynb")
    assert nb_path.exists()
    nb = nbformat.read(str(nb_path), as_version=4)
    nbformat.validate(nb)
    full_src = "\n".join(c.source for c in nb.cells)
    for required in [
        "assert_no_split_leakage",
        "anonymise_text",
        "train_and_evaluate",
        "confusion_matrix",
        "ReviewABSAInference",
        "DRIVE_BASE_DIR",
    ]:
        assert required in full_src, f"Missing {required} in notebook"


def test_trained_model_inference_if_present() -> None:
    model_dir = Path("artifacts/models/pathpilot-roberta-absa")
    if not (model_dir / "config.json").exists():
        pytest.skip("Local trained weights not present in gitignored artifacts/models/")
    engine = ReviewABSAInference(model_dir=model_dir)
    payload = engine.analyse_reviews(
        reviews=[
            {
                "review_id": "smoke_1",
                "text": "Great lighting near the metro gate, though the side lane feels deserted.",
                "source_id": "smoke_test",
                "source_date": "2026-11-25T23:00:00+05:30",
                "candidate_aspects": ["lighting", "isolation"],
            }
        ],
        location_id="kmc_ward_063_park_street",
        allow_mock=False,
    )
    assert payload["is_mock"] is False
    assert len(payload["results"]) == 2
    for res in payload["results"]:
        assert res["sentiment"] in POLARITIES
        assert 0.0 <= res["confidence"] <= 1.0
        assert "safety_score" not in res


