"""Unit Tests for Audited Dataset & Licence Manifest Registries.
Owner: Debajyoti (Audio ML Lead)
"""

import os
import csv
import json
import pytest

from ml.audio.src.model import PathPilotAudioCNN

DATASET_REGISTER_PATH = "ml/audio/registry/dataset_acquisition_register.csv"
LICENCE_MANIFEST_PATH = "ml/audio/registry/licence_manifest.csv"
LABEL_MAPPING_PATH = "ml/audio/registry/label_mapping.json"


def test_dataset_acquisition_register_schema_and_contents():
    assert os.path.exists(DATASET_REGISTER_PATH), "Dataset acquisition register missing!"

    expected_headers = [
        "dataset_id",
        "owner",
        "name",
        "source_url",
        "version",
        "access_date",
        "licence_or_terms",
        "allowed_use",
        "attribution",
        "checksum",
        "storage_path",
        "notes",
    ]

    with open(DATASET_REGISTER_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == expected_headers

        rows = list(reader)
        assert len(rows) >= 3, "Expected at least ESC-50, UrbanSound8K, and FSD50K"

        dataset_names = [r["name"] for r in rows]
        assert any("ESC-50" in name for name in dataset_names)
        assert any("UrbanSound8K" in name for name in dataset_names)
        assert any("FSD50K" in name for name in dataset_names)

        for r in rows:
            assert r["owner"] == "Debajyoti"
            assert r["licence_or_terms"] != ""
            assert r["attribution"] != ""


def test_licence_manifest_schema_and_allowlist_rules():
    assert os.path.exists(LICENCE_MANIFEST_PATH), "Licence manifest missing!"

    expected_headers = [
        "clip_id",
        "source_dataset",
        "source_label",
        "mapped_label",
        "licence",
        "allowed_use",
        "sample_rate",
        "duration_seconds",
        "split_group",
        "included",
        "exclusion_reason",
    ]

    with open(LICENCE_MANIFEST_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == expected_headers

        rows = list(reader)
        assert len(rows) > 0

        included_count = 0
        excluded_count = 0

        for r in rows:
            if r["included"].upper() == "TRUE":
                included_count += 1
                assert r["exclusion_reason"] == "none"
                assert r["mapped_label"] in PathPilotAudioCNN.CLASS_NAMES
                assert r["split_group"] != ""
            else:
                excluded_count += 1
                # Must have an explicit and transparent exclusion reason
                assert r["exclusion_reason"] != "none"
                assert len(r["exclusion_reason"]) > 3

        assert included_count > 0, "No clips marked included=TRUE"
        assert excluded_count > 0, "No clips marked included=FALSE (needed to document exclusions)"


def test_label_mapping_json_taxonomy_and_prohibited_terms():
    assert os.path.exists(LABEL_MAPPING_PATH), "Label mapping JSON missing!"

    with open(LABEL_MAPPING_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Check prohibited labels policy
    assert "taxonomy_policy" in data
    prohibited = data["taxonomy_policy"]["strictly_prohibited_labels"]
    assert "assault" in prohibited
    assert "assault_detected" in prohibited
    assert "distress" in prohibited

    # Check target classes
    assert "target_classes" in data
    assert len(data["target_classes"]) == 5
    target_names = [c["name"] for c in data["target_classes"]]
    assert set(target_names) == set(PathPilotAudioCNN.CLASS_NAMES)

    # Check source dataset mappings
    assert "source_mappings" in data
    assert "ESC-50" in data["source_mappings"]
    assert "UrbanSound8K" in data["source_mappings"]
    assert "FSD50K" in data["source_mappings"]
