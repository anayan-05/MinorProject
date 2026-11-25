"""Unit Tests for API Contract Schema and Mock Fixture Validation.
Owner: Debajyoti (Audio ML Lead)
"""

import os
import json
import pytest

SCHEMA_PATH = "docs/audio/api_contract.json"
FIXTURE_PATH = "ml/audio/fixtures/mock_audio_analysis_response.json"


def test_api_contract_schema_is_valid():
    assert os.path.exists(SCHEMA_PATH), "API contract schema missing!"

    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema = json.load(f)

    assert schema["title"] == "PathPilotAudioAnalyseResponse"
    assert "required" in schema
    assert "requires_user_confirmation" in schema["required"]
    assert "detected_cues" in schema["required"]
    assert "limitations" in schema["required"]

    # Check that requires_user_confirmation has const: true
    user_conf_prop = schema["properties"]["requires_user_confirmation"]
    assert user_conf_prop.get("const") is True


def test_mock_fixture_validates_against_contract():
    assert os.path.exists(FIXTURE_PATH), "Mock fixture JSON missing!"

    with open(FIXTURE_PATH, "r", encoding="utf-8") as f:
        fixture = json.load(f)

    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema = json.load(f)

    # Check required fields
    for field in schema["required"]:
        assert field in fixture, f"Missing required field in mock fixture: {field}"

    assert fixture["status"] in ["success", "error"]
    assert fixture["is_mock"] is True
    assert fixture["requires_user_confirmation"] is True
    assert fixture["schema_version"] == "1.0.0"
    assert isinstance(fixture["detected_cues"], list)
    assert len(fixture["detected_cues"]) > 0

    cue_item = fixture["detected_cues"][0]
    assert "cue" in cue_item
    assert "confidence" in cue_item
    assert "time_offsets" in cue_item
    assert cue_item["cue"] in [
        "screaming_shouting",
        "alarm_siren",
        "glass_breaking",
        "impact_crash",
    ]
    assert 0.0 <= cue_item["confidence"] <= 1.0
    assert len(cue_item["time_offsets"]) == 2

    assert len(fixture["limitations"]) > 0
    # Must explicitly state non-detection of assault in limitations
    assert any("assault" in lim.lower() for lim in fixture["limitations"])
