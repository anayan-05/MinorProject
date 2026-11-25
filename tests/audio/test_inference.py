"""Unit Tests for Audio Inference Engine and Safety Invariants.
Owner: Debajyoti (Audio ML Lead)
"""

import pytest
import numpy as np

from ml.audio.src.inference import AudioInferenceEngine
from ml.audio.src.model import PathPilotAudioCNN

try:
    import torch
    HAS_TORCH = True
except (ImportError, OSError):
    HAS_TORCH = False
    torch = None


@pytest.fixture
def engine():
    return AudioInferenceEngine(is_mock=False)


@pytest.fixture
def mock_engine():
    return AudioInferenceEngine(is_mock=True)


def test_inference_engine_contract_structure(engine):
    preprocessor = engine.preprocessor
    scream_waveform = preprocessor.generate_synthetic_cue("screaming_shouting")

    response = engine.analyse(scream_waveform)

    # Verify all required keys in API contract
    required_keys = [
        "status",
        "is_mock",
        "model_version",
        "schema_version",
        "generated_at",
        "detected_cues",
        "requires_user_confirmation",
        "limitations",
        "warnings",
    ]
    for key in required_keys:
        assert key in response, f"Missing key: {key}"

    assert response["status"] == "success"
    assert response["is_mock"] is False
    assert response["schema_version"] == "1.0.0"
    assert isinstance(response["limitations"], list)
    assert len(response["limitations"]) > 0


def test_requires_user_confirmation_is_strictly_true(engine, mock_engine):
    # Safety Invariant: requires_user_confirmation must ALWAYS be True
    preprocessor = engine.preprocessor

    for cue_name in ["screaming_shouting", "alarm_siren", "background_normal"]:
        wf = preprocessor.generate_synthetic_cue(cue_name)
        res = engine.analyse(wf)
        assert res["requires_user_confirmation"] is True, "requires_user_confirmation violated!"

    dummy = torch.zeros(1, 48000) if HAS_TORCH else np.zeros((1, 48000), dtype=np.float32)
    mock_res = mock_engine.analyse(dummy)
    assert mock_res["requires_user_confirmation"] is True


def test_detected_cues_validity_and_offsets(engine):
    preprocessor = engine.preprocessor
    siren_wf = preprocessor.generate_synthetic_cue("alarm_siren")

    response = engine.analyse(siren_wf, override_threshold=0.30)
    assert "detected_cues" in response
    assert isinstance(response["detected_cues"], list)

    for item in response["detected_cues"]:
        assert "cue" in item
        assert "confidence" in item
        assert "time_offsets" in item

        # Cue must belong to valid acoustic classes
        assert item["cue"] in PathPilotAudioCNN.CLASS_NAMES
        assert item["cue"] != "background_normal"

        # Confidence bounds
        assert 0.0 <= item["confidence"] <= 1.0

        # Time offsets
        assert len(item["time_offsets"]) == 2
        assert item["time_offsets"][0] == 0.0
        assert item["time_offsets"][1] == 3.0


def test_mock_engine_mode(mock_engine):
    dummy = torch.zeros(1, 48000) if HAS_TORCH else np.zeros((1, 48000), dtype=np.float32)
    res = mock_engine.analyse(
        dummy,
        mock_override_cue="glass_breaking",
    )
    assert res["status"] == "success"
    assert res["is_mock"] is True
    assert len(res["detected_cues"]) == 1
    assert res["detected_cues"][0]["cue"] == "glass_breaking"
    assert res["requires_user_confirmation"] is True


def test_safety_guardrail_rejects_assault_queries(engine):
    with pytest.raises(ValueError) as excinfo:
        engine.reject_assault_query()
    assert "SAFETY VIOLATION" in str(excinfo.value)
    assert "assault" in str(excinfo.value)
