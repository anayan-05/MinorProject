"""Inference Engine for PathPilot Audio ML Subsystem.
Owner: Debajyoti (Audio ML)

Implements the official PathPilot API Contract for POST /api/v1/audio/analyse:
- Accepts raw audio file path, waveform array, or binary buffer
- Converts to 16kHz mono log-Mel spectrogram
- Runs inference through PathPilotAudioCNN (PyTorch or pure-NumPy runtime)
- Applies confidence threshold (default 0.60)
- Returns structured JSON adhering to the PathPilot integration contract:
    - status
    - is_mock
    - model_version
    - schema_version
    - generated_at (ISO 8601)
    - detected_cues (cue, confidence, time_offsets)
    - requires_user_confirmation: true (IMMUTABLE: ALWAYS TRUE)
    - limitations
    - warnings
- Strictly rejects any query for 'assault_detected' or automated SOS triggers.
"""

import os
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union
import numpy as np

try:
    from .preprocess import AudioPreprocessor
    from .model import PathPilotAudioCNN
except ImportError:
    from preprocess import AudioPreprocessor
    from model import PathPilotAudioCNN

try:
    import torch
    HAS_TORCH = True
except (ImportError, OSError):
    HAS_TORCH = False
    torch = None


class AudioInferenceEngine:
    """Production inference engine for PathPilot acoustic cue analysis."""

    SCHEMA_VERSION = "1.0.0"
    DEFAULT_MODEL_VERSION = "v1.0.0-compact-cnn"
    DEFAULT_CONFIDENCE_THRESHOLD = 0.60

    STANDARD_LIMITATIONS = [
        "Acoustic cues are statistical sound classifications, not proof of danger, crime, or distress.",
        "Model does NOT detect assault, intent, or domestic violence.",
        "Accuracy may degrade under heavy wind, traffic rumble, or low-cost microphone clipping.",
        "Advisory assistance only; requires explicit user confirmation before taking any action.",
        "Audio alone never triggers SOS or emergency contact notifications.",
    ]

    def __init__(
        self,
        weights_path: Optional[str] = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        device: Optional[str] = None,
        is_mock: bool = False,
    ):
        self.confidence_threshold = confidence_threshold
        self.is_mock = is_mock
        self.preprocessor = AudioPreprocessor()
        self.model = PathPilotAudioCNN(num_classes=5)
        self.model_version = self.DEFAULT_MODEL_VERSION

        if HAS_TORCH and torch is not None:
            try:
                self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
                if weights_path and os.path.exists(weights_path) and not is_mock:
                    checkpoint = torch.load(weights_path, map_location=self.device)
                    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                        self.model.load_state_dict(checkpoint["model_state_dict"])
                        self.model_version = checkpoint.get("model_version", self.DEFAULT_MODEL_VERSION)
                    elif isinstance(checkpoint, dict):
                        self.model.load_state_dict(checkpoint)
                self.model.to(self.device)
            except Exception:
                self.device = None
        else:
            self.device = None

        self.model.eval()

    def analyse(
        self,
        audio_input: Any,
        sample_rate: Optional[int] = None,
        override_threshold: Optional[float] = None,
        mock_override_cue: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Runs acoustic cue analysis on audio input and returns formatted contract response."""
        now_iso = datetime.now(timezone.utc).isoformat()
        threshold = override_threshold or self.confidence_threshold
        warnings: List[str] = []

        if self.is_mock:
            detected_cues = []
            if mock_override_cue and mock_override_cue in PathPilotAudioCNN.CLASS_NAMES:
                if mock_override_cue != "background_normal":
                    detected_cues.append({
                        "cue": mock_override_cue,
                        "confidence": 0.88,
                        "time_offsets": [0.0, 3.0],
                    })
            return {
                "status": "success",
                "is_mock": True,
                "model_version": f"{self.model_version}-mock",
                "schema_version": self.SCHEMA_VERSION,
                "generated_at": now_iso,
                "detected_cues": detected_cues,
                "requires_user_confirmation": True,  # Mandatory safety invariant
                "limitations": self.STANDARD_LIMITATIONS,
                "warnings": ["Payload generated in mock mode for integration testing."],
            }

        # 1. Standardise and preprocess audio
        try:
            waveform = self.preprocessor.load_and_standardise(audio_input, input_sample_rate=sample_rate)
        except Exception as e:
            return {
                "status": "error",
                "is_mock": False,
                "model_version": self.model_version,
                "schema_version": self.SCHEMA_VERSION,
                "generated_at": now_iso,
                "detected_cues": [],
                "requires_user_confirmation": True,
                "limitations": self.STANDARD_LIMITATIONS,
                "warnings": [f"Audio decoding failure: {str(e)}"],
            }

        # 2. Extract Log-Mel Spectrogram
        log_mel = self.preprocessor.compute_log_mel(waveform)

        # 3. Model Forward Pass
        probs = self.model.predict_proba(log_mel)
        if HAS_TORCH and torch is not None and isinstance(probs, torch.Tensor):
            probabilities = probs[0].detach().cpu().numpy()
        else:
            probabilities = probs[0] if probs.ndim == 2 else probs

        # 4. Filter cues exceeding cautious threshold
        detected_cues = []
        for idx in range(1, len(PathPilotAudioCNN.CLASS_NAMES)):
            class_name = PathPilotAudioCNN.CLASS_NAMES[idx]
            conf = float(probabilities[idx])
            if conf >= threshold:
                detected_cues.append({
                    "cue": class_name,
                    "confidence": round(conf, 4),
                    "time_offsets": [0.0, round(self.preprocessor.target_duration, 1)],
                })

        if not detected_cues:
            bg_conf = float(probabilities[0])
            if bg_conf < 0.50:
                warnings.append("Acoustic environment has high uncertainty or low signal-to-noise ratio.")

        # Sort detected cues by descending confidence
        detected_cues.sort(key=lambda x: x["confidence"], reverse=True)

        return {
            "status": "success",
            "is_mock": False,
            "model_version": self.model_version,
            "schema_version": self.SCHEMA_VERSION,
            "generated_at": now_iso,
            "detected_cues": detected_cues,
            "requires_user_confirmation": True,  # Mandatory safety invariant: NEVER false
            "limitations": self.STANDARD_LIMITATIONS,
            "warnings": warnings,
        }

    def reject_assault_query(self) -> None:
        """Explicit safety guardrail demonstrating that assault/distress queries are forbidden."""
        raise ValueError(
            "SAFETY VIOLATION: PathPilot strictly rejects queries for 'assault' or 'distress'. "
            "Acoustic cues are non-causal physical audio patterns (e.g. siren, glass break, screaming). "
            "They must never be used to declare assault or trigger automated dispatch."
        )
