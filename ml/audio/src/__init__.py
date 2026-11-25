"""PathPilot Audio Machine Learning Package.
Owned by Debajyoti (Audio ML Lead).

Subsystem providing cautious, user-authorised, advisory audio cue detection.
Strictly adheres to PathPilot safety and privacy guidelines:
- Advisory acoustic cues only; never claims danger or distress
- Never detects assault
- requires_user_confirmation: true is always enforced
- Audio never auto-triggers SOS
"""

from .preprocess import AudioPreprocessor
from .model import PathPilotAudioCNN
from .inference import AudioInferenceEngine

__version__ = "1.0.0"
__all__ = ["AudioPreprocessor", "PathPilotAudioCNN", "AudioInferenceEngine"]
