# PathPilot Audio Machine Learning Documentation
**Lead:** Debajyoti (Audio ML Lead)  
**Branch:** `feature/audio-model`  
**Paths:** `ml/audio/`, `docs/audio/`, `tests/audio/`

---

## 1. System Overview
The PathPilot Audio ML subsystem provides optional, user-authorised, advisory acoustic cue analysis for the PathPilot women-traveller safety application in Kolkata. It monitors environmental audio when explicitly started by the user and detects physical acoustic sound events (e.g. screams, sirens, breaking glass, impacts).

### Non-Negotiable Safety & Privacy Invariants
1. **Advisory Cues Only:** The model predicts physical sound events. It does **not** detect "assault", "distress", "fear", or "danger".
2. **Mandatory User Confirmation:** All inference outputs return `requires_user_confirmation: true`. Audio cues **never** trigger automatic SOS, emergency dispatch, or silent notifications.
3. **User-Started & Visible:** Recording only occurs when explicitly initiated by the user. An Android foreground service notification remains visible at all times with a one-tap "Stop" action.
4. **Zero Audio Retention:** Processing is performed in ephemeral memory buffers; raw audio waveforms are discarded immediately after inference and are never stored on device flash or backend disk.
5. **Rigorous Licencing:** All training data (ESC-50, UrbanSound8K, FSD50K) are audited for licence compliance (e.g., CC BY-NC) and tracked in a clip-level manifest.

---

## 2. Directory Structure
```text
ml/audio/
  registry/
    dataset_acquisition_register.csv   # Dataset metadata, terms, and storage tracking
    licence_manifest.csv              # Clip-level allowlist, terms, and split groups
    label_mapping.json                # Source-to-target acoustic taxonomy mapping
  src/
    __init__.py
    preprocess.py                     # 16kHz mono resampling, log-Mel spectrograms, augmentations
    model.py                          # Compact PathPilotAudioCNN (~105k params, < 450 KB)
    train.py                          # Grouped-split, class-weighted training pipeline
    evaluate.py                       # Macro-F1, confusion matrix, noise robustness, FAR analysis
    inference.py                      # Integration engine implementing /api/v1/audio/analyse
  training/
    pathpilot_audio_cue_detection.ipynb  # Runnable Google Colab notebook
  fixtures/
    mock_audio_analysis_response.json # Sample JSON contract fixture for mobile integration

docs/audio/
  README.md                           # This overview document
  api_contract.json                   # OpenAPI / JSON Schema definition
  api_contract.md                     # Human-readable endpoint documentation
  model_card.md                       # Comprehensive Model Card
  dataset_licence_registry.md         # In-depth legal & licensing audit
  label_mapping_guide.md              # Taxonomy and justification against "assault" labels
  evaluation_report.md                # Quantitative metrics, FAR, noise stress tests
  privacy_and_android_constraints.md  # Hand-off guidelines for Deep (Android Lead)

tests/audio/
  test_preprocess.py                  # Audio standardisation & spectrogram unit tests
  test_model.py                       # Model architecture, forward pass, footprint tests
  test_inference.py                   # Contract validation & safety invariant tests
  test_licence_manifest.py            # Manifest integrity & licence compliance tests
  test_api_contract.py                # Schema validation for API payloads
```

---

## 3. Quickstart & Verification

### Running Unit Tests
```bash
pytest tests/audio -v
```

### Running Model Training & Evaluation
```bash
python -m ml.audio.src.train
python -c "from ml.audio.src.evaluate import run_full_evaluation; print(run_full_evaluation())"
```

### Running Inference via API Client
```python
from ml.audio.src.inference import AudioInferenceEngine

engine = AudioInferenceEngine(is_mock=False)
result = engine.analyse("sample_recording.wav")
assert result["requires_user_confirmation"] is True
```
