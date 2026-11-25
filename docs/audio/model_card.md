# Model Card: PathPilot SE-ResNet Audio Guardian (`v1.1.0-resnet-se-attention`)
**Owner:** Debajyoti (Audio ML Lead, PathPilot Project)  
**Date:** November 2026 | **Version:** 1.1.0 | **Model Type:** Residual 2D CNN with Squeeze-and-Excitation (SE) Channel Attention  

---

## 1. Model Details
- **Architecture:** 4-stage Residual Convolutional Neural Network with Squeeze-and-Excitation (SE) channel attention blocks, residual identity skip connections, Batch Normalisation, ReLU, 2D Dropout, Global Average Pooling, and a 2-layer Dense classification head.
- **Input Representation:** 128-band Log-Mel Spectrogram (Window: 3.0s, $16\text{ kHz}$ mono, $\text{FFT}=1024$, $\text{Hop}=512$, $50\text{--}8000\text{ Hz}$).
- **Parameter Count:** 326,245 parameters (100% trainable).
- **Model Footprint:** $1.245\text{ MB}$ (PyTorch state dict uncompressed).
- **Target Hardware:** Mobile ARM CPUs (via PyTorch Mobile / ONNX / TFLite) and x86/x64 edge backend instances.
- **Licence:** Creative Commons Attribution-NonCommercial 3.0 (derived from ESC-50 / UrbanSound8K training subsets for academic & prototype demonstration).

---

## 2. Intended Use
- **Primary Domain:** Advisory environmental sound analysis for women travellers navigating public streets and urban transit routes in Kolkata.
- **Intended Interaction:** Active, user-started listening sessions (e.g. while walking alone through poorly lit or unfamiliar corridors).
- **Output Cues:**
  - `background_normal`: Ambient urban sounds (traffic, conversation, laughter, footsteps).
  - `screaming_shouting`: High-energy vocal events (screams, loud shouting, distress calls).
  - `alarm_siren`: Emergency vehicle sirens, car alarms, repetitive horns.
  - `glass_breaking`: High-frequency shattering of glass windows, bottles, or storefronts.
  - `impact_crash`: Blunt mechanical collisions, vehicle impacts.

---

## 3. Explicit Non-Goals & Prohibited Uses
- **NO Assault Detection:** The model **cannot** and **does not** detect "assault", "violence", "threat level", or "intent". Physical acoustic events (such as shouting or breaking glass) can occur during sporting events, cinema, construction, or benign human celebration.
- **NO Automated SOS Dispatch:** Acoustic cues alone must **never** automatically notify police, emergency contacts, or trigger an SOS sequence.
- **NO Covert or Always-On Monitoring:** The model must never run without prominent visual indication (Android foreground service persistent banner) or without prior user authorization.

---

## 4. Training & Data Provenance
Training and evaluation leverage audited subsets of established environmental audio research datasets:
1. **ESC-50 (Piczak, 2015):** 2,000 environmental recordings across 50 categories. Folds 1--3 used for training, Fold 4 for validation, Fold 5 for held-out testing.
2. **UrbanSound8K (Salamon et al., 2014):** 8,732 urban sound excerpts across 10 classes in 10 predefined folds.
3. **FSD50K (Fonseca et al., 2021):** Clip-level allowlist filtering for permissive CC0 and CC-BY audio samples.
4. **Mixup & Augmentation:** Trained with Mixup data augmentation ($\alpha=0.2$) and SpecAugment to handle complex acoustic background noise typical of Kolkata streets.
5. **Data Leakage Mitigation:** Strictly grouped by original recording session / fold ID. No slices or windows from the same audio capture appear across train and test sets.

---

## 5. Performance Metrics & Evaluation Summary
Held-out evaluation conducted on completely untouched test split (ESC-50 Fold 5 + Blended Cues):

| Acoustic Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| `background_normal` | 0.922 | 0.979 | 0.950 | 96 |
| `screaming_shouting` | 1.000 | 1.000 | 1.000 | 8 |
| `alarm_siren` | 1.000 | 0.875 | 0.933 | 24 |
| `glass_breaking` | 1.000 | 0.375 | 0.546 | 8 |
| `impact_crash` | 0.889 | 1.000 | 0.941 | 16 |
| **Macro Average** | **0.962** | **0.846** | **0.874** | **152** |

### Robustness & False Alarm Rate (FAR)
- **Ambient Urban Noise False Alarm Rate:** $\mathbf{0.0\%}$ ($0$ false triggers out of $30$ adversarial trials, well below the $\le 5.0\%$ threshold).
- **Noise Degradation (SNR):**
  - Clean / $30\text{ dB SNR}$: $\text{Macro-F1} = 0.723$
  - $20\text{ dB SNR}$ (Moderate ambient noise): $\text{Macro-F1} = 0.717$
  - $10\text{ dB SNR}$ (Heavy street traffic): $\text{Macro-F1} = 0.706$
  - $5\text{ dB SNR}$ (Extreme noise): $\text{Macro-F1} = 0.652$

### Latency Profile (Single Core Intel / AMD x86_64)
- **$p50$ Latency:** $8.5\text{ ms}$ (End-to-End audio pipeline)
- **$p95$ Latency:** $13.9\text{ ms}$
- **Budget Target:** $< 50\text{ ms}$ (**PASSED**)

---

## 6. Ethical Considerations & Privacy Guarantees
1. **Human-in-the-Loop:** All downstream safety decisions remain under the explicit agency of the human traveller (`requires_user_confirmation: true`).
2. **Ephemeral Audio Buffering:** Android app audio buffers are held strictly in transient RAM and overwritten every 3.0 seconds. No user voice or environmental recording is ever persisted to non-volatile storage.
3. **No Biometric Identification:** The model extracts spectral audio characteristics only; it does not perform voice recognition, speaker identification, or linguistic speech-to-text transcription.
