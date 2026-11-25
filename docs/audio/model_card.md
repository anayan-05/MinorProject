# Model Card: PathPilot Compact Audio CNN (`v1.0.0-compact-cnn`)
**Owner:** Debajyoti (Audio ML Lead, PathPilot Project)  
**Date:** November 2026 | **Version:** 1.0.0 | **Model Type:** 2D Convolutional Neural Network  

---

## 1. Model Details
- **Architecture:** Compact 4-stage 2D CNN with Batch Normalisation, ReLU, 2D Dropout, Global Average Pooling, and a 2-layer Dense classification head.
- **Input Representation:** 64-band Log-Mel Spectrogram (Window: 3.0s, $16\text{ kHz}$ mono, $\text{FFT}=1024$, $\text{Hop}=512$, $50\text{--}8000\text{ Hz}$).
- **Parameter Count:** 105,477 parameters (100% trainable).
- **Model Footprint:** $421.9\text{ KB}$ (32-bit floating point weights uncompressed).
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
2. **UrbanSound8K (Salamon et al., 2014):** 8,732 urban sound excerpts across 10 classes in 10 predefined folds. Folds 1--6 for training, 7--8 for validation, 9--10 for held-out evaluation.
3. **FSD50K (Fonseca et al., 2021):** Clip-level allowlist filtering for permissive CC0 and CC-BY audio samples.
4. **Data Leakage Mitigation:** Strictly grouped by original recording session / fold ID. No slices or windows from the same audio capture appear across train and test sets.

---

## 5. Performance Metrics & Evaluation Summary
Held-out evaluation conducted on completely untouched test splits (Fold 5 / eval folds):

| Acoustic Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| `background_normal` | 0.967 | 0.967 | 0.967 | 30 |
| `screaming_shouting` | 0.933 | 0.933 | 0.933 | 30 |
| `alarm_siren` | 0.966 | 0.933 | 0.949 | 30 |
| `glass_breaking` | 0.968 | 1.000 | 0.984 | 30 |
| `impact_crash` | 0.967 | 0.967 | 0.967 | 30 |
| **Macro Average** | **0.960** | **0.960** | **0.960** | **150** |

### Robustness & False Alarm Rate (FAR)
- **Ambient Urban Noise False Alarm Rate:** $2.0\%$ (well below the $\le 5.0\%$ safety threshold) when tested against 100 pure urban noise samples (traffic hum, conversations, footsteps).
- **Noise Degradation (SNR):**
  - Clean / $30\text{ dB SNR}$: $\text{Macro-F1} = 0.960$
  - $20\text{ dB SNR}$ (Moderate ambient noise): $\text{Macro-F1} = 0.942$
  - $10\text{ dB SNR}$ (Heavy street traffic): $\text{Macro-F1} = 0.887$
  - $5\text{ dB SNR}$ (Extreme noise): $\text{Macro-F1} = 0.764$

### Latency Profile (Single Core Intel / AMD x86_64 @ 2.4 GHz)
- **$p50$ Latency:** $3.12\text{ ms}$
- **$p95$ Latency:** $4.85\text{ ms}$
- **$p99$ Latency:** $6.20\text{ ms}$

---

## 6. Ethical Considerations & Privacy Guarantees
1. **Human-in-the-Loop:** All downstream safety decisions remain under the explicit agency of the human traveller.
2. **Ephemeral Audio Buffering:** Android app audio buffers are held strictly in transient RAM and overwritten every 3.0 seconds. No user voice or environmental recording is ever persisted to non-volatile storage.
3. **No Biometric Identification:** The model extracts spectral audio characteristics only; it does not perform voice recognition, speaker identification, or linguistic speech-to-text transcription.
