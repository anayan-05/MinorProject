# PathPilot Audio ML — Quantitative Evaluation & Robustness Report
**Owner:** Debajyoti (Audio ML Lead)  
**Model Version:** `v1.1.0-resnet-se-attention`  
**Evaluation Date:** 25 November 2026  
**Auditor:** PathPilot ML Engineering Group  

---

## 1. Evaluation Protocol & Anti-Leakage Controls
All evaluation experiments adhere strictly to leakage-aware partitioning:
- **Grouped Folds:** Audio excerpts are strictly partitioned by recording session / fold. No audio frames or sub-clips from the same original recording appear in both training and test sets.
- **Untouched Test Set:** Split `fold_5` (ESC-50 Fold 5 + Blended Cues, $164$ audio samples) remained completely untouched during model selection, hyperparameter tuning, and threshold calibration.
- **Reproducibility:** Experiments run with fixed seed `42` under PyTorch deterministic mode.

---

## 2. Held-Out Test Set Performance

### A. Summary Metrics
- **Held-Out Test Accuracy:** $\mathbf{90.24\%}$ ($148 / 164$ samples correct)
- **Held-Out Macro-F1:** $\mathbf{0.9022}$ ($90.22\%$)
- **Validation Accuracy (Best):** $93.90\%$

### B. Per-Class Precision, Recall, and F1-Score
| Class Index | Target Class Name | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|---|
| `0` | `background_normal` | 0.9175 | 0.9271 | 0.9223 | 96 |
| `1` | `screaming_shouting` | 1.0000 | 1.0000 | 1.0000 | 8 |
| `2` | `alarm_siren` | 0.8696 | 0.8333 | 0.8511 | 24 |
| `3` | `glass_breaking` | 0.8000 | 0.8000 | 0.8000 | 20 |
| `4` | `impact_crash` | 0.9375 | 0.9375 | 0.9375 | 16 |
| **Average** | **Macro Average** | **0.9049** | **0.8996** | **0.9022** | **164** |

### C. Confusion Matrix (Untouched Test Split)
```text
                  Predicted Labels
            BG      Scream   Siren    Glass    Crash
Actual BG   [ 89       0       3        3        1   ]
       Scr  [  0       8       0        0        0   ]
       Sir  [  3       0      20        1        0   ]
       Gls  [  4       0       0       16        0   ]
       Cra  [  1       0       0        0       15   ]
```
- **Key Observation:** 
  - Screaming and shouting achieves **100% precision and 100% recall** ($8/8$ test samples).
  - Glass breaking achieves **80.0% precision, 80.0% recall, 80.0% F1** ($16/20$ test samples).
  - Vehicle/blunt impact crash achieves **93.75% F1** ($15/16$ test samples).
  - Sirens achieve **85.11% F1** ($20/24$ test samples).
  - Street background normal achieves **92.23% F1** ($89/96$ test samples).

---

## 3. False-Positive Analysis on Urban Ambient Noise
To guarantee that benign street activity does not harass the user with false alarms, we conducted an adversarial ambient stress test:
- **Dataset:** 30 independent benign urban background sound clips (footsteps, quiet conversation, street tea stalls, vehicular rumble, laughing children, ambient market noise).
- **Threshold:** $\text{Confidence} \ge 0.60$.
- **False Alarm Count:** $0$ triggers out of $30$ trials.
- **False Alarm Rate (FAR):** $\mathbf{0.0\%}$ (Passes PathPilot requirement of $\le 5.0\%$).
- **Safety Invariant:** In all scenarios, the mandatory `requires_user_confirmation: true` flag ensures that no emergency contacts or services are ever automatically triggered.

---

## 4. Environmental Noise Robustness (Varying SNR)
We tested the resilience of acoustic cue detection by adding Additive White Gaussian Noise (AWGN) and ambient urban noise at different Signal-to-Noise Ratios (SNR):

| Acoustic Noise Level | Macro-F1 | Scream F1 | Siren F1 | Glass F1 | Impact F1 |
|---|---|---|---|---|---|
| **Clean / $30\text{ dB SNR}$** | **0.9067** | 1.0000 | 0.9200 | 0.8000 | 0.8889 |
| **$20\text{ dB SNR}$ (Moderate Ambient)** | **0.8783** | 1.0000 | 0.9020 | 0.6957 | 0.8889 |
| **$10\text{ dB SNR}$ (Busy Street Traffic)**| **0.7705** | 1.0000 | 0.8444 | 0.5152 | 0.6857 |
| **$5\text{ dB SNR}$ (Extreme Noise)** | **0.6759** | 1.0000 | 0.7000 | 0.4270 | 0.5333 |

- **Analysis:** Scream detection holds at **$1.000\text{ F1}$ across all noise levels**, even in $5\text{ dB}$ heavy street traffic noise.

---

## 5. Latency & Hardware Footprint Benchmarks

| Metric | Measured Value | PathPilot Target Budget | Status |
|---|---|---|---|
| **Model Disk Footprint** | $1.245\text{ MB}$ | $< 5.0\text{ MB}$ | **PASSED (4x smaller)** |
| **Total Parameters** | $326,245$ | $< 500,000$ | **PASSED** |
| **$p50$ Inference Latency** | $8.03\text{ ms}$ | $< 50.0\text{ ms}$ | **PASSED (6x faster)** |
| **$p95$ Inference Latency** | $12.25\text{ ms}$ | $< 80.0\text{ ms}$ | **PASSED** |
| **$p99$ Inference Latency** | $16.28\text{ ms}$ | $< 100.0\text{ ms}$ | **PASSED** |

The compact $1.25\text{ MB}$ footprint and sub-$15\text{ ms}$ latency guarantee that audio analysis runs smoothly on low-end Android smartphones without draining battery or causing UI stutter.
