# PathPilot Review NLP — Evaluation Report

Owner: Anayan (`feature/nlp-model`)  
Generated from: `artifacts/models/pathpilot-roberta-absa/training_metrics.json` (`2026-11-25T17:33:48Z`)

## 1. Run Configuration & Dataset Hygiene
- **Base Model:** `distilroberta-base` (`RobertaForSequenceClassification`, 3-class polarity)
- **Model Version:** `pathpilot-distilroberta-base-absa-v1.0.0`
- **Device:** CPU (`freeze_bottom_layers=5`, `7,680,771` trainable / `82,120,707` total parameters)
- **Hyperparameters:** `epochs=10`, `batch_size=32`, `lr=5e-5` (linear warmup + decay), `max_length=96`, `seed=42`
- **Class Weights (inverse sqrt frequency):** `[1.0277, 1.3321, 0.6402]` for `[negative, neutral, positive]`
- **Dataset:** SemEval-2014 Task 4 Restaurants (`aspectCategory` polarity; benchmark/transfer verification only)
  - Train XML: `3,041` sentences -> `3,713` pairs (`195` `conflict` excluded -> `3,518` kept; `12` rows PII-sanitised)
  - Test Gold XML: `800` sentences -> `1,025` pairs (`52` `conflict` excluded -> `973` kept; `2` rows PII-sanitised)
  - Deduplication & Leakage Removal: `17` exact train duplicates removed, `1` cross-test leakage row removed, `0` cross-val leakage rows
  - Final Leakage-Verified Splits (`assert_no_split_leakage` passed): `train=3,000`, `val=500`, `test=973`

## 2. Per-Epoch Training & Validation Metrics (Epochs 1–10)
| Epoch | Train Loss | Train Acc | Train Macro-F1 | Val Loss | Val Acc | Val Macro-F1 | Val Weighted-F1 | Neg F1 (Val) | Neu F1 (Val) | Pos F1 (Val) | LR | Time (s) | Best |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1.0313 | 0.5917 | 0.3109 | 0.8717 | 0.6580 | 0.4289 | 0.5779 | 0.2238 | 0.2766 | 0.7864 | 5.00e-05 | 149.61 | Yes |
| 2 | 0.7144 | 0.7290 | 0.6454 | 0.5658 | 0.8320 | 0.7721 | 0.8311 | 0.7706 | 0.6483 | 0.8974 | 4.44e-05 | 138.75 | Yes |
| 3 | 0.5946 | 0.7817 | 0.7207 | 0.5160 | 0.8320 | 0.7795 | 0.8312 | 0.7611 | 0.6846 | 0.8928 | 3.89e-05 | 142.78 | Yes |
| 4 | 0.5451 | 0.8017 | 0.7426 | 0.4907 | 0.8380 | 0.7833 | 0.8394 | 0.7805 | 0.6667 | 0.9028 | 3.33e-05 | 186.19 | Yes |
| 5 | 0.5162 | 0.8057 | 0.7494 | 0.4778 | 0.8300 | 0.7775 | 0.8334 | 0.7680 | 0.6667 | 0.8978 | 2.78e-05 | 138.31 | No |
| 6 | 0.4828 | 0.8230 | 0.7706 | 0.4903 | 0.8440 | 0.7889 | 0.8454 | 0.7811 | 0.6753 | 0.9103 | 2.22e-05 | 241.26 | Yes |
| 7 | 0.4662 | 0.8197 | 0.7714 | 0.4908 | 0.8460 | 0.7911 | 0.8474 | 0.7759 | 0.6839 | 0.9135 | 1.67e-05 | 775.27 | Yes |
| 8 | 0.4554 | 0.8300 | 0.7831 | 0.4806 | 0.8440 | 0.7886 | 0.8469 | 0.7863 | 0.6667 | 0.9127 | 1.11e-05 | 458.72 | No |
| **9** | **0.4402** | **0.8350** | **0.7875** | **0.4795** | **0.8480** | **0.7925** | **0.8497** | **0.7931** | **0.6710** | **0.9135** | **5.56e-06** | **341.65** | **Best** |
| 10 | 0.4369 | 0.8377 | 0.7930 | 0.4777 | 0.8460 | 0.7884 | 0.8480 | 0.7881 | 0.6623 | 0.9148 | 0.00e+00 | 272.64 | No |

## 3. Held-Out Gold Test Evaluation (Best Checkpoint: Epoch 9)
- **Test Loss:** `0.3713`
- **Test Accuracy:** `0.8746` (`973` samples)
- **Test Macro-F1:** `0.7985`
- **Test Weighted-F1:** `0.8766`

### Per-Class Held-Out Test Metrics
| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| `negative` | 0.8122 | 0.8378 | 0.8248 | 222 |
| `neutral` | 0.6095 | 0.6809 | 0.6432 | 94 |
| `positive` | 0.9405 | 0.9148 | 0.9275 | 657 |

### Per-Aspect Held-Out Test Metrics
| Aspect | Accuracy | Macro-F1 | Support |
|---|---|---|---|
| `ambience` | 0.8857 | 0.7867 | 105 |
| `anecdotes/miscellaneous` | 0.8128 | 0.7960 | 219 |
| `food` | 0.8731 | 0.7182 | 402 |
| `price` | 0.9000 | 0.8193 | 80 |
| `service` | 0.9401 | 0.7938 | 167 |

### Confusion Matrix (`rows = true`, `cols = predicted`)
| True \ Pred | `negative` | `neutral` | `positive` |
|---|---|---|---|
| `negative` | 186 | 17 | 19 |
| `neutral` | 11 | 64 | 19 |
| `positive` | 32 | 24 | 601 |

## 4. Error Analysis & Inference Smoke Test
- **Primary error modes observed in top-15 high-confidence errors:**
  1. **Mixed-aspect polarity bleed:** e.g., *"Great food but the service was dreadful!"* evaluated for `food` (`true=positive`, `pred=negative`, `conf=0.9904`) when negative adjectives for another aspect dominate the sequence.
  2. **Neutral vs negative boundary:** e.g., *"The food was mediocre and the service was severely slow."* evaluated for `food` (`true=neutral`, `pred=negative`, `conf=0.9892`).
  3. **Sarcasm / rhetorical questions:** e.g., *"Did I mention that the coffee is OUTSTANDING?"* (`true=positive`, `pred=negative`, `conf=0.9372`).
- **Reload & CPU Smoke Test:**
  - Artifact size on disk (`artifacts/models/pathpilot-roberta-absa/`): `316.68 MB` (gitignored)
  - Single-sample CPU inference latency: `155.43 ms` (`pred_sentiment=positive`, `confidence=0.9791`)
