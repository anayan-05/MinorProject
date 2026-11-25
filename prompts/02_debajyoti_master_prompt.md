# DEBAJYOTI — MASTER PROMPT: AUDIO ML
Work on `feature/audio-model`, paths `ml/audio/`, `docs/audio/`, audio tests. Read shared context and handbook.

History/context: PathPilot is a Kolkata-first women-traveller safety Android app. Audio is optional, user-authorised and advisory only; it is not an assault detector. The Android app must install via USB. Five Antigravity agents coordinate through GitHub, not shared local chat.

Data: investigate official ESC-50, UrbanSound8K, FSD50K; AudioSet only with authorised access and terms review. ESC-50 is CC BY-NC; FSD50K licences vary by clip; verify UrbanSound8K terms. Maintain clip-level licence/label manifest. Map only genuine source labels; possible cues may include vocalisation, shout/call, impact/crash, alarm/siren, glass breaking only if supported. Do not invent assault/distress labels.

Mandatory notebook: `ml/audio/training/pathpilot_audio_cue_detection.ipynb`, fresh-Colab runnable. Include setup/versions/seeds; authorised data load; licence validation; file integrity/sample rate/channels/duration; label mapping; grouped split by original recording; resampling/mono/fixed duration; log-Mel spectrogram parameters; compact CNN baseline; training; held-out macro-F1/per-class precision/recall/F1/confusion matrix; false-positive/negative analysis; background-noise robustness; export; model size/latency; inference demo; CPU smoke test. No fake metrics. Save weights outside Git.

Deliver preprocess/train/inference modules, label mapping, licence register, model card, API contract, evaluation report and mock fixture. Inference returns possible cue/time offsets/confidence/model version/limitations and `requires_user_confirmation: true`; never `assault_detected`.

Privacy: user-started visible stoppable recording only; no covert/always-on capture; audio never auto-triggers SOS. Coordinate Android permission/foreground service with Deep; define retention and deletion. No raw audio, datasets, weights or secrets in Git. Test, diff, commit and PR.
