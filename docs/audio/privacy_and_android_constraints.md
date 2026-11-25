# PathPilot Audio ML — Mobile Privacy & Android Integration Constraints
**Author:** Debajyoti (Audio ML Lead)  
**Target Recipient:** Deep (Android App / Integration Lead)  
**Subject:** Native Android Permissions, Foreground Service, Ephemeral Memory, and UI Confirmation Workflow  

---

## 1. Executive Summary for Android Integration
Audio monitoring in PathPilot is **purely optional, user-started, and advisory**. It must never operate silently, covertly, or in the background without clear visual status.

This document outlines the strict native Android architectural constraints required to maintain user trust, privacy, and full compliance with Google Play and Android security policies.

---

## 2. Native Permissions & Android Manifest Specification

### A. Android Manifest Permissions (`AndroidManifest.xml`)
```xml
<!-- Required for user-authorised audio sampling -->
<uses-permission android:name="android.permission.RECORD_AUDIO" />

<!-- Required for ongoing visible foreground capture on Android 9+ (API 28+) -->
<uses-permission android:name="android.permission.FOREGROUND_SERVICE" />

<!-- Required on Android 14+ (API 34+) for microphone foreground services -->
<uses-permission android:name="android.permission.FOREGROUND_SERVICE_MICROPHONE" />
```

### B. Foreground Service Declaration
```xml
<service
    android:name=".services.AudioSafetyForegroundService"
    android:foregroundServiceType="microphone"
    android:exported="false" />
```

### C. Runtime Permission Request Flow
Before calling `AudioRecord` or starting the service, Deep must present an in-app Educational Rationale Sheet:
> **"PathPilot Audio Guardian (Optional)"**  
> *"PathPilot can monitor your immediate acoustic surroundings to detect acute sounds like sirens, breaking glass, or shouting. All processing occurs ephemerally in volatile memory. No audio is ever recorded, saved, or uploaded for speech recognition. You can stop monitoring at any moment."*

If the user denies permission:
- The app must gracefully fall back to geospatial and review navigation only.
- Audio toggle in settings must remain `OFF` without continuous nagging.

---

## 3. Persistent Foreground Notification (Android Mandatory)
While audio capture is running, Android requires a persistent notification. PathPilot mandates:
1. **Notification Title:** `PathPilot Audio Guardian Active`
2. **Body:** `Listening for acoustic safety cues (Advisory only)`
3. **Notification Priority:** `PRIORITY_LOW` (non-intrusive sound/vibration, but persistent in the shade)
4. **Action Button (Crucial):** `[Stop Monitoring]` — tapping this button must immediately stop the microphone, release the hardware, and terminate the foreground service.

---

## 4. Ephemeral Memory Architecture & Zero Raw Storage
To prevent privacy breaches or subpoena risks, raw user audio must **never touch non-volatile storage** (no SQLite, no internal flash, no SD card).

```text
[ Android Microphone ]
         │ (AudioRecord.read)
         ▼
[ Volatile RAM Circular Buffer ] (3.0 seconds = 96 KB PCM)
         │
         ├──> [ Feature Extraction: Log-Mel Spectrogram ]
         │
         ▼ (IMMEDIATE FLUSH)
[ Buffer Zeroed / Overwritten in RAM ] (0 bytes written to disk)
```

- **Buffer Size:** $3.0\text{ seconds} \times 16,000\text{ samples/sec} \times 2\text{ bytes (16-bit PCM)} = 96,000\text{ bytes} \approx 93.75\text{ KB}$.
- **Destruction Lifecycle:** Once the spectrogram is computed or dispatched to `POST /api/v1/audio/analyse`, the byte buffer in RAM must be zeroed out.

---

## 5. UI Interaction & Confirmation Protocol

When the API or on-device model returns a detected cue (e.g. `screaming_shouting` with $88\%$ confidence):

### Step 1: The Advisory Banner
The client must display an advisory bottom sheet or banner:
```text
┌────────────────────────────────────────────────────────┐
│  ⚠️ Acoustic Alert                                     │
│  Possible sound detected: Shouting / Screaming         │
│  Confidence: 88% (Advisory)                            │
│                                                        │
│  Are you safe?                                         │
│                                                        │
│  [ I am Safe ]              [ View Emergency Res. ]   │
│                                                        │
│  [ Open SOS Confirmation Screen ]                      │
└────────────────────────────────────────────────────────┘
```

### Step 2: User Action Handlers
- **`[ I am Safe ]`**: Immediately dismisses the sheet. Audio monitoring continues normally.
- **`[ View Emergency Resources ]`**: Opens Ayushman & Amisa's emergency tab (displaying nearest verified 24/7 hospitals, police booths, open pharmacies).
- **`[ Open SOS Confirmation Screen ]`**: Navigates to the user-confirmed SOS preview screen.

### Step 3: Absolute Safety Guardrail
- **Under NO circumstances may the Android app automatically trigger an SOS call, SMS, or dispatch based on audio alone.**
- The `requires_user_confirmation: true` flag in the API response is an absolute constraint.

---

## 6. Offline & Low-Power Handling
- **Airplane Mode / Offline:** If network is unavailable, the React Native app should use the on-device TorchScript/TFLite model (`artifacts/models/pathpilot_audio_cnn_v1.torchscript.pt`) for local inference.
- **Battery Optimization:** If battery drops below $15\%$, prompt the user to pause audio monitoring to preserve phone battery for navigation and emergency calling.
