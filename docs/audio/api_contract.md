# PathPilot Audio ML — API Contract Specification

**Endpoint:** `POST /api/v1/audio/analyse`  
**Owner:** Debajyoti (Audio ML Lead)  
**Consumers:** Deep (Android App Integration), Ayushman (Backend Orchestration)  
**Schema Version:** `1.0.0`

---

## 1. Description & Purpose
Processes an optional, user-authorised 3.0-second environmental audio buffer captured on the user's Android phone. The endpoint classifies acoustic sound events into cautious, objective categories and returns statistical probabilities.

### Critical Safety Invariants
1. **Never Automatic SOS:** Detection of an acoustic cue (e.g. `screaming_shouting` or `glass_breaking`) **never** automatically triggers emergency dispatch, SMS alerts, or contact notifications.
2. **Mandatory Confirmation Flag:** Every successful response contains `"requires_user_confirmation": true`. The mobile UI must display an advisory confirmation modal (e.g., *"Possible sound detected: Siren. Are you okay?"*).
3. **Never Claims Assault:** The endpoint strictly refuses to predict "assault", "distress", or "crime".

---

## 2. Request Specification

### HTTP Headers
```http
POST /api/v1/audio/analyse HTTP/1.1
Host: api.pathpilot.app
Content-Type: multipart/form-data; boundary=----WebKitFormBoundaryXYZ
Accept: application/json
Authorization: Bearer <user_session_token>
```

### Request Payload (`multipart/form-data`)
| Field Name | Type | Required | Description |
|---|---|---|---|
| `audio_file` | Binary (WAV/PCM) | Yes | 16,000 Hz, 16-bit Mono PCM audio, recommended duration 3.0s (min 1.0s, max 5.0s). |
| `confidence_threshold` | Float | No | Minimum confidence required to emit a cue (default: `0.60`). |
| `is_mock` | Boolean | No | If `true`, returns fixture payload without invoking CNN inference. |
| `device_timestamp` | String (ISO 8601) | No | Timestamp of local capture on the Android client. |

---

## 3. Response Specification (`application/json`)

### Success Response (`200 OK`)
```json
{
  "status": "success",
  "is_mock": false,
  "model_version": "v1.0.0-compact-cnn",
  "schema_version": "1.0.0",
  "generated_at": "2026-11-24T14:15:30.124Z",
  "detected_cues": [
    {
      "cue": "screaming_shouting",
      "confidence": 0.8842,
      "time_offsets": [0.0, 3.0]
    }
  ],
  "requires_user_confirmation": true,
  "limitations": [
    "Acoustic cues are statistical sound classifications, not proof of danger, crime, or distress.",
    "Model does NOT detect assault, intent, or domestic violence.",
    "Accuracy may degrade under heavy wind, traffic rumble, or low-cost microphone clipping.",
    "Advisory assistance only; requires explicit user confirmation before taking any action.",
    "Audio alone never triggers SOS or emergency contact notifications."
  ],
  "warnings": []
}
```

### Ambiguous / Ambient Environment Response (`200 OK`)
When no acute safety cues cross the threshold:
```json
{
  "status": "success",
  "is_mock": false,
  "model_version": "v1.0.0-compact-cnn",
  "schema_version": "1.0.0",
  "generated_at": "2026-11-24T14:16:00.000Z",
  "detected_cues": [],
  "requires_user_confirmation": true,
  "limitations": [
    "Acoustic cues are statistical sound classifications, not proof of danger, crime, or distress.",
    "Model does NOT detect assault, intent, or domestic violence.",
    "Advisory assistance only; requires explicit user confirmation before taking any action.",
    "Audio alone never triggers SOS or emergency contact notifications."
  ],
  "warnings": [
    "Acoustic environment classified as normal ambient background."
  ]
}
```

### Error Responses
- **`400 Bad Request`**: Audio payload corrupted, unparseable format, or missing binary data.
- **`413 Payload Too Large`**: Audio duration exceeds 10.0 seconds.
- **`422 Unprocessable Entity`**: Client attempted to query forbidden labels (e.g. `detect_assault: true`).

---

## 4. Client Integration Checklist for Deep (Android Lead)
- [ ] Show visible persistent notification with a red "Stop Recording" button while audio is active.
- [ ] When `detected_cues` is non-empty, prompt user with: *"PathPilot heard a possible sound ([Cue Name]). Are you safe?"*
- [ ] Action buttons: `[I'm Safe]` (dismisses advisory), `[View Police/Hospitals]`, `[SOS Options]`.
- [ ] Destroy local audio byte buffer immediately after `POST` request completes.
