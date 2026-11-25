# PathPilot — Travel-Safety Companion for Women Travellers

Kolkata-first AI-assisted travel-safety Android prototype for women travellers (earlier working concept name: SafeRoam). It combines explainable geospatial evidence, review-based aspect/sentiment intelligence, optional user-authorised audio cue analysis, alternative route generation, emergency resources and a user-confirmed SOS/share workflow.

## Team Ownership & Branches
- `feature/nlp-model` — **Anayan**: Project lead + Review NLP / RoBERTa ABSA (`ml/nlp/`, `docs/nlp/`)
- `feature/audio-model` — **Debajyoti**: Audio ML / CNN cue detection (`ml/audio/`, `docs/audio/`, `tests/audio/`)
- `feature/geospatial` — **Amisa**: OSM / VIIRS / official geospatial data & explainable safety evidence (`geospatial/`, `docs/geospatial/`)
- `feature/mobile-ui` — **Deep**: React Native TypeScript Android app & USB APK build (`mobile/`, `docs/mobile/`)
- `feature/navigation-sos` — **Ayushman**: Route alternatives, emergency resources & user-confirmed SOS (`navigation/`, `emergency/`)

## Reference & Rules
- Use assigned branches only; never push directly to `main`, force-push, or commit secrets, personal data, raw datasets/audio, or large model weights.

## Audio ML Subsystem (`ml/audio/`, `docs/audio/`, `tests/audio/`)
Owned by **Debajyoti**. Provides optional, advisory, user-authorised acoustic cue detection (`screaming_shouting`, `alarm_siren`, `glass_breaking`, `impact_crash`, `background_normal`) mapped from genuine public audio datasets (ESC-50, UrbanSound8K, FSD50K).

### Strict Safety & Privacy Principles
1. **Advisory Acoustic Cues Only:** The model detects objective physical sound events, NOT danger or "assault". No "distress/assault" labels are invented or inferred.
2. **User Authorization & Consent:** Audio recording is exclusively user-started, visibly indicated via Android foreground service notifications, and can be stopped at any moment.
3. **Never Automatic SOS:** Audio cue detection requires explicit user confirmation (`requires_user_confirmation: true`) and NEVER automatically triggers emergency dispatch or SOS.
4. **Zero Data Retention:** Audio is processed ephemerally in-memory buffers; raw user audio is discarded immediately following inference.
5. **Licence Compliance:** Datasets (ESC-50, UrbanSound8K, FSD50K) are audited for licence constraints (e.g., CC BY-NC) and tracked in a clip-level manifest.
