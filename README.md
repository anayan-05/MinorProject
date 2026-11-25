# PathPilot

Kolkata-first AI-assisted travel-safety Android prototype for women travellers (earlier working concept name: SafeRoam).

## Team Ownership & Branches
- `feature/nlp-model` — **Anayan**: Project lead + Review NLP / RoBERTa ABSA (`ml/nlp/`, `docs/nlp/`)
- `feature/audio-model` — **Debajyoti**: Audio ML / CNN cue detection (`ml/audio/`, `docs/audio/`)
- `feature/geospatial` — **Amisa**: OSM / VIIRS / official geospatial data & explainable safety evidence (`geospatial/`, `docs/geospatial/`)
- `feature/mobile-ui` — **Deep**: React Native TypeScript Android app & USB APK build (`mobile/`, `docs/mobile/`)
- `feature/navigation-sos` — **Ayushman**: Route alternatives, emergency resources & user-confirmed SOS (`navigation/`, `emergency/`)

## Reference & Rules
- Read [`handbook/PathPilot_Complete_Handbook.md`](handbook/PathPilot_Complete_Handbook.md) before coding.
- Use assigned branches only; never push directly to `main`, force-push, or commit secrets, personal data, raw datasets/audio, or large model weights.
