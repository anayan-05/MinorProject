# PATHPILOT — COMPLETE TEAM EXECUTION HANDBOOK
Version 1.0 | 18 November 2026

## 1. Project definition and history
PathPilot (earlier working concept name: SafeRoam) is an AI-assisted travel-safety companion for women travellers, initially focused on Kolkata, West Bengal. The goal is a genuinely working Android prototype installed on physical phones using USB debugging, not a Streamlit-only demo. It combines explainable geospatial evidence, review-based aspect/sentiment intelligence, optional user-authorised audio cue analysis, alternative route generation, emergency resources and a user-confirmed SOS/share workflow.

The user will provide UI design references later. Deep must start app scaffolding and integration now, but must not finalise visual branding, colours, typography, spacing or screen composition until those references arrive. Once provided, combine the user's design with accessible, safety-oriented interaction improvements.

### Product scope
MVP:
1. Android app opens on a physical phone.
2. User selects current or manual origin and destination.
3. App displays map and up to three genuinely distinct routes: safest-estimated, balanced, fastest.
4. Each route shows estimated time/distance, estimated safety score (if evidence supports it), score components, evidence coverage, data timestamps and warnings.
5. Nearby mapped shops and essential amenities are shown; listed opening hours are evaluated at estimated arrival time.
6. Review-derived local observations are displayed with source/date context.
7. Emergency resources are discoverable from verified records.
8. SOS/share previews what will be shared, requires explicit confirmation and supports cancellation.
9. Permission, offline, stale-data and no-route states are handled.
10. Debug APK builds and installs over USB.

Stretch after MVP: user-started audio cue model; on-device inference; accounts; community contribution; public cloud deployment.

Non-goals: guaranteed safety, crime prediction, live police tracking, real-time crowd feeds, automatic emergency dispatch, covert/always-on recording, fabricated traffic/shop/crime data.

## 2. Team and strict ownership
| Member | Role | Owns | Does not own |
|---|---|---|---|
| Anayan | Project lead + NLP | review annotation, RoBERTa ABSA, Colab notebook, inference API, project coordination | geospatial score, routing, audio, mobile UI |
| Debajyoti | Audio ML | audio dataset/licence registry, CNN notebook, inference API | hidden recording, SOS automation, app permissions |
| Amisa | Geospatial | study boundary, OSM/VIIRS/official-data pipeline, spatial features, explainable evidence/score | route candidate algorithm, UI |
| Deep | Android app/integration | React Native app, map, screens, permissions, API integration, APK | model training, inventing data |
| Ayushman | Routing/emergency/SOS | road graph routing, alternative paths, route ranking, emergency resources, user-confirmed SOS backend | spatial source truth, mobile visual design |

All agents must coordinate via versioned API contracts. No agent should silently redefine another owner's schema or overwrite another member's files.

## 3. Architecture
```text
React Native Android App (Deep)
  ├─ GPS/location permission + map
  ├─ origin/destination + route comparison
  ├─ evidence/review panels
  ├─ optional user-started audio capture
  └─ emergency resources + confirmed SOS
             |
             v
FastAPI backend (/api/v1)
  ├─ route service ----------> Ayushman
  ├─ geospatial evidence ----> Amisa
  ├─ review analysis --------> Anayan
  ├─ audio analysis ---------> Debajyoti
  └─ emergency/SOS ----------> Ayushman
             |
             v
OSM roads/POIs • VIIRS (if accessible) • official data (if granular)
permitted/annotated reviews • licensed audio datasets • model artefacts
```

## 4. Project methodology and safety logic
### Route options
- **Safest estimated:** prioritises route-segment evidence, essential amenities and shops reported open at estimated ETA, plus environmental/review evidence.
- **Balanced:** compromises estimated safety, time and distance.
- **Fastest:** prioritises estimated time but still displays evidence and warnings.

Generate candidates with OSMnx/NetworkX, e.g. Yen's k-shortest loopless paths. Filter near-duplicates using shared-edge overlap. Return fewer than three if fewer than three distinct feasible paths exist.

Conceptual cost:
`C(route) = alpha * normalised_distance + beta * normalised_time + gamma * estimated_route_risk`
Weights are provisional, configurable and must be sensitivity-tested. Score, evidence coverage, uncertainty and data freshness are separate.

### Evidence inputs
- OSM road connectivity and mapped POIs.
- Nearby shops/commercial activity.
- Essential amenities: police facilities, hospitals/clinics, pharmacies, transit and selected public facilities.
- OSM `opening_hours` evaluated at ETA.
- VIIRS nighttime radiance only with QA metadata.
- Review-derived perceptions: lighting, crowd/activity, police presence, transport access, road condition, isolation, commercial activity.
- Crime data only if official source granularity genuinely supports the geography/time period being analysed.

### Critical truthfulness constraints
- OSM opening hours are a listed schedule, not live confirmation. Label `reported open based on listed hours`; missing/unparseable means `unknown`, not closed.
- VIIRS Black Marble VNP46A2 is roughly 500m scale satellite-observed nighttime radiance, not proof of pedestrian-level street lighting.
- Crime totals must not be redistributed into small grid cells and presented as observed incidents. Keep data at actual supported scale or omit from segment scoring.
- Reviews are perceptions, not verified facts.
- Amenity proximity does not guarantee access, staffing or safety.
- Audio cue is not proof of danger and never automatically triggers SOS.
- Never claim all information is real-time. Show source timestamps and stale/unknown states.

## 5. Detailed work assignment — Anayan (NLP/reviews)
Branch: `feature/nlp-model`; paths: `ml/nlp/`, `docs/nlp/`, NLP tests.

1. Acquire official SemEval-2014 Task 4 ABSA data; optionally SemEval-2016 Task 5 if official release/terms are confirmed. Record version, source and terms. Use only for benchmark/transfer experiments; restaurant/laptop domains are not Kolkata travel-safety ground truth.
2. Identify local review sources whose terms permit intended collection/use, or use consented/manual contributions. Do not indiscriminately scrape Google Maps or bypass platform restrictions.
3. Anonymise names, phone numbers, emails, usernames and other identifiers. Record source, date, location ID/area, licence/terms.
4. Annotate aspects: `lighting`, `crowd_activity`, `police_presence`, `transport_access`, `road_condition`, `isolation`, `commercial_activity`, `general_environment`; polarity `positive`, `negative`, `neutral`.
5. Pilot 100–200 snippets; double-annotate a subset; calculate agreement; adjudicate; freeze guidelines. Split by location/source where feasible and prevent duplicates across splits.
6. Build `ml/nlp/training/pathpilot_review_absa.ipynb`, runnable top-to-bottom in fresh Colab: setup/seeds, configurable Drive path, validation, privacy cleaning, duplicates, label distribution, grouped split/leakage checks, RoBERTa tokenisation, model training, held-out evaluation, error analysis, saving and inference demo.
7. Report macro-F1, per-class/per-aspect precision/recall/F1, confusion matrix, support and failure cases. Keep final test untouched. Never invent metrics.
8. Deliver `preprocess.py`, `train.py`, `inference.py`, labels map, annotation guide, provenance register, model card, API contract and mock fixture.
9. Inference returns aspect, sentiment, model confidence, evidence text, source date, model version and limitations. It must not calculate final safety score.
10. Coordinate aspect names and location IDs with Amisa; provide fixture to Deep/backend.

## 6. Detailed work assignment — Debajyoti (audio ML)
Branch: `feature/audio-model`; paths: `ml/audio/`, `docs/audio/`, audio tests.

1. Investigate ESC-50, UrbanSound8K, FSD50K; optionally AudioSet only with authorised access and terms review.
2. Verify licences before use. ESC-50 is CC BY-NC; suitability for any commercial future use must be assessed. FSD50K licence can vary per clip; build clip-level allowlist. Verify UrbanSound8K terms. Never assume redistribution rights.
3. Map only actual source labels to cautious cue categories (e.g. human vocalisation, shout/call, impact/crash, alarm/siren, glass breaking only if source supports it). Do not invent “assault” or “distress” ground truth.
4. Build `ml/audio/training/pathpilot_audio_cue_detection.ipynb`, runnable in fresh Colab: setup/seeds, data/manifest validation, sample rate/channels/duration, licence check, grouped split by original recording, resampling/mono, log-Mel spectrogram, compact CNN, training, macro-F1/per-class metrics/confusion matrix, false-positive analysis, noise robustness, export and CPU smoke test.
5. Measure model size and inference latency. Save weights outside Git. Never fabricate metrics.
6. Deliver preprocess/train/inference modules, label mapping, licence manifest, model card, API contract, evaluation report and fixture.
7. Inference returns possible cue, confidence, time offsets, model version, limitations and `requires_user_confirmation: true`.
8. Audio must be user-started, visible and stoppable. No covert/always-on recording. Audio alone never triggers SOS. Coordinate native Android permission/foreground-service constraints with Deep.

## 7. Detailed work assignment — Amisa (geospatial)
Branch: `feature/geospatial`; paths: `geospatial/`, `docs/geospatial/`, spatial tests.

1. Confirm study boundary with Anayan and record source/date/CRS. KMC and Kolkata Police jurisdiction are not interchangeable.
2. Download OSM regional extract from Geofabrik or use responsible cached Overpass queries. Extract road/path network, shops/commercial POIs, police facilities, hospitals/clinics, pharmacies, transit stops and selected amenities. Preserve ODbL attribution, query and timestamp.
3. Parse `opening_hours` at route ETA. Output `reported_open`, `reported_closed`, `unknown`; missing/unparseable is unknown. This is schedule data, not live verification.
4. Investigate NASA Black Marble VNP46A2 via NASA Earthdata. Confirm version, dates, HDF-EOS5, QA flags and roughly 500m resolution. Use quality flags. If access is blocked, document and omit; do not fabricate.
5. Investigate NCRB, Kolkata Police, West Bengal Police and data.gov.in. Include only official data at its actual spatial/time granularity. Do not spread aggregate counts into grid cells.
6. Consume Anayan review outputs as perception evidence.
7. Build acquisition, preprocessing, opening-hours, spatial join, road-segment feature and scoring scripts. Candidate features: amenity/shop proximity, reported-open coverage at ETA, connectivity/isolation proxies, POI density, VIIRS with QA, review signals and crime only if valid.
8. Keep score, components, evidence coverage, uncertainty and timestamps separate. Missing data is not automatically safe. Run sensitivity analysis.
9. Deliver source/licence register, data dictionary, QA maps/tables, tests and versioned API contract to Ayushman.

## 8. Detailed work assignment — Deep (Android)
Branch: `feature/mobile-ui`; paths: `mobile/`, `docs/mobile/`, mobile tests.

Start now:
1. Create React Native + TypeScript app with native Android build support.
2. Set navigation, typed API client, versioned response types, mock mode and loading/error/empty/offline states.
3. Validate map SDK (MapLibre React Native candidate); it needs native build support, so do not assume Expo Go compatibility.
4. Implement location permission and manual-origin fallback.
5. Build screen shells: onboarding/permissions, map/home, destination search, route alternatives, route detail/evidence, reviews, emergency resources, audio consent/start/stop, SOS preview/confirm/cancel, privacy/settings.
6. Use mock payloads labelled `is_mock: true` until actual APIs are ready.
7. Show route profile, estimated time/distance, score, components, evidence coverage, timestamps, amenities and unknown opening status.
8. Do not claim SOS sent unless backend confirms. Demo SOS must say test/demo.
9. Wait for user's visual references before final colours, typography, spacing, icons and layouts. Then combine references with accessible safety interaction suggestions.
10. Build/install debug APK on physical phone; document versions and commands.

## 9. Detailed work assignment — Ayushman (routing/emergency/SOS)
Branch: `feature/navigation-sos`; paths: `navigation/`, `emergency/`, tests and agreed docs.

1. Consume Amisa's road graph and segment feature output; do not duplicate geospatial extraction.
2. Build OSMnx/NetworkX graph loader and origin/destination snapping.
3. Generate loopless candidates (e.g. Yen's k-shortest paths), estimate time with documented road-class assumptions; no live traffic claim without provider.
4. Filter near-duplicates by edge overlap.
5. Rank safest-estimated, balanced and fastest using configurable distance/time/risk terms. Safest profile gives high priority to route-segment evidence and essential amenities/shops reported open at ETA. Unknown opening hours stay unknown.
6. Return route geometry, distance, estimated duration, component breakdown, evidence coverage, timestamps and warnings. Return fewer than three if not enough distinct routes.
7. Emergency resources must come from verified OSM/official records; never invent phone numbers/open status.
8. SOS flow: preview location/recipient/data -> explicit confirmation -> send/test -> confirmed status; cancel and failure states. Demo endpoint labelled test. Never auto-trigger from audio.
9. Deliver route engine, emergency lookup, SOS service, API contract and tests on toy graphs (no route, disconnected graph, route diversity, ETA opening-hour boundary, missing evidence).

## 10. Dataset acquisition register
This is a plan, not a claim that files are already downloaded. Every member must verify current official source, exact version, terms/licence and intended use; save a manifest with URL, access date, version, licence, attribution, checksum, allowed use and storage path.

| Owner | Dataset/source | Purpose | Critical caveat |
|---|---|---|---|
| Anayan | SemEval-2014 Task 4 ABSA official release | benchmark/transfer | restaurant/laptop domain, not travel-safety truth |
| Anayan | SemEval-2016 Task 5 (optional official release) | benchmark/transfer | verify release and terms |
| Anayan | permitted/consented local Kolkata review snippets | project-specific annotations | no prohibited scraping; anonymise |
| Debajyoti | ESC-50 official release | environmental audio baseline | CC BY-NC restrictions |
| Debajyoti | UrbanSound8K official release | urban sound baseline | inspect terms and label fit |
| Debajyoti | FSD50K official release | sound event data | clip-level licence variation |
| Debajyoti | AudioSet (optional) | extra sound event data | authorised access and terms |
| Amisa | OSM/Geofabrik | roads and POIs | ODbL attribution; tags incomplete |
| Amisa | NASA Black Marble VNP46A2 | coarse nighttime radiance | QA flags, ~500m, not streetlight proof |
| Amisa | NCRB/Kolkata Police/WB Police/data.gov.in | crime context if valid | often aggregated; don't downscale |
| Ayushman | Amisa processed road graph | routing | single shared geospatial pipeline |
| Deep | mock fixtures | UI integration | clearly label mock; no training datasets |

Store raw/large data outside Git. Model weights in Drive or approved artifact store. Commit only scripts, manifests and small licence-compatible test fixtures.

## 11. Integration contracts
All APIs under `/api/v1`. JSON. WGS84 GeoJSON coordinate order `[longitude, latitude]`. Distances metres, duration seconds, timestamps ISO 8601 with timezone. Include schema version, generated time, source/model versions, warnings and `is_mock`. Unknown is explicit `unknown`/`null`, never silently zero.

Proposed endpoints:
- `GET /api/v1/health`
- `POST /api/v1/routes/alternatives`
- `GET /api/v1/places/nearby`
- `POST /api/v1/reviews/analyse`
- `POST /api/v1/audio/analyse`
- `POST /api/v1/sos/test`
- `POST /api/v1/sos/share` only if real authorised share is implemented.

Route response must contain route ID/profile/geometry/distance/duration/estimated score/evidence coverage/components/amenity summary/open-closed-unknown counts/source timestamps/warnings. NLP response contains aspects/sentiment/confidence/evidence text/source date/model version/limitations. Audio response contains cue/confidence/time offsets/`requires_user_confirmation`. Geospatial response contains segment ID/features/amenity proximity/opening status/source dates/score components/evidence coverage. SOS response must distinguish test, sent, failed and cancelled.

## 12. GitHub and Antigravity workflow
Anayan creates private repo `PathPilot`, invites the other four with Write access, enables main-branch protection/PR review where available, commits this pack, and creates branches:
- `feature/nlp-model` — Anayan
- `feature/audio-model` — Debajyoti
- `feature/geospatial` — Amisa
- `feature/mobile-ui` — Deep
- `feature/navigation-sos` — Ayushman

Each member authenticates their own GitHub account in their Antigravity environment, clones repo and works on own branch. Five Antigravity sessions do not share local files/chat history automatically; GitHub is source of truth. A prompt cannot grant itself repo permissions or invite members.

Every agent must inspect repo/status before edits, propose a plan, stay in owned paths, run tests, inspect `git diff`, then commit/push and open PR. Never push directly to main, force-push, rewrite shared history, delete branches or commit `.env`, credentials, signing keys, raw datasets, reviews, audio, large weights or personal location traces. Human owner approves permissions, destructive operations and merges.

Suggested `.gitignore`:
```gitignore
.env
.env.*
!.env.example
__pycache__/
.ipynb_checkpoints/
.venv/
node_modules/
build/
dist/
android/.gradle/
*.apk
*.aab
data/raw/**
data/processed/**
artifacts/models/**
*.pt
*.pth
*.ckpt
*.onnx
*.tflite
*.wav
*.mp3
*.flac
```

## 13. Phases and order of work
Phase 0 — Repo/team: private repo, collaborators, branches, docs, `.gitignore`, boundary decision, API conventions. Exit: everyone can clone and open test PR.
Phase 1 — Feasibility: validate dataset licences/access, OSM boundary, VIIRS availability, crime-data granularity, review annotation and audio labels; Deep scaffolds app; Ayushman proves route candidates on toy graph. Exit: data register/contracts.
Phase 2 — Baselines: NLP annotation + model, audio baseline, geospatial pipeline, route alternatives, mobile screens with mocks. Exit: reproducible notebooks, QA and tests.
Phase 3 — Integration: NLP -> geospatial; geospatial -> routing; route/emergency -> app; optional audio -> consent flow. Exit: local end-to-end journey.
Phase 4 — Device test: build APK, USB install, test selected journeys, permissions, stale data, no network and SOS demo. Exit: test report.
Phase 5 — Final demo/report: freeze versions, rerun tests, architecture/report/demo script/limitations/ethics. Exit: repeatable APK build.

## 14. Android build and real-time plan
Recommended: React Native + TypeScript native Android build; MapLibre React Native or another compatible map SDK; FastAPI; OSMnx/NetworkX; Python GIS/ML. Start local backend on laptop and phone connected by USB. Enable Developer Options and USB debugging, connect phone, accept authorisation, run `adb devices`, start backend, optionally run `adb reverse tcp:8000 tcp:8000`, build debug APK and install with `adb install path/to/app-debug.apk`.

Test: permission granted/denied, manual origin, GPS unavailable, backend offline, no route, one/two alternatives only, missing hours, stale sources, audio denied/start/stop, SOS cancel and demo mode.

Real-time capability truth:
- GPS can be live while permission granted.
- Route recalculation can be on-demand/periodic; ETA remains estimated unless a traffic provider exists.
- OSM opening hours are schedule-based, not live.
- VIIRS is periodic, not live.
- Crime publications are usually delayed/aggregated.
- Reviews update only when new permitted data are collected.
- Audio is live only during explicit visible capture.
Do not market all inputs as real-time.

## 15. Acceptance criteria
Anayan: annotation/provenance, leakage-aware split, notebook, held-out metrics/error analysis, inference contract.
Debajyoti: licence manifest, justified labels, notebook, held-out evaluation/false-positive analysis, confirmation-required inference.
Amisa: boundary/CRS, OSM provenance, opening-hours unknown semantics, VIIRS QA if used, crime scale validity, segment features, score/evidence and QA.
Ayushman: connected distinct routes, profile ranking, ETA amenities, resource provenance, SOS confirmation and tests.
Deep: APK builds/installs, core journey works, permissions/errors handled, APIs integrated, design references applied.
Whole product: no invented real-time claims, no safety guarantee, data attribution, mocks labelled, no secrets/personal data in Git.

## 16. Risks and mitigations
Sparse local reviews -> annotation pilot and transparent coverage. Missing hours -> unknown. Coarse crime -> exclude from segment scoring. VIIRS resolution mismatch -> coarse context only. Audio false positives -> confirmation, no auto-SOS. Native Android build -> test early on physical phone. Git conflicts -> ownership/contracts/PR. Free cloud sleep/limits -> local USB prototype first.
