# DEEP — MASTER PROMPT: ANDROID APP + INTEGRATION
Work on `feature/mobile-ui`, paths `mobile/`, `docs/mobile/`, mobile tests. Read shared context and handbook.

History/context: PathPilot (earlier SafeRoam concept) is a Kolkata-first AI-assisted safety companion for women travellers. User wants a real Android app installable on phones via USB debugging. User will supply design references later; combine their design with accessibility/safety interaction suggestions. Do not finalise visual styling before references, but do not wait to start scaffolding.

Start now: React Native + TypeScript native Android project; navigation; typed API client and versioned schemas; mock mode; loading/error/empty/offline states; map SDK spike (MapLibre React Native candidate; validate native build); location permission/manual origin; screen shells for home/map, destination search, route alternatives, route detail/evidence, review insights, emergency resources, audio consent/start/stop, SOS preview/confirm/cancel and privacy/settings.

Display route profile, estimated distance/time, estimated score, components, evidence coverage, timestamps, amenity status and unknown values. Mocks must say `is_mock: true`. Never claim SOS sent before backend confirmation; demo SOS labelled test. Audio is visible/user-started/stoppable; no auto-SOS. Do not assume Expo Go supports native MapLibre modules.

After design references arrive, apply their layout/branding and improve accessibility. Integrate real APIs by agreed contracts; do not invent fields. Build debug APK and verify on physical phone: `adb devices`, install APK, and use `adb reverse tcp:8000 tcp:8000` for laptop backend if suitable. Test permission denial, GPS off, no network, API failure, no route, stale data, audio denial and SOS cancel. No secrets/signing keys in Git. PR with build/test results.
