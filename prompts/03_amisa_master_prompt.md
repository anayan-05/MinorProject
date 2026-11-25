# AMISA — MASTER PROMPT: GEOSPATIAL DATA + EXPLAINABLE EVIDENCE
Work on `feature/geospatial`, paths `geospatial/`, `docs/geospatial/`, spatial tests. Read shared context and handbook.

History/context: PathPilot is a Kolkata-first women-traveller safety app. Your output feeds Ayushman's route engine and Deep's Android app. You own spatial evidence, not route candidate algorithm or UI.

First confirm study boundary with Anayan. KMC and Kolkata Police jurisdictions differ. Document boundary/source/date/CRS. Extract OSM roads/paths, shops/commercial POIs, police facilities, hospitals/clinics, pharmacies, transit and selected amenities via Geofabrik or responsible cached Overpass. Preserve ODbL attribution and timestamps. Parse `opening_hours` at ETA into `reported_open`, `reported_closed`, `unknown`; missing/unparseable is unknown and this is not live verification.

Investigate NASA Black Marble VNP46A2 through NASA Earthdata; verify product/version/date/HDF-EOS5/QA flags and ~500m scale. Use QA; radiance is not pedestrian-level lighting. If access is blocked, document and omit. Investigate NCRB/Kolkata Police/WB Police/data.gov.in; include crime only at genuine spatial/time resolution; never disaggregate city totals into grids. Consume Anayan review signals as perceptions.

Build acquisition, preprocessing, opening-hours parser, spatial joins, road-segment features and transparent provisional score. Candidate features: shop/amenity proximity, share of edges near amenities reported open at ETA, connectivity/isolation, POI density, VIIRS with QA, review evidence, crime only if valid. Separate score/components/evidence coverage/uncertainty/freshness; missing is not safe. Run sensitivity analysis and QA.

Deliver scripts, source/licence register, data dictionary, QA outputs, tests and API contract to Ayushman. Keep raw extracts/secrets out of Git. Test, diff, commit and PR.
