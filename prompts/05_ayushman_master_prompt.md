# AYUSHMAN — MASTER PROMPT: ROUTING + EMERGENCY RESOURCES + SOS
Work on `feature/navigation-sos`, paths `navigation/`, `emergency/`, tests and agreed docs. Read shared context and handbook.

History/context: PathPilot (formerly SafeRoam concept) is a Kolkata-first women-traveller safety app. Amisa owns geospatial data/evidence; you consume her outputs. Deep owns Android UI. Target is Android app installable via USB. Five agents integrate via GitHub and versioned contracts.

Build OSMnx/NetworkX road graph loader, origin/destination snapping, loopless candidate paths (e.g. Yen's k-shortest), documented travel-time estimates (no live traffic claim), route diversity filter by shared-edge overlap, and route ranking. Up to three routes: safest_estimated (prioritise segment evidence and essential amenities/shops reported open at ETA), balanced (safety/time/distance trade-off), fastest (time priority but retain warnings). Missing opening hours remain unknown. Return fewer than three if not enough distinct paths.

Use configurable conceptual cost: `C(route)=alpha*normalised_distance + beta*normalised_time + gamma*estimated_route_risk`. Keep score, evidence coverage, freshness and uncertainty separate. Include geometry, distance/time, component breakdown, timestamps, amenity summary and warnings.

Emergency resources must come from verified OSM/official records; never invent contact numbers or open status. SOS must preview recipient/location, require explicit confirmation, support cancel/failure and distinguish demo/test from real share. Audio never auto-triggers SOS.

Deliver road graph, candidate routes, optimiser, emergency lookup, SOS service, API contract and toy-graph tests for disconnected graph/no route/diversity/ETA opening boundary/missing evidence. Coordinate contracts with Amisa and Deep. No raw data/secrets in Git. Test, diff, commit and PR.
