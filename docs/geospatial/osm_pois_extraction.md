# Kolkata District OSM POIs & Essential Amenities Extraction

## Purpose
Extracts point and polygon OpenStreetMap POIs and essential amenities for **Kolkata District** from the Eastern Zone `.osm.pbf` extract, clipped precisely to the Survey of India Kolkata District boundary (`EPSG:4326`).

## Extracted Categories (`poi_category`)
1. `police` — `amenity=police`
2. `hospital` — `amenity=hospital`
3. `clinic` — `amenity=clinic`
4. `pharmacy` — `amenity=pharmacy`
5. `restaurant` — `amenity=restaurant`
6. `shop` — `shop=*` (excluding `shop=no`)
7. `bus_stop` — `highway=bus_stop`
8. `metro_station` — `railway=station` (or `public_transport=station`) with subway/metro tags (`station=subway`, `subway=yes`, or Metro operator/network). Track-level `railway=stop` positions and `railway=subway_entrance` gates are excluded to avoid double-counting stations.

## Inputs (Reused from `geospatial.osm.common`)
- **OSM PBF extract:** `C:\Users\amisa\Downloads\PathPilot_DATA\OSM\eastern-zone-261124.osm.pbf`
- **District boundary:** `C:\Users\amisa\Downloads\PathPilot_Data\Boundaries\Kolkata_District\kolkata_district_boundary.geojson` (`EPSG:4326`)

## Outputs (Stored Outside Git)
- **Clipped POIs GeoJSON:** `C:\Users\amisa\Downloads\PathPilot_DATA\Processed\OSM\kolkata_district_pois.geojson`
- **Extraction metadata:** `C:\Users\amisa\Downloads\PathPilot_DATA\Processed\OSM\kolkata_district_pois.metadata.json`

## How to Run
```powershell
C:\Users\amisa\Downloads\PathPilot_GIS\.venv\Scripts\python.exe -m geospatial.osm.extract_pois
```

## Preserved Attributes
- Core requested tags: `name`, `amenity`, `shop`, `highway`, `railway`, `opening_hours`, `operator`, `id`, `osm_type`, `geometry`
- Additional context: `poi_category`, `station`, `subway`, `public_transport`, `addr:street`, `addr:postcode`, `phone`, `website`, `timestamp`, `version`, `tags`
- Missing names or attributes remain `null` without discarding valid features.

## Attribution
Map data © OpenStreetMap contributors, available under the Open Database License (ODbL). Administrative boundary: Survey of India (SOI).
