# Kolkata District OSM Pedestrian Roads Extraction

## Purpose
Extracts pedestrian-accessible roads and paths for **Kolkata District** from the Eastern Zone OpenStreetMap (`.osm.pbf`) extract, clipped precisely to the Survey of India Kolkata District boundary (`EPSG:4326`).

## Inputs
- **OSM PBF extract:** `C:\Users\amisa\Downloads\PathPilot_DATA\OSM\eastern-zone-261124.osm.pbf`
- **District boundary:** `C:\Users\amisa\Downloads\PathPilot_Data\Boundaries\Kolkata_District\kolkata_district_boundary.geojson` (`EPSG:4326`, valid `MultiPolygon`)

## Outputs (Stored Outside Git)
- **Clipped roads GeoJSON:** `C:\Users\amisa\Downloads\PathPilot_DATA\Processed\OSM\kolkata_district_pedestrian_roads.geojson`
- **Extraction metadata:** `C:\Users\amisa\Downloads\PathPilot_DATA\Processed\OSM\kolkata_district_pedestrian_roads.metadata.json`

## How to Run
Using the existing GIS virtual environment (`PathPilot_GIS\.venv`):

```powershell
C:\Users\amisa\Downloads\PathPilot_GIS\.venv\Scripts\python.exe -m geospatial.osm.extract_roads
```

## Preserved OSM Tags
- Required walkability/safety tags: `highway`, `lit`, `surface`, `name`, `sidewalk`, `opening_hours`
- Additional road context: `foot`, `footway`, `access`, `oneway`, `bridge`, `tunnel`, `lanes`, `width`, `smoothness`, `service`, `segregated`, `id`, `osm_type`, `timestamp`, `version`, `tags`
- Derived metric length: `clipped_length_m` (computed in `EPSG:32645` UTM Zone 45N after clipping; exported geometries remain in `EPSG:4326`)

## Attribution
Map data © OpenStreetMap contributors, available under the Open Database License (ODbL). Administrative boundary: Survey of India (SOI).
