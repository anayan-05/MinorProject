# Kolkata District Monthly Nighttime Lights (NASA VIIRS VNP46A3 Collection 2.0, 2025)

## Overview
This directory contains **Dataset B** for the PathPilot geospatial pipeline: monthly snow-free nighttime radiance composites for **Kolkata District, West Bengal, India** for **January–December 2025**, derived from NASA Black Marble **VNP46A3 Collection 2.0** (`archive/allData/5200/VNP46A3`), tile **`h26v06`**.

### Why VNP46A3 Was Selected Over VNP46A2
`VNP46A3` provides **monthly lunar-BRDF-adjusted composites** (12 granules for a full year) rather than daily `VNP46A2` swaths/grids (~365 granules/year), making it far more manageable for annual urban nighttime-lighting analysis while preserving NASA's standardized monthly compositing and quality flags.

> **Important Quality-Flag Distinction:** Because `VNP46A3` is a monthly composite product, it **does not contain the daily `QF_Cloud_Mask` layer** found in `VNP46A2`. No daily cloud-mask filtering was performed on these files; instead, quality filtering uses `VNP46A3`'s native composite quality layer (`AllAngle_Composite_Snow_Free_Quality`).

## Source & Study Area
- **NASA Product**: `VNP46A3` (VIIRS/NPP Lunar BRDF-Adjusted Nighttime Lights Monthly L3 Global 15 arc-second Linear Lat Lon Grid)
- **Collection**: `002` (`5200`)
- **Tile**: `h26v06` (`80°E–90°E`, `20°N–30°N`, `2400 × 2400` pixels, `HE5_GCTP_GEO` in `EPSG:4326`)
- **Source Archive**: `https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/5200/VNP46A3/2025/`
- **Study Area Boundary**: `data/geospatial/boundaries/kolkata_district_boundary.geojson` (`EPSG:4326`, 1 valid `MultiPolygon` feature)

## Selected SDS Layers (`HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields`)
- **Input Radiance**: `AllAngle_Composite_Snow_Free` (`float32`, `_FillValue = -999.9`, `scale_factor = 1.0`, `offset = 0.0`, units: `nW/(cm² sr)` / `nWatts/(cm^2 sr)`)
- **Composite Quality**: `AllAngle_Composite_Snow_Free_Quality` (`uint8`, `0 = Good quality`, `1 = Poor quality`, `2 = Gap filled`, `255 = Fill value`)
- **Observation Count**: `AllAngle_Composite_Snow_Free_Num` (`uint16`, `_FillValue = 65535`)
- **Radiance Std Dev**: `AllAngle_Composite_Snow_Free_Std` (`float32`, `_FillValue = -999.9`, units: `nW/(cm² sr)`)
- **Land/Water Mask**: `Land_Water_Mask` (`uint8`, `0 = Land & Desert`, `1 = Land no Desert`, `2 = Inland Water`, `3 = Sea Water`, `5 = Coastal`, `_FillValue = 255`)

## Processing Workflow Summary
1. **HDF5 Inspection**: Verified all 28 SDS layers, `HE5_GCTP_GEO` 15-arc-second geographic grid (`EPSG:4326`), `scale_factor = 1.0`, `offset = 0.0`, and `_FillValue` attributes on the January 2025 sample before processing all 12 months.
2. **Strict Quality Filtering**: Retained only pixels where `AllAngle_Composite_Snow_Free_Quality == 0` (Good quality), finite, `!= _FillValue (-999.9)`, and `>= 0.0`. Quality values `1` (Poor), `2` (Gap filled), and `255` (Fill) are masked to `NaN` (`NoData`). No missing pixels are interpolated or filled.
3. **Reprojection & Clipping**: Reprojected from `EPSG:4326` to **`EPSG:32645`** (WGS 84 / UTM Zone 45N) on a canonical **`450 m × 450 m`** grid (`48 × 54` pixels) using **nearest-neighbor resampling** (`Resampling.nearest`) to preserve original NASA radiance values without spatial interpolation, then clipped to pixel centers inside the Kolkata District boundary (`964` district pixels).
4. **Monthly Stacking & Statistics**: Stacked all 12 months (`2025-01` through `2025-12`) into a single 12-band GeoTIFF and computed monthly summary statistics (`mean`, `median`, `min`, `max`, `std`, valid and quality-flag pixel counts). Individual clipped monthly layers (`60` GeoTIFFs for radiance, quality, observation count, std, and land/water mask) are preserved separately outside Git in `C:\Users\amisa\Downloads\PathPilot_DATA\Processed\VIIRS\2025\monthly\`.

## Outputs in This Directory
- `kolkata_viirs_2025_monthly_radiance.tif`: 12-band `float32` GeoTIFF (`EPSG:32645`, `450 m` resolution, `48 × 54` pixels, `NoData = NaN`, bands `2025-01`–`2025-12`).
- `kolkata_viirs_2025_monthly_statistics.csv`: Monthly statistics and quality-flag pixel counts for all 12 months.
- `kolkata_viirs_2025_metadata.json`: Full provenance metadata including SDS names, processing rules, CRS/transform, and SHA-256 checksums for all 12 source HDF5 granules.

## Validation Results
- **Structure & Alignment**: Verified `12` bands ordered January–December 2025, `EPSG:32645`, `450.0 m` resolution, `48 × 54` dimensions, `0` finite pixels outside the Kolkata District boundary, and exact CRS/transform/dimension alignment across all `60` preserved monthly GeoTIFFs.
- **CSV & Checksum Consistency**: Recalculating statistics directly from `kolkata_viirs_2025_monthly_radiance.tif` matched `kolkata_viirs_2025_monthly_statistics.csv` with a maximum absolute difference of `0.0`. All `12 / 12` source granule SHA-256 checksums were verified.
- **Monsoon Observations (June–August 2025)**:
  - **July 2025 (`2025-07`)** has **`0` valid `Quality == 0` pixels** (`169` Quality `1`, `795` Quality `255` in `EPSG:32645`; `0–1` total observations per pixel in the raw NASA HDF5 granule due to persistent monsoon cloud cover). July 2025 is preserved as `NoData` (`NaN`), **not** zero radiance.
  - **June 2025 (`2025-06`)** (`126` valid pixels) and **August 2025 (`2025-08`)** (`281` valid pixels) also reflect reduced valid coverage due to monsoon clouds.

## Scale, Limitations & Separation from OSM `lit`
- **Spatial Scale**: VIIRS Day/Night Band (`~500 m` scale; `15 arc-seconds` native, `450 m` projected) measures **broad ambient nighttime radiance** over city blocks and neighborhoods—it **cannot identify individual streetlights** and should be kept conceptually and analytically separate from street-level OpenStreetMap `lit` attributes.
- **Monthly Composite Limitations**: Monthly composites smooth out day-to-day variations (power outages, events, weather) and remain vulnerable to persistent seasonal cloud cover (as seen in June–August 2025), atmospheric scattering, and stray-light/viewing-angle effects.

## Attribution & Reproducibility
Data source: NASA Level-1 and Atmosphere Archive & Distribution System Distributed Active Archive Center (LAADS DAAC), Black Marble `VNP46A3` Collection 2.0 (`https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/5200/VNP46A3/`). Administrative boundary: National Water Data Portal (NWDP) / GSI / SOI.

See [docs/geospatial/viirs_nighttime_lights_processing.md](../../../docs/geospatial/viirs_nighttime_lights_processing.md) for the full processing and validation reference.

```powershell
# Download 2025 VNP46A3 files (requires EARTHDATA_TOKEN env var or ~/.netrc)
C:\Users\amisa\Downloads\PathPilot_GIS\.venv\Scripts\python.exe -m geospatial.viirs.download_viirs --workers 3

# Process all 12 months into GeoTIFF, CSV, and metadata
C:\Users\amisa\Downloads\PathPilot_GIS\.venv\Scripts\python.exe -m geospatial.viirs.process_viirs

# Run unit tests
C:\Users\amisa\Downloads\PathPilot_GIS\.venv\Scripts\python.exe -m unittest discover -s tests/geospatial
```
