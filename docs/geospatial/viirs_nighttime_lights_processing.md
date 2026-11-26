# Kolkata District NASA VIIRS Black Marble (`VNP46A3`) Processing

## Purpose
Produces **Dataset B** for PathPilot: a 12-band monthly nighttime radiance GeoTIFF (`EPSG:32645`, `450 m` resolution), monthly summary statistics CSV, and provenance metadata for **Kolkata District** across **January–December 2025** using NASA Black Marble **VNP46A3 Collection 2.0** (`tile h26v06`).

## Product Selection: `VNP46A3` vs. `VNP46A2`
- **`VNP46A3` (Selected)**: Monthly lunar-BRDF-adjusted nighttime lights composite (`12` granules/year). It is well-suited and manageable for annual district-scale temporal analysis.
- **`VNP46A2` (Not Used)**: Daily nighttime lights product (`~365` granules/year).
- **No Daily Cloud Mask (`QF_Cloud_Mask`)**: `VNP46A3` **does not provide the daily `QF_Cloud_Mask` layer** present in `VNP46A2`. Consequently, daily cloud-mask filtering was **not** performed; quality screening relies on `VNP46A3`'s monthly composite quality flag (`AllAngle_Composite_Snow_Free_Quality`).

## Inputs & Storage
- **Source Product**: NASA Black Marble `VNP46A3`, Collection `002` (`5200`), Tile `h26v06` (`80°E–90°E`, `20°N–30°N`, `2400 × 2400` pixels, `HE5_GCTP_GEO` in `EPSG:4326`)
- **Archive URL**: `https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/5200/VNP46A3/2025/`
- **Raw HDF5 Directory (Outside Git)**: `C:\Users\amisa\Downloads\PathPilot_DATA\VIIRS\VNP46A3\2025\`
- **District Boundary**: `data/geospatial/boundaries/kolkata_district_boundary.geojson` (`EPSG:4326`, 1 valid `MultiPolygon`)

### HDF5 SDS Layers Used (`HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields`)
| Role | SDS Name | Dtype | `_FillValue` | Scale / Offset | Units |
|---|---|---|---|---|---|
| Primary Radiance | `AllAngle_Composite_Snow_Free` | `float32` | `-999.9` | `1.0` / `0.0` | `nW/(cm² sr)` |
| Composite Quality | `AllAngle_Composite_Snow_Free_Quality` | `uint8` | `255` | `1.0` / `0.0` | Flag (`0`=Good, `1`=Poor, `2`=Gap-filled, `255`=Fill) |
| Observation Count | `AllAngle_Composite_Snow_Free_Num` | `uint16` | `65535` | `1.0` / `0.0` | Count (`0–65534`) |
| Radiance Std Dev | `AllAngle_Composite_Snow_Free_Std` | `float32` | `-999.9` | `1.0` / `0.0` | `nW/(cm² sr)` |
| Land/Water Mask | `Land_Water_Mask` | `uint8` | `255` | `1.0` / `0.0` | Class (`0,1`=Land, `2`=Inland Water, `3`=Sea, `5`=Coastal) |

## Processing Workflow
1. **Sample Verification (`geospatial.viirs.download_viirs`)**: Downloaded the January 2025 (`2025-01`, DOY `001`) granule via authenticated NASA Earthdata Login (`EARTHDATA_TOKEN`), inspected `StructMetadata.0` and all 28 SDS layers, and verified `HE5_GCTP_GEO` (`EPSG:4326`) georeferencing, scale factors, and fill values before downloading February–December 2025.
2. **Quality Filtering (`geospatial.viirs.process_viirs`)**: For the primary radiance output, retained only pixels where `AllAngle_Composite_Snow_Free_Quality == 0`, value is finite, `!= _FillValue (-999.9)`, and `>= 0.0`. Quality values `1`, `2`, and `255` are masked to `NaN` (`NoData`). No values are interpolated or gap-filled.
3. **Reprojection & Clipping**: Built a canonical **`EPSG:32645`** (WGS 84 / UTM Zone 45N) grid at **`450.0 m × 450.0 m`** resolution (`48 × 54` pixels) snapped around Kolkata District. Reprojected each month using **nearest-neighbor resampling** (`Resampling.nearest`) to preserve original NASA radiance values, then clipped to pixel centers inside the Kolkata District `MultiPolygon` (`all_touched=False`, `964` district pixels).
4. **Monthly Stacking & Preserved Layers**:
   - Exported the 12-band stack to `data/geospatial/viirs/kolkata_viirs_2025_monthly_radiance.tif` (`float32`, `NoData = NaN`, bands `2025-01`–`2025-12`).
   - Preserved `60` individual monthly GeoTIFFs (radiance, quality, observation count, std, land/water mask × 12 months) outside Git in `C:\Users\amisa\Downloads\PathPilot_DATA\Processed\VIIRS\2025\monthly\`.
5. **Statistical Analysis & Metadata**: Computed monthly `mean`, `median`, `min`, `max`, `std`, valid pixel count, and quality-flag pixel counts into `data/geospatial/viirs/kolkata_viirs_2025_monthly_statistics.csv`, and recorded provenance + SHA-256 checksums in `data/geospatial/viirs/kolkata_viirs_2025_metadata.json`.

## Validation Results
- **Raster Integrity**: Confirmed `12` bands ordered `2025-01` to `2025-12`, `EPSG:32645`, `450.0 m` resolution, `48 × 54` dimensions, `0` finite pixels outside the Kolkata District boundary, and identical CRS/transform/dimensions across all `60` preserved monthly GeoTIFFs.
- **CSV & Checksum Verification**: Independent recalculation of all monthly statistics from the 12-band GeoTIFF matched the CSV with a maximum absolute difference of `0.0`. All `12 / 12` source HDF5 SHA-256 checksums matched.
- **Monsoon Cloud Gaps**:
  - **July 2025 (`2025-07`)**: **`0` valid `Quality == 0` pixels** (`169` Quality `1`, `795` Quality `255` in `EPSG:32645`; native `EPSG:4326` HDF5 inspection confirmed `0–1` cloud-free observations across Kolkata District due to monsoon cloud cover). Preserved as **`NoData` (`NaN`)**, not `0.0` radiance.
  - **June 2025 (`2025-06`)** (`126` valid pixels) and **August 2025 (`2025-08`)** (`281` valid pixels) also reflect reduced monsoon clear-sky coverage.

## Limitations & Relationship to OSM `lit`
- **Spatial Scale (~500 m)**: VIIRS DNB captures **broad ambient nighttime radiance** at ~`450–500 m` scale and **cannot resolve individual streetlights** or sidewalk-level illumination. Keep VIIRS radiance separate from street-level OpenStreetMap `lit` tags during modeling and analysis.
- **Monthly Composite Limitations**: Monthly compositing smooths sub-monthly temporal variability and can produce partial or complete `NoData` gaps during persistent cloud seasons (e.g., June–August monsoon in Kolkata).

## How to Run
```powershell
C:\Users\amisa\Downloads\PathPilot_GIS\.venv\Scripts\python.exe -m geospatial.viirs.download_viirs --workers 3
C:\Users\amisa\Downloads\PathPilot_GIS\.venv\Scripts\python.exe -m geospatial.viirs.process_viirs
C:\Users\amisa\Downloads\PathPilot_GIS\.venv\Scripts\python.exe -m unittest discover -s tests/geospatial -v
```

## Attribution
NASA VIIRS Black Marble `VNP46A3` Collection 2.0, distributed by NASA LAADS DAAC (`https://ladsweb.modaps.eosdis.nasa.gov/`). Kolkata District boundary: National Water Data Portal (NWDP) / Geological Survey of India (GSI) / Survey of India (SOI).
