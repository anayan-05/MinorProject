"""Process NASA VIIRS Black Marble VNP46A3 Collection 2.0 monthly HDF5 files for Kolkata District."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from rasterio.transform import Affine, from_origin
from rasterio.warp import Resampling, reproject

from geospatial.viirs.common import (
    COLLECTION,
    COLLECTION_ARCHIVE_ID,
    DEFAULT_BOUNDARY_PATH,
    DEFAULT_MONTHLY_PROCESSED_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RAW_DIR,
    MONTH_DOYS_2025,
    PRODUCT,
    SOURCE_BASE_URL,
    TARGET_CRS,
    TILE,
    YEAR,
    file_sha256,
    load_and_validate_boundary,
)

HDF5_GRID_GROUP = "HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields"
SDS_RADIANCE = "AllAngle_Composite_Snow_Free"
SDS_QUALITY = "AllAngle_Composite_Snow_Free_Quality"
SDS_NUM_OBS = "AllAngle_Composite_Snow_Free_Num"
SDS_STD = "AllAngle_Composite_Snow_Free_Std"
SDS_LAND_WATER = "Land_Water_Mask"

RADIANCE_UNITS = "nWatts/(cm^2 sr)"
TARGET_PIXEL_SIZE_M = 450.0


def _decode_attr(val: Any) -> Any:
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="replace")
    if hasattr(val, "tolist"):
        val = val.tolist()
    if isinstance(val, list) and len(val) == 1:
        return _decode_attr(val[0])
    return val


def read_vnp46a3_month(h5_path: Path) -> dict[str, Any]:
    """Read required VNP46A3 SDS layers and georeferencing metadata from an HDF5 file."""
    if not h5_path.is_file():
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    with h5py.File(h5_path, "r") as hf:
        if HDF5_GRID_GROUP not in hf:
            raise KeyError(f"Missing expected HDF5 group {HDF5_GRID_GROUP} in {h5_path}")
        group = hf[HDF5_GRID_GROUP]

        required = [SDS_RADIANCE, SDS_QUALITY, SDS_NUM_OBS, SDS_STD, SDS_LAND_WATER, "lat", "lon"]
        missing = [name for name in required if name not in group]
        if missing:
            raise KeyError(f"Missing required SDS layers {missing} in {h5_path}")

        root_attrs = {k: _decode_attr(v) for k, v in hf.attrs.items()}
        lat = group["lat"][:]
        lon = group["lon"][:]

        # Verify 15-arc-second geographic grid (HE5_HDFE_GD_UL)
        west = float(root_attrs.get("WestBoundingCoord", lon[0]))
        north = float(root_attrs.get("NorthBoundingCoord", lat[0]))
        dlat = abs(float(lat[1] - lat[0]))
        dlon = abs(float(lon[1] - lon[0]))
        src_transform = from_origin(west, north, dlon, dlat)

        rad_ds = group[SDS_RADIANCE]
        rad_fill = float(_decode_attr(rad_ds.attrs["_FillValue"]))
        rad_scale = float(_decode_attr(rad_ds.attrs.get("scale_factor", 1.0)))
        rad_offset = float(_decode_attr(rad_ds.attrs.get("offset", 0.0)))
        rad_units = str(_decode_attr(rad_ds.attrs.get("units", RADIANCE_UNITS)))
        raw_rad = rad_ds[:].astype(np.float32)

        qual_ds = group[SDS_QUALITY]
        qual_fill = int(_decode_attr(qual_ds.attrs["_FillValue"]))
        quality = qual_ds[:].astype(np.uint8)

        num_ds = group[SDS_NUM_OBS]
        num_fill = int(_decode_attr(num_ds.attrs["_FillValue"]))
        num_obs = num_ds[:].astype(np.uint16)

        std_ds = group[SDS_STD]
        std_fill = float(_decode_attr(std_ds.attrs["_FillValue"]))
        std_scale = float(_decode_attr(std_ds.attrs.get("scale_factor", 1.0)))
        std_offset = float(_decode_attr(std_ds.attrs.get("offset", 0.0)))
        raw_std = std_ds[:].astype(np.float32)

        lw_ds = group[SDS_LAND_WATER]
        lw_fill = int(_decode_attr(lw_ds.attrs["_FillValue"]))
        land_water = lw_ds[:].astype(np.uint8)

    # Apply documented fill-value handling, scale/offset, valid_range (>= 0.0), and Quality == 0
    rad_scaled = raw_rad * rad_scale + rad_offset
    valid_rad_mask = (
        (quality == 0)
        & np.isfinite(raw_rad)
        & (~np.isclose(raw_rad, rad_fill, rtol=1e-5, atol=1e-5))
        & (rad_scaled >= 0.0)
    )
    strict_radiance = np.where(valid_rad_mask, rad_scaled, np.nan).astype(np.float32)

    std_scaled = raw_std * std_scale + std_offset
    valid_std_mask = (
        (quality == 0)
        & np.isfinite(raw_std)
        & (~np.isclose(raw_std, std_fill, rtol=1e-5, atol=1e-5))
        & (std_scaled >= 0.0)
    )
    strict_std = np.where(valid_std_mask, std_scaled, np.nan).astype(np.float32)

    return {
        "src_crs": "EPSG:4326",
        "src_transform": src_transform,
        "root_attrs": root_attrs,
        "radiance": strict_radiance,
        "quality": quality,
        "quality_fill": qual_fill,
        "num_obs": num_obs,
        "num_obs_fill": num_fill,
        "std": strict_std,
        "land_water": land_water,
        "land_water_fill": lw_fill,
        "units": rad_units,
    }


def build_target_grid(
    boundary_gdf: Any,
    target_crs: str = TARGET_CRS,
    pixel_size: float = TARGET_PIXEL_SIZE_M,
) -> tuple[Affine, int, int, np.ndarray]:
    """Create a canonical target grid in target_crs and boolean district mask."""
    boundary_proj = boundary_gdf.to_crs(target_crs)
    minx, miny, maxx, maxy = boundary_proj.total_bounds

    # Snap extent to pixel_size multiples with 1-pixel padding
    left = math.floor(minx / pixel_size - 1) * pixel_size
    bottom = math.floor(miny / pixel_size - 1) * pixel_size
    right = math.ceil(maxx / pixel_size + 1) * pixel_size
    top = math.ceil(maxy / pixel_size + 1) * pixel_size

    width = int(round((right - left) / pixel_size))
    height = int(round((top - bottom) / pixel_size))
    transform = from_origin(left, top, pixel_size, pixel_size)

    # True where pixel center falls inside Kolkata District MultiPolygon
    inside_mask = geometry_mask(
        boundary_proj.geometry,
        out_shape=(height, width),
        transform=transform,
        invert=True,
        all_touched=False,
    )
    return transform, width, height, inside_mask


def reproject_and_clip(
    array: np.ndarray,
    src_crs: str,
    src_transform: Affine,
    dst_crs: str,
    dst_transform: Affine,
    width: int,
    height: int,
    inside_mask: np.ndarray,
    src_nodata: float | int,
    dst_nodata: float | int,
    dtype: Any,
) -> np.ndarray:
    """Reproject a 2D array to the canonical target grid using nearest-neighbor (no interpolation) and clip."""
    dst = np.full((height, width), dst_nodata, dtype=dtype)
    reproject(
        source=array,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        src_nodata=src_nodata,
        dst_nodata=dst_nodata,
        resampling=Resampling.nearest,
    )
    return np.where(inside_mask, dst, dst_nodata).astype(dtype)


def _write_single_band_tif(
    path: Path,
    data: np.ndarray,
    crs: str,
    transform: Affine,
    nodata: float | int,
    description: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
        compress="deflate",
    ) as dst:
        dst.write(data, 1)
        dst.set_band_description(1, description)


def find_month_h5(raw_dir: Path, month: int, year: int = YEAR, tile: str = TILE) -> Path:
    """Locate the downloaded HDF5 file for a specific month in raw_dir."""
    doy = MONTH_DOYS_2025[month]
    matches = sorted(raw_dir.glob(f"{PRODUCT}.A{year}{doy}.{tile}.*.h5"))
    if not matches:
        raise FileNotFoundError(
            f"Missing raw HDF5 file for {year}-{month:02d} (DOY {doy}, tile {tile}) in {raw_dir}"
        )
    return matches[-1]


def process_all_months(
    raw_dir: Path = DEFAULT_RAW_DIR,
    boundary_path: Path = DEFAULT_BOUNDARY_PATH,
    monthly_dir: Path = DEFAULT_MONTHLY_PROCESSED_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    year: int = YEAR,
    tile: str = TILE,
) -> dict[str, Any]:
    """Process all 12 monthly VNP46A3 files into clipped GeoTIFFs, statistics CSV, and metadata."""
    boundary = load_and_validate_boundary(boundary_path)
    dst_transform, width, height, inside_mask = build_target_grid(boundary)
    total_district_pixels = int(np.count_nonzero(inside_mask))

    monthly_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    radiance_stack = np.full((12, height, width), np.nan, dtype=np.float32)
    stats_rows: list[dict[str, Any]] = []
    source_files_meta: list[dict[str, Any]] = []
    detected_units = RADIANCE_UNITS

    for month in range(1, 13):
        h5_path = find_month_h5(raw_dir, month=month, year=year, tile=tile)
        doy = MONTH_DOYS_2025[month]
        year_month = f"{year}-{month:02d}"
        data = read_vnp46a3_month(h5_path)
        detected_units = data["units"]

        rad_clip = reproject_and_clip(
            data["radiance"],
            data["src_crs"],
            data["src_transform"],
            TARGET_CRS,
            dst_transform,
            width,
            height,
            inside_mask,
            src_nodata=np.nan,
            dst_nodata=np.nan,
            dtype=np.float32,
        )
        qual_clip = reproject_and_clip(
            data["quality"],
            data["src_crs"],
            data["src_transform"],
            TARGET_CRS,
            dst_transform,
            width,
            height,
            inside_mask,
            src_nodata=data["quality_fill"],
            dst_nodata=255,
            dtype=np.uint8,
        )
        num_clip = reproject_and_clip(
            data["num_obs"],
            data["src_crs"],
            data["src_transform"],
            TARGET_CRS,
            dst_transform,
            width,
            height,
            inside_mask,
            src_nodata=data["num_obs_fill"],
            dst_nodata=65535,
            dtype=np.uint16,
        )
        std_clip = reproject_and_clip(
            data["std"],
            data["src_crs"],
            data["src_transform"],
            TARGET_CRS,
            dst_transform,
            width,
            height,
            inside_mask,
            src_nodata=np.nan,
            dst_nodata=np.nan,
            dtype=np.float32,
        )
        lw_clip = reproject_and_clip(
            data["land_water"],
            data["src_crs"],
            data["src_transform"],
            TARGET_CRS,
            dst_transform,
            width,
            height,
            inside_mask,
            src_nodata=data["land_water_fill"],
            dst_nodata=255,
            dtype=np.uint8,
        )

        # Preserve individual monthly observations separately outside Git
        _write_single_band_tif(
            monthly_dir / f"kolkata_viirs_{year}_{month:02d}_radiance.tif",
            rad_clip,
            TARGET_CRS,
            dst_transform,
            np.nan,
            f"{year_month} strict snow-free radiance (Quality==0)",
        )
        _write_single_band_tif(
            monthly_dir / f"kolkata_viirs_{year}_{month:02d}_quality.tif",
            qual_clip,
            TARGET_CRS,
            dst_transform,
            255,
            f"{year_month} AllAngle_Composite_Snow_Free_Quality",
        )
        _write_single_band_tif(
            monthly_dir / f"kolkata_viirs_{year}_{month:02d}_num_obs.tif",
            num_clip,
            TARGET_CRS,
            dst_transform,
            65535,
            f"{year_month} AllAngle_Composite_Snow_Free_Num",
        )
        _write_single_band_tif(
            monthly_dir / f"kolkata_viirs_{year}_{month:02d}_std.tif",
            std_clip,
            TARGET_CRS,
            dst_transform,
            np.nan,
            f"{year_month} AllAngle_Composite_Snow_Free_Std (Quality==0)",
        )
        _write_single_band_tif(
            monthly_dir / f"kolkata_viirs_{year}_{month:02d}_land_water.tif",
            lw_clip,
            TARGET_CRS,
            dst_transform,
            255,
            f"{year_month} Land_Water_Mask",
        )

        radiance_stack[month - 1] = rad_clip

        valid_vals = rad_clip[inside_mask & np.isfinite(rad_clip)]
        qual_in = qual_clip[inside_mask]
        stats_rows.append(
            {
                "month": month,
                "year_month": year_month,
                "doy": doy,
                "source_file": h5_path.name,
                "total_district_pixels": total_district_pixels,
                "valid_pixel_count": int(valid_vals.size),
                "quality_0_count": int(np.count_nonzero(qual_in == 0)),
                "quality_1_count": int(np.count_nonzero(qual_in == 1)),
                "quality_2_count": int(np.count_nonzero(qual_in == 2)),
                "quality_255_count": int(np.count_nonzero(qual_in == 255)),
                "mean_radiance": float(np.mean(valid_vals)) if valid_vals.size else float("nan"),
                "median_radiance": float(np.median(valid_vals)) if valid_vals.size else float("nan"),
                "min_radiance": float(np.min(valid_vals)) if valid_vals.size else float("nan"),
                "max_radiance": float(np.max(valid_vals)) if valid_vals.size else float("nan"),
                "std_radiance": float(np.std(valid_vals)) if valid_vals.size else float("nan"),
                "units": detected_units,
            }
        )

        source_files_meta.append(
            {
                "month": month,
                "year_month": year_month,
                "doy": doy,
                "filename": h5_path.name,
                "sha256": file_sha256(h5_path),
                "range_beginning_date": data["root_attrs"].get("RangeBeginningDate"),
                "range_ending_date": data["root_attrs"].get("RangeEndingDate"),
                "local_granule_id": data["root_attrs"].get("LocalGranuleID"),
            }
        )

    # Write 12-band monthly GeoTIFF
    stack_path = output_dir / f"kolkata_viirs_{year}_monthly_radiance.tif"
    with rasterio.open(
        stack_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=12,
        dtype=np.float32,
        crs=TARGET_CRS,
        transform=dst_transform,
        nodata=np.nan,
        compress="deflate",
    ) as dst:
        for idx in range(12):
            dst.write(radiance_stack[idx], idx + 1)
            dst.set_band_description(idx + 1, f"{year}-{idx + 1:02d}")
            dst.update_tags(
                idx + 1,
                year_month=f"{year}-{idx + 1:02d}",
                doy=MONTH_DOYS_2025[idx + 1],
                sds=SDS_RADIANCE,
                quality_filter="AllAngle_Composite_Snow_Free_Quality == 0",
                units=detected_units,
            )

    # Write statistics CSV
    csv_path = output_dir / f"kolkata_viirs_{year}_monthly_statistics.csv"
    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(csv_path, index=False)

    # Write metadata JSON
    metadata_path = output_dir / f"kolkata_viirs_{year}_metadata.json"
    metadata = {
        "dataset": "PathPilot Dataset B - Kolkata District Monthly Nighttime Lights",
        "nasa_product": PRODUCT,
        "collection": COLLECTION,
        "archive_collection_id": COLLECTION_ARCHIVE_ID,
        "tile": tile,
        "year": year,
        "months_covered": [f"{year}-{m:02d}" for m in range(1, 13)],
        "source_url": f"{SOURCE_BASE_URL}/{year}/",
        "boundary_file": str(boundary_path.relative_to(boundary_path.parents[3]))
        if len(boundary_path.parents) > 3
        else str(boundary_path),
        "crs": TARGET_CRS,
        "pixel_size_meters": TARGET_PIXEL_SIZE_M,
        "dimensions": {"height": height, "width": width, "bands": 12},
        "transform": list(dst_transform)[:6],
        "nodata": "NaN",
        "resampling": "nearest (no spatial interpolation)",
        "clipping": "pixel center inside Kolkata District MultiPolygon (all_touched=False)",
        "sds_layers": {
            "radiance": SDS_RADIANCE,
            "quality": SDS_QUALITY,
            "num_observations": SDS_NUM_OBS,
            "radiance_std": SDS_STD,
            "land_water_mask": SDS_LAND_WATER,
        },
        "quality_filtering": (
            "Strict primary radiance retains only pixels where "
            "AllAngle_Composite_Snow_Free_Quality == 0, finite, != _FillValue (-999.9), "
            "and >= 0.0. Quality values 1 (Poor), 2 (Gap filled), and 255 (Fill) are masked "
            "as NoData in the primary radiance output. Original quality, observation count, "
            "std, and land/water mask layers are preserved separately per month."
        ),
        "units": detected_units,
        "scale_factor": 1.0,
        "offset": 0.0,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "outputs": {
            "monthly_stack_geotiff": stack_path.name,
            "monthly_statistics_csv": csv_path.name,
            "monthly_preserved_dir": str(monthly_dir),
        },
        "source_granules": source_files_meta,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "stack_path": stack_path,
        "csv_path": csv_path,
        "metadata_path": metadata_path,
        "stats_df": stats_df,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Process 2025 VNP46A3 monthly files for Kolkata.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--boundary", type=Path, default=DEFAULT_BOUNDARY_PATH)
    parser.add_argument("--monthly-dir", type=Path, default=DEFAULT_MONTHLY_PROCESSED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    result = process_all_months(
        raw_dir=args.raw_dir,
        boundary_path=args.boundary,
        monthly_dir=args.monthly_dir,
        output_dir=args.output_dir,
    )
    print(f"Wrote 12-band GeoTIFF: {result['stack_path']}")
    print(f"Wrote statistics CSV : {result['csv_path']}")
    print(f"Wrote metadata JSON  : {result['metadata_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
