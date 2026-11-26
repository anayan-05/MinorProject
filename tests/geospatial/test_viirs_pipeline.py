"""Unit tests for NASA VIIRS VNP46A3 monthly pipeline using synthetic fixtures."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import geopandas as gpd
import h5py
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import MultiPolygon, Polygon

from geospatial.viirs import download_viirs, process_viirs
from geospatial.viirs.common import MONTH_DOYS_2025


def _make_synthetic_boundary(path: Path) -> Path:
    """Create a small valid MultiPolygon in EPSG:4326 inside [88.0, 88.5] x [22.0, 22.5]."""
    poly = Polygon(
        [
            (88.30, 22.50),
            (88.40, 22.50),
            (88.40, 22.60),
            (88.30, 22.60),
            (88.30, 22.50),
        ]
    )
    gdf = gpd.GeoDataFrame(
        {"district": ["Kolkata"]},
        geometry=[MultiPolygon([poly])],
        crs="EPSG:4326",
    )
    gdf.to_file(path, driver="GeoJSON")
    return path


def _write_synthetic_vnp46a3_h5(
    path: Path,
    month: int,
    base_radiance: float = 10.0,
) -> Path:
    """Write a minimal synthetic VNP46A3-structured HDF5 file covering [88.0, 88.5] x [22.2, 22.7]."""
    nrows, ncols = 120, 120
    step = 1.0 / 240.0
    west, north = 88.0, 22.7

    lat = north - np.arange(nrows, dtype=np.float64) * step
    lon = west + np.arange(ncols, dtype=np.float64) * step

    rad = np.full((nrows, ncols), base_radiance + month, dtype=np.float32)
    qual = np.zeros((nrows, ncols), dtype=np.uint8)
    num_obs = np.full((nrows, ncols), 10, dtype=np.uint16)
    std = np.full((nrows, ncols), 1.5, dtype=np.float32)
    lw = np.ones((nrows, ncols), dtype=np.uint8)

    # Inject quality 1, 2, 255 and fill-value pixels
    qual[0:10, 0:10] = 1
    qual[10:20, 0:10] = 2
    qual[20:30, 0:10] = 255
    rad[20:30, 0:10] = -999.9

    with h5py.File(path, "w") as hf:
        hf.attrs["WestBoundingCoord"] = west
        hf.attrs["NorthBoundingCoord"] = north
        hf.attrs["RangeBeginningDate"] = f"2025-{month:02d}-01"
        hf.attrs["RangeEndingDate"] = f"2025-{month:02d}-28"
        hf.attrs["LocalGranuleID"] = path.name

        grp = hf.create_group(process_viirs.HDF5_GRID_GROUP)
        grp.create_dataset("lat", data=lat)
        grp.create_dataset("lon", data=lon)

        ds_rad = grp.create_dataset(process_viirs.SDS_RADIANCE, data=rad)
        ds_rad.attrs["_FillValue"] = np.float32(-999.9)
        ds_rad.attrs["scale_factor"] = np.float64(1.0)
        ds_rad.attrs["offset"] = np.float64(0.0)
        ds_rad.attrs["units"] = b"nWatts/(cm^2 sr)"

        ds_qual = grp.create_dataset(process_viirs.SDS_QUALITY, data=qual)
        ds_qual.attrs["_FillValue"] = np.uint8(255)

        ds_num = grp.create_dataset(process_viirs.SDS_NUM_OBS, data=num_obs)
        ds_num.attrs["_FillValue"] = np.uint16(65535)

        ds_std = grp.create_dataset(process_viirs.SDS_STD, data=std)
        ds_std.attrs["_FillValue"] = np.float32(-999.9)
        ds_std.attrs["scale_factor"] = np.float64(1.0)
        ds_std.attrs["offset"] = np.float64(0.0)

        ds_lw = grp.create_dataset(process_viirs.SDS_LAND_WATER, data=lw)
        ds_lw.attrs["_FillValue"] = np.uint8(255)

    return path


class TestViirsPipeline(unittest.TestCase):
    def test_strict_quality_filtering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            h5_path = _write_synthetic_vnp46a3_h5(Path(tmp) / "sample.h5", month=1)
            data = process_viirs.read_vnp46a3_month(h5_path)

            rad = data["radiance"]
            # Quality 1, 2, 255 rows must be NaN
            self.assertTrue(np.all(np.isnan(rad[0:30, 0:10])))
            # Quality 0 pixel must retain exact value (10.0 + 1 = 11.0)
            self.assertAlmostEqual(float(rad[50, 50]), 11.0, places=5)
            # Original quality & num_obs preserved
            self.assertEqual(int(data["quality"][0, 0]), 1)
            self.assertEqual(int(data["quality"][10, 0]), 2)
            self.assertEqual(int(data["num_obs"][50, 50]), 10)

    def test_process_all_months_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_dir = tmp_path / "raw"
            raw_dir.mkdir()
            monthly_dir = tmp_path / "monthly"
            out_dir = tmp_path / "out"
            boundary_path = _make_synthetic_boundary(tmp_path / "boundary.geojson")

            for month in range(1, 13):
                doy = MONTH_DOYS_2025[month]
                fname = f"VNP46A3.A2025{doy}.h26v06.002.2026064000000.h5"
                _write_synthetic_vnp46a3_h5(raw_dir / fname, month=month)

            result = process_viirs.process_all_months(
                raw_dir=raw_dir,
                boundary_path=boundary_path,
                monthly_dir=monthly_dir,
                output_dir=out_dir,
            )

            stack_path = result["stack_path"]
            csv_path = result["csv_path"]
            meta_path = result["metadata_path"]

            self.assertTrue(stack_path.is_file())
            self.assertTrue(csv_path.is_file())
            self.assertTrue(meta_path.is_file())

            with rasterio.open(stack_path) as src:
                self.assertEqual(src.count, 12)
                self.assertEqual(src.crs.to_epsg(), 32645)
                self.assertEqual(
                    list(src.descriptions),
                    [f"2025-{m:02d}" for m in range(1, 13)],
                )
                band1 = src.read(1)
                band12 = src.read(12)
                self.assertTrue(np.isnan(band1[0, 0]))
                self.assertAlmostEqual(float(np.nanmean(band1)), 11.0, places=4)
                self.assertAlmostEqual(float(np.nanmean(band12)), 22.0, places=4)

            df = pd.read_csv(csv_path)
            self.assertEqual(len(df), 12)
            self.assertEqual(list(df["month"]), list(range(1, 13)))
            self.assertTrue((df["valid_pixel_count"] > 0).all())

            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.assertEqual(meta["nasa_product"], "VNP46A3")
            self.assertEqual(meta["collection"], "002")
            self.assertEqual(meta["tile"], "h26v06")
            self.assertEqual(len(meta["source_granules"]), 12)

            # Verify separate monthly preserved files exist (12 * 5 = 60)
            self.assertEqual(len(list(monthly_dir.glob("*.tif"))), 60)

    def test_missing_credentials_raises_runtime_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch("geospatial.viirs.download_viirs.get_earthdata_token", return_value=None),
                patch(
                    "geospatial.viirs.download_viirs.get_earthdata_netrc_credentials",
                    return_value=None,
                ),
            ):
                with self.assertRaises(RuntimeError):
                    download_viirs.download_month(1, raw_dir=Path(tmp))


if __name__ == "__main__":
    unittest.main()
