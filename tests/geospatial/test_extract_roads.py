"""Unit tests for Kolkata District OSM pedestrian road extraction."""

from __future__ import annotations

import json
import struct
import sys
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import geopandas as gpd
from pyrosm.proto import fileformat_pb2, osmformat_pb2
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiPolygon,
    Point,
    Polygon,
)

from geospatial.osm import extract_roads


def _make_sample_boundary_gdf(crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    """Create a tiny valid MultiPolygon roughly around central Kolkata."""
    poly = Polygon(
        [
            (88.34, 22.54),
            (88.38, 22.54),
            (88.38, 22.58),
            (88.34, 22.58),
            (88.34, 22.54),
        ]
    )
    return gpd.GeoDataFrame(
        {
            "district": ["Kolkata"],
            "src_agency": ["Survey of India (SOI)"],
        },
        geometry=[MultiPolygon([poly])],
        crs=crs,
    )


def _write_synthetic_pbf_header(
    path: Path,
    bbox: tuple[float, float, float, float] | None = (88.0, 22.0, 89.0, 23.0),
    replication_ts: int = 1764019283,
    compressed: bool = True,
    blob_type: str = "OSMHeader",
) -> Path:
    """Write a minimal valid binary PBF header block (~70 bytes) to disk."""
    hb = osmformat_pb2.HeaderBlock()
    if bbox is not None:
        left, bottom, right, top = bbox
        hb.bbox.left = int(round(left * 1e9))
        hb.bbox.bottom = int(round(bottom * 1e9))
        hb.bbox.right = int(round(right * 1e9))
        hb.bbox.top = int(round(top * 1e9))
    hb.osmosis_replication_timestamp = replication_ts
    hb.writingprogram = "test-writer/1.0"
    raw_hb = hb.SerializeToString()

    blob = fileformat_pb2.Blob()
    if compressed:
        blob.zlib_data = zlib.compress(raw_hb)
    else:
        blob.raw = raw_hb
    blob.raw_size = len(raw_hb)
    raw_blob = blob.SerializeToString()

    bh = fileformat_pb2.BlobHeader()
    bh.type = blob_type
    bh.datasize = len(raw_blob)
    raw_bh = bh.SerializeToString()

    with path.open("wb") as handle:
        handle.write(struct.pack(">I", len(raw_bh)))
        handle.write(raw_bh)
        handle.write(raw_blob)
    return path


def _make_sample_roads_gdf(
    include_optional_tags: bool = True,
    crs: str | None = "EPSG:4326",
) -> gpd.GeoDataFrame:
    """Create synthetic road lines (inside, crossing, outside, border-touch)."""
    data: dict[str, list] = {
        "id": [1, 2, 3, 4],
        "osm_type": ["way", "way", "way", "way"],
        "highway": ["residential", "footway", "primary", "path"],
        "name": ["Park Street", None, "Outside Road", "Border Touch"],
        "timestamp": [1700000000, 1700000001, 1700000002, 1700000003],
        "version": [3, 1, 2, 1],
        "tags": ['{"highway":"residential"}', None, None, None],
    }
    if include_optional_tags:
        data["lit"] = ["yes", None, "no", None]
        data["surface"] = ["asphalt", "paved", None, None]
        data["sidewalk"] = ["both", None, None, None]
        data["opening_hours"] = [None, "06:00-22:00", None, None]

    geoms = [
        # 1. Completely inside boundary
        LineString([(88.35, 22.55), (88.37, 22.55)]),
        # 2. Crosses eastern edge (88.38) -> should be clipped at 88.38
        LineString([(88.36, 22.56), (88.41, 22.56)]),
        # 3. Completely outside boundary -> should be dropped
        LineString([(88.50, 22.60), (88.52, 22.61)]),
        # 4. Touches boundary only at a single point (88.34, 22.54) -> point artifact dropped
        LineString([(88.32, 22.52), (88.34, 22.54)]),
    ]
    return gpd.GeoDataFrame(data, geometry=geoms, crs=crs)


class TestBoundaryValidation(unittest.TestCase):
    """Tests for load_and_validate_boundary."""

    def test_valid_epsg4326_multipolygon(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "boundary.geojson"
            _make_sample_boundary_gdf().to_file(path, driver="GeoJSON")
            loaded = extract_roads.load_and_validate_boundary(path)
            self.assertEqual(loaded.crs.to_epsg(), 4326)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded.geometry.geom_type.iloc[0], "MultiPolygon")

    def test_missing_boundary_file_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                extract_roads.load_and_validate_boundary(Path(tmp) / "missing.geojson")

    def test_empty_boundary_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.geojson"
            empty_gdf = gpd.GeoDataFrame({"name": []}, geometry=[], crs="EPSG:4326")
            empty_gdf.to_file(path, driver="GeoJSON")
            with self.assertRaises(ValueError):
                extract_roads.load_and_validate_boundary(path)

    def test_non_4326_crs_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "utm.gpkg"
            utm_gdf = _make_sample_boundary_gdf().to_crs("EPSG:32645")
            utm_gdf.to_file(path, driver="GPKG")
            with self.assertRaises(ValueError):
                extract_roads.load_and_validate_boundary(path)

    def test_non_polygon_geometry_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "line.geojson"
            line_gdf = gpd.GeoDataFrame(
                {"name": ["not a polygon"]},
                geometry=[LineString([(88.34, 22.54), (88.38, 22.58)])],
                crs="EPSG:4326",
            )
            line_gdf.to_file(path, driver="GeoJSON")
            with self.assertRaises(ValueError):
                extract_roads.load_and_validate_boundary(path)

    def test_invalid_bowtie_polygon_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "invalid.geojson"
            bowtie = Polygon(
                [(88.34, 22.54), (88.38, 22.58), (88.38, 22.54), (88.34, 22.58), (88.34, 22.54)]
            )
            bad_gdf = gpd.GeoDataFrame({"name": ["bowtie"]}, geometry=[bowtie], crs="EPSG:4326")
            bad_gdf.to_file(path, driver="GeoJSON")
            with self.assertRaises(ValueError):
                extract_roads.load_and_validate_boundary(path)


class TestPbfCoverageValidation(unittest.TestCase):
    """Tests for read_pbf_header and verify_pbf_covers_boundary."""

    def test_read_compressed_and_raw_pbf_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for compressed in (True, False):
                pbf = _write_synthetic_pbf_header(
                    Path(tmp) / f"sample_{compressed}.osm.pbf",
                    bbox=(88.0, 22.0, 89.0, 23.0),
                    compressed=compressed,
                )
                header = extract_roads.read_pbf_header(pbf)
                self.assertEqual(header["writingprogram"], "test-writer/1.0")
                self.assertIsNotNone(header["osmosis_replication_timestamp_utc"])
                for actual, expected in zip(header["bbox"], [88.0, 22.0, 89.0, 23.0]):
                    self.assertAlmostEqual(actual, expected, places=6)

    def test_missing_pbf_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                extract_roads.read_pbf_header(Path(tmp) / "missing.osm.pbf")

    def test_wrong_first_blob_type_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pbf = _write_synthetic_pbf_header(
                Path(tmp) / "bad.osm.pbf",
                blob_type="OSMData",
            )
            with self.assertRaises(ValueError):
                extract_roads.read_pbf_header(pbf)

    def test_verify_pbf_covers_boundary_passes(self) -> None:
        boundary = _make_sample_boundary_gdf()
        result = extract_roads.verify_pbf_covers_boundary(
            {"bbox": [88.0, 22.0, 89.0, 23.0]},
            boundary,
        )
        self.assertTrue(result["covered"])

    def test_verify_pbf_outside_boundary_raises(self) -> None:
        boundary = _make_sample_boundary_gdf()
        with self.assertRaises(ValueError):
            extract_roads.verify_pbf_covers_boundary(
                {"bbox": [88.35, 22.0, 89.0, 23.0]},  # misses west edge 88.34
                boundary,
            )

    def test_verify_pbf_missing_bbox_raises(self) -> None:
        boundary = _make_sample_boundary_gdf()
        with self.assertRaises(ValueError):
            extract_roads.verify_pbf_covers_boundary({"bbox": None}, boundary)


class TestClippingAndMissingTags(unittest.TestCase):
    """Tests for clipping, geometry normalization, and missing OSM tags."""

    def test_clip_trims_crossing_road_and_drops_outside_and_point_artifacts(self) -> None:
        boundary = _make_sample_boundary_gdf()
        roads = _make_sample_roads_gdf(include_optional_tags=True)

        clipped, stats = extract_roads.clip_and_prepare_roads(roads, boundary)

        self.assertEqual(stats["pre_clip_feature_count"], 4)
        self.assertEqual(stats["dropped_non_linear_border_artifacts"], 1)
        self.assertEqual(stats["post_clip_feature_count"], 2)
        self.assertEqual(len(clipped), 2)
        self.assertEqual(clipped.crs.to_epsg(), 4326)

        # Crossing road (id=2) must stop at eastern boundary 88.38
        clipped_bounds = clipped.total_bounds
        self.assertLessEqual(clipped_bounds[2], 88.38 + 1e-9)
        self.assertTrue((clipped["clipped_length_m"] > 0).all())

    def test_missing_osm_tag_columns_are_preserved_as_null(self) -> None:
        boundary = _make_sample_boundary_gdf()
        roads_no_optional = _make_sample_roads_gdf(include_optional_tags=False)

        clipped, _ = extract_roads.clip_and_prepare_roads(roads_no_optional, boundary)

        for col in extract_roads.PRESERVED_COLUMNS:
            self.assertIn(col, clipped.columns)
        self.assertTrue(clipped["opening_hours"].isna().all())
        self.assertTrue(clipped["lit"].isna().all())
        self.assertTrue(clipped["surface"].isna().all())
        self.assertTrue(clipped["sidewalk"].isna().all())

    def test_extract_linear_geometry_handles_geometry_collection_and_empty(self) -> None:
        gc = GeometryCollection(
            [Point(88.34, 22.54), LineString([(88.35, 22.55), (88.36, 22.55)])]
        )
        extracted = extract_roads._extract_linear_geometry(gc)
        self.assertIsInstance(extracted, LineString)

        self.assertIsNone(extract_roads._extract_linear_geometry(Point(88.34, 22.54)))
        self.assertIsNone(extract_roads._extract_linear_geometry(LineString()))
        self.assertIsNone(extract_roads._extract_linear_geometry(None))

    def test_no_roads_inside_boundary_raises(self) -> None:
        boundary = _make_sample_boundary_gdf()
        outside_only = gpd.GeoDataFrame(
            {"id": [99], "highway": ["residential"]},
            geometry=[LineString([(89.0, 23.0), (89.1, 23.1)])],
            crs="EPSG:4326",
        )
        with self.assertRaises(ValueError):
            extract_roads.clip_and_prepare_roads(outside_only, boundary)

    def test_extract_pedestrian_roads_with_mocked_pyrosm_empty_raises(self) -> None:
        boundary = _make_sample_boundary_gdf()
        mock_osm = MagicMock()
        mock_osm.get_network.return_value = gpd.GeoDataFrame(
            {"highway": []}, geometry=[], crs="EPSG:4326"
        )
        with patch("geospatial.osm.extract_roads.pyrosm.OSM", return_value=mock_osm):
            with self.assertRaises(ValueError):
                extract_roads.extract_pedestrian_roads(Path("dummy.osm.pbf"), boundary)


class TestExportAndValidation(unittest.TestCase):
    """Tests for GeoJSON export, CRS validation, and geometry checks."""

    def test_export_and_validate_success(self) -> None:
        boundary = _make_sample_boundary_gdf()
        roads = _make_sample_roads_gdf(include_optional_tags=False)
        clipped, _ = extract_roads.clip_and_prepare_roads(roads, boundary)

        with tempfile.TemporaryDirectory() as tmp:
            out_geojson = Path(tmp) / "roads.geojson"
            validation = extract_roads.export_and_validate(clipped, boundary, out_geojson)

            self.assertTrue(out_geojson.is_file())
            self.assertTrue(validation["verified_on_disk"])
            self.assertEqual(validation["crs"], "EPSG:4326")
            self.assertEqual(validation["feature_count"], 2)
            self.assertEqual(validation["invalid_geometry_count"], 0)
            self.assertEqual(validation["empty_geometry_count"], 0)
            self.assertTrue(validation["within_boundary_bbox"])
            self.assertEqual(validation["required_tag_non_null_counts"]["opening_hours"], 0)
            self.assertEqual(validation["required_tag_non_null_counts"]["highway"], 2)

    def test_export_and_validate_rejects_empty_geometry(self) -> None:
        boundary = _make_sample_boundary_gdf()
        bad_roads = gpd.GeoDataFrame(
            {"highway": ["residential"], "clipped_length_m": [0.0]},
            geometry=[LineString()],
            crs="EPSG:4326",
        )
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                extract_roads.export_and_validate(
                    bad_roads, boundary, Path(tmp) / "bad.geojson"
                )

    def test_export_and_validate_rejects_non_line_geometry(self) -> None:
        boundary = _make_sample_boundary_gdf()
        poly_roads = gpd.GeoDataFrame(
            {"highway": ["residential"], "clipped_length_m": [10.0]},
            geometry=[Point(88.35, 22.55)],
            crs="EPSG:4326",
        )
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                extract_roads.export_and_validate(
                    poly_roads, boundary, Path(tmp) / "point.geojson"
                )

    def test_export_and_validate_rejects_out_of_bounds_road(self) -> None:
        boundary = _make_sample_boundary_gdf()
        far_roads = gpd.GeoDataFrame(
            {"highway": ["residential"], "clipped_length_m": [100.0]},
            geometry=[LineString([(88.50, 22.60), (88.52, 22.61)])],
            crs="EPSG:4326",
        )
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                extract_roads.export_and_validate(
                    far_roads, boundary, Path(tmp) / "far.geojson"
                )

    def test_end_to_end_main_with_mocked_pyrosm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            boundary_path = tmp_path / "boundary.geojson"
            pbf_path = tmp_path / "mini.osm.pbf"
            output_path = tmp_path / "out" / "roads.geojson"
            metadata_path = tmp_path / "out" / "roads.metadata.json"

            _make_sample_boundary_gdf().to_file(boundary_path, driver="GeoJSON")
            _write_synthetic_pbf_header(pbf_path, bbox=(88.0, 22.0, 89.0, 23.0))

            mock_osm = MagicMock()
            mock_osm.get_network.return_value = _make_sample_roads_gdf(
                include_optional_tags=False
            )

            argv = [
                "extract_roads.py",
                "--pbf",
                str(pbf_path),
                "--boundary",
                str(boundary_path),
                "--output",
                str(output_path),
                "--metadata-output",
                str(metadata_path),
            ]
            with (
                patch("geospatial.osm.extract_roads.pyrosm.OSM", return_value=mock_osm),
                patch.object(sys, "argv", argv),
            ):
                extract_roads.main()

            self.assertTrue(output_path.is_file())
            self.assertTrue(metadata_path.is_file())
            meta = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(meta["crs"], "EPSG:4326")
            self.assertEqual(meta["feature_count"], 2)
            self.assertIn("OpenStreetMap contributors", meta["attribution"])
            self.assertIn("extraction_date_utc", meta)


if __name__ == "__main__":
    unittest.main()
