"""Unit tests for Kolkata District OSM POI and amenity extraction."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import geopandas as gpd
from shapely.geometry import GeometryCollection, LineString, Point, Polygon

from geospatial.osm import extract_pois
from tests.geospatial.test_extract_roads import (
    _make_sample_boundary_gdf,
    _write_synthetic_pbf_header,
)


def _make_sample_pois_gdf(include_optional_cols: bool = True) -> gpd.GeoDataFrame:
    """Create synthetic POIs covering all 8 categories, polygon clipping, and duplicates."""
    rows = [
        # 1. Police (Point inside)
        {
            "id": 101,
            "osm_type": "node",
            "name": "Park Street PS",
            "amenity": "police",
            "shop": None,
            "highway": None,
            "railway": None,
            "station": None,
            "subway": None,
            "public_transport": None,
            "opening_hours": "24/7",
            "operator": "Kolkata Police",
            "geometry": Point(88.35, 22.55),
        },
        # 2. Hospital (Polygon crossing eastern border 88.38 -> clipped polygon)
        {
            "id": 102,
            "osm_type": "way",
            "name": "City Hospital",
            "amenity": "hospital",
            "shop": None,
            "highway": None,
            "railway": None,
            "station": None,
            "subway": None,
            "public_transport": None,
            "opening_hours": "24/7",
            "operator": None,
            "geometry": Polygon(
                [
                    (88.375, 22.55),
                    (88.385, 22.55),
                    (88.385, 22.56),
                    (88.375, 22.56),
                    (88.375, 22.55),
                ]
            ),
        },
        # 3. Clinic (missing name -> must STILL be kept)
        {
            "id": 103,
            "osm_type": "node",
            "name": None,
            "amenity": "clinic",
            "shop": None,
            "highway": None,
            "railway": None,
            "station": None,
            "subway": None,
            "public_transport": None,
            "opening_hours": None,
            "operator": None,
            "geometry": Point(88.36, 22.55),
        },
        # 4. Pharmacy
        {
            "id": 104,
            "osm_type": "node",
            "name": "Care Pharmacy",
            "amenity": "pharmacy",
            "shop": None,
            "highway": None,
            "railway": None,
            "station": None,
            "subway": None,
            "public_transport": None,
            "opening_hours": "08:00-22:00",
            "operator": None,
            "geometry": Point(88.36, 22.56),
        },
        # 5. Restaurant
        {
            "id": 105,
            "osm_type": "node",
            "name": "Bengal Spice",
            "amenity": "restaurant",
            "shop": None,
            "highway": None,
            "railway": None,
            "station": None,
            "subway": None,
            "public_transport": None,
            "opening_hours": "11:00-23:00",
            "operator": None,
            "geometry": Point(88.355, 22.565),
        },
        # 6. Shop
        {
            "id": 106,
            "osm_type": "node",
            "name": "Corner Bakery",
            "amenity": "cafe",
            "shop": "bakery",
            "highway": None,
            "railway": None,
            "station": None,
            "subway": None,
            "public_transport": None,
            "opening_hours": None,
            "operator": None,
            "geometry": Point(88.365, 22.565),
        },
        # 7. Bus stop (unnamed)
        {
            "id": 107,
            "osm_type": "node",
            "name": None,
            "amenity": None,
            "shop": None,
            "highway": "bus_stop",
            "railway": None,
            "station": None,
            "subway": None,
            "public_transport": None,
            "opening_hours": None,
            "operator": "WBTC",
            "geometry": Point(88.352, 22.552),
        },
        # 8. Metro station
        {
            "id": 108,
            "osm_type": "node",
            "name": "Park Street Metro",
            "amenity": None,
            "shop": None,
            "highway": None,
            "railway": "station",
            "station": "subway",
            "subway": "yes",
            "public_transport": "station",
            "opening_hours": None,
            "operator": "Metro Railway, Kolkata",
            "geometry": Point(88.351, 22.554),
        },
        # 9. Duplicate OSM (osm_type, id) of Police -> must be removed
        {
            "id": 101,
            "osm_type": "node",
            "name": "Park Street PS",
            "amenity": "police",
            "shop": None,
            "highway": None,
            "railway": None,
            "station": None,
            "subway": None,
            "public_transport": None,
            "opening_hours": "24/7",
            "operator": "Kolkata Police",
            "geometry": Point(88.35, 22.55),
        },
        # 10. Non-target mainline railway station (no subway tag) -> must be excluded
        {
            "id": 109,
            "osm_type": "node",
            "name": "Suburban Rail",
            "amenity": None,
            "shop": None,
            "highway": None,
            "railway": "station",
            "station": None,
            "subway": None,
            "public_transport": None,
            "opening_hours": None,
            "operator": "ER",
            "geometry": Point(88.37, 22.57),
        },
        # 11. Outside boundary -> must be dropped by clip
        {
            "id": 110,
            "osm_type": "node",
            "name": "Outside Shop",
            "amenity": None,
            "shop": "supermarket",
            "highway": None,
            "railway": None,
            "station": None,
            "subway": None,
            "public_transport": None,
            "opening_hours": None,
            "operator": None,
            "geometry": Point(88.50, 22.60),
        },
    ]
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    if not include_optional_cols:
        gdf = gdf.drop(columns=["opening_hours", "operator"])
    return gdf


class TestPoiClassificationAndGeometry(unittest.TestCase):
    """Tests for POI category rules, clipping, deduplication, and missing attributes."""

    def test_classify_all_eight_categories_and_exclusions(self) -> None:
        gdf = _make_sample_pois_gdf()
        categories = [extract_pois.classify_poi_category(row) for _, row in gdf.iterrows()]
        self.assertEqual(
            categories[:8],
            [
                "police",
                "hospital",
                "clinic",
                "pharmacy",
                "restaurant",
                "shop",
                "bus_stop",
                "metro_station",
            ],
        )
        # Mainline railway station without subway tag is excluded
        self.assertIsNone(categories[9])

    def test_filter_clip_and_dedupe_keeps_point_and_clipped_polygon_and_missing_names(self) -> None:
        boundary = _make_sample_boundary_gdf()
        raw_pois = _make_sample_pois_gdf()

        clipped, stats = extract_pois.filter_clip_and_dedupe_pois(raw_pois, boundary)

        self.assertEqual(stats["raw_pyrosm_rows"], 11)
        self.assertEqual(stats["excluded_non_target_rows"], 1)
        self.assertEqual(stats["duplicates_removed_by_osm_id"], 1)
        self.assertEqual(stats["post_clip_feature_count"], 8)
        self.assertEqual(clipped.crs.to_epsg(), 4326)

        # Clipped hospital polygon must not extend past eastern boundary 88.38
        hospital_geom = clipped[clipped["poi_category"] == "hospital"].geometry.iloc[0]
        self.assertEqual(hospital_geom.geom_type, "Polygon")
        self.assertLessEqual(hospital_geom.bounds[2], 88.38 + 1e-9)

        # Missing names (clinic & bus_stop) are preserved rather than dropped
        self.assertEqual(int(clipped["name"].isna().sum()), 2)

    def test_missing_optional_columns_preserved_as_null(self) -> None:
        boundary = _make_sample_boundary_gdf()
        raw_pois = _make_sample_pois_gdf(include_optional_cols=False)

        clipped, _ = extract_pois.filter_clip_and_dedupe_pois(raw_pois, boundary)

        for col in extract_pois.PRESERVED_POI_COLUMNS:
            self.assertIn(col, clipped.columns)
        self.assertTrue(clipped["opening_hours"].isna().all())
        self.assertTrue(clipped["operator"].isna().all())

    def test_normalize_poi_geometry_handles_collection_and_drops_lines(self) -> None:
        poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 0)])
        gc = GeometryCollection([LineString([(0, 0), (1, 1)]), poly])
        self.assertEqual(extract_pois._normalize_poi_geometry(gc).geom_type, "Polygon")
        self.assertIsNone(extract_pois._normalize_poi_geometry(LineString([(0, 0), (1, 1)])))
        self.assertIsNone(extract_pois._normalize_poi_geometry(Point()))


class TestPoiExportAndEndToEnd(unittest.TestCase):
    """Tests for GeoJSON export validation and mocked end-to-end pipeline."""

    def test_export_and_validate_pois_success(self) -> None:
        boundary = _make_sample_boundary_gdf()
        clipped, _ = extract_pois.filter_clip_and_dedupe_pois(
            _make_sample_pois_gdf(), boundary
        )

        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "pois.geojson"
            validation = extract_pois.export_and_validate_pois(clipped, boundary, out_path)

            self.assertTrue(out_path.is_file())
            self.assertEqual(validation["crs"], "EPSG:4326")
            self.assertEqual(validation["feature_count"], 8)
            self.assertEqual(validation["invalid_geometry_count"], 0)
            self.assertEqual(validation["empty_geometry_count"], 0)
            self.assertEqual(validation["geometry_types"], {"Point": 7, "Polygon": 1})
            for cat in extract_pois.POI_CATEGORIES:
                self.assertEqual(validation["category_counts"][cat], 1)

    def test_export_rejects_wrong_crs_and_empty_geom(self) -> None:
        boundary = _make_sample_boundary_gdf()
        bad_crs = gpd.GeoDataFrame(
            {"poi_category": ["police"]},
            geometry=[Point(88.35, 22.55)],
            crs="EPSG:32645",
        )
        empty_geom = gpd.GeoDataFrame(
            {"poi_category": ["police"]},
            geometry=[Point()],
            crs="EPSG:4326",
        )
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                extract_pois.export_and_validate_pois(
                    bad_crs, boundary, Path(tmp) / "bad_crs.geojson"
                )
            with self.assertRaises(ValueError):
                extract_pois.export_and_validate_pois(
                    empty_geom, boundary, Path(tmp) / "empty.geojson"
                )

    def test_end_to_end_main_with_mocked_pyrosm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            boundary_path = tmp_path / "boundary.geojson"
            pbf_path = tmp_path / "mini.osm.pbf"
            output_path = tmp_path / "out" / "pois.geojson"
            metadata_path = tmp_path / "out" / "pois.metadata.json"

            _make_sample_boundary_gdf().to_file(boundary_path, driver="GeoJSON")
            _write_synthetic_pbf_header(pbf_path, bbox=(88.0, 22.0, 89.0, 23.0))

            mock_osm = MagicMock()
            mock_osm.get_pois.return_value = _make_sample_pois_gdf()

            argv = [
                "extract_pois.py",
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
                patch("geospatial.osm.extract_pois.pyrosm.OSM", return_value=mock_osm),
                patch.object(sys, "argv", argv),
            ):
                extract_pois.main()

            self.assertTrue(output_path.is_file())
            self.assertTrue(metadata_path.is_file())
            meta = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(meta["crs"], "EPSG:4326")
            self.assertEqual(meta["feature_count"], 8)
            self.assertIn("OpenStreetMap contributors", meta["attribution"])
            self.assertEqual(sum(meta["category_counts"].values()), 8)


if __name__ == "__main__":
    unittest.main()
