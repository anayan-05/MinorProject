"""Extract pedestrian-accessible OSM roads clipped to the Kolkata District boundary.

This script implements the first geospatial dataset step for PathPilot:
1. Validate the Kolkata District boundary GeoJSON (must be EPSG:4326 and valid).
2. Read the Eastern Zone OSM PBF header and verify it covers Kolkata District.
3. Extract pedestrian-accessible roads and paths with Pyrosm (`network_type="walking"`),
   preserving key OSM safety/walkability tags (`highway`, `lit`, `surface`, `name`,
   `sidewalk`, `opening_hours`, etc.).
4. Clip the extracted road geometries to the exact Kolkata District MultiPolygon.
5. Validate output CRS, geometries, and feature counts, export to GeoJSON outside
   the Git repository, and write a concise extraction metadata JSON file.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import pyrosm
from shapely.geometry import GeometryCollection, LineString, MultiLineString
from shapely.geometry.base import BaseGeometry

from geospatial.osm.common import (
    DEFAULT_BOUNDARY_PATH,
    DEFAULT_PBF_PATH,
    KOLKATA_METRIC_CRS,
    OSM_ATTRIBUTION,
    TARGET_CRS,
    file_sha256,
    load_and_validate_boundary,
    read_pbf_header,
    verify_pbf_covers_boundary,
)

DEFAULT_OUTPUT_PATH = Path(
    r"C:\Users\amisa\Downloads\PathPilot_DATA\Processed\OSM\kolkata_district_pedestrian_roads.geojson"
)
DEFAULT_METADATA_PATH = Path(
    r"C:\Users\amisa\Downloads\PathPilot_DATA\Processed\OSM\kolkata_district_pedestrian_roads.metadata.json"
)

REQUIRED_TAG_COLUMNS = [
    "highway",
    "lit",
    "surface",
    "name",
    "sidewalk",
    "opening_hours",
]

PRESERVED_COLUMNS = [
    "id",
    "osm_type",
    "highway",
    "name",
    "lit",
    "surface",
    "sidewalk",
    "opening_hours",
    "foot",
    "footway",
    "access",
    "oneway",
    "bridge",
    "tunnel",
    "lanes",
    "width",
    "smoothness",
    "service",
    "segregated",
    "timestamp",
    "version",
    "tags",
    "clipped_length_m",
    "geometry",
]


def _extract_linear_geometry(geom: BaseGeometry | None) -> BaseGeometry | None:
    """Keep only linear geometries after clipping; return None for point/empty artifacts."""
    if geom is None or geom.is_empty:
        return None
    if isinstance(geom, (LineString, MultiLineString)):
        return geom
    if isinstance(geom, GeometryCollection):
        lines: list[LineString] = []
        for part in geom.geoms:
            if isinstance(part, LineString) and not part.is_empty:
                lines.append(part)
            elif isinstance(part, MultiLineString) and not part.is_empty:
                lines.extend(line for line in part.geoms if not line.is_empty)
        if len(lines) == 1:
            return lines[0]
        if len(lines) > 1:
            return MultiLineString(lines)
    return None


def extract_pedestrian_roads(
    pbf_path: Path, boundary: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Read pedestrian-accessible OSM roads within the boundary bounding box."""
    boundary_geom = boundary.geometry.union_all()
    osm = pyrosm.OSM(str(pbf_path), bounding_box=boundary_geom)

    # `network_type="walking"` follows OSMnx/OSM pedestrian access rules.
    # `extra_attributes` ensures `opening_hours` is surfaced as a column when present.
    roads = osm.get_network(
        network_type="walking",
        extra_attributes=["opening_hours"],
    )

    if roads is None or roads.empty:
        raise ValueError("Pyrosm returned 0 walking road features inside the bounding box.")

    if roads.crs is None:
        roads = roads.set_crs(TARGET_CRS)
    elif roads.crs.to_epsg() != 4326:
        roads = roads.to_crs(TARGET_CRS)

    return roads


def clip_and_prepare_roads(
    roads: gpd.GeoDataFrame, boundary: gpd.GeoDataFrame
) -> tuple[gpd.GeoDataFrame, dict[str, int]]:
    """Clip roads to exact district polygon, normalize columns, and compute lengths."""
    pre_clip_count = len(roads)
    clipped = gpd.clip(roads, boundary)

    # Clipping a line by a polygon can occasionally produce a Point where a road
    # only touches the border. Keep only actual linear road segments.
    clipped = clipped.copy()
    clipped["geometry"] = clipped.geometry.apply(_extract_linear_geometry)
    dropped_non_linear = int(clipped["geometry"].isna().sum())
    clipped = clipped[clipped["geometry"].notna()].copy()
    clipped = clipped[~clipped.geometry.is_empty].copy()

    if clipped.empty:
        raise ValueError("No linear road features remained after clipping to the district boundary.")

    # Compute clipped segment length in metres using Kolkata's UTM zone (EPSG:32645).
    metric_lengths = clipped.to_crs(KOLKATA_METRIC_CRS).geometry.length
    clipped["clipped_length_m"] = metric_lengths.round(2)

    # Ensure every expected column exists (preserving None/null when absent in OSM).
    for col in PRESERVED_COLUMNS:
        if col not in clipped.columns:
            clipped[col] = None

    clipped = clipped[PRESERVED_COLUMNS].reset_index(drop=True)

    # Convert timestamp (uint32 Unix epoch) to nullable integer / clean JSON-safe type.
    if "timestamp" in clipped.columns:
        clipped["timestamp"] = clipped["timestamp"].astype("Int64")
    if "version" in clipped.columns:
        clipped["version"] = clipped["version"].astype("Int64")
    if "id" in clipped.columns:
        clipped["id"] = clipped["id"].astype("Int64")

    clipped = gpd.GeoDataFrame(clipped, geometry="geometry", crs=TARGET_CRS)

    stats = {
        "pre_clip_feature_count": pre_clip_count,
        "dropped_non_linear_border_artifacts": dropped_non_linear,
        "post_clip_feature_count": len(clipped),
    }
    return clipped, stats


def export_and_validate(
    clipped: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame,
    output_path: Path,
) -> dict[str, Any]:
    """Export clipped roads to GeoJSON and re-read from disk to verify integrity."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clipped.to_file(output_path, driver="GeoJSON")

    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise IOError(f"Exported GeoJSON is missing or empty: {output_path}")

    verified = gpd.read_file(output_path)
    if verified.empty:
        raise ValueError("Re-read GeoJSON has 0 features.")

    if verified.crs is None or verified.crs.to_epsg() != 4326:
        raise ValueError(f"Output GeoJSON CRS must be EPSG:4326, found: {verified.crs}")

    geom_types = verified.geometry.geom_type.value_counts().to_dict()
    allowed_geom_types = {"LineString", "MultiLineString"}
    unexpected_types = set(geom_types.keys()) - allowed_geom_types
    if unexpected_types:
        raise ValueError(f"Unexpected output geometry types: {sorted(unexpected_types)}")

    invalid_count = int((~verified.geometry.is_valid).sum())
    empty_count = int(verified.geometry.is_empty.sum())
    if invalid_count > 0 or empty_count > 0:
        raise ValueError(
            f"Output validation failed: invalid={invalid_count}, empty={empty_count}"
        )

    b_minx, b_miny, b_maxx, b_maxy = [float(v) for v in boundary.total_bounds]
    o_minx, o_miny, o_maxx, o_maxy = [float(v) for v in verified.total_bounds]
    tol = 1e-6
    within_boundary_bbox = (
        o_minx >= b_minx - tol
        and o_miny >= b_miny - tol
        and o_maxx <= b_maxx + tol
        and o_maxy <= b_maxy + tol
    )
    if not within_boundary_bbox:
        raise ValueError(
            f"Output bounds {[o_minx, o_miny, o_maxx, o_maxy]} exceed boundary bounds {[b_minx, b_miny, b_maxx, b_maxy]}"
        )

    tag_non_null_counts = {
        tag: int(verified[tag].notna().sum()) if tag in verified.columns else 0
        for tag in REQUIRED_TAG_COLUMNS
    }

    return {
        "verified_on_disk": True,
        "output_size_bytes": output_path.stat().st_size,
        "output_sha256": file_sha256(output_path),
        "crs": f"EPSG:{verified.crs.to_epsg()}",
        "feature_count": int(len(verified)),
        "geometry_types": {str(k): int(v) for k, v in geom_types.items()},
        "invalid_geometry_count": invalid_count,
        "empty_geometry_count": empty_count,
        "within_boundary_bbox": within_boundary_bbox,
        "output_bounds_epsg4326": [o_minx, o_miny, o_maxx, o_maxy],
        "total_clipped_length_km": round(float(verified["clipped_length_m"].sum()) / 1000.0, 2),
        "required_tag_non_null_counts": tag_non_null_counts,
    }


def build_metadata(
    pbf_path: Path,
    boundary_path: Path,
    output_path: Path,
    boundary: gpd.GeoDataFrame,
    pbf_header: dict[str, Any],
    coverage_check: dict[str, Any],
    clip_stats: dict[str, int],
    validation: dict[str, Any],
) -> dict[str, Any]:
    """Build provenance and validation metadata for the extracted OSM road dataset."""
    src_agency = (
        str(boundary["src_agency"].iloc[0])
        if "src_agency" in boundary.columns
        else "Survey of India"
    )
    district_name = (
        str(boundary["district"].iloc[0])
        if "district" in boundary.columns
        else "Kolkata"
    )

    return {
        "dataset": "kolkata_district_pedestrian_roads",
        "study_area": f"{district_name} District, West Bengal, India",
        "extraction_date_utc": dt.datetime.now(tz=dt.timezone.utc).isoformat(),
        "crs": validation["crs"],
        "feature_count": validation["feature_count"],
        "sources": {
            "osm_pbf": {
                "path": str(pbf_path),
                "size_bytes": pbf_path.stat().st_size,
                "sha256": file_sha256(pbf_path),
                "osmosis_replication_timestamp_utc": pbf_header[
                    "osmosis_replication_timestamp_utc"
                ],
                "writingprogram": pbf_header["writingprogram"],
                "bbox_epsg4326": pbf_header["bbox"],
                "network_filter": "pyrosm walking (pedestrian-accessible roads and paths)",
            },
            "boundary": {
                "path": str(boundary_path),
                "source_agency": src_agency,
                "crs": f"EPSG:{boundary.crs.to_epsg()}",
                "sha256": file_sha256(boundary_path),
                "bbox_epsg4326": coverage_check["boundary_bbox_epsg4326"],
            },
        },
        "attribution": OSM_ATTRIBUTION,
        "pbf_coverage_check": coverage_check,
        "clipping_stats": clip_stats,
        "preserved_required_tags": REQUIRED_TAG_COLUMNS,
        "validation": validation,
        "output_geojson": str(output_path),
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the Kolkata OSM road extraction pipeline."""
    parser = argparse.ArgumentParser(
        description="Extract pedestrian-accessible OSM roads clipped to Kolkata District."
    )
    parser.add_argument("--pbf", type=Path, default=DEFAULT_PBF_PATH)
    parser.add_argument("--boundary", type=Path, default=DEFAULT_BOUNDARY_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--metadata-output", type=Path, default=DEFAULT_METADATA_PATH)
    return parser.parse_args()


def main() -> None:
    """Run the full OSM road extraction, clipping, validation, and export pipeline."""
    args = parse_args()

    print(f"[1/6] Loading and validating boundary: {args.boundary}")
    boundary = load_and_validate_boundary(args.boundary)

    print(f"[2/6] Reading PBF header and checking boundary coverage: {args.pbf}")
    pbf_header = read_pbf_header(args.pbf)
    coverage_check = verify_pbf_covers_boundary(pbf_header, boundary)
    print(f"      PBF bbox {pbf_header['bbox']} covers Kolkata District.")

    print("[3/6] Extracting pedestrian-accessible roads via Pyrosm...")
    roads = extract_pedestrian_roads(args.pbf, boundary)
    print(f"      Extracted {len(roads)} road features within bounding box.")

    print("[4/6] Clipping roads to exact Kolkata District MultiPolygon...")
    clipped, clip_stats = clip_and_prepare_roads(roads, boundary)
    print(
        f"      Kept {clip_stats['post_clip_feature_count']} linear features "
        f"(dropped {clip_stats['dropped_non_linear_border_artifacts']} border-point artifacts)."
    )

    print(f"[5/6] Exporting GeoJSON and validating output on disk: {args.output}")
    validation = export_and_validate(clipped, boundary, args.output)

    print(f"[6/6] Writing metadata JSON: {args.metadata_output}")
    metadata = build_metadata(
        pbf_path=args.pbf,
        boundary_path=args.boundary,
        output_path=args.output,
        boundary=boundary,
        pbf_header=pbf_header,
        coverage_check=coverage_check,
        clip_stats=clip_stats,
        validation=validation,
    )
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\n--- Extraction Summary ---")
    print(f"Output GeoJSON : {args.output}")
    print(f"Metadata JSON  : {args.metadata_output}")
    print(f"CRS            : {validation['crs']}")
    print(f"Feature count  : {validation['feature_count']}")
    print(f"Total length   : {validation['total_clipped_length_km']} km")
    print(f"Geometry types : {validation['geometry_types']}")
    print(f"Tag coverage   : {validation['required_tag_non_null_counts']}")


if __name__ == "__main__":
    main()
