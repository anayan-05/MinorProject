"""Extract OSM POIs and essential amenities clipped to the Kolkata District boundary.

Extracts point and polygon features for eight PathPilot safety/amenity categories:
- police: amenity=police
- hospital: amenity=hospital
- clinic: amenity=clinic
- pharmacy: amenity=pharmacy
- restaurant: amenity=restaurant
- shop: shop=*
- bus_stop: highway=bus_stop
- metro_station: railway=station (or public_transport=station) with subway/metro tags

Outputs a single EPSG:4326 GeoJSON and extraction metadata JSON outside Git.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import pyrosm
from shapely.geometry import (
    GeometryCollection,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)
from shapely.geometry.base import BaseGeometry

from geospatial.osm.common import (
    DEFAULT_BOUNDARY_PATH,
    DEFAULT_PBF_PATH,
    OSM_ATTRIBUTION,
    TARGET_CRS,
    file_sha256,
    load_and_validate_boundary,
    read_pbf_header,
    verify_pbf_covers_boundary,
)

DEFAULT_POI_OUTPUT_PATH = Path(
    r"C:\Users\amisa\Downloads\PathPilot_DATA\Processed\OSM\kolkata_district_pois.geojson"
)
DEFAULT_POI_METADATA_PATH = Path(
    r"C:\Users\amisa\Downloads\PathPilot_DATA\Processed\OSM\kolkata_district_pois.metadata.json"
)

POI_CATEGORIES = [
    "police",
    "hospital",
    "clinic",
    "pharmacy",
    "restaurant",
    "shop",
    "bus_stop",
    "metro_station",
]

POI_CUSTOM_FILTER: dict[str, Any] = {
    "amenity": ["police", "hospital", "clinic", "pharmacy", "restaurant"],
    "shop": True,
    "highway": ["bus_stop"],
    "railway": ["station"],
    "station": ["subway"],
    "subway": ["yes"],
    "public_transport": ["station"],
}

POI_EXTRA_ATTRIBUTES = [
    "highway",
    "railway",
    "station",
    "subway",
    "public_transport",
    "opening_hours",
    "operator",
    "network",
]

PRESERVED_POI_COLUMNS = [
    "id",
    "osm_type",
    "poi_category",
    "name",
    "amenity",
    "shop",
    "highway",
    "railway",
    "station",
    "subway",
    "public_transport",
    "opening_hours",
    "operator",
    "addr:street",
    "addr:postcode",
    "phone",
    "website",
    "timestamp",
    "version",
    "tags",
    "geometry",
]

REQUIRED_POI_ATTRIBUTE_COLUMNS = [
    "name",
    "amenity",
    "shop",
    "highway",
    "railway",
    "opening_hours",
    "operator",
    "id",
    "osm_type",
]


def _clean_str(value: Any) -> str | None:
    """Normalize missing or blank string values to None."""
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text if text else None


def classify_poi_category(row: pd.Series) -> str | None:
    """Assign a single primary PathPilot POI category, or None if out of scope."""
    amenity = (_clean_str(row.get("amenity")) or "").lower()
    shop = _clean_str(row.get("shop"))
    highway = (_clean_str(row.get("highway")) or "").lower()
    railway = (_clean_str(row.get("railway")) or "").lower()
    station = (_clean_str(row.get("station")) or "").lower()
    subway = (_clean_str(row.get("subway")) or "").lower()
    public_transport = (_clean_str(row.get("public_transport")) or "").lower()
    operator = (_clean_str(row.get("operator")) or "").lower()
    network = (_clean_str(row.get("network")) or "").lower()

    if amenity in {"police", "hospital", "clinic", "pharmacy"}:
        return amenity

    is_subway_tagged = (
        station == "subway"
        or subway == "yes"
        or "metro" in operator
        or "metro" in network
    )
    if (railway == "station" or public_transport == "station") and is_subway_tagged:
        return "metro_station"

    if highway == "bus_stop":
        return "bus_stop"

    if amenity == "restaurant":
        return "restaurant"

    if shop is not None and shop.lower() != "no":
        return "shop"

    return None


def _normalize_poi_geometry(geom: BaseGeometry | None) -> BaseGeometry | None:
    """Keep valid Point/MultiPoint/Polygon/MultiPolygon geometries after clipping."""
    if geom is None or geom.is_empty:
        return None
    if isinstance(geom, (Point, MultiPoint, Polygon, MultiPolygon)):
        return geom
    if isinstance(geom, GeometryCollection):
        polygons: list[Polygon] = []
        points: list[Point] = []
        for part in geom.geoms:
            if isinstance(part, Polygon) and not part.is_empty:
                polygons.append(part)
            elif isinstance(part, MultiPolygon) and not part.is_empty:
                polygons.extend(p for p in part.geoms if not p.is_empty)
            elif isinstance(part, Point) and not part.is_empty:
                points.append(part)
            elif isinstance(part, MultiPoint) and not part.is_empty:
                points.extend(pt for pt in part.geoms if not pt.is_empty)
        if len(polygons) == 1:
            return polygons[0]
        if len(polygons) > 1:
            return MultiPolygon(polygons)
        if len(points) == 1:
            return points[0]
        if len(points) > 1:
            return MultiPoint(points)
    return None


def extract_raw_pois(
    pbf_path: Path, boundary: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Extract target POI nodes/ways/polygons within the Kolkata bounding box."""
    boundary_geom = boundary.geometry.union_all()
    osm = pyrosm.OSM(str(pbf_path), bounding_box=boundary_geom)

    pois = osm.get_pois(
        custom_filter=POI_CUSTOM_FILTER,
        extra_attributes=POI_EXTRA_ATTRIBUTES,
    )

    if pois is None or pois.empty:
        raise ValueError("Pyrosm returned 0 POI features inside the bounding box.")

    if pois.crs is None:
        pois = pois.set_crs(TARGET_CRS)
    elif pois.crs.to_epsg() != 4326:
        pois = pois.to_crs(TARGET_CRS)

    return pois


def filter_clip_and_dedupe_pois(
    pois: gpd.GeoDataFrame, boundary: gpd.GeoDataFrame
) -> tuple[gpd.GeoDataFrame, dict[str, int]]:
    """Classify into target categories, deduplicate, and clip to Kolkata District."""
    raw_count = len(pois)
    df = pois.copy()

    # Ensure every expected attribute column exists so classification & export never KeyError.
    for col in PRESERVED_POI_COLUMNS + ["network"]:
        if col not in df.columns:
            df[col] = None

    df["poi_category"] = df.apply(classify_poi_category, axis=1)
    df = df[df["poi_category"].notna()].copy()
    matched_category_count = len(df)

    if df.empty:
        raise ValueError("No features matched the requested 8 POI categories.")

    # Deduplicate by OSM element identity (osm_type, id) when available.
    pre_id_dedupe = len(df)
    has_osm_key = df["osm_type"].notna() & df["id"].notna()
    with_key = df[has_osm_key].drop_duplicates(subset=["osm_type", "id"], keep="first")
    without_key = df[~has_osm_key]
    df = pd.concat([with_key, without_key], ignore_index=True)
    df = gpd.GeoDataFrame(df, geometry="geometry", crs=TARGET_CRS)
    removed_osm_id_dupes = pre_id_dedupe - len(df)

    # Clip to exact Kolkata District MultiPolygon.
    clipped = gpd.clip(df, boundary).copy()
    clipped["geometry"] = clipped.geometry.apply(_normalize_poi_geometry)
    dropped_non_poi_artifacts = int(clipped["geometry"].isna().sum())
    clipped = clipped[clipped["geometry"].notna()].copy()
    clipped = clipped[~clipped.geometry.is_empty].copy()

    if clipped.empty:
        raise ValueError("No POI features remained after clipping to the district boundary.")

    # Remove exact duplicate (poi_category, normalized_name, geometry) rows if any exist.
    pre_geom_dedupe = len(clipped)
    clipped["_norm_name"] = clipped["name"].apply(
        lambda v: (_clean_str(v) or "").lower()
    )
    clipped["_geom_wkb"] = clipped.geometry.to_wkb()
    clipped = clipped.drop_duplicates(
        subset=["poi_category", "_norm_name", "_geom_wkb"], keep="first"
    ).copy()
    removed_geom_dupes = pre_geom_dedupe - len(clipped)

    clipped = clipped[PRESERVED_POI_COLUMNS].reset_index(drop=True)

    for int_col in ("id", "timestamp", "version"):
        if int_col in clipped.columns:
            clipped[int_col] = pd.to_numeric(clipped[int_col], errors="coerce").astype("Int64")

    clipped = gpd.GeoDataFrame(clipped, geometry="geometry", crs=TARGET_CRS)

    stats = {
        "raw_pyrosm_rows": raw_count,
        "matched_target_category_rows": matched_category_count,
        "excluded_non_target_rows": raw_count - matched_category_count,
        "duplicates_removed_by_osm_id": int(removed_osm_id_dupes),
        "duplicates_removed_by_exact_geometry": int(removed_geom_dupes),
        "dropped_clip_artifacts": dropped_non_poi_artifacts,
        "post_clip_feature_count": int(len(clipped)),
    }
    return clipped, stats


def export_and_validate_pois(
    pois: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame,
    output_path: Path,
) -> dict[str, Any]:
    """Export clipped POIs to GeoJSON and re-read from disk to validate integrity."""
    if pois.crs is None or pois.crs.to_epsg() != 4326:
        raise ValueError(f"POI GeoDataFrame CRS must be EPSG:4326, found: {pois.crs}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pois.to_file(output_path, driver="GeoJSON")

    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise IOError(f"Exported POI GeoJSON is missing or empty: {output_path}")

    verified = gpd.read_file(output_path)
    if verified.empty:
        raise ValueError("Re-read POI GeoJSON has 0 features.")

    if verified.crs is None or verified.crs.to_epsg() != 4326:
        raise ValueError(f"Output POI GeoJSON CRS must be EPSG:4326, found: {verified.crs}")

    geom_types = verified.geometry.geom_type.value_counts().to_dict()
    allowed_geom_types = {"Point", "MultiPoint", "Polygon", "MultiPolygon"}
    unexpected_types = set(geom_types.keys()) - allowed_geom_types
    if unexpected_types:
        raise ValueError(f"Unexpected POI geometry types: {sorted(unexpected_types)}")

    invalid_count = int((~verified.geometry.is_valid).sum())
    empty_count = int(verified.geometry.is_empty.sum())
    if invalid_count > 0 or empty_count > 0:
        raise ValueError(
            f"POI validation failed: invalid={invalid_count}, empty={empty_count}"
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
            f"POI output bounds {[o_minx, o_miny, o_maxx, o_maxy]} exceed boundary bounds {[b_minx, b_miny, b_maxx, b_maxy]}"
        )

    category_counts = {
        cat: int((verified["poi_category"] == cat).sum())
        for cat in POI_CATEGORIES
    }
    attribute_non_null_counts = {
        col: int(verified[col].notna().sum()) if col in verified.columns else 0
        for col in REQUIRED_POI_ATTRIBUTE_COLUMNS
    }

    return {
        "verified_on_disk": True,
        "output_size_bytes": output_path.stat().st_size,
        "output_sha256": file_sha256(output_path),
        "crs": f"EPSG:{verified.crs.to_epsg()}",
        "feature_count": int(len(verified)),
        "category_counts": category_counts,
        "geometry_types": {str(k): int(v) for k, v in geom_types.items()},
        "invalid_geometry_count": invalid_count,
        "empty_geometry_count": empty_count,
        "within_boundary_bbox": within_boundary_bbox,
        "output_bounds_epsg4326": [o_minx, o_miny, o_maxx, o_maxy],
        "attribute_non_null_counts": attribute_non_null_counts,
    }


def build_poi_metadata(
    pbf_path: Path,
    boundary_path: Path,
    output_path: Path,
    boundary: gpd.GeoDataFrame,
    pbf_header: dict[str, Any],
    coverage_check: dict[str, Any],
    processing_stats: dict[str, int],
    validation: dict[str, Any],
) -> dict[str, Any]:
    """Build provenance, category count, and validation metadata for POI outputs."""
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
        "dataset": "kolkata_district_pois",
        "study_area": f"{district_name} District, West Bengal, India",
        "extraction_date_utc": dt.datetime.now(tz=dt.timezone.utc).isoformat(),
        "crs": validation["crs"],
        "feature_count": validation["feature_count"],
        "category_counts": validation["category_counts"],
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
                "custom_filter": POI_CUSTOM_FILTER,
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
        "processing_stats": processing_stats,
        "validation": validation,
        "output_geojson": str(output_path),
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the Kolkata OSM POI extraction pipeline."""
    parser = argparse.ArgumentParser(
        description="Extract OSM POIs and essential amenities clipped to Kolkata District."
    )
    parser.add_argument("--pbf", type=Path, default=DEFAULT_PBF_PATH)
    parser.add_argument("--boundary", type=Path, default=DEFAULT_BOUNDARY_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_POI_OUTPUT_PATH)
    parser.add_argument("--metadata-output", type=Path, default=DEFAULT_POI_METADATA_PATH)
    return parser.parse_args()


def main() -> None:
    """Run the full OSM POI extraction, deduplication, clipping, and validation flow."""
    args = parse_args()

    print(f"[1/6] Loading and validating boundary: {args.boundary}")
    boundary = load_and_validate_boundary(args.boundary)

    print(f"[2/6] Reading PBF header and checking boundary coverage: {args.pbf}")
    pbf_header = read_pbf_header(args.pbf)
    coverage_check = verify_pbf_covers_boundary(pbf_header, boundary)
    print(f"      PBF bbox {pbf_header['bbox']} covers Kolkata District.")

    print("[3/6] Extracting target POI features via Pyrosm...")
    raw_pois = extract_raw_pois(args.pbf, boundary)
    print(f"      Pyrosm returned {len(raw_pois)} raw candidate features.")

    print("[4/6] Classifying, deduplicating, and clipping to Kolkata District...")
    clipped_pois, stats = filter_clip_and_dedupe_pois(raw_pois, boundary)
    print(
        f"      Kept {stats['post_clip_feature_count']} POIs "
        f"(excluded {stats['excluded_non_target_rows']} non-target rows, "
        f"removed {stats['duplicates_removed_by_osm_id']} ID dupes / "
        f"{stats['duplicates_removed_by_exact_geometry']} geometry dupes)."
    )

    print(f"[5/6] Exporting GeoJSON and validating output on disk: {args.output}")
    validation = export_and_validate_pois(clipped_pois, boundary, args.output)

    print(f"[6/6] Writing metadata JSON: {args.metadata_output}")
    metadata = build_poi_metadata(
        pbf_path=args.pbf,
        boundary_path=args.boundary,
        output_path=args.output,
        boundary=boundary,
        pbf_header=pbf_header,
        coverage_check=coverage_check,
        processing_stats=stats,
        validation=validation,
    )
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\n--- POI Extraction Summary ---")
    print(f"Output GeoJSON  : {args.output}")
    print(f"Metadata JSON   : {args.metadata_output}")
    print(f"CRS             : {validation['crs']}")
    print(f"Total features  : {validation['feature_count']}")
    print(f"Geometry types  : {validation['geometry_types']}")
    print(f"Category counts : {validation['category_counts']}")
    print(f"Attr non-null   : {validation['attribute_non_null_counts']}")


if __name__ == "__main__":
    main()
