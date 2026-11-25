"""Shared configuration, boundary validation, and PBF header helpers for OSM extraction."""

from __future__ import annotations

import datetime as dt
import hashlib
import struct
import zlib
from pathlib import Path
from typing import Any

import geopandas as gpd
from pyrosm.proto import fileformat_pb2, osmformat_pb2

DEFAULT_PBF_PATH = Path(
    r"C:\Users\amisa\Downloads\PathPilot_DATA\OSM\eastern-zone-261124.osm.pbf"
)
DEFAULT_BOUNDARY_PATH = Path(
    r"C:\Users\amisa\Downloads\PathPilot_Data\Boundaries\Kolkata_District\kolkata_district_boundary.geojson"
)

# Standard metric CRS for Kolkata (WGS 84 / UTM Zone 45N) for metre-based checks.
# All exported GeoJSON outputs stay in EPSG:4326 per PathPilot API conventions.
KOLKATA_METRIC_CRS = "EPSG:32645"
TARGET_CRS = "EPSG:4326"

OSM_ATTRIBUTION = (
    "Map data © OpenStreetMap contributors, available under the Open Database "
    "License (ODbL). Administrative boundary: Survey of India (SOI)."
)


def file_sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of a file for provenance tracking."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_and_validate_boundary(boundary_path: Path) -> gpd.GeoDataFrame:
    """Load the district boundary GeoJSON and verify CRS and geometry validity."""
    if not boundary_path.is_file():
        raise FileNotFoundError(f"Boundary file not found: {boundary_path}")

    boundary = gpd.read_file(boundary_path)
    if boundary.empty:
        raise ValueError(f"Boundary file contains no features: {boundary_path}")

    if boundary.crs is None or boundary.crs.to_epsg() != 4326:
        raise ValueError(
            f"Boundary CRS must be EPSG:4326, found: {boundary.crs}"
        )

    valid_types = {"Polygon", "MultiPolygon"}
    found_types = set(boundary.geometry.geom_type.unique())
    if not found_types.issubset(valid_types):
        raise ValueError(
            f"Boundary must contain Polygon/MultiPolygon geometries, found: {sorted(found_types)}"
        )

    if boundary.geometry.is_empty.any() or not boundary.geometry.is_valid.all():
        raise ValueError("Boundary contains empty or invalid geometries.")

    return boundary


def read_pbf_header(pbf_path: Path) -> dict[str, Any]:
    """Read the first OSMHeader block from a .osm.pbf file to inspect extent & timestamp."""
    if not pbf_path.is_file():
        raise FileNotFoundError(f"OSM PBF file not found: {pbf_path}")

    with pbf_path.open("rb") as handle:
        raw_size = handle.read(4)
        if len(raw_size) != 4:
            raise ValueError(f"Invalid PBF file (truncated header size): {pbf_path}")

        header_size = struct.unpack(">I", raw_size)[0]
        blob_header = fileformat_pb2.BlobHeader()
        blob_header.ParseFromString(handle.read(header_size))

        if blob_header.type != "OSMHeader":
            raise ValueError(
                f"Expected OSMHeader as first PBF blob, found: {blob_header.type}"
            )

        blob = fileformat_pb2.Blob()
        blob.ParseFromString(handle.read(blob_header.datasize))

    if blob.HasField("zlib_data"):
        raw_header = zlib.decompress(blob.zlib_data)
    elif blob.HasField("raw"):
        raw_header = blob.raw
    else:
        raise ValueError("Unsupported PBF header compression format.")

    header_block = osmformat_pb2.HeaderBlock()
    header_block.ParseFromString(raw_header)

    bbox = None
    if header_block.HasField("bbox"):
        # OSM PBF HeaderBBox stores coordinates in nanodegrees (1e-9 degrees).
        bbox = [
            header_block.bbox.left / 1e9,
            header_block.bbox.bottom / 1e9,
            header_block.bbox.right / 1e9,
            header_block.bbox.top / 1e9,
        ]

    repl_ts = None
    if header_block.osmosis_replication_timestamp:
        repl_ts = dt.datetime.fromtimestamp(
            header_block.osmosis_replication_timestamp, tz=dt.timezone.utc
        ).isoformat()

    return {
        "bbox": bbox,
        "osmosis_replication_timestamp_utc": repl_ts,
        "writingprogram": header_block.writingprogram or None,
        "source": header_block.source or None,
    }


def verify_pbf_covers_boundary(
    pbf_header: dict[str, Any], boundary: gpd.GeoDataFrame
) -> dict[str, Any]:
    """Check that the PBF extract bounding box fully covers the study boundary."""
    pbf_bbox = pbf_header.get("bbox")
    if not pbf_bbox:
        raise ValueError("PBF header does not contain a bounding box.")

    b_minx, b_miny, b_maxx, b_maxy = [float(v) for v in boundary.total_bounds]
    p_minx, p_miny, p_maxx, p_maxy = [float(v) for v in pbf_bbox]

    covered = (
        p_minx <= b_minx
        and p_miny <= b_miny
        and p_maxx >= b_maxx
        and p_maxy >= b_maxy
    )
    if not covered:
        raise ValueError(
            f"PBF bbox {pbf_bbox} does not cover boundary bbox {[b_minx, b_miny, b_maxx, b_maxy]}"
        )

    return {
        "covered": True,
        "pbf_bbox_epsg4326": pbf_bbox,
        "boundary_bbox_epsg4326": [b_minx, b_miny, b_maxx, b_maxy],
    }
