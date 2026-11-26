"""Shared configuration, boundary loading, and Earthdata auth helpers for VNP46A3."""

from __future__ import annotations

import hashlib
import netrc
import os
from pathlib import Path

import geopandas as gpd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BOUNDARY_PATH = (
    REPO_ROOT / "data" / "geospatial" / "boundaries" / "kolkata_district_boundary.geojson"
)
DEFAULT_RAW_DIR = Path(r"C:\Users\amisa\Downloads\PathPilot_DATA\VIIRS\VNP46A3\2025")
DEFAULT_MONTHLY_PROCESSED_DIR = Path(
    r"C:\Users\amisa\Downloads\PathPilot_DATA\Processed\VIIRS\2025\monthly"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "geospatial" / "viirs"

PRODUCT = "VNP46A3"
COLLECTION = "002"
COLLECTION_ARCHIVE_ID = "5200"
TILE = "h26v06"
YEAR = 2025
TARGET_CRS = "EPSG:32645"
SOURCE_BASE_URL = "https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/5200/VNP46A3"

# 2025 is a non-leap year; VNP46A3 monthly composites are indexed by the first DOY of each month.
MONTH_DOYS_2025: dict[int, str] = {
    1: "001",
    2: "032",
    3: "060",
    4: "091",
    5: "121",
    6: "152",
    7: "182",
    8: "213",
    9: "244",
    10: "274",
    11: "305",
    12: "335",
}


def file_sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of a file for provenance tracking."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_and_validate_boundary(boundary_path: Path) -> gpd.GeoDataFrame:
    """Load the Kolkata District boundary GeoJSON and verify CRS and geometry validity."""
    if not boundary_path.is_file():
        raise FileNotFoundError(f"Boundary file not found: {boundary_path}")

    boundary = gpd.read_file(boundary_path)
    if boundary.empty:
        raise ValueError(f"Boundary file contains no features: {boundary_path}")

    if boundary.crs is None or boundary.crs.to_epsg() != 4326:
        raise ValueError(f"Boundary CRS must be EPSG:4326, found: {boundary.crs}")

    valid_types = {"Polygon", "MultiPolygon"}
    found_types = set(boundary.geometry.geom_type.unique())
    if not found_types.issubset(valid_types):
        raise ValueError(
            f"Boundary must contain Polygon/MultiPolygon geometries, found: {sorted(found_types)}"
        )

    if boundary.geometry.is_empty.any() or not boundary.geometry.is_valid.all():
        raise ValueError("Boundary contains empty or invalid geometries.")

    return boundary


def get_earthdata_token() -> str | None:
    """Read NASA Earthdata / LAADS Bearer token from standard environment variables."""
    for var_name in ("EARTHDATA_TOKEN", "LAADS_TOKEN", "NASA_EARTHDATA_TOKEN"):
        value = os.environ.get(var_name, "").strip()
        if value:
            return value
    return None


def get_earthdata_netrc_credentials() -> tuple[str, str] | None:
    """Read NASA Earthdata Login credentials from ~/.netrc or ~/_netrc if present."""
    for filename in (".netrc", "_netrc"):
        candidate = Path.home() / filename
        if candidate.is_file():
            parsed = netrc.netrc(candidate)
            for host in ("urs.earthdata.nasa.gov", "ladsweb.modaps.eosdis.nasa.gov"):
                auth = parsed.authenticators(host)
                if auth is not None:
                    login, _, password = auth
                    if login and password:
                        return (login, password)
    return None
