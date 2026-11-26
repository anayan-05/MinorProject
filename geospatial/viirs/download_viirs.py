"""Download and inspect NASA VIIRS Black Marble VNP46A3 Collection 2.0 HDF5 files."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import h5py
import requests

from geospatial.viirs.common import (
    DEFAULT_RAW_DIR,
    MONTH_DOYS_2025,
    SOURCE_BASE_URL,
    TILE,
    YEAR,
    get_earthdata_netrc_credentials,
    get_earthdata_token,
)


def find_tile_filename(
    session: requests.Session,
    year: int,
    doy: str,
    tile: str = TILE,
    base_url: str = SOURCE_BASE_URL,
) -> tuple[str, str]:
    """Query LAADS DAAC directory JSON for a given year/DOY and return (filename, url)."""
    dir_json_url = f"{base_url}/{year}/{doy}.json"
    resp = session.get(dir_json_url, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(
            f"Failed to list LAADS DAAC directory {dir_json_url}: "
            f"HTTP {resp.status_code} {resp.reason}"
        )

    payload = resp.json()
    entries = payload.get("content", payload) if isinstance(payload, dict) else payload
    if not isinstance(entries, list):
        raise ValueError(f"Unexpected directory listing format from {dir_json_url}")

    matches = [
        entry["name"]
        for entry in entries
        if isinstance(entry, dict)
        and isinstance(entry.get("name"), str)
        and f".{tile}." in entry["name"]
        and entry["name"].endswith(".h5")
    ]
    if not matches:
        raise FileNotFoundError(
            f"No VNP46A3 .h5 file found for tile {tile} in {dir_json_url}"
        )
    if len(matches) > 1:
        matches.sort()

    filename = matches[-1]
    return filename, f"{base_url}/{year}/{doy}/{filename}"


def _build_authenticated_session() -> tuple[requests.Session, str]:
    """Create a requests Session configured with Earthdata Bearer token or .netrc."""
    session = requests.Session()
    token = get_earthdata_token()
    if token:
        session.headers.update({"Authorization": f"Bearer {token}"})
        return session, "bearer_token"

    creds = get_earthdata_netrc_credentials()
    if creds:
        session.auth = creds
        return session, "netrc"

    raise RuntimeError(
        "NASA Earthdata Login credentials not found. "
        "Set EARTHDATA_TOKEN (or LAADS_TOKEN) environment variable with your Earthdata "
        "Bearer token, or configure ~/.netrc (_netrc on Windows) for urs.earthdata.nasa.gov."
    )


def download_month(
    month: int,
    raw_dir: Path = DEFAULT_RAW_DIR,
    year: int = YEAR,
    tile: str = TILE,
    overwrite: bool = False,
) -> Path:
    """Download a single monthly VNP46A3 HDF5 file for 2025 into raw_dir."""
    if month not in MONTH_DOYS_2025:
        raise ValueError(f"Month must be 1..12, got {month}")

    doy = MONTH_DOYS_2025[month]
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded locally before requiring auth
    existing = sorted(raw_dir.glob(f"VNP46A3.A{year}{doy}.{tile}.*.h5"))
    if existing and not overwrite:
        return existing[-1]

    session, auth_mode = _build_authenticated_session()
    filename, file_url = find_tile_filename(session, year=year, doy=doy, tile=tile)
    dest_path = raw_dir / filename
    if dest_path.is_file() and not overwrite:
        return dest_path

    tmp_path = raw_dir / f"{filename}.part"
    curl_exe = shutil.which("curl.exe") or shutil.which("curl")
    if curl_exe and auth_mode == "bearer_token":
        token = get_earthdata_token()
        # Pass header via stdin config (-K -) so the token never appears in process args.
        curl_cfg = f'header = "Authorization: Bearer {token}"\n'
        cmd = [
            curl_exe,
            "-fSL",
            "-sS",
            "-C",
            "-",
            "--connect-timeout",
            "30",
            "--speed-time",
            "60",
            "--speed-limit",
            "1024",
            "-K",
            "-",
            "-o",
            str(tmp_path),
            file_url,
        ]
        print(f"Downloading {filename} via curl...", flush=True)
        proc = subprocess.run(cmd, input=curl_cfg, text=True, capture_output=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"curl download failed for {file_url} (exit {proc.returncode}): "
                f"{proc.stderr.strip()}"
            )
        if not tmp_path.is_file() or tmp_path.stat().st_size < 1_000_000:
            raise RuntimeError(
                f"Downloaded file {tmp_path} is missing or unexpectedly small "
                f"({tmp_path.stat().st_size if tmp_path.is_file() else 0} bytes)."
            )
    else:
        with session.get(
            file_url, stream=True, timeout=(30, 60), allow_redirects=True
        ) as resp:
            if resp.status_code in (401, 403):
                raise RuntimeError(
                    f"NASA Earthdata authentication failed ({auth_mode}) for {file_url}: "
                    f"HTTP {resp.status_code} {resp.reason}"
                )
            if resp.status_code != 200:
                raise RuntimeError(
                    f"Download failed for {file_url}: HTTP {resp.status_code} {resp.reason}"
                )
            content_type = resp.headers.get("Content-Type", "")
            if "text/html" in content_type.lower():
                raise RuntimeError(
                    f"Received HTML login/redirect page instead of HDF5 binary from {file_url} "
                    f"(final URL: {resp.url}). Check Earthdata Login authorization."
                )

            total_bytes = int(resp.headers.get("Content-Length", "0") or 0)
            print(
                f"Downloading {filename} ({total_bytes / (1024 * 1024):.1f} MB)...",
                flush=True,
            )
            with tmp_path.open("wb") as handle:
                for chunk in resp.iter_content(chunk_size=65536):
                    if chunk:
                        handle.write(chunk)

    tmp_path.replace(dest_path)
    return dest_path


def _decode_attr(val: Any) -> Any:
    """Convert HDF5 attribute values into JSON-serializable Python types."""
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="replace")
    if hasattr(val, "tolist"):
        val = val.tolist()
    if isinstance(val, list):
        return [_decode_attr(item) for item in val]
    return val


def inspect_hdf5(h5_path: Path) -> dict[str, Any]:
    """Inspect all groups, SDS datasets, attributes, and StructMetadata in a VNP46A3 HDF5 file."""
    if not h5_path.is_file():
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    datasets: dict[str, dict[str, Any]] = {}
    struct_metadata: str | None = None

    with h5py.File(h5_path, "r") as hf:
        root_attrs = {k: _decode_attr(v) for k, v in hf.attrs.items()}

        def visitor(name: str, obj: Any) -> None:
            nonlocal struct_metadata
            if isinstance(obj, h5py.Dataset):
                if name.endswith("StructMetadata.0"):
                    raw = obj[()]
                    struct_metadata = (
                        raw.decode("utf-8", errors="replace")
                        if isinstance(raw, bytes)
                        else str(raw)
                    )
                    return
                datasets[name] = {
                    "shape": list(obj.shape),
                    "dtype": str(obj.dtype),
                    "attrs": {k: _decode_attr(v) for k, v in obj.attrs.items()},
                }

        hf.visititems(visitor)

    return {
        "file": str(h5_path),
        "root_attrs": root_attrs,
        "struct_metadata": struct_metadata,
        "datasets": datasets,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download and inspect VNP46A3 HDF5 files.")
    parser.add_argument(
        "--month",
        type=int,
        help="Single month (1-12) to download. Omit to download all 12 months.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Directory for raw downloaded HDF5 files (outside Git).",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Inspect and print HDF5 SDS layers and metadata for the downloaded file(s).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download even if the file already exists locally.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of concurrent downloads when fetching multiple months.",
    )
    args = parser.parse_args(argv)

    months = [args.month] if args.month else list(range(1, 13))
    results: dict[int, Path] = {}
    if args.workers > 1 and len(months) > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            future_map = {
                pool.submit(
                    download_month,
                    m,
                    raw_dir=args.raw_dir,
                    overwrite=args.overwrite,
                ): m
                for m in months
            }
            for fut in as_completed(future_map):
                m = future_map[fut]
                results[m] = fut.result()
                print(f"Completed month {m:02d}: {results[m].name}", flush=True)
    else:
        for m in months:
            results[m] = download_month(
                m, raw_dir=args.raw_dir, overwrite=args.overwrite
            )

    for month in sorted(results):
        path = results[month]
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"Month {month:02d}: {path} ({size_mb:.2f} MB)")
        if args.inspect:
            info = inspect_hdf5(path)
            print(json.dumps(info, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
