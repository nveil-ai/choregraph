# SPDX-FileCopyrightText: 2026 NVEIL SAS
# SPDX-FileContributor: Pierre Jacquet
# SPDX-FileContributor: Guillaume Franque
# SPDX-License-Identifier: AGPL-3.0-or-later

"""URL fetcher module for downloading remote data sources to local files.

Supports direct-file URLs (e.g., CSV, JSON) and API endpoints.
Downloads are saved as local files so the existing pipeline (CSV
characterization, Kedro catalog, hash-based caching) works unchanged.
"""

import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import unquote, urlparse

import requests

from .security import safe_path, sanitize_filename

logger = logging.getLogger(__name__)

# Safety limit: 500 MB
MAX_DOWNLOAD_SIZE = 500 * 1024 * 1024

# Content-Type to file extension mapping
_CONTENT_TYPE_MAP = {
    "text/csv": ".csv",
    "application/csv": ".csv",
    "application/json": ".json",
    "application/geo+json": ".json",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "application/vnd.ms-excel": ".xls",
    "text/plain": ".csv",  # best guess for plain text
}


def _filename_from_content_disposition(header: str) -> Optional[str]:
    """Extract filename from Content-Disposition header."""
    if not header:
        return None
    match = re.search(r'filename\*?=["\']?(?:UTF-8\'\')?([^"\';\s]+)', header, re.IGNORECASE)
    if match:
        return unquote(match.group(1))
    return None


def _filename_from_url(url: str) -> Optional[str]:
    """Derive a filename from the URL path."""
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")
    if path:
        # Normalize the path to prevent directory traversal
        name = Path(path).name
        if "." in name:
            # Sanitize the filename to remove any dangerous characters
            return sanitize_filename(name)
    return None


def _infer_extension(content_type: Optional[str]) -> str:
    """Map Content-Type header to a file extension."""
    if not content_type:
        return ".csv"
    # Take the base type (ignore charset etc.)
    base = content_type.split(";")[0].strip().lower()
    return _CONTENT_TYPE_MAP.get(base, ".csv")


def fetch_url(
    url: str,
    dest_dir: Path,
    filename: Optional[str] = None,
    timeout: Tuple[int, int] = (30, 120),
) -> Tuple[Path, str]:
    """Download a URL to a local file.

    Args:
        url: The URL to fetch.
        dest_dir: Directory to save the file into.
        filename: Override filename. If None, derived from URL/headers.
        timeout: (connect_timeout, read_timeout) in seconds.

    Returns:
        Tuple of (local_file_path, detected_format).

    Raises:
        ValueError: If the download exceeds the size limit or if the path is unsafe.
        requests.RequestError: On network errors.
    """
    dest_dir = safe_path(Path(dest_dir))
    dest_dir.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, stream=True, timeout=timeout)
    response.raise_for_status()

    # Determine filename
    if not filename:
        filename = _filename_from_content_disposition(
            response.headers.get("Content-Disposition")
        )
    if not filename:
        filename = _filename_from_url(url)
    if not filename:
        ext = _infer_extension(response.headers.get("Content-Type"))
        filename = f"url_data{ext}"

    # Sanitize the filename to prevent directory traversal
    filename = sanitize_filename(filename)

    # Infer format from extension
    ext = Path(filename).suffix.lstrip(".").upper()
    fmt = ext if ext else "CSV"

    dest_path = dest_dir / filename

    # Stream download with size check
    downloaded = 0
    size_exceeded = False
    with open(safe_path(dest_path), "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            downloaded += len(chunk)
            if downloaded > MAX_DOWNLOAD_SIZE:
                size_exceeded = True
                break
            f.write(chunk)

    if size_exceeded:
        safe_path(dest_path).unlink(missing_ok=True)
        raise ValueError(
            f"Download exceeds {MAX_DOWNLOAD_SIZE // (1024*1024)}MB limit."
        )

    logger.info(f"Downloaded {url} -> {dest_path} ({downloaded} bytes, format={fmt})")
    return dest_path, fmt


def fetch_inputs(inputs: list, workspace_path: Path) -> int:
    """Download all URL-based inputs, updating their location in-place.

    Args:
        inputs: List of InputSpec objects to process.
        workspace_path: Root workspace directory for saving files.

    Returns:
        Number of inputs successfully fetched.
    """
    count = 0
    for inp in inputs:
        if not inp.url:
            continue
        try:
            dest_path, fmt = fetch_url(inp.url, workspace_path)
            inp.location = str(dest_path)
            if not inp.format or inp.format == "CSV":
                inp.format = fmt
            count += 1
            logger.info(f"Fetched URL input '{inp.label}': {inp.url} -> {dest_path}")
        except Exception as e:
            logger.error(f"Failed to fetch URL input '{inp.label}' ({inp.url}): {e}")
    return count
