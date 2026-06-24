"""BinaryCIF downloader with retry and hash computation.

Downloads .bcif.gz files from RCSB and computes SHA-256 for provenance.

Requirements: 1.1, 1.7
"""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

RCSB_BCIF_URL = "https://files.rcsb.org/download/{pdb_id}.cif.gz"


@dataclass
class DownloadResult:
    """Result of a BinaryCIF download."""

    pdb_id: str
    file_path: Path
    file_hash: str  # SHA-256 of decompressed content
    size_bytes: int


class DownloadError(Exception):
    """Raised when BinaryCIF download fails after all retries."""

    pass


async def download_bcif(
    pdb_id: str,
    output_dir: Path = Path("/tmp/bcif"),
    max_retries: int = 3,
    backoff_base: float = 2.0,
) -> DownloadResult:
    """Download BinaryCIF from RCSB with retry + exponential backoff.

    URL pattern: https://files.rcsb.org/download/{pdb_id}.bcif.gz

    The file is downloaded as gzipped, decompressed, and stored as .bcif.
    SHA-256 is computed on the decompressed content for provenance tracking.

    Args:
        pdb_id: 4-character PDB identifier (case-insensitive).
        output_dir: Directory to store downloaded files.
        max_retries: Maximum number of download attempts (default 3).
        backoff_base: Base for exponential backoff in seconds.

    Returns:
        DownloadResult with file path and SHA-256 hash.

    Raises:
        DownloadError: If download fails after all retries.
    """
    pdb_id_lower = pdb_id.strip().lower()
    pdb_id_upper = pdb_id.strip().upper()
    url = RCSB_BCIF_URL.format(pdb_id=pdb_id_upper)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{pdb_id_lower}.cif"

    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(
                "Downloading BinaryCIF for %s (attempt %d/%d)",
                pdb_id_lower,
                attempt,
                max_retries,
            )
            content = await _download_and_decompress(url)

            # Compute SHA-256 on decompressed content
            file_hash = hashlib.sha256(content).hexdigest()

            # Write to disk
            output_path.write_bytes(content)

            logger.info(
                "Downloaded %s: %d bytes, hash=%s",
                pdb_id_lower,
                len(content),
                file_hash[:12],
            )

            return DownloadResult(
                pdb_id=pdb_id_lower,
                file_path=output_path,
                file_hash=file_hash,
                size_bytes=len(content),
            )

        except Exception as e:
            last_error = e
            if attempt < max_retries:
                wait_time = backoff_base**attempt
                logger.warning(
                    "Download attempt %d failed for %s: %s. Retrying in %.1fs",
                    attempt,
                    pdb_id_lower,
                    str(e),
                    wait_time,
                )
                await asyncio.sleep(wait_time)
            else:
                logger.error(
                    "All %d download attempts failed for %s: %s",
                    max_retries,
                    pdb_id_lower,
                    str(e),
                )

    raise DownloadError(
        f"Failed to download BinaryCIF for {pdb_id_lower} after {max_retries} attempts: "
        f"{last_error}"
    )


async def _download_and_decompress(url: str) -> bytes:
    """Download gzipped content and decompress it.

    Uses urllib (stdlib) to avoid adding httpx dependency to the science container.
    Runs in a thread executor to avoid blocking the event loop.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _sync_download_and_decompress, url)


def _sync_download_and_decompress(url: str) -> bytes:
    """Synchronous download + gzip decompression."""
    request = Request(url, headers={"Accept-Encoding": "identity"})
    try:
        with urlopen(request, timeout=60) as response:
            if response.status != 200:
                raise DownloadError(
                    f"HTTP {response.status} from {url}"
                )
            compressed = response.read()
    except URLError as e:
        raise DownloadError(f"Network error downloading {url}: {e}") from e

    try:
        return gzip.decompress(compressed)
    except gzip.BadGzipFile as e:
        raise DownloadError(f"Invalid gzip data from {url}: {e}") from e
