"""Download the public evaluation corpus listed in corpus/manifest.json.

Usage:
    python scripts/download_corpus.py            # fetch missing files
    python scripts/download_corpus.py --verify   # only check SHA-256 sums

The sums of the files the eval was measured on live in corpus/SHA256SUMS, so a
changed upstream page is reported instead of silently shifting the results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

CORPUS_DIR = Path(__file__).resolve().parent.parent / "corpus"
USER_AGENT = "rag-with-citations-corpus-fetcher/1.0"


def sha256_of(path: Path) -> str:
    """Return the hex SHA-256 digest of a file.

    Args:
        path: File to hash.

    Returns:
        Lowercase hex digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(url: str, destination: Path) -> None:
    """Download one URL to a destination path.

    Args:
        url: Public URL from the manifest.
        destination: Where to write the bytes (parent dirs are created).
    """
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 - fixed manifest URLs
        data = response.read()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)


def load_sums() -> dict[str, str]:
    """Read corpus/SHA256SUMS into a path -> digest mapping (empty if absent).

    Returns:
        Mapping of manifest-relative path to recorded digest.
    """
    sums_file = CORPUS_DIR / "SHA256SUMS"
    if not sums_file.exists():
        return {}
    pairs = (line.split(maxsplit=1) for line in sums_file.read_text().splitlines() if line.strip())
    return {path.strip(): digest for digest, path in pairs}


def main() -> int:
    """Fetch missing corpus files and compare them with the recorded sums.

    Returns:
        Process exit code: 0 when every file matches, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="do not download, only verify")
    args = parser.parse_args()
    manifest = json.loads((CORPUS_DIR / "manifest.json").read_text())
    recorded, mismatches = load_sums(), 0
    for entry in manifest["files"]:
        target = CORPUS_DIR / entry["path"]
        if not target.exists() and not args.verify:
            print(f"fetch  {entry['url']}")
            fetch(entry["url"], target)
        expected = recorded.get(entry["path"])
        actual = sha256_of(target) if target.exists() else "missing"
        if expected and expected != actual:
            mismatches += 1
            print(f"CHANGED {entry['path']} (recorded {expected[:12]}, now {actual[:12]})")
    print(f"{len(manifest['files'])} files, {mismatches} mismatches")
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
