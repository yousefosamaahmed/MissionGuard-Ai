from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

ZENODO_RECORD_ID = "15108715"
ZENODO_API = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD_DIR = PROJECT_ROOT / "data" / "opssat" / "raw"


def fetch_metadata() -> dict:
    with urllib.request.urlopen(ZENODO_API, timeout=60) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="List or download official OPSSAT-AD files from the Zenodo record."
    )
    parser.add_argument("--download", action="store_true", help="Download all record files.")
    parser.add_argument("--file", help="Download one exact filename instead of all files.")
    args = parser.parse_args()

    metadata = fetch_metadata()
    files = metadata.get("files", [])
    print(f"Record: {metadata.get('metadata', {}).get('title', 'OPSSAT-AD')}")
    print(f"Official record: https://zenodo.org/records/{ZENODO_RECORD_ID}")
    for entry in files:
        print(f"- {entry['key']} ({entry.get('size', 0) / 1024 / 1024:.2f} MB)")

    if not args.download and not args.file:
        print("\nUse --download or --file FILENAME to retrieve data.")
        return

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    selected = [entry for entry in files if not args.file or entry["key"] == args.file]
    if args.file and not selected:
        raise SystemExit(f"File not found in Zenodo record: {args.file}")

    for entry in selected:
        url = entry.get("links", {}).get("self") or entry.get("links", {}).get("download")
        if not url:
            print(f"Skipping {entry['key']}: no download URL")
            continue
        destination = DOWNLOAD_DIR / entry["key"]
        print(f"Downloading {entry['key']} -> {destination}")
        urllib.request.urlretrieve(url, destination)

    print("\nDownload complete. Preserve the original license and cite the dataset paper.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"OPSSAT preparation failed: {exc}", file=sys.stderr)
        raise
