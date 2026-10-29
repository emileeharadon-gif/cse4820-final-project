"""Download Kaggle metadata, then fetch the MP3 links in its index CSV.

Run with --limit 1 to check a link, or --metadata-only to skip audio.
Existing MP3s are skipped on reruns. Failures are saved beside the audio.
"""

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = "samhiatt/avian-vocalizations-partitioned-data"


def is_mp3(path):
    if not path.is_file():
        return False
    with path.open("rb") as handle:
        header = handle.read(3)
    return header == b"ID3" or (len(header) >= 2 and header[0] == 255 and header[1] & 224 == 224)


def download_recordings(index, output, limit, workers):
    import requests

    with index.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not {"file_name", "file_url"}.issubset(reader.fieldnames or []):
            raise ValueError("The index needs file_name and file_url columns")
        rows = list(reader)
    if limit is not None:
        rows = rows[:limit]
    names = [row["file_name"] for row in rows]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate file names in index; inspect the metadata first")
    for name in names:
        if not name or Path(name).name != name or not name.lower().endswith(".mp3"):
            raise ValueError(f"Invalid MP3 filename in index: {name!r}")
    output.mkdir(parents=True, exist_ok=True)

    def fetch(row):
        target = output / row["file_name"]
        if is_mp3(target):
            return row, "cached", ""
        partial = target.with_suffix(".mp3.part")
        for attempt in range(3):
            try:
                with requests.get(row["file_url"], stream=True, timeout=(15, 60)) as response:
                    response.raise_for_status()
                    with partial.open("wb") as handle:
                        for chunk in response.iter_content(chunk_size=65536):
                            if chunk:
                                handle.write(chunk)
                if not is_mp3(partial):
                    raise ValueError("The URL returned no recognizable MP3 audio (possibly a web page)")
                partial.replace(target)
                return row, "downloaded", ""
            except (requests.RequestException, OSError, ValueError) as error:
                failure = str(error)
                partial.unlink(missing_ok=True)
                status = getattr(getattr(error, "response", None), "status_code", None)
                if status in {401, 403, 404, 410}:
                    break
                if attempt < 2:
                    time.sleep(2 ** attempt)
        return row, "failed", failure

    counts = {"downloaded": 0, "cached": 0, "failed": 0}
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for number, (row, status, error) in enumerate(executor.map(fetch, rows), start=1):
            counts[status] += 1
            if status == "failed":
                failures.append({"file_name": row["file_name"], "file_url": row["file_url"], "error": error})
            if number == 1 or number % 25 == 0 or number == len(rows):
                print(f"MP3s {number}/{len(rows)}: {counts}", flush=True)
    with (output / "download_failures.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file_name", "file_url", "error"])
        writer.writeheader()
        writer.writerows(failures)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Kaggle owner/dataset handle")
    parser.add_argument("--output", type=Path, help="Destination; defaults to data/raw/<dataset-name>")
    parser.add_argument("--metadata-only", action="store_true", help="Skip fetching the CSV's MP3 URLs")
    parser.add_argument("--limit", type=int, help="Fetch only the first N recordings (use 1 to check a link)")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent audio downloads (default: 4)")
    args = parser.parse_args()
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--workers and --limit must be positive")
    dataset_name = args.dataset.split("/")
    if len(dataset_name) < 2 or not all(dataset_name[:2]) or dataset_name[1] in {".", ".."}:
        parser.error("--dataset must be a Kaggle owner/dataset handle")
    output = (args.output or PROJECT_ROOT / "data" / "raw" / dataset_name[1]).resolve()

    # Keep Kaggle's local cache inside the project, too.
    os.environ.setdefault("KAGGLEHUB_CACHE", str(PROJECT_ROOT / "data" / ".kagglehub-cache"))
    try:
        import kagglehub
    except ModuleNotFoundError:
        parser.exit(1, "Install the downloader first: python -m pip install -r requirements-download.txt\n")

    print(f"Downloading {args.dataset}", flush=True)
    print(f"Destination: {output}", flush=True)
    # Public datasets can be downloaded without an interactive login. KaggleHub
    # also uses existing credentials or KAGGLE_API_TOKEN when authentication is needed.
    try:
        path = Path(kagglehub.dataset_download(args.dataset, output_dir=str(output)))
    except Exception as error:
        parser.exit(1, f"Download failed ({type(error).__name__}). Check your network and the dataset URL.\n"
                    "If Kaggle requires authentication, configure KAGGLE_API_TOKEN locally.\n")

    print(f"Dataset downloaded and extracted to: {path}")
    download_counts = None
    if not args.metadata_only:
        index = path / "xeno-canto_ca-nv_index.csv"
        if not index.is_file():
            parser.exit(1, "The expected xeno-canto_ca-nv_index.csv was not found in this dataset.\n")
        try:
            download_counts = download_recordings(index, path / "audio", args.limit, args.workers)
        except (OSError, ValueError) as error:
            parser.exit(1, f"Could not prepare the audio download: {error}\n")
    # Also support direct execution: python datadownload/kaggledownload.py.
    sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.inspect_data import inspect_dataset

    try:
        summary = inspect_dataset(path.resolve(), PROJECT_ROOT / "data" / "inventory")
    except (OSError, ValueError) as error:
        parser.exit(1, f"Download completed, but inventory failed: {error}\n")
    print(f"Audio files found: {summary['audio_files']}")
    print(f"Metadata tables found: {len(summary['metadata_tables'])}")
    print("Inventory saved in data/inventory/")
    if not summary["audio_files"]:
        print("No MP3s are available locally yet.")
    if download_counts and download_counts["failed"]:
        parser.exit(1, "Some recordings could not be fetched. See audio/download_failures.csv in the dataset folder.\n")


if __name__ == "__main__":
    main()
