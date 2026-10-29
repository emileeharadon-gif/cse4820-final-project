"""Inventory an extracted dataset before deciding how its labels should be read.

Uses only the Python standard library. Does not modify source files or infer
species labels from folder names; those must be checked against the metadata.
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"}


def inspect_dataset(root: Path, output: Path) -> dict:
    if not root.is_dir():
        raise ValueError(f"Dataset folder does not exist: {root}")
    if output == root or root.is_relative_to(output):
        raise ValueError("Output must not be the dataset folder or its ancestor.")

    audio = []
    metadata = []
    archives = []
    folders = Counter()
    extensions = Counter()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_relative_to(output):
            continue
        relative = path.relative_to(root).as_posix()
        suffix = path.suffix.lower()
        if suffix in AUDIO_EXTENSIONS:
            parent = path.parent.relative_to(root).as_posix()
            audio.append({"path": relative, "parent_folder": parent,
                          "filename": path.name, "bytes": path.stat().st_size})
            folders[parent] += 1
            extensions[suffix] += 1
        elif suffix in {".csv", ".tsv"}:
            entry = {"path": relative}
            try:
                with path.open(encoding="utf-8-sig", newline="") as handle:
                    reader = csv.reader(handle, delimiter="\t" if suffix == ".tsv" else ",")
                    entry["columns"] = next(reader, [])
                    entry["rows"] = sum(1 for row in reader if row)
            except (OSError, UnicodeError, csv.Error) as error:
                entry["error"] = str(error)
            metadata.append(entry)
        elif suffix in {".zip", ".gz", ".tar", ".7z"}:
            archives.append(relative)

    output.mkdir(parents=True, exist_ok=True)
    with (output / "audio_inventory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "parent_folder", "filename", "bytes"])
        writer.writeheader()
        writer.writerows(audio)
    summary = {
        "dataset_root": str(root),
        "audio_files": len(audio),
        "audio_extensions": dict(sorted(extensions.items())),
        "audio_by_folder": dict(sorted(folders.items())),
        "metadata_tables": metadata,
        "archives": archives,
        "note": "Folder counts are not verified species counts. Audio has not been decoded or validated.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw"), help="Extracted dataset folder")
    parser.add_argument("--output", type=Path, default=Path("data/inventory"))
    args = parser.parse_args()
    try:
        summary = inspect_dataset(args.root.resolve(), args.output.resolve())
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"Audio files: {summary['audio_files']}")
    print(f"Metadata tables: {len(summary['metadata_tables'])}")
    print(f"Inventory saved to: {args.output}")
    if not summary["audio_files"]:
        print("No extracted audio found. Download/extract the dataset, then run this command again.")
        if summary["archives"]:
            print("Archives found: " + ", ".join(summary["archives"]))
        raise SystemExit(1)
    print("Next: inspect summary.json and metadata to verify species labels and source recording IDs.")


if __name__ == "__main__":
    main()
