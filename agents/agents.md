# Project handoff for teammates and coding agents

Last verified: October 28, 2026. Working branch: `adib-branch`.
This is a snapshot of completed work and remaining tasks. Recheck the files and
counts before relying on them later. Ask a teammate's agent to read this file
explicitly; this nested, lowercase filename may not be discovered automatically.

## Project context

The project classifies bird species from audio and compares four models:

| File | Planned model | Input |
| --- | --- | --- |
| `models/logreg.py` | Logistic regression | Pooled MFCC features |
| `models/rf.py` | Random forest | The same pooled MFCC features |
| `models/cnnmax.py` | CNN with max pooling | Log-mel spectrograms |
| `models/cnn.py` | CNN without pooling | The same log-mel spectrograms |

Adib has taken responsibility for both CNNs. All four model files are currently
empty. The proposal calls for selecting 10-15 species and comparing accuracy,
macro F1, confusion matrices, and per-class precision and recall using shared
held-out recordings. Final species selection and preprocessing are not done.

## Files added or updated

### `datadownload/kaggledownload.py`

The downloader was initially at the repository root and has since been moved
into `datadownload/`. Its project-root calculation and inspection-script import
were corrected to support direct execution from this location.

The script:

1. Downloads and extracts Kaggle dataset
   `samhiatt/avian-vocalizations-partitioned-data` using `kagglehub`.
2. Defaults to `data/raw/avian-vocalizations-partitioned-data/` beneath the
   repository root, independent of the terminal's working directory.
3. Reads `file_name` and `file_url` from `xeno-canto_ca-nv_index.csv`.
4. Fetches the recording URLs using `requests`, following redirects, with four
   concurrent workers by default.
5. Streams downloads to temporary `.mp3.part` files, checks their headers, then
   renames successful files to their final `.mp3` names.
6. Skips existing files that pass its MP3 header check.
7. Makes up to three attempts for retryable failures. HTTP 401, 403, 404, and 410
   stop retries for that recording. Connection/read timeouts are 15/60 seconds.
8. Prints progress for the first result, every 25 results, and the last result.
   Progress counts processed rows, including cached files and failures.
9. Writes `audio/download_failures.csv` when the batch finishes.
10. Generates the inventory using `scripts/inspect_data.py` and exits with status
    1 if any attempted recording failed. A nonzero status can therefore coexist
    with a mostly successful download.

The MP3 check accepts an ID3 header or an MPEG frame-sync prefix. It is a basic
format check, not full decoding, a checksum, or a guarantee of audio quality.

Public access worked without calling `kagglehub.login()`. Existing credentials
or a locally configured `KAGGLE_API_TOKEN` can be used if authentication becomes
necessary. No credentials are embedded in the source. The default Kaggle cache
is set to `data/.kagglehub-cache/` unless `KAGGLEHUB_CACHE` is already configured.

Supported options:

| Option | Meaning |
| --- | --- |
| `--metadata-only` | Download metadata and inventory it; skip audio fetching |
| `--limit N` | Process only the first N index rows; useful for a smoke check |
| `--workers N` | Number of concurrent audio downloads |
| `--output PATH` | Override the extracted dataset destination |
| `--dataset HANDLE` | Override the Kaggle handle |

Audio fetching expects the named index CSV and its columns, so `--dataset` does
not make this a generic downloader for unrelated dataset layouts. Each completed
audio batch overwrites the failure log with failures from that invocation;
running with `--limit` does not preserve a full-dataset failure report.

### `scripts/inspect_data.py`

A standard-library-only inventory tool. It recursively lists audio paths,
filenames, parent folders, and byte sizes; counts files by extension and folder;
records CSV/TSV column names and row counts; and lists archives still present.
It does not infer species from folders, modify source audio, decode recordings,
create splits, or extract features. Empty audio input produces exit status 1.

Generated outputs:

- `data/inventory/audio_inventory.csv`: local audio files and sizes.
- `data/inventory/summary.json`: counts, metadata table descriptions, and root path.

### `requirements-download.txt` and `.venv/`

The downloader dependencies are `kagglehub==1.0.3` and `requests>=2.32,<3`.
They were installed in a local `.venv/`; the local Python version is 3.13.1.
The environment is not shared through Git. This dependency list covers downloading
only; audio preprocessing, scikit-learn, and a CNN framework still need setup.

### `.gitignore` and `data/raw/.gitkeep`

Git ignores `.venv/`, Python caches, downloaded raw files, generated features,
inventory reports, the local Kaggle cache, checkpoints, common model artifacts,
`.env`, and `kaggle.json`. The `.gitkeep` preserves the raw-data directory without
adding recordings to Git. Teammates must obtain their own local data.

A setup README was added earlier but is no longer present in the working tree.
The current commands and handoff details are recorded here.

## Verified download results

The full download invocation has finished; it is not still running.

| Check | Result |
| --- | --- |
| Recordings listed in index | 2,730 |
| Species, grouped by genus and species | 91 |
| Index recordings per species | 30 each |
| Downloaded MP3s | 2,698 (98.8%) |
| Missing recordings / logged failures | 32 |
| Unexpected MP3 filenames | 0 |
| Remaining `.part` files | 0 |
| Saved MP3s passing header check | 2,698 |
| Total saved audio bytes | 1,472,442,736 (about 1.37 GiB) |

Failure categories: one HTTP 404 (`XC146295.mp3`), one HTTP 500
(`XC307090.mp3`), and 30 responses rejected by the MP3 header check. The latter
are not yet proven to be HTML, unavailable recordings, or another audio format;
their actual responses need investigation if these recordings are needed.

The proposal assumed unequal class counts. This particular index is balanced
before failures: 30 recordings per species. Recount usable recordings after
decoding and filtering. Do not report all 2,730 as successfully downloaded.

## Local dataset layout and metadata

```text
data/
  raw/
    .gitkeep
    avian-vocalizations-partitioned-data/
      xeno-canto_ca-nv_index.csv
      train_file_ids.csv
      test_file_ids.csv
      audio/
        XC17804.mp3
        ...
        download_failures.csv
  inventory/
    audio_inventory.csv
    summary.json
```

The Kaggle partitioned package contains metadata, not bundled recordings.
Audio was fetched directly from its CSV URLs. No companion audio package is
needed for this approach.

Relevant index columns include `file_id`, `file_name`, `file_url`, `genus`,
`species`, `english_cname`, `full_name`, `duration_seconds`, `license`, `recordist`,
and location information. `full_name` can contain subspecies detail; verify the
label definition rather than assuming every distinct full name is a species.

The supplied split files have `file_id` and `label` columns:

- `train_file_ids.csv`: 1,820 rows.
- `test_file_ids.csv`: 910 rows.

These are the dataset author's existing assignments, not newly created project
splits. Verify label correspondence and filter to available files before use.
There is no project validation split yet.

## Commands for teammates

Run these from the repository root. Network access is required for downloading.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-download.txt
python datadownload/kaggledownload.py
```

Rerunning skips saved MP3s and retries missing ones; it also checks Kaggle for
the metadata package. No need to delete the existing audio folder.

To inspect existing files without downloading:

```bash
python3 scripts/inspect_data.py --root data/raw/avian-vocalizations-partitioned-data
```

To check one download on a fresh setup:

```bash
python datadownload/kaggledownload.py --limit 1
```

## Verification already performed

- Checked the finished process and compared local filenames with the full index.
- Checked the headers of all 2,698 saved MP3s and checked for partial files.
- Exercised inventory generation with synthetic audio filenames and CSV metadata.
- Exercised streamed downloads, cached reruns, HTTP failures, partial cleanup,
  and rejection of HTML-like responses with mocked network responses.
- Verified the moved downloader's repository-root calculation, output destination,
  and inventory generation using a temporary dataset and a mocked Kaggle download.
- Checked downloader CLI help, Python compilation, package dependency consistency,
  and Git whitespace checks during implementation.

These were targeted checks run during the session, not a committed test suite.
No full audio decoding, feature extraction, model training, or model evaluation
has been performed.

## Next data-preparation steps

1. Decode local recordings to verify readability, actual durations, sample rates,
   and channels. Log unusable files separately from download failures.
2. Investigate the missing responses if needed and build a manifest joining
   available audio to verified species labels and original recording IDs.
3. Select and save a fixed set of 10-15 species. With equal class counts, document
   a deterministic tie-breaking rule rather than claiming a unique top 10.
4. Inspect the supplied train/test assignments. Preserve a suitable existing
   held-out test partition and derive validation from training, or document any
   justified alternative. All models must use the same recording assignments.
5. Keep every segment or duplicate of an original recording in one split. Split
   by source recording before creating segments to prevent data leakage.
6. Choose and save sample rate, mono conversion, clip length, crop/padding policy,
   spectrogram parameters, random seed, and a shared species-to-number mapping.
7. Extract pooled MFCC features for both classical models and fixed-size log-mel
   spectrogram arrays for both CNNs from the same selected audio segments.
8. Fit dataset-level normalization or scaling on training data only. Verify
   shapes, labels, finite values, per-class counts, split separation, and example
   spectrograms before model training.

## Git and collaboration state

As of this handoff, the additions have not been committed or pushed. The original
root-level model placeholders show as deleted, while `models/` contains the four
empty placeholders. Preserve that user reorganization when reviewing or staging
changes. Do not assume another branch or teammate checkout already contains this
work or the ignored local data. Read the current working tree before editing.
