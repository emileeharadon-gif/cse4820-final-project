"""Shared data preparation for all four models (CSE 4820 bird-audio project).

Follows the project proposal:
  * keep the 10-15 species with the most available recordings,
  * one stratified train / validation / test split shared by every model,
  * pooled MFCC features (Logistic Regression, Random Forest),
  * fixed-size log-mel spectrograms (both CNNs), from the SAME clips.

Leakage rule: the split is made per ORIGINAL RECORDING, before any clip is cut,
so clips from one recording never land in two splits.

Usage (from the repo root):
    python scripts/prepare_data.py                      # 12 species, defaults
    python scripts/prepare_data.py --n-species 15
Outputs go to data/processed/ (git-ignored).
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "data" / "raw" / "avian-vocalizations-partitioned-data"
DEFAULT_OUT = PROJECT_ROOT / "data" / "processed"

# Audio / feature settings (saved to config.json so every model reads the same ones).
SR = 22050
CLIP_SECONDS = 5.0
MIN_TAIL_SECONDS = 2.5      # a leftover tail shorter than this is dropped (longer ones are padded)
N_FFT = 2048
HOP = 512
N_MELS = 128
N_MFCC = 20
FMAX = SR // 2
SEED = 42


def load_index(data_dir):
    """Rows of the Kaggle index CSV that have a downloaded MP3."""
    index = data_dir / "xeno-canto_ca-nv_index.csv"
    audio_dir = data_dir / "audio"
    rows = []
    with index.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            path = audio_dir / row["file_name"]
            if path.is_file():
                genus, species = row["genus"].strip(), row["species"].strip()
                rows.append({
                    "file_name": row["file_name"],
                    "path": str(path),
                    "species": f"{genus} {species}",       # label = genus + species (not full_name,
                    "common_name": row.get("english_cname", ""),  # which can carry subspecies text)
                })
    return rows


def decode(path):
    """Decode to mono float32 at SR. Returns None if the file is unreadable."""
    import librosa
    try:
        y, _ = librosa.load(path, sr=SR, mono=True)
    except Exception:
        return None
    if y.size == 0 or not np.isfinite(y).all():
        return None
    return y.astype(np.float32)


def check_one(row):
    y = decode(row["path"])
    if y is None:
        return row, None
    return row, len(y) / SR


def select_species(usable, n_species):
    """Top n species by number of USABLE recordings.

    The Kaggle index has 30 recordings per species, so many species tie. Ties are
    broken deterministically by total usable audio (seconds) and then by name.
    """
    stats = {}
    for row in usable:
        s = stats.setdefault(row["species"], {"n": 0, "seconds": 0.0})
        s["n"] += 1
        s["seconds"] += row["seconds"]
    ranked = sorted(stats.items(), key=lambda kv: (-kv[1]["n"], -kv[1]["seconds"], kv[0]))
    return [name for name, _ in ranked[:n_species]], ranked


def stratified_split(rows, seed, val_frac, test_frac):
    """Stratified recording-level split. Returns {file_name: 'train'|'val'|'test'}."""
    from sklearn.model_selection import train_test_split
    names = [r["file_name"] for r in rows]
    labels = [r["species"] for r in rows]
    trainval, test, y_trainval, _ = train_test_split(
        names, labels, test_size=test_frac, stratify=labels, random_state=seed)
    train, val = train_test_split(
        trainval, test_size=val_frac / (1.0 - test_frac), stratify=y_trainval, random_state=seed)
    split = {n: "train" for n in train}
    split.update({n: "val" for n in val})
    split.update({n: "test" for n in test})
    return split


def cut_clips(y, clip_len, min_tail, max_clips):
    """Non-overlapping fixed-length clips; short audio is zero-padded to one clip."""
    clips = []
    for start in range(0, len(y), clip_len):
        piece = y[start:start + clip_len]
        if len(piece) < clip_len:
            if clips and len(piece) < min_tail:
                break
            piece = np.pad(piece, (0, clip_len - len(piece)))
        clips.append(piece)
        if len(clips) >= max_clips:
            break
    return clips


def features_for_recording(row):
    """Returns (mfcc_pooled [n,2*N_MFCC], logmel [n,N_MELS,T]) for one recording."""
    import librosa
    y = decode(row["path"])
    clips = cut_clips(y, int(CLIP_SECONDS * SR), int(MIN_TAIL_SECONDS * SR), row["max_clips"])
    pooled, mels = [], []
    for clip in clips:
        mel = librosa.feature.melspectrogram(
            y=clip, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, fmax=FMAX)
        logmel = librosa.power_to_db(mel, ref=1.0, top_db=80.0).astype(np.float32)
        mfcc = librosa.feature.mfcc(S=logmel, n_mfcc=N_MFCC)
        pooled.append(np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)]))  # pooled over time
        mels.append(logmel)
    return np.asarray(pooled, dtype=np.float32), np.asarray(mels, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-species", type=int, default=12, help="Proposal range is 10-15 (default 12)")
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--max-clips-per-recording", type=int, default=10,
                        help="Cap so long recordings do not dominate a class (default 10 = 50 s)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--jobs", type=int, default=-1, help="Parallel workers (-1 = all cores)")
    args = parser.parse_args()
    if not 10 <= args.n_species <= 15:
        parser.error("The proposal specifies 10 to 15 species; use --n-species within that range")

    from joblib import Parallel, delayed
    args.out.mkdir(parents=True, exist_ok=True)

    # 1. Which recordings exist locally, and which actually decode?
    rows = load_index(args.data_dir)
    if not rows:
        sys.exit(f"No downloaded MP3s found under {args.data_dir / 'audio'}. Run datadownload/kaggledownload.py first.")
    print(f"Index recordings present on disk: {len(rows)}", flush=True)
    checked = Parallel(n_jobs=args.jobs, verbose=0)(delayed(check_one)(r) for r in rows)
    usable = [dict(r, seconds=sec) for r, sec in checked if sec is not None]
    unreadable = [r["file_name"] for r, sec in checked if sec is None]
    print(f"Decodable: {len(usable)}  Unreadable: {len(unreadable)}", flush=True)
    (args.out / "unreadable_audio.txt").write_text("\n".join(unreadable) + ("\n" if unreadable else ""))

    # 2. Species subset: most usable recordings, deterministic tie-break.
    species, ranked = select_species(usable, args.n_species)
    with (args.out / "species_ranking.csv").open("w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["rank", "species", "usable_recordings", "usable_seconds", "selected"])
        for i, (name, s) in enumerate(ranked, 1):
            w.writerow([i, name, s["n"], round(s["seconds"], 1), name in species])
    label_map = {name: i for i, name in enumerate(sorted(species))}
    kept = [r for r in usable if r["species"] in label_map]
    print(f"Selected {len(species)} species, {len(kept)} recordings", flush=True)

    # 3. ONE stratified recording-level split shared by all four models.
    split = stratified_split(kept, args.seed, args.val_frac, args.test_frac)

    # 4. Clips + features, built from the same audio for every model.
    for r in kept:
        r["max_clips"] = args.max_clips_per_recording
    feats = Parallel(n_jobs=args.jobs, verbose=0)(delayed(features_for_recording)(r) for r in kept)

    X_mfcc, X_mel, y, rec_ids, splits = [], [], [], [], []
    for r, (pooled, mels) in zip(kept, feats):
        n = len(pooled)
        X_mfcc.append(pooled)
        X_mel.append(mels)
        y += [label_map[r["species"]]] * n
        rec_ids += [r["file_name"]] * n
        splits += [split[r["file_name"]]] * n
    X_mfcc, X_mel = np.concatenate(X_mfcc), np.concatenate(X_mel)
    y, rec_ids, splits = np.array(y), np.array(rec_ids), np.array(splits)

    # Dataset-level normalisation statistics come from TRAIN clips only.
    train = splits == "train"
    mel_mean, mel_std = float(X_mel[train].mean()), float(X_mel[train].std() + 1e-8)

    np.savez_compressed(
        args.out / "features.npz",
        mfcc_pooled=X_mfcc, logmel=X_mel, y=y, recording_id=rec_ids, split=splits,
        mel_mean=mel_mean, mel_std=mel_std)
    with (args.out / "recordings.csv").open("w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["file_name", "species", "label", "split", "seconds"])
        for r in kept:
            w.writerow([r["file_name"], r["species"], label_map[r["species"]], split[r["file_name"]], round(r["seconds"], 2)])
    config = {
        "label_map": label_map, "sample_rate": SR, "clip_seconds": CLIP_SECONDS,
        "min_tail_seconds": MIN_TAIL_SECONDS, "n_fft": N_FFT, "hop_length": HOP, "n_mels": N_MELS,
        "n_mfcc": N_MFCC, "fmax": FMAX, "seed": args.seed, "n_species": len(label_map),
        "split_fractions": {"train": 1 - args.val_frac - args.test_frac, "val": args.val_frac, "test": args.test_frac},
        "max_clips_per_recording": args.max_clips_per_recording,
        "mel_mean": mel_mean, "mel_std": mel_std,
        "logmel_shape": list(X_mel.shape[1:]),
    }
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    # 5. Sanity report (also useful as numbers for the midpoint report).
    print("\nRecordings / clips per split:")
    for s in ("train", "val", "test"):
        print(f"  {s:5s} recordings={len(set(rec_ids[splits == s])):4d}  clips={int((splits == s).sum()):5d}")
    overlap = [set(rec_ids[splits == a]) & set(rec_ids[splits == b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))]
    assert not any(overlap), "Leakage: a recording appears in more than one split"
    print(f"\nlog-mel clip shape: {X_mel.shape[1:]}  MFCC feature dim: {X_mfcc.shape[1]}")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
