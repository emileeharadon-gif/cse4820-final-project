"""Shared loading + evaluation so all four models are scored identically.

Metrics follow the proposal: accuracy, macro F1, confusion matrix, and per-class
precision/recall, all computed on the same held-out test set.
"""

import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS = PROJECT_ROOT / "results"


def load_processed(processed=PROCESSED):
    """Returns (npz arrays, config dict). Run scripts/prepare_data.py first."""
    features = processed / "features.npz"
    if not features.is_file():
        raise SystemExit("data/processed/features.npz not found. Run: python scripts/prepare_data.py")
    data = np.load(features)
    config = json.loads((processed / "config.json").read_text())
    return data, config


def split_arrays(data, key):
    """Train / val / test slices of one feature array, plus matching labels."""
    out = {}
    for name in ("train", "val", "test"):
        mask = data["split"] == name
        out[name] = (data[key][mask], data["y"][mask])
    return out


def evaluate(model_name, y_true, y_pred, config, extra=None, results_dir=RESULTS):
    """Compute the proposal's metrics, save JSON + confusion-matrix PNG, return the dict."""
    from sklearn.metrics import (accuracy_score, classification_report,
                                 confusion_matrix, f1_score)

    names = [n for n, _ in sorted(config["label_map"].items(), key=lambda kv: kv[1])]
    labels = list(range(len(names)))
    report = classification_report(y_true, y_pred, labels=labels, target_names=names,
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    result = {
        "model": model_name,
        "test_clips": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class": {n: {"precision": report[n]["precision"], "recall": report[n]["recall"],
                          "f1": report[n]["f1-score"], "support": int(report[n]["support"])} for n in names},
        "confusion_matrix": cm.tolist(),
        "classes": names,
    }
    if extra:
        result.update(extra)

    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"{model_name}_results.json").write_text(json.dumps(result, indent=2) + "\n")

    # Numbers first: print them and keep the JSON even if plotting is broken.
    print(f"\n[{model_name}] test accuracy = {result['accuracy']:.3f}   macro F1 = {result['macro_f1']:.3f}")
    print(f"{'species':32s} precision  recall  support")
    for n in names:
        p = result["per_class"][n]
        print(f"{n:32s} {p['precision']:9.3f} {p['recall']:7.3f} {p['support']:8d}")
    print(f"Saved results/{model_name}_results.json")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(labels, names, rotation=60, ha="right", fontsize=7)
        ax.set_yticks(labels, names, fontsize=7)
        ax.set_xlabel("Predicted species")
        ax.set_ylabel("True species")
        ax.set_title(f"{model_name}: test confusion matrix")
        for i in labels:
            for j in labels:
                if cm[i, j]:
                    ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                            color="white" if cm[i, j] > cm.max() / 2 else "black")
        fig.tight_layout()
        fig.savefig(results_dir / f"{model_name}_confusion.png", dpi=150)
        plt.close(fig)
        print(f"Saved results/{model_name}_confusion.png")
    except Exception as error:  # e.g. matplotlib built for NumPy 1.x under NumPy 2
        print(f"(Confusion-matrix image skipped: {type(error).__name__}. "
              f"Fix with: python -m pip install --upgrade matplotlib. "
              f"The matrix is still in the JSON.)")
    return result
