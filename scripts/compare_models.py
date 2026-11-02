"""Compare saved baseline and CNN clip-level test results without retraining."""

import argparse
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
MODELS = [("logreg", "Logistic regression"), ("rf", "Random forest"),
          ("cnnmax", "CNN with max pooling"), ("cnn", "CNN without pooling")]


def compare(results_dir):
    results = {key: json.loads((results_dir / f"{key}_results.json").read_text()) for key, _ in MODELS}
    reference = results["logreg"]
    for key, result in results.items():
        if result["classes"] != reference["classes"] or result["test_clips"] != reference["test_clips"]:
            raise ValueError(f"{key}: species or test clip count differs from the baseline")
        for species in reference["classes"]:
            if result["per_class"][species]["support"] != reference["per_class"][species]["support"]:
                raise ValueError(f"{key}: test support differs for {species}")
    if results["cnn"]["feature_sha256"] != results["cnnmax"]["feature_sha256"]:
        raise ValueError("CNNs were trained on different feature archives")
    if results["cnn"]["assignment_sha256"] != results["cnnmax"]["assignment_sha256"]:
        raise ValueError("CNN recording assignments differ")

    rows = []
    for key, title in MODELS:
        result = results[key]
        rows.append({"model": key, "name": title, "test_clips": result["test_clips"],
                     "accuracy": result["accuracy"], "macro_f1": result["macro_f1"],
                     "val_macro_f1": result["val_macro_f1"],
                     "parameters": result.get("parameters"), "best_epoch": result.get("best_epoch"),
                     "epochs_run": result.get("epochs_run"),
                     "training_seconds": result.get("training_seconds"),
                     "source": "local CNN run" if key in {"cnn", "cnnmax"} else "Rakan's saved baseline result"})
    comparison = {
        "evaluation_level": "clip", "test_clips": reference["test_clips"],
        "species": len(reference["classes"]), "models": rows,
        "cnn_feature_sha256": results["cnn"]["feature_sha256"],
        "cnn_assignment_sha256": results["cnn"]["assignment_sha256"],
        "provenance_note": "CNNs use the same supplied archive. Baseline class order, test count, and per-class "
                           "supports match; the saved baseline reports do not contain feature hashes.",
    }

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["Logistic\nregression", "Random\nforest", "CNN with\nmax pooling", "CNN without\npooling"]
    colors = ["#717780", "#a6aab0", "#3274a1", "#8061a6"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, scale, title in [(axes[0], "accuracy", 100, "Test accuracy (%)"),
                                 (axes[1], "macro_f1", 1, "Test macro F1")]:
        values = [row[key] * scale for row in rows]
        bars = ax.bar(names, values, color=colors)
        ax.bar_label(bars, fmt="%.2f" if scale == 100 else "%.3f", padding=3)
        ax.set(title=title, ylim=(0, scale))
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle(f"Bird species classification: {len(reference['classes'])} species, {reference['test_clips']} held-out clips")
    fig.tight_layout()
    fig.savefig(results_dir / "model_comparison.png", dpi=170)
    plt.close(fig)

    lines = ["# Four-model comparison", "",
             f"Clip-level test evaluation on {reference['test_clips']} clips across {len(reference['classes'])} species.", "",
             "| Model | Validation macro F1 | Test accuracy | Test macro F1 |",
             "| --- | ---: | ---: | ---: |"]
    for row in rows:
        lines.append(f"| {row['name']} | {row['val_macro_f1']:.3f} | {row['accuracy']:.2%} | {row['macro_f1']:.3f} |")
    pooled, plain = results["cnnmax"], results["cnn"]
    lines.extend(["", "## CNN experiment", "",
                  "Both CNNs have 355,084 trainable parameters, channels 16/32/64, and the same convolutional "
                  "classification head. The pooled model uses stride-one convolutions followed by max pooling; "
                  "the other uses stride-two convolutions and contains no pooling or dense layers.", "",
                  "Both architectures use the same normalized spectrograms, training-only class weights, seeded "
                  "batch order, optimizer, and per-trial training budget. Each was tried with the same two "
                  "configurations: default (learning rate 0.001, weight decay 0.0001, dropout 0.3) and "
                  "regularized (learning rate 0.0003, weight decay 0.001, dropout 0.5). Checkpoints and "
                  "configurations are selected by validation macro F1. Test results do not select them.", ""])
    for title, result in [("With max pooling", pooled), ("Without pooling", plain)]:
        lines.append(f"- {title}: best epoch {result['best_epoch']} of {result['epochs_run']} completed; "
                     f"training and validation took {result['training_seconds'] / 60:.1f} minutes on {result['device']}.")
        if "tuning" in result:
            lines.append(f"  Selected configuration: `{result['tuning']['selected_trial']}`; "
                         f"both trials together took {result['tuning']['total_training_seconds'] / 60:.1f} minutes.")
    lines.extend(["", f"Max pooling minus no pooling: **{(pooled['accuracy']-plain['accuracy'])*100:+.2f} "
                  f"percentage points** in accuracy and **{pooled['macro_f1']-plain['macro_f1']:+.3f}** macro F1.", "",
                  "## Reproducibility and limits", "",
                  f"- Feature archive SHA256: `{comparison['cnn_feature_sha256']}`.",
                  f"- Ordered clip/recording/label/split SHA256: `{comparison['cnn_assignment_sha256']}`.",
                  "- The archive contains 1,896 training, 398 validation, and 370 test clips from 252/54/54 recordings.",
                  "- No original recording crosses splits. The archived normalization values match config.json.",
                  "- Baseline metrics come from Rakan's existing result files; their class order and per-class test "
                  "counts match the supplied archive. The original baseline reports do not record feature hashes.",
                  "- These are two single-seed runs on one split, not evidence of a statistically significant "
                  "difference or a universal advantage for either architecture.",
                  "- Clip-level scoring weights recordings that supply more clips more heavily. "
                  "It does not measure one prediction per original recording.",
                  "- Exact numerical reproduction across PyTorch versions and hardware is not guaranteed. "
                  "Run settings, package versions, and fingerprints are saved in each CNN result JSON.", "",
                  "## Local artifacts", "",
                  "- `checkpoints/cnnmax_best.pt` and `checkpoints/cnn_best.pt`: selected weights and configuration (Git-ignored).",
                  "- `results/cnnmax_results.json` and `results/cnn_results.json`: test metrics and run metadata.",
                  "- `results/*_learning_curves.png`: training and validation plots.",
                  "- `checkpoints/*_training.csv` and `checkpoints/*_run.json`: detailed local histories and metadata (Git-ignored).",
                  "- `checkpoints/*_test_predictions.csv`: exact held-out clip IDs, true labels, predictions, and confidence (Git-ignored).",
                  "- `results/model_comparison.md` and `.png`: numerical and visual comparison.",
                  "- `checkpoints/experiment_archive/`: preserved intermediate trials and one-off selection helper (Git-ignored).", ""])
    (results_dir / "model_comparison.md").write_text("\n".join(lines))
    print("\n".join(lines[:10]))
    print(f"\nSaved comparison files to {results_dir}")
    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=PROJECT_ROOT / "results")
    compare(parser.parse_args().results)
