"""Shared architecture, data validation, training, and evaluation for both CNNs.

The feature archive and config come from the baseline experiment. Checkpoint
selection uses validation macro F1; test predictions are made after selection.
"""

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sklearn
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset

if __package__:
    from .common import evaluate
else:
    from common import evaluate

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))


class BirdCNN(nn.Module):
    """Matched three-block CNNs: only the downsampling operation changes.

    No dense layers or average pooling are used. A convolution over the final
    grid produces one raw class score per species for CrossEntropyLoss.
    """

    def __init__(self, num_classes, input_shape, dropout, use_max_pool):
        super().__init__()
        height, width = input_shape
        if height < 8 or width < 8 or height % 8 or width % 8:
            raise ValueError("Spectrogram dimensions must be positive multiples of eight")
        self.input_shape = tuple(input_shape)
        self.num_classes = num_classes
        self.dropout = dropout
        self.use_max_pool = use_max_pool
        layers = []
        channels = (1, 16, 32, 64)
        for incoming, outgoing in zip(channels, channels[1:]):
            layers.extend([
                nn.Conv2d(incoming, outgoing, 3, stride=1 if use_max_pool else 2, padding=1),
                nn.ReLU(),
            ])
            if use_max_pool:
                layers.append(nn.MaxPool2d(2))
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2d(64, num_classes, kernel_size=(height // 8, width // 8)),
        )

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 1 or tuple(x.shape[2:]) != self.input_shape:
            raise ValueError(f"Expected [batch, 1, {self.input_shape[0]}, {self.input_shape[1]}]")
        return self.classifier(self.features(x)).flatten(1)


@dataclass
class PreparedData:
    x: torch.Tensor
    y: torch.Tensor
    split: np.ndarray
    recording_ids: np.ndarray
    config: dict
    feature_sha256: str
    assignment_sha256: str

    def indices(self, split):
        return np.flatnonzero(self.split == split)

    def summary(self):
        return {
            split: {"clips": int((self.split == split).sum()),
                    "recordings": len(set(self.recording_ids[self.split == split]))}
            for split in ("train", "val", "test")
        }


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_prepared(processed):
    config = json.loads((processed / "config.json").read_text())
    feature_path = processed / "features.npz"
    if not feature_path.is_file():
        raise ValueError(f"Put the shared features.npz in {processed}; do not regenerate the baseline splits")
    with np.load(feature_path, allow_pickle=False) as archive:
        mel = np.array(archive["logmel"], dtype=np.float32, copy=True)
        labels = archive["y"]
        splits = archive["split"]
        recording_ids = archive["recording_id"]
        mean, std = float(archive["mel_mean"]), float(archive["mel_std"])
    classes = len(config["label_map"])
    if set(config["label_map"].values()) != set(range(classes)):
        raise ValueError("Species labels must be consecutive integers starting at zero")
    if config["n_species"] != classes:
        raise ValueError("Species count disagrees with the label map")
    if mel.ndim != 3 or tuple(mel.shape[1:]) != tuple(config["logmel_shape"]):
        raise ValueError("Spectrogram shape disagrees with config.json")
    if any(a.shape != (len(mel),) for a in [labels, splits, recording_ids]):
        raise ValueError("Features, labels, recording IDs, and split assignments do not align")
    if labels.dtype.kind not in "iu" or set(labels.tolist()) != set(range(classes)):
        raise ValueError("Archive labels disagree with config.json")
    if set(splits.tolist()) != {"train", "val", "test"}:
        raise ValueError("Expected nonempty train, val, and test splits")
    if not np.isfinite(mel).all() or not np.isfinite([mean, std]).all() or std <= 0:
        raise ValueError("Spectrograms and normalization statistics must be finite, with std > 0")
    if mean != config["mel_mean"] or std != config["mel_std"]:
        raise ValueError("Normalization statistics disagree with config.json")
    assignments = {}
    for recording, label, split in zip(recording_ids, labels, splits):
        assignment = (int(label), str(split))
        if recording in assignments and assignments[recording] != assignment:
            raise ValueError(f"Recording has conflicting labels or crosses splits: {recording}")
        assignments[recording] = assignment
    for split in ("train", "val", "test"):
        if set(labels[splits == split].tolist()) != set(range(classes)):
            raise ValueError(f"Missing a species in the {split} split")
    assignment_bytes = json.dumps(
        [[str(r), int(y), str(s)] for r, y, s in zip(recording_ids, labels, splits)],
        separators=(",", ":"),
    ).encode()
    # These statistics came from TRAIN clips in Rakan's preparation run.
    mel -= mean
    mel /= std
    return PreparedData(
        torch.from_numpy(mel[:, None]), torch.from_numpy(labels.astype(np.int64)),
        splits, recording_ids, config, file_sha256(feature_path),
        hashlib.sha256(assignment_bytes).hexdigest(),
    )


def choose_device(requested):
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable in this session; use --device cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; use --device cpu or auto")
    return torch.device(requested)


def make_loader(data, split, batch_size, seed):
    dataset = Subset(TensorDataset(data.x, data.y), data.indices(split).tolist())
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=split == "train",
                      num_workers=0, generator=generator)


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, total = 0.0, 0
    truth, predictions = [], []
    with torch.set_grad_enabled(training):
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Training produced a non-finite loss")
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(labels)
            total += len(labels)
            truth.extend(labels.detach().cpu().tolist())
            predictions.extend(logits.detach().argmax(1).cpu().tolist())
    return {
        "loss": total_loss / total,
        "accuracy": float(accuracy_score(truth, predictions)),
        "macro_f1": float(f1_score(truth, predictions, labels=range(model.num_classes),
                                   average="macro", zero_division=0)),
    }


def write_history(history, path):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def plot_history(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for split in ("train", "val"):
        axes[0].plot(epochs, [row[f"{split}_loss"] for row in history], label=split)
        axes[1].plot(epochs, [row[f"{split}_macro_f1"] for row in history], label=split)
    for ax, title in zip(axes, ["Cross-entropy loss", "Macro F1"]):
        ax.set(xlabel="Epoch", title=title)
        ax.legend()
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def test_checkpoint(checkpoint, data, device, results_dir, batch_size=32):
    """Evaluate a selected checkpoint without training or changing its weights."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload["feature_sha256"] != data.feature_sha256 or payload["config"] != data.config:
        raise ValueError("Checkpoint was trained with different features or configuration")
    model = BirdCNN(**payload["architecture"]).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    predicted, confidence = [], []
    with torch.inference_mode():
        for x, _ in make_loader(data, "test", batch_size, payload["seed"]):
            probabilities = model(x.to(device)).softmax(1)
            values, labels = probabilities.max(1)
            predicted.extend(labels.cpu().tolist())
            confidence.extend(values.cpu().tolist())
    indices = data.indices("test")
    truth = data.y[indices].numpy()
    model_name = payload["model"]
    result = evaluate(model_name, truth, np.asarray(predicted), data.config,
                      extra=payload["run_summary"], results_dir=results_dir)
    with (checkpoint.parent / f"{model_name}_test_predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["clip_index", "recording_id", "true_label", "predicted_label", "confidence"])
        for index, expected, actual, score in zip(indices, truth, predicted, confidence):
            writer.writerow([int(index), data.recording_ids[index], int(expected), actual, score])
    return result


def run(model_class, model_name):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--results", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--checkpoints", type=Path, default=PROJECT_ROOT / "checkpoints")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--train-only", action="store_true", help="Save a checkpoint without evaluating test clips")
    parser.add_argument("--evaluate-only", action="store_true", help="Load the saved checkpoint and evaluate")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing training run")
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.batch_size, args.threads) < 1 or args.lr <= 0:
        parser.error("Epochs, patience, batch size, threads, and learning rate must be positive")
    if args.weight_decay < 0 or not 0 <= args.dropout < 1:
        parser.error("Weight decay must be nonnegative and dropout must be in [0, 1)")
    if args.train_only and args.evaluate_only:
        parser.error("Choose --train-only or --evaluate-only")
    args.results.mkdir(parents=True, exist_ok=True)
    args.checkpoints.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoints / f"{model_name}_best.pt"
    if checkpoint.exists() and not (args.overwrite or args.evaluate_only):
        parser.error(f"{checkpoint} exists; use --evaluate-only or explicitly --overwrite")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = choose_device(args.device)
    data = load_prepared(args.processed)
    print(f"Device: {device}; data: {data.summary()}", flush=True)
    print(f"Feature SHA256: {data.feature_sha256}", flush=True)
    if args.evaluate_only:
        test_checkpoint(checkpoint, data, device, args.results, args.batch_size)
        return
    model = model_class(len(data.config["label_map"]), tuple(data.config["logmel_shape"]), args.dropout).to(device)
    parameters = sum(p.numel() for p in model.parameters())
    print(f"{model_name}: {parameters:,} trainable parameters", flush=True)
    train_loader = make_loader(data, "train", args.batch_size, args.seed)
    val_loader = make_loader(data, "val", args.batch_size, args.seed)
    train_labels = data.y[data.indices("train")]
    counts = torch.bincount(train_labels, minlength=model.num_classes).float()
    weights = (len(train_labels) / (model.num_classes * counts)).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    versions = {"python": platform.python_version(), "torch": str(torch.__version__),
                "numpy": np.__version__, "scikit_learn": sklearn.__version__}
    history, best_f1, best_epoch = [], -1.0, 0
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        tick = time.perf_counter()
        train = run_epoch(model, train_loader, criterion, device, optimizer)
        val = run_epoch(model, val_loader, criterion, device)
        scheduler.step(val["macro_f1"])
        history.append({"epoch": epoch, **{f"train_{k}": v for k, v in train.items()},
                        **{f"val_{k}": v for k, v in val.items()},
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "seconds": time.perf_counter() - tick})
        write_history(history, args.checkpoints / f"{model_name}_training.csv")
        improved = val["macro_f1"] > best_f1
        if improved:
            best_f1, best_epoch = val["macro_f1"], epoch
            payload = {
                "model": model_name, "seed": args.seed, "config": data.config,
                "architecture": {"num_classes": model.num_classes, "input_shape": list(model.input_shape),
                                 "dropout": model.dropout, "use_max_pool": model.use_max_pool},
                "model_state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "feature_sha256": data.feature_sha256,
            }
            torch.save(payload, checkpoint)
        print(f"{model_name} epoch {epoch:02d}/{args.epochs}: train loss={train['loss']:.4f}, "
              f"val loss={val['loss']:.4f}, val F1={val['macro_f1']:.4f}, "
              f"best={best_f1:.4f} at {best_epoch}, {history[-1]['seconds']:.1f}s", flush=True)
        if epoch - best_epoch >= args.patience:
            print(f"Early stopping after {args.patience} epochs without a new validation best", flush=True)
            break
    summary = {
        "val_macro_f1": best_f1, "best_epoch": best_epoch, "epochs_run": len(history),
        "parameters": parameters, "device": str(device), "training_seconds": time.perf_counter() - started,
        "feature_sha256": data.feature_sha256, "assignment_sha256": data.assignment_sha256,
        "split_counts": data.summary(), "evaluation_level": "clip", "versions": versions,
        "training_settings": {"max_epochs": args.epochs, "patience": args.patience,
                              "batch_size": args.batch_size, "learning_rate": args.lr,
                              "weight_decay": args.weight_decay, "dropout": args.dropout,
                              "seed": args.seed, "class_weighting": "balanced, training clips only",
                              "optimizer": "Adam", "scheduler": "ReduceLROnPlateau(factor=0.5, patience=3)"},
    }
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["run_summary"] = summary
    torch.save(payload, checkpoint)
    (args.checkpoints / f"{model_name}_run.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_history(history, args.results / f"{model_name}_learning_curves.png")
    print(f"Saved best checkpoint: {checkpoint}", flush=True)
    if not args.train_only:
        test_checkpoint(checkpoint, data, device, args.results, args.batch_size)

