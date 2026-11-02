# Project handoff

Last updated: November 2, 2026. Current work is on `cnn-branch`, based on
Rakan's `rakan-baselines` commit `3ee8d0e`.

## User constraints

- Do not commit, push, merge, or post GitHub comments unless the user explicitly
  changes that instruction. All CNN changes and results are currently local.
- Keep the repository small. Intermediate trials, checkpoints, detailed logs,
  and caches stay in Git-ignored folders.
- Emilee plans to merge Rakan's branch and then this dependent CNN branch into
  main. Preserve the user's branch organization and existing baseline files.

## Current project state

The project compares 12-species bird-call classification using Logistic
Regression and Random Forest on pooled MFCCs, and two CNNs on log-mel
spectrograms. All four models are implemented and have preliminary results.

Rakan added `scripts/prepare_data.py`, `models/common.py`, the two classical
models, and their saved results. His latest commit adds
`data/processed/config.json`. His preprocessing code selects usable species,
splits original recordings before clipping, and extracts both feature types.

Adib's CNN branch adds:

- `models/cnn.py`: strided-convolution CNN without pooling.
- `models/cnnmax.py`: CNN with max pooling.
- `models/cnn_common.py`: matched architecture, validated data loading, training,
  early stopping, checkpoint reload, and evaluation through Rakan's evaluator.
- `requirements-cnn.txt`: pinned dependencies used for local training.
- `scripts/compare_models.py`: builds the final comparison report and figure.
- `tests/test_cnns.py`: architecture, gradients, tiny-batch learning, checkpoint
  reload, data alignment, and leakage/config rejection tests.
- Final CNN metrics, confusion matrices, learning curves, and comparison in
  `results/`. See `results/model_comparison.md` for measured scores and limits.

Both CNNs have 355,084 parameters and matching initial weights. The main
architectural difference is stride-two convolution versus stride-one convolution
followed by max pooling. Both use a convolutional classifier, with no dense or
average-pooling layers. Neither CNN outperformed the saved classical baselines
on this test split. Learning curves show overfitting; do not describe the CNNs as
proven improvements.

## Shared prepared dataset

Rakan supplied the completed feature archive. It was placed locally as
`data/processed/features 1.npz` and renamed to `features.npz` for the loader.
The archive is ignored by Git; teammates must obtain it separately.

- Archive: `data/processed/features.npz` (244,468,157 bytes).
- SHA256: `2c9e4813833708c4913c21c6b4b4d32c6078676605a14e748cda880b3a430e5d`.
- Matching configuration: `data/processed/config.json` (already tracked).
- 12 species, 360 original recordings, 2,664 clips.
- Training: 252 recordings / 1,896 clips.
- Validation: 54 recordings / 398 clips.
- Test: 54 recordings / 370 clips.
- No recording crosses splits; normalization matches the supplied config.
- MFCCs are 40-dimensional; log-mel inputs are 128x216 before the channel axis.

The archive contains features, labels, recording IDs, split assignments, and
normalization statistics. CNNs use it directly without regenerating raw audio.
Baseline class order and every per-class test count agree with this archive;
the older baseline reports do not contain feature hashes. Do not claim their
feature arrays were independently reconstructed or their training rerun here.

## Local runs and outputs

Both CNNs were trained on Apple MPS using the same two validation configurations:

1. Default: Adam, learning rate 0.001, weight decay 0.0001, dropout 0.3.
2. Regularized: learning rate 0.0003, weight decay 0.001, dropout 0.5.

Both use batch size 32, seed 42, balanced training-only class weights, a plateau
scheduler, a 40-epoch ceiling, and early stopping after eight epochs without a
new best validation macro F1. Defaults won validation for both architectures.
Selected epochs: max pooling 2, no pooling 9. Test evaluation followed selection.
These are single-seed, clip-level experiments, not significance tests.

`results/` keeps only final paper-facing CNN artifacts and the existing baseline
artifacts. Detailed training CSVs, run summaries, per-clip predictions, and saved
weights are under ignored `checkpoints/`. Intermediate trials and the one-off
selection helper are preserved in `checkpoints/experiment_archive/`.

## Commands

From the repository root:

```bash
source .venv/bin/activate
python -m pip install -r requirements-cnn.txt
python -m unittest discover -s tests -v
python models/cnnmax.py --evaluate-only
python models/cnn.py --evaluate-only
python scripts/compare_models.py
```

To retrain, use the relevant model script with `--train-only`. Existing
checkpoints require explicit `--overwrite`, or use a separate `--checkpoints`
directory. `--device auto` chooses CUDA, MPS, or CPU. The execution sandbox may
hide MPS; local GPU runs required execution outside it. No network is needed
for training once packages and the feature archive are present.

## Earlier acquisition work

`datadownload/kaggledownload.py` downloads Kaggle metadata and fetches the CSV's
MP3 links; `scripts/inspect_data.py` inventories local files. The October 28
acquisition saved 2,698 of 2,730 recordings across 91 species. Its 32 failures
were one HTTP 404, one HTTP 500, and 30 responses failing the MP3 header check.
A header failure does not prove the original recording is corrupt. The index
had 30 recordings per species before exclusions. The prepared archive supplied
later is now the shared input for modeling.

Do not regenerate that archive for one model and silently compare against the
old test scores. Preserve source recording separation and use validation data
for model choices. The original detailed acquisition handoff is archived locally
under `checkpoints/experiment_archive/agents_oct28.md`.
