# Four-model comparison

Clip-level test evaluation on 370 clips across 12 species.

| Model | Validation macro F1 | Test accuracy | Test macro F1 |
| --- | ---: | ---: | ---: |
| Logistic regression | 0.271 | 36.76% | 0.372 |
| Random forest | 0.271 | 34.32% | 0.338 |
| CNN with max pooling | 0.277 | 27.30% | 0.285 |
| CNN without pooling | 0.236 | 32.43% | 0.320 |

## CNN experiment

Both CNNs have 355,084 trainable parameters, channels 16/32/64, and the same convolutional classification head. The pooled model uses stride-one convolutions followed by max pooling; the other uses stride-two convolutions and contains no pooling or dense layers.

Both architectures use the same normalized spectrograms, training-only class weights, seeded batch order, optimizer, and per-trial training budget. Each was tried with the same two configurations: default (learning rate 0.001, weight decay 0.0001, dropout 0.3) and regularized (learning rate 0.0003, weight decay 0.001, dropout 0.5). Checkpoints and configurations are selected by validation macro F1. Test results do not select them.

- With max pooling: best epoch 2 of 10 completed; training and validation took 0.5 minutes on mps.
  Selected configuration: `default`; both trials together took 1.2 minutes.
- Without pooling: best epoch 9 of 17 completed; training and validation took 0.3 minutes on mps.
  Selected configuration: `default`; both trials together took 0.6 minutes.

Max pooling minus no pooling: **-5.14 percentage points** in accuracy and **-0.035** macro F1.

## Reproducibility and limits

- Feature archive SHA256: `2c9e4813833708c4913c21c6b4b4d32c6078676605a14e748cda880b3a430e5d`.
- Ordered clip/recording/label/split SHA256: `3144a172d8e5ebff6ef408aad09c54d273650c17bf1fb6351fa5d0f07f398cc4`.
- The archive contains 1,896 training, 398 validation, and 370 test clips from 252/54/54 recordings.
- No original recording crosses splits. The archived normalization values match config.json.
- Baseline metrics come from Rakan's existing result files; their class order and per-class test counts match the supplied archive. The original baseline reports do not record feature hashes.
- These are two single-seed runs on one split, not evidence of a statistically significant difference or a universal advantage for either architecture.
- Clip-level scoring weights recordings that supply more clips more heavily. It does not measure one prediction per original recording.
- Exact numerical reproduction across PyTorch versions and hardware is not guaranteed. Run settings, package versions, and fingerprints are saved in each CNN result JSON.

## Local artifacts

- `checkpoints/cnnmax_best.pt` and `checkpoints/cnn_best.pt`: selected weights and configuration (Git-ignored).
- `results/cnnmax_results.json` and `results/cnn_results.json`: test metrics and run metadata.
- `results/*_learning_curves.png`: training and validation plots.
- `checkpoints/*_training.csv` and `checkpoints/*_run.json`: detailed local histories and metadata (Git-ignored).
- `checkpoints/*_test_predictions.csv`: exact held-out clip IDs, true labels, predictions, and confidence (Git-ignored).
- `results/model_comparison.md` and `.png`: numerical and visual comparison.
- `checkpoints/experiment_archive/`: preserved intermediate trials and one-off selection helper (Git-ignored).
