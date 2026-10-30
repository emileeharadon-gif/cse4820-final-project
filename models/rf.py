"""Random Forest on the SAME pooled MFCC features (proposal model 2).

Run from the repo root after scripts/prepare_data.py:
    python models/rf.py

Hyperparameters are chosen on the validation split (macro F1); the test split
is used once, for the final score.
"""

import itertools

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from common import evaluate, load_processed, split_arrays

MODEL_NAME = "rf"
GRID = {"n_estimators": [100, 300], "max_depth": [None, 20]}


def build(params, seed):
    return RandomForestClassifier(**params, class_weight="balanced", n_jobs=-1, random_state=seed)


def main():
    data, config = load_processed()
    parts = split_arrays(data, "mfcc_pooled")
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = parts["train"], parts["val"], parts["test"]

    best, best_f1 = None, -1.0
    for values in itertools.product(*GRID.values()):
        params = dict(zip(GRID, values))
        model = build(params, config["seed"]).fit(x_train, y_train)
        f1 = f1_score(y_val, model.predict(x_val), average="macro", zero_division=0)
        print(f"{params}  val macro F1 = {f1:.3f}")
        if f1 > best_f1:
            best, best_f1 = params, f1

    model = build(best, config["seed"]).fit(x_train, y_train)
    evaluate(MODEL_NAME, y_test, model.predict(x_test), config,
             extra={"best_params": best, "val_macro_f1": best_f1})


if __name__ == "__main__":
    main()
