"""Logistic Regression on pooled MFCC features (proposal model 1, the simple baseline).

Run from the repo root after scripts/prepare_data.py:
    python models/logreg.py

The regularisation strength C is chosen on the validation split (macro F1);
the test split is used once, for the final score.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import evaluate, load_processed, split_arrays

MODEL_NAME = "logreg"
C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]


def build(c, seed):
    # StandardScaler lives inside the pipeline, so it is fit on training data only.
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c, max_iter=5000, class_weight="balanced", random_state=seed),
    )


def main():
    data, config = load_processed()
    parts = split_arrays(data, "mfcc_pooled")
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = parts["train"], parts["val"], parts["test"]

    best_c, best_f1 = None, -1.0
    for c in C_GRID:
        model = build(c, config["seed"]).fit(x_train, y_train)
        f1 = f1_score(y_val, model.predict(x_val), average="macro", zero_division=0)
        print(f"C={c:<7} val macro F1 = {f1:.3f}")
        if f1 > best_f1:
            best_c, best_f1 = c, f1

    model = build(best_c, config["seed"]).fit(x_train, y_train)
    evaluate(MODEL_NAME, y_test, model.predict(x_test), config,
             extra={"best_params": {"C": best_c}, "val_macro_f1": best_f1})


if __name__ == "__main__":
    main()
