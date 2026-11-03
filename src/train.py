"""
Phase 3: baseline evaluation.

Trains and scores the two baselines on validation. Reports macro-F1 for:
  - majority class (the floor)
  - TF-IDF + LR with class_weight='balanced'
  - TF-IDF + LR with class_weight=None

The test set is not loaded. Validation only.
"""

import joblib
import numpy as np
from sklearn.metrics import f1_score, classification_report

from src.config import CLASS_NAMES, OUTPUTS, MODEL_DIR
from src.data import load_train_val_test
from src.baseline import (
    fit_majority_baseline,
    predict_majority_baseline,
    fit_tfidf_baseline,
    predict_tfidf_baseline,
)

if __name__ == "__main__":
    df_train, df_val, _ = load_train_val_test()
    y_val = df_val["label"]

    # Majority baseline
    majority = fit_majority_baseline(df_train)
    majority_preds = predict_majority_baseline(majority, df_val)
    majority_f1 = f1_score(y_val, majority_preds, average="macro")
    np.save(OUTPUTS / "majority_val_preds.npy", majority_preds.astype(np.int64))

    # TF-IDF + LR (balanced)
    tfidf_baseline = fit_tfidf_baseline(df_train, class_weight="balanced")
    tfidf_balanced_preds = predict_tfidf_baseline(tfidf_baseline, df_val)
    tfidf_balanced_f1 = f1_score(y_val, tfidf_balanced_preds, average="macro")
    np.save(OUTPUTS / "tfidf_balanced_val_preds.npy", tfidf_balanced_preds.astype(np.int64))
    joblib.dump(tfidf_baseline, MODEL_DIR / "tfidf_balanced_baseline.joblib", compress=3)

    # TF-IDF + LR (plain, no class weighting)
    tfidf_plain = fit_tfidf_baseline(df_train, class_weight=None)
    tfidf_plain_preds = predict_tfidf_baseline(tfidf_plain, df_val)
    tfidf_plain_f1 = f1_score(y_val, tfidf_plain_preds, average="macro")
    np.save(OUTPUTS / "tfidf_plain_val_preds.npy", tfidf_plain_preds.astype(np.int64))

    # Summary
    print(f"majority class            macro-F1 = {majority_f1:.4f}")
    print(f"TF-IDF + LR (balanced)    macro-F1 = {tfidf_balanced_f1:.4f}  "
          f"(+{tfidf_balanced_f1 - majority_f1:.4f} vs majority)")
    print(f"TF-IDF + LR (plain)       macro-F1 = {tfidf_plain_f1:.4f}  "
          f"(+{tfidf_plain_f1 - majority_f1:.4f} vs majority)")

    # Per-class breakdown on the balanced variant
    print("\nPer-class report (TF-IDF + LR, balanced):")
    print(classification_report(
        y_val, tfidf_balanced_preds,
        target_names=CLASS_NAMES,
        zero_division=0,
    ))