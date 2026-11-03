"""
Final evaluation on the held-out test set.

Run exactly once, at the end of the project. Does NOT log to experiments.csv:
test results are not an experiment row, they are the final verdict. All
hyperparameter and model decisions were made on validation; test is evaluated
only after those decisions were frozen.

Loads the run 002 DistilBERT artifact from outputs/models/distilbert_run_002/,
evaluates it on test, fits the TF-IDF baseline on train, evaluates the same
baseline on test, and writes a report to outputs/final_evaluation.txt.

Run from project root:
    python -m src.final_evaluation
"""

from datetime import datetime

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from src.config import (
    CLASS_NAMES,
    FINAL_DISTILBERT_DIR,
    OUTPUTS,
)
from src.data import load_train_val_test
from src.baseline import fit_tfidf_baseline, predict_tfidf_baseline
from src.model import predict_distilbert

# Reference numbers from the experiment log (runs 001 and 002), included so
# the final verdict sits alongside the validation numbers that drove model
# selection. Transcribed, not recomputed — val is on record, test is sacred.
VAL_DISTILBERT_MACRO_F1 = 0.9427
VAL_TFIDF_MACRO_F1 = 0.9102
VAL_DISTILBERT_PROCEDURE_F1 = 0.9114
VAL_TFIDF_PROCEDURE_F1 = 0.78

PROC_IDX = CLASS_NAMES.index("Procedure")


if __name__ == "__main__":
    if not FINAL_DISTILBERT_DIR.exists():
        print(f"ERROR: artifact directory not found: {FINAL_DISTILBERT_DIR}")
        print("Expected HuggingFace Trainer to have saved model + tokenizer here.")
        raise SystemExit(1)

    # Load the run 002 artifact. HuggingFace Trainer saves a directory,
    # not a .pkl — config.json, model.safetensors, tokenizer files.
    tokenizer = AutoTokenizer.from_pretrained(str(FINAL_DISTILBERT_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(FINAL_DISTILBERT_DIR)
    )
    artifact = {"tokenizer": tokenizer, "model": model}

    # df_val is unpacked to underscore: not used here.
    df_train, _, df_test = load_train_val_test()
    y_test = df_test["label"]

    # ---- DistilBERT predictions on test ----
    distilbert_preds = predict_distilbert(artifact, df_test)

    distilbert_test_macro_f1 = f1_score(
        y_test, distilbert_preds, average="macro", zero_division=0
    )
    distilbert_test_accuracy = accuracy_score(y_test, distilbert_preds)
    distilbert_per_class = f1_score(
        y_test, distilbert_preds, average=None, zero_division=0
    )
    distilbert_procedure_f1 = distilbert_per_class[PROC_IDX]

    # ---- TF-IDF baseline: fit on train, predict on test ----
    tfidf_baseline = fit_tfidf_baseline(df_train, class_weight="balanced")
    tfidf_preds = predict_tfidf_baseline(tfidf_baseline, df_test)

    tfidf_test_macro_f1 = f1_score(
        y_test, tfidf_preds, average="macro", zero_division=0
    )
    tfidf_per_class = f1_score(
        y_test, tfidf_preds, average=None, zero_division=0
    )
    tfidf_procedure_f1 = tfidf_per_class[PROC_IDX]

    lift = distilbert_test_macro_f1 - tfidf_test_macro_f1

    # ---- Build the report ----
    lines: list[str] = []
    def add(line: str = "") -> None:
        lines.append(line)

    add("Final Evaluation Report")
    add("=" * 60)
    add(f"Date: {datetime.now().isoformat(timespec='seconds')}")
    add("Model: Run 002 — DistilBERT 3ep "
        "(lr=5e-5, max_len=128, class-weighted CE)")
    add("Baseline: TF-IDF + Logistic Regression (class_weight='balanced')")
    add("Split: train=3,389  val=848  test=4,241")
    add("Test evaluated once at end of project. No decisions made from test.")
    add()
    add("Test macro-F1:")
    add(f"    DistilBERT (Run 002):      {distilbert_test_macro_f1:.4f}")
    add(f"    TF-IDF + LR (baseline):    {tfidf_test_macro_f1:.4f}")
    add(f"    Lift over baseline:        {lift:+.4f} points")
    add()
    add(f"Test accuracy (DistilBERT):    {distilbert_test_accuracy:.4f}")
    add()
    add("Validation macro-F1 (for comparison — from experiment log):")
    add(f"    DistilBERT (Run 002):      {VAL_DISTILBERT_MACRO_F1:.4f}")
    add(f"    TF-IDF + LR (baseline):    {VAL_TFIDF_MACRO_F1:.4f}")
    add()
    add("Procedure F1 (primary improvement target):")
    add(f"    Test (DistilBERT):         {distilbert_procedure_f1:.4f}")
    add(f"    Test (TF-IDF baseline):    {tfidf_procedure_f1:.4f}")
    add(f"    Val (DistilBERT):          {VAL_DISTILBERT_PROCEDURE_F1:.4f}")
    add(f"    Val (TF-IDF):              {VAL_TFIDF_PROCEDURE_F1:.4f}")
    add()
    add("Per-class F1 (test) — DistilBERT (Run 002):")
    add(classification_report(
        y_test, distilbert_preds,
        target_names=CLASS_NAMES,
        zero_division=0,
        digits=4,
    ))
    add("Per-class F1 (test) — TF-IDF baseline:")
    add(classification_report(
        y_test, tfidf_preds,
        target_names=CLASS_NAMES,
        zero_division=0,
        digits=4,
    ))

    report = "\n".join(lines)

    out_path = OUTPUTS / "final_evaluation.txt"
    out_path.write_text(report)
    print(f"Report written to {out_path}\n")
    print(report)