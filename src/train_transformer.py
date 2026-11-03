"""
Phase 4: DistilBERT fine-tuning for NOTAM classification.

Multiple experiments share this runner. Each invocation takes a run label
and epoch count, writes to its own output directory, and logs its own row.

    python -m src.train_transformer --run-label 002 --epochs 3
    python -m src.train_transformer --run-label 003 --epochs 5
    python -m src.train_transformer --smoke

The model architecture, tokenizer, data loading, and metrics are shared
across all runs. Only epochs, output path, and log notes vary per label.
"""

import argparse

from sklearn.metrics import classification_report, f1_score

from src.config import CLASS_NAMES
from src.data import load_train_val_test
from src.experiments import log_experiment
from src.model import fit_distilbert, predict_distilbert

BASELINE_VAL_MACRO_F1 = 0.9102


def _run_config_string(epochs: int) -> str:
    return (
        f"distilbert-base-uncased; {epochs} epochs; lr=5e-5; batch=16; "
        f"max_len=128; class-weighted CE; best-by-val-macro-F1"
    )


def _run_notes_string(label: str, epochs: int) -> str:
    if label == "003":
        return (
            "Run 003: 5-epoch fine-tune to test whether longer training "
            "improves tail classes further. Epoch count change also lengthens "
            "the LR decay window (5e-5 → 0 over 5 epochs vs 3), so comparison "
            "is end-state 5-epoch vs 3-epoch, not purely more steps at same LR."
        )
    return (
        "Run 002: first transformer run. Tests whether fine-tuning "
        "improves tail classes (Procedure, Communication_and_Radar, "
        "Parking) over TF-IDF + LR."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-label", default="002",
                        help="Short label, used in output dir and log. e.g. 002")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Training epochs. Default 3 (run 002 config).")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: 1 epoch, no logging.")
    args = parser.parse_args()

    smoke = args.smoke
    epochs = 1 if smoke else args.epochs
    label = "smoke" if smoke else args.run_label
    output_subdir = f"distilbert_run_{label}"

    df_train, df_val, _ = load_train_val_test()
    y_val = df_val["label"]
    y_train = df_train["label"]

    print(f"[{'SMOKE' if smoke else label}] training for {epochs} epoch(s)")
    artifact = fit_distilbert(
        df_train, df_val,
        epochs=epochs,
        learning_rate=5e-5,
        batch_size=16,
        output_subdir=output_subdir,
    )

    train_preds = predict_distilbert(artifact, df_train)
    val_preds = predict_distilbert(artifact, df_val)

    train_macro_f1 = f1_score(y_train, train_preds, average="macro", zero_division=0)
    val_macro_f1 = f1_score(y_val, val_preds, average="macro", zero_division=0)
    per_class = f1_score(y_val, val_preds, average=None, zero_division=0)

    idx = {n: i for i, n in enumerate(CLASS_NAMES)}
    proc_f1 = per_class[idx["Procedure"]]
    cr_f1 = per_class[idx["Communication_and_Radar"]]
    park_f1 = per_class[idx["Parking"]]

    print(f"\n===== Run {label} results =====")
    print(f"  Train macro-F1:      {train_macro_f1:.4f}")
    print(f"  Val   macro-F1:      {val_macro_f1:.4f}")
    print(f"  Val   Procedure F1:  {proc_f1:.4f}")
    print(f"  Val   CommRadar F1:  {cr_f1:.4f}")
    print(f"  Val   Parking F1:    {park_f1:.4f}")
    print(f"  Baseline (run 001):  {BASELINE_VAL_MACRO_F1:.4f}")
    print(f"  Lift vs baseline:   {val_macro_f1 - BASELINE_VAL_MACRO_F1:+.4f}")

    print("\nPer-class report (val):")
    print(classification_report(
        y_val, val_preds, target_names=CLASS_NAMES, zero_division=0,
    ))

    if not smoke:
        run_id = log_experiment(
            approach=f"DistilBERT fine-tuned ({epochs} epochs)",
            config=_run_config_string(epochs),
            baseline_val_macro_f1=BASELINE_VAL_MACRO_F1,
            notes=_run_notes_string(label, epochs),
            train_macro_f1=float(train_macro_f1),
            val_macro_f1=float(val_macro_f1),
            val_procedure_f1=float(proc_f1),
            val_commradar_f1=float(cr_f1),
            val_parking_f1=float(park_f1),
        )
        print(f"\nLogged as run_id={run_id}")
    else:
        print("\n[SMOKE] not logged to experiments.csv")