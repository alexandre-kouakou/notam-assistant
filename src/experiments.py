"""
Experiment log for the NOTAM classifier.

Each call to log_experiment appends one row to outputs/experiments.csv.
Run IDs are integers, monotonically increasing across the file's lifetime.

Design notes:
  - Required fields are the ones every run has: run_id, timestamp, approach,
    config, baseline_val_macro_f1, notes.
  - Optional fields are passed as **optional_metrics. This lets run 001
    (Phase 3 baseline, no train metric) and runs 004/005 (LLM, no train
    metric) write rows with blank cells for the metrics they don't have,
    without a special case per approach.
  - The per-class F1 columns (val_procedure_f1, val_commradar_f1,
    val_parking_f1) are the ones the reframed target cares about. Aggregate
    macro-F1 alone could hide a transformer that scores well overall while
    dropping Procedure, so those classes are logged explicitly.
  - baseline_val_macro_f1 is logged on every row so lift is computable from
    the CSV alone, without cross-referencing any other file.
  - FIELDS defines the canonical column order. Every row is reindexed against
    it before write, so appended rows cannot silently misalign with the
    existing header. This is the fix for the class of bug where an appended
    row has a different key set and pandas writes it in that row's order.
"""

import pandas as pd
from datetime import datetime

from src.config import OUTPUTS


# Canonical column order. Every row is written against this schema.
FIELDS = [
    "run_id",
    "timestamp",
    "approach",
    "config",
    "baseline_val_macro_f1",
    "train_macro_f1",
    "val_macro_f1",
    "val_procedure_f1",
    "val_commradar_f1",
    "val_parking_f1",
    "notes",
]


def log_experiment(
    approach: str,
    config: str,
    baseline_val_macro_f1: float,
    notes: str = "",
    **optional_metrics,
) -> int:
    """
    Append one experiment row to outputs/experiments.csv.

    Parameters
    ----------
    approach : str
        Short model identifier, e.g. "TF-IDF + LR (balanced)" or
        "DistilBERT fine-tuned".
    config : str
        Short description of the run's configuration: hyperparameters,
        prompt variant, epoch count, learning rate. Must be specific enough
        that a reader could reproduce the run from this string plus the
        git commit it corresponds to.
    baseline_val_macro_f1 : float
        Validation macro-F1 of the reference baseline for this project
        (TF-IDF + LR, balanced, 0.9102). Logged on every row so lift is
        computable from the CSV alone.
    notes : str
        Free-text context. Should say what the run *tests*, not just what
        the run *is*. "3 epochs, lr=5e-5" belongs in config; "tests whether
        longer training improves the tail without overfitting" belongs in
        notes.
    **optional_metrics :
        Any additional numeric fields. Recognized columns in this project:
            train_macro_f1        — training-set macro-F1 (blank for 001/004/005)
            val_macro_f1          — validation-set macro-F1
            val_procedure_f1      — Procedure class F1 on validation
            val_commradar_f1      — Communication_and_Radar F1 on validation
            val_parking_f1        — Parking class F1 on validation
        Columns missing on a given row are left blank (NaN after read_csv).

    Returns
    -------
    int
        The run_id assigned to this row.
    """
    csv_path = OUTPUTS / "experiments.csv"

    # Determine next run_id
    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        next_run_id = int(existing["run_id"].max()) + 1
    else:
        next_run_id = 1

    # Required fields, present on every row
    row = {
        "run_id": next_run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "approach": approach,
        "config": config,
        "baseline_val_macro_f1": baseline_val_macro_f1,
        "notes": notes,
    }

    # Optional metrics — only the ones the caller passed
    row.update(optional_metrics)

    # Reindex against the canonical schema so column order is fixed regardless
    # of which optional metrics were supplied. Missing keys become NaN.
    record = pd.DataFrame([row]).reindex(columns=FIELDS)

    write_header = not csv_path.exists()
    record.to_csv(csv_path, mode="a", header=write_header, index=False)

    return next_run_id