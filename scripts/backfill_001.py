"""
One-time backfill: log the Phase 3 TF-IDF baseline as run 001.

The Phase 3 evaluation ran before the experiment log existed. This script
records it as row 001 with the numbers from the Phase 3 terminal output, so
the log has an honest starting point. Do not re-run this script — it will
create a duplicate row.
"""

from src.experiments import log_experiment

if __name__ == "__main__":
    run_id = log_experiment(
        approach="TF-IDF + LR (balanced)",
        config="TfidfVectorizer defaults; LogisticRegression(max_iter=1000, class_weight='balanced')",
        baseline_val_macro_f1=0.0239,   # majority-class floor on validation
        val_macro_f1=0.9102,
        val_procedure_f1=0.78,
        val_commradar_f1=0.87,
        val_parking_f1=0.88,
        notes=(
            "Phase 3 baseline. Aggregate task is largely vocabulary-driven; "
            "residual difficulty is in Procedure, Communication_and_Radar, "
            "Parking. Training-set macro-F1 not logged for this run."
        ),
    )
    print(f"Backfilled run_id={run_id}")