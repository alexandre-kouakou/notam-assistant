"""
Phase 4, run 004: Claude zero-shot classification of NOTAMs.

Sends each validation NOTAM to Claude with only the class names as context.
Scores against the same val labels used by runs 001-003. Test set is never
loaded.

Run from project root:
    python -m src.run_llm --run-label 004
    python -m src.run_llm --run-label 004 --model claude-3-5-haiku-20241022
    python -m src.run_llm --run-label 005 --prompt-variant few_shot
"""

import argparse

import numpy as np
from sklearn.metrics import classification_report, f1_score

from src.config import CLASS_NAMES, OUTPUTS
from src.data import load_train_val_test
from src.experiments import log_experiment
from src.llm_classifier import DEFAULT_MODEL, LLMClassifier

BASELINE_VAL_MACRO_F1 = 0.9102


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Claude model name. Override if the default 404s.")
    parser.add_argument("--run-label", default="004")
    parser.add_argument("--prompt-variant", default="zero_shot",
                        choices=["zero_shot", "few_shot"],
                        help="Prompt variant to use for classification.")
    args = parser.parse_args()


    df_train, df_val, _ = load_train_val_test()
    y_val = df_val["label"].values
    texts = df_val["text"].tolist()

    if args.prompt_variant == "few_shot":
    # Two examples per class, sampled from train only (no val leakage).
    # Random sample to give the LLM typical examples of each class; a
    # hand-curated "hard cases" set would be a different experiment.
        few_shot_examples = {
            CLASS_NAMES[i]: df_train[df_train["label"] == i]["text"]
                .sample(n=2, random_state=42).tolist()
            for i in range(len(CLASS_NAMES))
        }
    else:
        few_shot_examples = None

    cache_path = OUTPUTS / f"llm_cache_{args.run_label}.jsonl"
    classifier = LLMClassifier(
        model=args.model,
        cache_path=cache_path,
        n_workers=8,
        prompt_variant=args.prompt_variant,
        few_shot_examples=few_shot_examples,
    )

    print(f"[{args.run_label}] classifying {len(texts)} val examples via "
          f"Claude ({classifier.model})")
    print(f"  cache: {cache_path} ({len(classifier._cache)} cached)")

    preds = np.array(classifier.predict(texts))

    parse_failures = int((preds == -1).sum())
    if parse_failures:
        print(f"  WARNING: {parse_failures} unparseable responses")

    valid_mask = preds != -1
    if valid_mask.sum() == 0:
        print("\nERROR: no valid predictions. Check "
              "outputs/llm_cache_004.jsonl for cached response contents.")
        raise SystemExit(1)

    y_valid = y_val[valid_mask]
    preds_valid = preds[valid_mask]
    macro_f1 = f1_score(y_valid, preds_valid, average="macro", zero_division=0)
    per_class = f1_score(y_valid, preds_valid, average=None, zero_division=0)
    idx = {n: i for i, n in enumerate(CLASS_NAMES)}

    print(f"\n===== Run {args.run_label} results =====")
    print(f"  Val macro-F1:        {macro_f1:.4f}")
    print(f"  Val Procedure F1:    {per_class[idx['Procedure']]:.4f}")
    print(f"  Val CommRadar F1:    {per_class[idx['Communication_and_Radar']]:.4f}")
    print(f"  Val Parking F1:      {per_class[idx['Parking']]:.4f}")
    print(f"  Baseline (run 001):  {BASELINE_VAL_MACRO_F1:.4f}")
    print(f"  Parse failures:      {parse_failures}/{len(texts)}")

    print("\nPer-class report (val):")
    print(classification_report(
        y_val, preds, target_names=CLASS_NAMES, zero_division=0,
    ))

    run_id = log_experiment(
        approach=f"LLM zero-shot (anthropic)",
        config=(
            f"prompt_variant={args.prompt_variant}; "
            f"{'definitions + 2 train examples per class' if args.prompt_variant == 'few_shot' else 'class names only'}; "
            f"model={classifier.model}"
        ),
        baseline_val_macro_f1=BASELINE_VAL_MACRO_F1,
        notes=(
            f"Run {args.run_label}: zero-shot Claude classification. Tests "
            "whether a general-purpose model without NOTAM-domain fine-tuning "
            "can match TF-IDF (0.91) or DistilBERT (0.94)."
        ),
        val_macro_f1=float(macro_f1),
        val_procedure_f1=float(per_class[idx["Procedure"]]),
        val_commradar_f1=float(per_class[idx["Communication_and_Radar"]]),
        val_parking_f1=float(per_class[idx["Parking"]]),
    )
    print(f"\nLogged as run_id={run_id}")