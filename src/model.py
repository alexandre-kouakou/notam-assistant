"""
Non-baseline models for NOTAM classification.

Currently: DistilBERT fine-tuning via HuggingFace Trainer.

Follows the fit/predict contract of src/baseline.py so Phase 4's runner can
treat baselines and models symmetrically:

    artifact = fit_distilbert(df_train, df_val)
    preds    = predict_distilbert(artifact, df_val)

The artifact is a dict holding the tokenizer, the fine-tuned model, the
Trainer (which carries training history), and the output directory. Same
shape-of-thing as fit_tfidf_baseline returning {"vectorizer", "classifier"}.
"""

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from src.config import (
    CLASS_NAMES,
    MAX_TOKEN_LENGTH,
    NUM_CLASSES,
    OUTPUTS,
    RANDOM_SEED,
)

MODEL_CHECKPOINT = "distilbert-base-uncased"

_tokenizer = None


def get_tokenizer():
    """Lazily load and cache the DistilBERT tokenizer."""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    return _tokenizer


def tokenize_function(examples):
    """
    Tokenize a batch of NOTAM texts for DistilBERT.

    Called by Dataset.map(batched=True). Expects a dict with a 'text' key
    containing a list of strings; returns input_ids and attention_mask.
    Padding to MAX_TOKEN_LENGTH happens here, so no data collator is needed
    downstream — Trainer's default collator stacks the padded tensors.
    """
    tokenizer = get_tokenizer()
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_TOKEN_LENGTH,
    )


def _to_hf_dataset(df):
    """
    Convert a pandas DataFrame to a tokenized HuggingFace Dataset.

    Drops the 'text' column after tokenization (no longer needed), renames
    'label' to 'labels' (the name sequence-classification models expect).
    """
    ds = Dataset.from_pandas(
        df[["text", "label"]].reset_index(drop=True),
        preserve_index=False,
    )
    ds = ds.map(tokenize_function, batched=True)
    ds = ds.remove_columns(["text"])
    ds = ds.rename_column("label", "labels")
    return ds


def compute_class_weights(df_train):
    """
    Inverse-frequency class weights, normalized to mean ~1.

    Sklearn's class_weight='balanced' recipe in tensor form: rare classes get
    proportionally more weight in the loss, but the overall scale is kept
    comparable to the unweighted loss so learning rate behaves predictably.
    """
    counts = df_train["label"].value_counts().sort_index().values
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(weights)
    return torch.tensor(weights, dtype=torch.float32)


class WeightedTrainer(Trainer):
    """
    Trainer subclass that applies class weights to the cross-entropy loss.

    Needed because the 13 NOTAM classes are ~4x imbalanced. Without
    weighting, the model optimizes for the large classes and Procedure /
    Parking / Communication_and_Radar — the classes the reframed target is
    about — get neglected.
    """

    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device)
        )
        loss = loss_fct(
            logits.view(-1, model.config.num_labels),
            labels.view(-1),
        )
        return (loss, outputs) if return_outputs else loss


_CLASS_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}


def compute_metrics(eval_pred):
    """
    Macro-F1 plus the three per-class F1s the reframed target tracks.

    Returned from the Trainer so each epoch's eval logs tail-class behavior
    directly — a model that scores well on aggregate while quietly losing
    Procedure is visible during training, not just after.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    per_class = f1_score(labels, preds, average=None, zero_division=0)
    return {
        "macro_f1": float(macro_f1),
        "procedure_f1": float(per_class[_CLASS_INDEX["Procedure"]]),
        "commradar_f1": float(per_class[_CLASS_INDEX["Communication_and_Radar"]]),
        "parking_f1": float(per_class[_CLASS_INDEX["Parking"]]),
    }


def fit_distilbert(
    df_train,
    df_val,
    epochs: int = 3,
    learning_rate: float = 5e-5,
    batch_size: int = 16,
    output_subdir: str = "distilbert_run_002",
):
    """
    Fine-tune DistilBERT on the NOTAM training split.

    Parameters
    ----------
    df_train, df_val : DataFrames with 'text' and 'label' columns.
    epochs : training epochs. 3 is the run-002 config; 1 for a smoke test.
    learning_rate : AdamW learning rate. 5e-5 is standard for BERT-family.
    batch_size : per-device batch size. 16 fits comfortably on MPS at 128 tokens.
    output_subdir : where checkpoints and the final model are saved, under
        outputs/models/. Overwritten on re-run, so pass a different name per
        experiment.

    Returns
    -------
    dict with keys:
        tokenizer  : the loaded tokenizer (needed by predict_distilbert)
        model      : the fine-tuned model, best checkpoint loaded
        trainer    : the Trainer object (holds training history in .state.log_history)
        output_dir : the directory where artifacts were saved
    """
    tokenizer = get_tokenizer()
    train_ds = _to_hf_dataset(df_train)
    val_ds = _to_hf_dataset(df_val)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=NUM_CLASSES,
        id2label={i: n for i, n in enumerate(CLASS_NAMES)},
        label2id={n: i for i, n in enumerate(CLASS_NAMES)},
    )

    output_dir = OUTPUTS / "models" / output_subdir

    # NOTE ON VERSION COMPAT:
    # - eval_strategy was renamed from evaluation_strategy in transformers 4.41.
    #   If you get "unexpected keyword argument 'eval_strategy'", rename it.
    # - tokenizer= in Trainer is deprecated in favor of processing_class= in
    #   transformers 4.40+. It still works with a deprecation warning; swap if
    #   the warning bothers you.
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,          # keeps all 3 epoch checkpoints for load_best
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        seed=RANDOM_SEED,
        report_to="none",
        logging_steps=50,
    )

    class_weights = compute_class_weights(df_train)

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Persist final (best-checkpoint) model + tokenizer so predict_distilbert
    # and future re-evaluations don't need to refit.
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    return {
        "tokenizer": tokenizer,
        "model": trainer.model,
        "trainer": trainer,
        "output_dir": str(output_dir),
    }


def predict_distilbert(artifact, df, batch_size: int = 32):
    """
    Predict class ids for raw text using a fitted DistilBERT artifact.

    Parameters
    ----------
    artifact : output of fit_distilbert.
    df : DataFrame with a 'text' column.
    batch_size : forward-pass batch size. 32 keeps MPS memory modest.

    Returns
    -------
    np.ndarray of shape (len(df),), predicted class ids.
    """
    tokenizer = artifact["tokenizer"]
    model = artifact["model"]

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device).eval()

    texts = df["text"].tolist()
    all_preds = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            padding="max_length",
            truncation=True,
            max_length=MAX_TOKEN_LENGTH,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        all_preds.append(torch.argmax(logits, dim=-1).cpu().numpy())

    return np.concatenate(all_preds)