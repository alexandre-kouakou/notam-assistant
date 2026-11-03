"""
Baseline models for NOTAM classification.

Two baselines:
1. Majority class — the trivial floor (always predicts the most common label).
2. TF-IDF + Logistic Regression — the real baseline any trained model must beat.

Both follow a fit/predict contract: fit_* learns from a training DataFrame,
predict_* scores any DataFrame. No file I/O, no data loading, no side effects.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.config import RANDOM_SEED


def fit_majority_baseline(df_train: pd.DataFrame) -> dict:
    """
    Learn the most frequent label in the training set.

    Parameters
    ----------
    df_train : DataFrame with a 'label' column.

    Returns
    -------
    dict with key 'majority_class' (int class id).
    """
    majority_class = int(df_train["label"].value_counts().idxmax())

    return {"majority_class": majority_class}


def predict_majority_baseline(baseline: dict, df: pd.DataFrame) -> np.ndarray:
    """
    Predict the majority class for every row.

    Parameters
    ----------
    baseline : output of fit_majority_baseline
    df : DataFrame with the same row count as the desired prediction array.

    Returns
    -------
    np.ndarray of shape (len(df),), every element equal to the majority class.
    """
    return np.full(len(df), baseline["majority_class"])


def fit_tfidf_baseline(
    df_train: pd.DataFrame, class_weight: str | None = "balanced"
) -> dict:
    """
    Fit a TF-IDF vectorizer and a Logistic Regression classifier.

    The vectorizer is fit on df_train['text'] ONLY. This is the leakage
    defense: validation and test text must never influence the vocabulary.

    Parameters
    ----------
    df_train : DataFrame with 'text' and 'label' columns.
    class_weight : passed to LogisticRegression. 'balanced' weights rare
        classes more heavily, which matters for macro-F1 on imbalanced data.

    Returns
    -------
    dict with keys 'vectorizer' and 'classifier'.
    """
    vectorizer = TfidfVectorizer()

    X_train = vectorizer.fit_transform(df_train["text"])

    classifier = LogisticRegression(
        random_state = RANDOM_SEED,
        max_iter = 1000,
        class_weight = class_weight
    )

    classifier.fit(X_train, df_train["label"])

    return {"vectorizer": vectorizer, "classifier": classifier}


def predict_tfidf_baseline(baseline: dict, df: pd.DataFrame) -> np.ndarray:
    """
    Predict class ids for raw text.

    Uses .transform() (not .fit_transform()) — the vectorizer's vocabulary
    is frozen from training.

    Parameters
    ----------
    baseline : output of fit_tfidf_baseline
    df : DataFrame with a 'text' column.

    Returns
    -------
    np.ndarray of shape (len(df),), predicted class ids.
    """
    X_test = baseline["vectorizer"].transform(df["text"])

    return baseline["classifier"].predict(X_test)