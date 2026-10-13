"""
Core sentiment-analysis code, shared by train_model.py, app.py and the tests.

Model: TF-IDF (single words + word pairs) -> Logistic Regression.
It predicts three classes: Negative, Neutral and Positive. If your training data has
no neutral examples, the model simply has two classes and never predicts Neutral.
"""
import json
import os
import pickle
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "sentiment_model.sav")
METRICS_PATH = os.path.join(BASE_DIR, "model", "metrics.json")

TOKEN_RE = re.compile(r"\w+")
MAX_HIGHLIGHT_WORDS = 120

NEGATIVE, NEUTRAL, POSITIVE = 0, 1, 2
CLASS_NAMES = ["Negative", "Neutral", "Positive"]
EMOJI = {"Negative": "😞", "Neutral": "😐", "Positive": "😊"}
LABEL_WORDS = {
    POSITIVE: {"positive", "pos", "1", "good", "true"},
    NEGATIVE: {"negative", "neg", "0", "bad", "false"},
    NEUTRAL: {"neutral", "neu", "mixed"},
}


def build_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b",
            sublinear_tf=True, min_df=1,
        )),
        ("clf", LogisticRegression(max_iter=1000, C=3.0, class_weight="balanced")),
    ])


def to_class(value):
    """'positive'/'pos'/1 -> 2, 'negative'/'neg'/0 -> 0, 'neutral' -> 1, anything else -> None."""
    text = str(value).strip().lower()
    for code, words in LABEL_WORDS.items():
        if text in words:
            return code
    return None


def load_dataset(path, text_col="text", label_col="label"):
    df = pd.read_csv(path)
    df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
    df["label"] = df["label"].map(to_class)
    return df.dropna().astype({"label": int}).reset_index(drop=True)


def train(df, test_size=0.2, seed=42):
    """Train on a DataFrame with 'text' and 'label' (0, 1 or 2). Returns (model, metrics)."""
    if not pd.api.types.is_numeric_dtype(df["label"]):   # accept words ("positive") as well as codes
        df = df.assign(label=df["label"].map(to_class)).dropna(subset=["label"])
        df["label"] = df["label"].astype(int)
    present = set(df["label"])
    if not {NEGATIVE, POSITIVE} <= present:
        raise ValueError("The data needs both positive and negative examples.")
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=test_size, stratify=df["label"], random_state=seed
    )
    model = build_pipeline().fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred, average="macro")),
        "confusion": confusion_matrix(y_test, pred, labels=[0, 1, 2]).tolist(),
        "classes": [CLASS_NAMES[c] for c in model.classes_],
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    return model, metrics


def save(model, metrics):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)


def load_model(path=MODEL_PATH):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_metrics(path=METRICS_PATH):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def class_probabilities(model, texts):
    """Array of shape (n, 3): probability of Negative, Neutral, Positive for each text."""
    raw = model.predict_proba(list(texts))
    out = np.zeros((len(raw), 3))
    out[:, model.classes_] = raw          # a class the model never saw stays at 0
    return out


def label_from_probabilities(probs):
    """The most likely class, as (name, emoji)."""
    name = CLASS_NAMES[int(np.argmax(probs))]
    return name, EMOJI[name]


def sentiment_score(model, texts):
    """One number per text: how much more positive than negative the model thinks it is."""
    scores = model.decision_function(list(texts))
    if scores.ndim == 1:                  # two-class model: score of the positive class
        return scores
    classes = list(model.classes_)
    return scores[:, classes.index(POSITIVE)] - scores[:, classes.index(NEGATIVE)]


def word_contributions(model, text):
    """
    Which words pushed the result? Each word is removed in turn and we see how much the
    positive-vs-negative score changes. Positive value = pushed towards positive, negative =
    towards negative. Returns [(word, value)] in text order, or None if the text is too long.
    """
    spans = [(m.start(), m.end(), m.group(0)) for m in TOKEN_RE.finditer(text)]
    if not spans or len(spans) > MAX_HIGHLIGHT_WORDS:
        return None
    variants = [text] + [text[:s] + " " + text[e:] for s, e, _ in spans]
    scores = sentiment_score(model, variants)
    return [(word, float(scores[0] - scores[i + 1])) for i, (_, _, word) in enumerate(spans)]


def top_words(model, n=12):
    """Most positive and most negative words/phrases learned by the model."""
    names = np.array(model.named_steps["tfidf"].get_feature_names_out())
    coef = model.named_steps["clf"].coef_
    classes = list(model.classes_)
    axis = coef[0] if coef.shape[0] == 1 else coef[classes.index(POSITIVE)] - coef[classes.index(NEGATIVE)]
    order = np.argsort(axis)
    pos = pd.Series(axis[order[::-1][:n]], index=names[order[::-1][:n]])
    neg = pd.Series(-axis[order[:n]], index=names[order[:n]])
    return pos, neg
