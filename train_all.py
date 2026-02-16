"""
python -m combined_pred_model.train_all

Trains and saves joblib artifacts for:
- topic (news coverage)
- intent
- sensationalism (MLP on features)
- sentiment (MLP on features)

Expected files (same as notebook):
- ./data/train2.tsv
- ./data/val2.tsv
"""
from __future__ import annotations

import os
import re
import warnings
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

import nltk
from nltk.tokenize import sent_tokenize

from .preditive_models_training.predictors import (
    read_tsv,
    text_of,
    extract_sentiment_features_from_statement,
    create_sentiment_feature_dataframe,
    evidence_anchors,
    ARTIFACT_DIR,
    TOPIC_ARTIFACT,
    INTENT_ARTIFACT,
    SENS_ARTIFACT,
    SENTIMENT_ARTIFACT,
)

warnings.filterwarnings("ignore")

def _ensure_dir():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

def _pick_col(df: pd.DataFrame, candidates: List[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of {candidates} found in columns: {list(df.columns)}")

def train_topic(train_path="./data/train2.tsv", val_path="./data/val2.tsv"):
    df_tr = read_tsv(train_path)
    ycol = _pick_col(df_tr, ["topic", "News Coverage", "news_coverage", "label_topic"])
    x = df_tr.apply(text_of, axis=1).tolist()
    y = df_tr[ycol].astype(str).tolist()

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, strip_accents="unicode", ngram_range=(1,2), max_features=50000)),
        ("clf", LinearSVC(random_state=42)),
    ])
    pipe.fit(x, y)
    joblib.dump(pipe, TOPIC_ARTIFACT)
    return pipe

def train_intent(train_path="./data/train2.tsv", val_path="./data/val2.tsv"):
    df_tr = read_tsv(train_path)
    ycol = _pick_col(df_tr, ["intent", "Intent", "label_intent"])
    # Use title/body when present, else fallback to text_of
    title_col = _pick_col(df_tr, ["title", "Title"]) if any(c in df_tr.columns for c in ["title","Title"]) else None
    body_col = _pick_col(df_tr, ["body", "Body", "content", "text"]) if any(c in df_tr.columns for c in ["body","Body","content","text"]) else None

    texts = []
    for _, r in df_tr.iterrows():
        if title_col and body_col:
            texts.append(f"{str(r[title_col])}\n\n{str(r[body_col])}")
        else:
            texts.append(text_of(r))
    y = df_tr[ycol].astype(str).tolist()

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, strip_accents="unicode", ngram_range=(1,2), max_features=50000)),
        ("clf", LinearSVC(random_state=42)),
    ])
    pipe.fit(texts, y)
    joblib.dump(pipe, INTENT_ARTIFACT)
    return pipe

def _build_feature_matrix(statements: List[str], extra_anchor_texts: List[str] | None = None):
    rows = []
    for i, s in enumerate(statements):
        feats = extract_sentiment_features_from_statement(s)
        if extra_anchor_texts is not None:
            anchors = evidence_anchors(extra_anchor_texts[i])
            feats.update({f"anchor_{k}": float(v) for k, v in anchors.items()})
        rows.append(feats)
    df = pd.DataFrame(rows).fillna(0.0)
    cols = sorted(df.columns)
    X = df[cols].to_numpy(dtype=np.float32)
    return X, cols

def train_sensationalism(train_path="./data/train2.tsv", val_path="./data/val2.tsv"):
    df_tr = read_tsv(train_path)
    ycol = _pick_col(df_tr, ["sensationalism", "Sensationalism", "label_sensationalism"])
    # Use statement/justification if present; else use full text
    stmt_col = _pick_col(df_tr, ["statement", "Statement"]) if any(c in df_tr.columns for c in ["statement","Statement"]) else None
    just_col = _pick_col(df_tr, ["justification", "Justification"]) if any(c in df_tr.columns for c in ["justification","Justification"]) else None

    statements = []
    anchor_texts = []
    for _, r in df_tr.iterrows():
        if stmt_col and just_col:
            s = str(r[stmt_col] or "")
            j = str(r[just_col] or "")
            statements.append(s + " " + j)
            anchor_texts.append(s + " " + j)
        else:
            t = text_of(r)
            statements.append(t)
            anchor_texts.append(t)

    y = df_tr[ycol].astype(str).tolist()
    X, cols = _build_feature_matrix(statements, extra_anchor_texts=anchor_texts)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    clf = MLPClassifier(hidden_layer_sizes=(128, 64), random_state=42, max_iter=500)
    clf.fit(Xs, y)

    joblib.dump({"scaler": scaler, "clf": clf, "cols": cols}, SENS_ARTIFACT)
    return clf

def train_sentiment(train_path="./data/train2.tsv", val_path="./data/val2.tsv"):
    df_tr = read_tsv(train_path)
    ycol = _pick_col(df_tr, ["sentiment", "Sentiment", "label_sentiment"])
    stmt_col = _pick_col(df_tr, ["statement", "Statement"]) if any(c in df_tr.columns for c in ["statement","Statement"]) else None

    statements = []
    for _, r in df_tr.iterrows():
        if stmt_col:
            statements.append(str(r[stmt_col] or ""))
        else:
            statements.append(text_of(r))

    y = df_tr[ycol].astype(str).tolist()
    X, cols = _build_feature_matrix(statements, extra_anchor_texts=None)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    clf = MLPClassifier(hidden_layer_sizes=(128, 64), random_state=42, max_iter=500)
    clf.fit(Xs, y)

    joblib.dump({"scaler": scaler, "clf": clf, "cols": cols}, SENTIMENT_ARTIFACT)
    return clf

def main():
    # NLTK downloads (safe if already present)
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    try:
        nltk.data.find("taggers/averaged_perceptron_tagger")
    except LookupError:
        nltk.download("averaged_perceptron_tagger")

    _ensure_dir()
    print("Training topic...")
    train_topic()
    print("Training intent...")
    train_intent()
    print("Training sensationalism...")
    train_sensationalism()
    print("Training sentiment...")
    train_sentiment()
    print("Done. Artifacts saved to:", ARTIFACT_DIR)

if __name__ == "__main__":
    main()
