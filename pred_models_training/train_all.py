"""

Trains and saves joblib artifacts for:
- topic (news coverage)
- intent
- sensationalism (MLP on features)
 
- ./data/pred_data/train2.tsv
- ./data/pred_data/val2.tsv
"""
from __future__ import annotations

import os
import re
import warnings
from typing import Dict, Tuple, List
from sklearn.svm import SVC
from scipy.sparse import hstack


import numpy as np
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from nrclex import NRCLex
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.utils.class_weight import compute_class_weight



import nltk
from nltk.tokenize import sent_tokenize

from .predictors import (
    read_tsv,
    text_of,
    evidence_anchors,
    ARTIFACT_DIR,
    TOPIC_ARTIFACT,
    INTENT_ARTIFACT,
    SENS_ARTIFACT,
    text_of,
    first_subject
)

warnings.filterwarnings("ignore")

def _ensure_dir():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

TOPIC_ARTIFACT = os.path.join(ARTIFACT_DIR, "topic_pipeline.joblib")
def train_topic(
    train_tsv: str = "./data/pred_data/train2.tsv",
    artifact_path: str = TOPIC_ARTIFACT,
    random_state: int = 42,
) -> dict:
    """
    Trains the News Coverage classifier using the 'subjects' column to 
    ensure parity with the original experimental model.
    """
    # 1. Load and Preprocess
    df_tr = read_tsv(train_tsv)
    X_tr = df_tr.apply(text_of, axis=1)
    
    # CRITICAL: Use "subjects" to match your original successful model
    y_tr = df_tr["subjects"].apply(first_subject) 

    # 2. Filter 'unknown' labels
    keep = y_tr.ne("unknown")
    X_tr, y_tr = X_tr[keep], y_tr[keep]

    # 3. Handle Class Imbalance
    classes = np.unique(y_tr)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_tr)
    wmap = {c: w for c, w in zip(classes, weights)}

    # 4. Build Pipeline (Identical Hyperparameters)
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.9,
            sublinear_tf=True,
        )),
        ("clf", LinearSVC(class_weight=wmap, random_state=random_state, dual=True)),
    ])

    # 5. Train and Persist
    pipe.fit(X_tr, y_tr)

    os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
    joblib.dump(pipe, artifact_path)

    return {
        "artifact_path": artifact_path,
        "n_train": int(len(X_tr)),
        "n_classes": int(len(classes)),
        "classes": list(map(str, classes)),
    }

INTENT_ARTIFACT = os.path.join(ARTIFACT_DIR, "intent_proto.joblib")

PROTOS = {
  "inform":   ["Officials said the department released a report with data and timelines."],
  "persuade": ["We should support this policy because it will improve outcomes."],
  "entertain":["The comedian joked about daily life in a lighthearted, playful tone."],
  "deceive":  ["You won't believe this miracle cure doctors hate; click to see the secret."]
}
CLASS_NAMES = ["inform","persuade","entertain","deceive"]

def train_intent(train_path="./data/pred_data/train2.tsv"):
    df = read_tsv(train_path)
    for c in ["statement","context","justification"]:
        df[c] = df[c].fillna("").astype(str).str.strip()

    texts = df.apply(text_of, axis=1).tolist()

    intent_tfidf = TfidfVectorizer(
        lowercase=True, strip_accents="unicode",
        analyzer="word", ngram_range=(1,2),
        min_df=3, max_df=0.95, sublinear_tf=True
    )
    intent_tfidf.fit(texts)

    proto_rows = []
    for name in CLASS_NAMES:
        pv = intent_tfidf.transform(PROTOS[name])
        pv_mean = np.asarray(pv.mean(axis=0))
        pv_norm = normalize(pv_mean, norm="l2", axis=1)
        proto_rows.append(pv_norm.ravel())
    PROTO_MAT = np.vstack(proto_rows)

    joblib.dump(
        {"vectorizer": intent_tfidf, "proto_mat": PROTO_MAT, "class_names": CLASS_NAMES},
        INTENT_ARTIFACT
    )
    return intent_tfidf

SENS_ARTIFACT = os.path.join(ARTIFACT_DIR, "sens_svc.joblib")

TOKEN_RE = re.compile(r"[A-Za-z]+")
EVIDENCE_PATTERNS = [
    r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b",
    r"\b(19|20)\d{2}\b",
    r"%",
    r"https?://",
    r"\"[^\"]+\"",
    r"\baccording to\b",
    r"\breport(ed|s)? by\b|\bstudy\b|\bsurvey\b",
]
EVIDENCE_RE = [re.compile(p, flags=re.I) for p in EVIDENCE_PATTERNS]

def evidence_anchors(text: str) -> float:
    s = str(text)
    hits = sum(len(r.findall(s)) for r in EVIDENCE_RE)
    toks = max(1, len(TOKEN_RE.findall(s)))
    dens = np.log1p(hits) / max(10, toks)
    return float(np.clip(dens, 0.0, 0.2))

SCORE_MAP = {
    "pants-fire": 5,
    "pants on fire": 5,
    "false": 4,
    "barely-true": 3,
    "half-true": 2,
    "mostly-true": 1,
    "true": 0,
}

THRESHOLD = 2.5  # >= => sensational

def train_sensationalism(train_path="./data/pred_data/train2.tsv"):
    df_tr = read_tsv(train_path)

    # Clean columns
    for c in ["statement", "context"]:
        df_tr[c] = df_tr[c].fillna("").astype(str)

    train_text = df_tr["statement"] + " " + df_tr["context"]

    tfidf = TfidfVectorizer(max_features=3000, stop_words="english", ngram_range=(1, 2))
    X_tfidf = tfidf.fit_transform(train_text)

    X_ev = df_tr["statement"].apply(evidence_anchors).values.reshape(-1, 1)
    X = hstack([X_tfidf, X_ev])

    # Map truthfulness label -> score -> binary sensational
    train_scores = df_tr["label"].astype(str).str.lower().map(SCORE_MAP).fillna(0)
    y = (train_scores >= THRESHOLD).astype(int).values

    model = SVC(kernel="linear", C=0.025, class_weight="balanced", probability=True)
    model.fit(X, y)

    joblib.dump(
        {
            "tfidf": tfidf,
            "model": model,
            "threshold": THRESHOLD,
            "score_map": SCORE_MAP,
            "evidence_patterns": EVIDENCE_PATTERNS,
        },
        SENS_ARTIFACT,
    )
    return model

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

if __name__ == "__main__":
    main()
