"""

Trains and saves joblib artifacts for:
- topic (news coverage)
- intent
- sensationalism (MLP on features)
- sentiment (MLP on features)
 
- ./pred_data/train2.tsv
- ./pred_data/val2.tsv
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
    extract_sentiment_features_from_statement,
    evidence_anchors,
    ARTIFACT_DIR,
    TOPIC_ARTIFACT,
    INTENT_ARTIFACT,
    SENS_ARTIFACT,
    SENTIMENT_ARTIFACT,
    text_of,
    first_subject
)

warnings.filterwarnings("ignore")

def _ensure_dir():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

TOPIC_ARTIFACT = os.path.join(ARTIFACT_DIR, "topic_pipeline.joblib")
def train_topic(
    train_tsv: str = "./pred_data/train2.tsv",
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

def train_intent(train_path="./pred_data/train2.tsv"):
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

def train_sensationalism(train_path="./pred_data/train2.tsv"):
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

SENTIMENT_ARTIFACT = os.path.join(ARTIFACT_DIR, "sentiment_rf.joblib")

ALPHA = 1.0
EPS = 1e-8
EMOTIONS_TO_ANALYZE = ["positive", "negative", "anger", "fear", "disgust",
                       "sadness", "joy", "anticipation", "trust", "surprise"]
_token_re = re.compile(r"[A-Za-z']+")

def _word_count(text):
    return len(_token_re.findall(str(text)))

def get_sentiment_class(vader_row):
    c = float(vader_row.get("compound", 0.0))
    if c >= 0.05:
        return "Positive"
    elif c <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def nrc_doc_score_from_text(text, alpha: float = ALPHA):
    emo = NRCLex(str(text))
    raw = emo.raw_emotion_scores or {}
    pos = raw.get("positive", 0)
    neg = raw.get("negative", 0)
    return float(np.log((pos + alpha) / (neg + alpha)))

def train_sentiment(train_path="./pred_data/train2.tsv", val_path="./pred_data/val2.tsv", text_col="statement"):
    df_tr = read_tsv(train_path)
    df_va = read_tsv(val_path)

    for df in (df_tr, df_va):
        df[text_col] = df[text_col].fillna("").astype(str)

    train_texts = df_tr[text_col].tolist()

    # Topic model artifacts (CountVectorizer + LDA)
    vectorizer = CountVectorizer(max_df=0.9, min_df=5, stop_words="english")
    X_train = vectorizer.fit_transform(train_texts)

    topic_model = LatentDirichletAllocation(n_components=20, random_state=42, learning_method="batch")
    topic_model.fit(X_train)

    theta_train = topic_model.transform(X_train)
    s_train = np.array([nrc_doc_score_from_text(s) for s in train_texts], dtype=float)

    weights_sum = theta_train.sum(axis=0) + EPS
    topic_mu = (theta_train.T @ s_train) / weights_sum

    diffs = s_train[:, None] - topic_mu[None, :]
    topic_var = (theta_train * diffs**2).sum(axis=0) / weights_sum
    topic_sigma = np.sqrt(topic_var + EPS)

    vader = SentimentIntensityAnalyzer()

    def extract_features(statement: str):
        s = "" if statement is None else str(statement)
        wc = _word_count(s)

        emo_obj = NRCLex(s)
        raw = emo_obj.raw_emotion_scores or {}
        pos = raw.get("positive", 0)
        neg = raw.get("negative", 0)
        emotion_logratio = float(np.log((pos + ALPHA) / (neg + ALPHA)))

        X = vectorizer.transform([s])
        theta = topic_model.transform(X)

        mu_hat = float((theta @ topic_mu)[0])
        var_hat = float((theta @ (topic_sigma ** 2))[0] + EPS)
        sd_hat = float(np.sqrt(var_hat))

        sent_dev_z = float((emotion_logratio - mu_hat) / sd_hat) if sd_hat > 0 else 0.0

        v = vader.polarity_scores(s)
        sentiment_value = get_sentiment_class(v)

        return {
            "emotion_logratio": emotion_logratio,
            "sent_dev_z": sent_dev_z,
            "sentiment_value": sentiment_value,
        }

    # Build train + val features
    tr_feat = pd.DataFrame([extract_features(s) for s in df_tr[text_col].tolist()])
    va_feat = pd.DataFrame([extract_features(s) for s in df_va[text_col].tolist()])

    features = ["emotion_logratio", "sent_dev_z"]
    X_tv = pd.concat([tr_feat[features], va_feat[features]], axis=0)
    y_tv = pd.concat([tr_feat["sentiment_value"], va_feat["sentiment_value"]], axis=0)

    sentiment_pipe = Pipeline(steps=[
        ("scale", StandardScaler(with_mean=False)),
        ("clf", RandomForestClassifier(
            max_depth=5, n_estimators=10, max_features=1, random_state=42
        ))
    ])
    sentiment_pipe.fit(X_tv, y_tv)

    joblib.dump({
        "vectorizer": vectorizer,
        "topic_model": topic_model,
        "topic_mu": topic_mu,
        "topic_sigma": topic_sigma,
        "features": features,
        "pipe": sentiment_pipe,
        "alpha": ALPHA,
        "eps": EPS,
    }, SENTIMENT_ARTIFACT)

    return sentiment_pipe

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
