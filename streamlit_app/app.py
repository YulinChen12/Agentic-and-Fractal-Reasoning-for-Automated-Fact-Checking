import streamlit as st
import pandas as pd
import numpy as np
import re
import csv
import torch
import warnings
import nltk
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import SVC, LinearSVC
from scipy.sparse import hstack
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize, StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import LatentDirichletAllocation
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast, AutoTokenizer, AutoModelForSequenceClassification
from nrclex import NRCLex
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from google import genai
from google.genai import types
from serpapi import GoogleSearch
import os
import json

# --- Configuration & Setup ---
st.set_page_config(
    page_title="News Analyzer AI",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for "fancy" look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        font-weight: bold;
    }
    h1 {
        color: #2c3e50;
        text-align: center;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        padding-bottom: 10px;
    }
    .stTextInput>div>div>input {
        border-radius: 5px;
    }
    .stTextArea>div>div>textarea {
        border-radius: 5px;
    }
    .report-container {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

st.title("📰 Intelligent News Analyzer")
st.markdown("<div style='text-align: center; color: #666; margin-bottom: 30px;'>AI-Powered Fact-Checking & Analysis Dashboard</div>", unsafe_allow_html=True)

# Suppress warnings
warnings.filterwarnings("ignore")

# Download NLTK data
@st.cache_resource(show_spinner=False)
def download_nltk_data():
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('averaged_perceptron_tagger')

download_nltk_data()

# API Keys
GOOGLE_API_KEY = "AIzaSyBwbwuHjIe97W7AkUIALTvRPqYMCwt-k6U"
SERPAPI_KEY = "2d4b3d3673b32b0d681d15159b267f4bb4d16fb9129b21b883bebacf62c0a2ca"

# Data Paths
DATA_PATH = './data/'

# --- Helper Functions ---
def read_tsv(path):
    COLS = ["id","label","statement","subjects","speaker","job_title",
            "state_info","party_affiliation","barely_true_cnt","false_cnt",
            "half_true_cnt","mostly_true_cnt","pants_on_fire_cnt","context","justification"]
    return pd.read_csv(path, sep="\t", header=None, names=COLS,
                       engine="python", quoting=csv.QUOTE_NONE, escapechar="\\",
                       on_bad_lines="skip")

def text_of(r):
    return " ".join([str(r.get("statement","")), str(r.get("context","")), str(r.get("justification",""))]).strip()

def first_subject(s):
    parts = re.split(r"[;,]", s) if isinstance(s,str) else []
    return parts[0].strip().lower() if parts and parts[0].strip() else "unknown"

# --- Model Loading & Training (Cached) ---

@st.cache_resource(show_spinner=False)
def load_news_coverage_model():
    # st.info("Loading News Coverage Model...")
    df_tr = read_tsv(DATA_PATH + "train2.tsv")
    
    X_tr = df_tr.apply(text_of, axis=1)
    y_tr = df_tr["subjects"].apply(first_subject)
    
    keep = y_tr.ne("unknown")
    X_tr, y_tr = X_tr[keep], y_tr[keep]
    
    classes = np.unique(y_tr)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_tr)
    wmap = {c:w for c,w in zip(classes, weights)}
    
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, strip_accents="unicode",
                                  analyzer="word", ngram_range=(1,2),
                                  min_df=2, max_df=0.9, sublinear_tf=True)),
        ("clf", LinearSVC(class_weight=wmap, random_state=42))
    ])
    
    pipe.fit(X_tr, y_tr)
    return pipe

@st.cache_resource(show_spinner=False)
def load_intent_model():
    # st.info("Loading Intent Classification Model...")
    df = read_tsv(DATA_PATH + "train2.tsv")
    for c in ["statement","context","justification"]:
        df[c] = df[c].fillna("").astype(str).str.strip()
    
    texts = df.apply(text_of, axis=1).tolist()
    
    intent_tfidf = TfidfVectorizer(
        lowercase=True, strip_accents="unicode",
        analyzer="word", ngram_range=(1,2),
        min_df=3, max_df=0.95, sublinear_tf=True
    )
    intent_tfidf.fit(texts) # Fit only
    
    PROTOS = {
      "inform":   ["Officials said the department released a report with data and timelines."],
      "persuade": ["We should support this policy because it will improve outcomes."],
      "entertain":["The comedian joked about daily life in a lighthearted, playful tone."],
      "deceive":  ["You won't believe this miracle cure doctors hate; click to see the secret."]
    }
    CLASS_NAMES = ["inform","persuade","entertain","deceive"]
    
    proto_rows = []
    for name in CLASS_NAMES:
        pv = intent_tfidf.transform(PROTOS[name])
        pv_mean = np.asarray(pv.mean(axis=0))
        pv_norm = normalize(pv_mean, norm="l2", axis=1)
        proto_rows.append(pv_norm.ravel())
    PROTO_MAT = np.vstack(proto_rows)
    
    return intent_tfidf, PROTO_MAT, CLASS_NAMES

@st.cache_resource(show_spinner=False)
def load_sensationalism_model():
    # st.info("Loading Sensationalism Model...")
    TOKEN_RE = re.compile(r"[A-Za-z]+")
    EVIDENCE_PATTERNS = [
        r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b",
        r"\b(19|20)\d{2}\b",
        r"%",
        r"https?://",
        r"\"[^\"]+\"",
        r"\baccording to\b",
        r"\breport(ed|s)? by\b|\bstudy\b|\bsurvey\b"
    ]
    EVIDENCE_RE = [re.compile(pat, flags=re.I) for pat in EVIDENCE_PATTERNS]

    def evidence_anchors(text):
        s = str(text)
        hits = sum(len(r.findall(s)) for r in EVIDENCE_RE)
        toks = max(1, len(TOKEN_RE.findall(s)))
        dens = np.log1p(hits) / max(10, toks)
        return float(np.clip(dens, 0.0, 0.2))
        
    df_tr = read_tsv(DATA_PATH + "train2.tsv")
    
    tfidf = TfidfVectorizer(max_features=3000, stop_words='english', ngram_range=(1,2))
    train_text = df_tr['statement'].astype(str) + " " + df_tr['context'].astype(str)
    
    X_train_tfidf = tfidf.fit_transform(train_text)
    X_train_ev = df_tr['statement'].apply(evidence_anchors).values.reshape(-1, 1)
    X_train_final = hstack([X_train_tfidf, X_train_ev])
    
    SCORE_MAP = {
        "pants-fire": 5, "false": 4, "barely-true": 3, 
        "half-true": 2, "mostly-true": 1, "true": 0
    }
    train_scores = df_tr['label'].map(SCORE_MAP).fillna(0)
    THRESHOLD = 2.5 
    y_train_binary = train_scores.apply(lambda x: 1 if x >= THRESHOLD else 0)
    
    rf_model = SVC(kernel="linear", C=0.025, class_weight="balanced", probability=True)
    rf_model.fit(X_train_final, y_train_binary)
    
    return rf_model, tfidf, evidence_anchors

@st.cache_resource(show_spinner=False)
def load_sentiment_model():
    # st.info("Loading Sentiment Analysis Model...")
    
    ALPHA = 1.0
    EPS = 1e-8
    EMOTIONS_TO_ANALYZE = ["positive", "negative"]
    _token_re = re.compile(r"[A-Za-z']+")
    
    def _word_count(text):
        return len(_token_re.findall(str(text)))
        
    def nrc_doc_score_from_text(text, alpha: float = ALPHA):
        emo = NRCLex(str(text))
        pos = emo.raw_emotion_scores.get('positive', 0)
        neg = emo.raw_emotion_scores.get('negative', 0)
        return float(np.log((pos + alpha) / (neg + alpha)))

    def get_sentiment_class(vader_row):
        c = float(vader_row.get("compound", 0.0))
        if c >= 0.05: return 'Positive'
        elif c <= -0.05: return 'Negative'
        else: return 'Neutral'
        
    class ModelArtifacts:
        def __init__(self, vectorizer, topic_model, topic_mu, topic_sigma):
            self.vectorizer = vectorizer
            self.topic_model = topic_model
            self.topic_mu = topic_mu
            self.topic_sigma = topic_sigma
            
    df_tr = read_tsv(DATA_PATH + "train2.tsv")
    df_va = read_tsv(DATA_PATH + "val2.tsv")
    
    # Train artifacts
    train_texts = df_tr["statement"].fillna("").astype(str).tolist()
    vectorizer = CountVectorizer(max_df=0.9, min_df=5, stop_words='english')
    X_train = vectorizer.fit_transform(train_texts)
    
    topic_model = LatentDirichletAllocation(n_components=20, random_state=42, learning_method="batch")
    topic_model.fit(X_train)
    
    theta_train = topic_model.transform(X_train)
    s_train = np.array([nrc_doc_score_from_text(s) for s in train_texts])
    
    weights_sum = theta_train.sum(axis=0) + EPS
    topic_mu = (theta_train.T @ s_train) / weights_sum
    diffs = s_train[:, None] - topic_mu[None, :]
    topic_var = (theta_train * diffs ** 2).sum(axis=0) / weights_sum
    topic_sigma = np.sqrt(topic_var + EPS)
    
    artifacts = ModelArtifacts(vectorizer, topic_model, topic_mu, topic_sigma)
    analyzer = SentimentIntensityAnalyzer()
    
    # Helper for feature extraction
    def extract_features(statement, arts, anz):
        s = "" if statement is None else str(statement)
        wc = _word_count(s)
        emo_obj = NRCLex(s)
        emo_counts = {e: int(emo_obj.raw_emotion_scores.get(e, 0)) for e in EMOTIONS_TO_ANALYZE}
        emotion_logratio = float(np.log((emo_counts["positive"] + ALPHA) / (emo_counts["negative"] + ALPHA)))
        s_val = emotion_logratio
        
        X = arts.vectorizer.transform([s])
        theta = arts.topic_model.transform(X)
        mu_hat = float((theta @ arts.topic_mu)[0])
        var_hat = float((theta @ (arts.topic_sigma ** 2))[0] + EPS)
        sd_hat = float(np.sqrt(var_hat))
        sent_dev_diff = float(s_val - mu_hat)
        sent_dev_z = float(sent_dev_diff / sd_hat) if sd_hat > 0 else 0.0
        
        vader = anz.polarity_scores(s)
        sentiment_value = get_sentiment_class(vader)
        
        return {
            "emotion_logratio": float(emotion_logratio),
            "sent_dev_z": float(sent_dev_z),
            "sentiment_value": sentiment_value
        }

    # Train Final Classifier
    def create_features_df(df):
        rows = df["statement"].apply(lambda x: extract_features(x, artifacts, analyzer))
        return pd.DataFrame(rows.tolist())

    train_feats = create_features_df(df_tr)
    valid_feats = create_features_df(df_va)
    
    features = ["emotion_logratio", "sent_dev_z"]
    X_tv = pd.concat([train_feats[features], valid_feats[features]], axis=0)
    y_tv = pd.concat([train_feats["sentiment_value"], valid_feats["sentiment_value"]], axis=0)
    
    clf = Pipeline(steps=[
        ("scale", StandardScaler(with_mean=False)),
        ("clf", RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, random_state=42))
    ])
    clf.fit(X_tv, y_tv)
    
    return clf, artifacts, analyzer, features, extract_features

@st.cache_resource(show_spinner=False)
def load_bert_models():
    # st.info("Loading Reputation and Stance Models (BERT)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Reputation
    try:
        rep_path = "./reputation_model"
        rep_model = DistilBertForSequenceClassification.from_pretrained(rep_path)
        rep_tok = DistilBertTokenizerFast.from_pretrained(rep_path)
        rep_model.to(device)
        rep_model.config.id2label = {0: "high", 1: "low", 2: "medium"}
    except Exception as e:
        print(f"Reputation model error: {e}")
        rep_model, rep_tok = None, None

    # Stance
    try:
        stance_path = "./stance_model"
        stance_tok = AutoTokenizer.from_pretrained(stance_path)
        stance_model = AutoModelForSequenceClassification.from_pretrained(stance_path)
        stance_model.to(device)
        stance_model.config.id2label = {0: "support", 1: "deny", 2: "neutral"}
    except Exception as e:
        print(f"Stance model error: {e}")
        stance_model, stance_tok = None, None
        
    return rep_model, rep_tok, stance_model, stance_tok, device

# --- Load All Models ---
loading_container = st.empty()
with loading_container.container():
    st.markdown("### Initializing AI Models...")
    p_bar = st.progress(0)
    status_text = st.empty()

    status_text.text("Loading News Coverage Model...")
    news_pipe = load_news_coverage_model()
    p_bar.progress(20)

    status_text.text("Loading Intent Classification Model...")
    intent_tfidf, PROTO_MAT, CLASS_NAMES = load_intent_model()
    p_bar.progress(40)

    status_text.text("Loading Sensationalism Model...")
    sens_model, sens_tfidf, evidence_anchors = load_sensationalism_model()
    p_bar.progress(60)

    status_text.text("Loading Sentiment Analysis Model...")
    sent_clf, sent_artifacts, sent_analyzer, sent_features, sent_extractor = load_sentiment_model()
    p_bar.progress(80)

    status_text.text("Loading Reputation & Stance Models...")
    rep_model, rep_tok, stance_model, stance_tok, device = load_bert_models()
    p_bar.progress(100)

loading_container.empty()

# --- Prediction Wrappers ---

def predict_news_coverage(text):
    pred = news_pipe.predict([text])[0]
    return {"topic": str(pred)}

def predict_intent(text):
    z = intent_tfidf.transform([text])
    zn = normalize(z, norm="l2", axis=1)
    scores = (zn @ PROTO_MAT.T).ravel()
    by_label = {CLASS_NAMES[i]: float(scores[i]) for i in range(4)}
    top_label = max(by_label, key=by_label.get)
    return {"primary_intent": top_label}

def predict_sensationalism(statement):
    # Justification omitted for simplicity in features as per nb logic roughly
    full_text = str(statement) # Context omitted in single sentence pred
    tfidf_vec = sens_tfidf.transform([full_text])
    ev_score = evidence_anchors(statement)
    ev_vec = np.array([[ev_score]])
    f_vec = hstack([tfidf_vec, ev_vec])
    
    p_sensational = float(sens_model.predict_proba(f_vec)[0, 1])
    label = "sensational" if p_sensational >= 0.5 else "neutral"
    return {"label": label}

def predict_sentiment(statement):
    f_dict = sent_extractor(statement, sent_artifacts, sent_analyzer)
    df = pd.DataFrame([f_dict])
    X = df[sent_features]
    pred = sent_clf.predict(X)[0]
    return {"sentiment": str(pred)}

def predict_reputation(sentences):
    if not rep_model or not rep_tok: return {"final_label": "medium"}
    if not sentences: return {"final_label": "medium"}
    
    results = []
    for sent in sentences:
        inputs = rep_tok(sent, truncation=True, padding=True, max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = rep_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            pred_id = torch.argmax(probs, dim=-1).item()
            label = rep_model.config.id2label[pred_id]
        results.append(label)
    
    return {"final_label": Counter(results).most_common(1)[0][0]}

def predict_stance(sentences):
    if not stance_model or not stance_tok: return {"final_label": "neutral"}
    if not sentences: return {"final_label": "neutral"}
    
    results = []
    for sent in sentences:
        inputs = stance_tok(sent, truncation=True, padding=True, max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = stance_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            pred_id = torch.argmax(probs, dim=-1).item()
            label = stance_model.config.id2label[pred_id]
        results.append(label)
    
    return {"final_label": Counter(results).most_common(1)[0][0]}


# --- Agent Tools ---

def analyze_complete_article(article_title: str = "", article_text: str = "") -> dict:
    """
    Complete article analysis using sentence-level voting from all 6 models.
    """
    results = {"predictions": {}}
    try:
        sentences = nltk.sent_tokenize(article_text or "")
    except Exception:
        sentences = []
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences and article_text.strip():
        sentences = [article_text.strip()]
    
    if not sentences:
        return {"error": "No valid sentences found."}

    # Voting
    votes = {k: [] for k in ["news", "intent", "sens", "sent"]}
    
    for s in sentences:
        try: votes["news"].append(predict_news_coverage(s)["topic"])
        except: pass
        try: votes["intent"].append(predict_intent(s)["primary_intent"])
        except: pass
        try: votes["sens"].append(predict_sensationalism(s)["label"])
        except: pass
        try: votes["sent"].append(predict_sentiment(s)["sentiment"])
        except: pass
        
    # Aggregation
    def get_maj(lst): return Counter(lst).most_common(1)[0][0] if lst else "error"
    
    results["predictions"]["news_coverage"] = {"topic": get_maj(votes["news"])}
    results["predictions"]["intent"] = {"primary_intent": get_maj(votes["intent"])}
    results["predictions"]["sensationalism"] = {"label": get_maj(votes["sens"])}
    results["predictions"]["sentiment"] = {"sentiment": get_maj(votes["sent"])}
    
    # Batch models
    try:
        results["predictions"]["reputation"] = {"level": predict_reputation(sentences)["final_label"]}
    except Exception as e:
        results["predictions"]["reputation"] = {"error": str(e)}
        
    try:
        results["predictions"]["stance"] = {"stance": predict_stance(sentences)["final_label"]}
    except Exception as e:
        results["predictions"]["stance"] = {"error": str(e)}

    return results["predictions"]

def serpapi_search(query: str) -> dict:
    """
    Uses SerpAPI to perform a web search.
    """
    if not SERPAPI_KEY:
        return {"error": "Missing API Key"}
        
    try:
        params = {
            "engine": "google",
            "q": query,
            "api_key": SERPAPI_KEY,
            "num": 5,
            "hl": "en"
        }
        search = GoogleSearch(params)
        results = search.get_dict()
        organic = results.get("organic_results", [])
        clean_results = []
        for r in organic:
            clean_results.append({
                "title": r.get("title"),
                "link": r.get("link"),
                "snippet": r.get("snippet"),
                "date": r.get("date", "Unknown")
            })
        return {"results": clean_results}
    except Exception as e:
        return {"error": f"Search failed: {str(e)}"}

# --- Few-Shot Helper Functions ---

def load_data(path):
    if not os.path.exists(path):
        return []
        
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, dict) and "articles" in data:
        return data["articles"]
    elif isinstance(data, list):
        return data
    else:
        return []

def format_few_shot_context(articles):
    """
    Formats the training articles into a prompt string.
    """
    output_text = "Here are reference examples of human-verified analysis:\\n\\n"
    
    for art in articles:
        # --- Inputs --- 
        headline = art.get('headline', 'No Title')
        body = art.get('text', 'No Text')
        source = art.get('news_source', 'Unknown Source')
        author = art.get('author', 'Unknown Author')
        date = art.get('date', 'Unknown Date')
        
        # --- GROUND TRUTH --- 
        topic = art.get('coverage', 'N/A')
        intent = art.get('intent', 'N/A')
        sensationalism = art.get('sensationalism', 'N/A')
        sentiment = art.get('sentiment', 'N/A')
        reputation = art.get('reputation', 'N/A')
        stance = art.get('stance', 'N/A')
        title_vs_body = art.get('title_vs_body', 'N/A')
        veracity = art.get('context_veracity', 'N/A')
        location = art.get('location', 'N/A')
        
        # --- INPUT BLOCK ---
        output_text += f"--- EXAMPLE INPUT ---\\n"
        output_text += f"Source: {source}\\n"
        output_text += f"Author: {author}\\n"
        output_text += f"Date: {date}\\n"
        output_text += f"Title: {headline}\\n"
        output_text += f"Body: {body}\\n\\n"
        
        # --- OUTPUT BLOCK ---
        output_text += f"--- EXAMPLE HUMAN LABELING OUTPUT ---\\n"
        output_text += f"- news_topic: {topic}\\n"
        output_text += f"- intent: {intent}\\n"
        output_text += f"- sensationalism: {sensationalism}\\n"
        output_text += f"- sentiment: {sentiment}\\n"
        output_text += f"- reputation: {reputation}\\n"
        output_text += f"- stance: {stance}\\n"
        output_text += f"- title_vs_body: {title_vs_body}\\n"
        output_text += f"- context_veracity: {veracity}\\n"
        output_text += f"- location: {location}\\n\\n"
        
    return output_text

# --- Gemini Agent ---

PROMPT_COT_GOOGLE = """
You are a senior investigative editor at an independent fact-checking newsroom.
You are mentoring a junior analyst and must produce a rigorous, transparent fact-check report for each article.

**FEW-SHOT EXAMPLES (GROUND TRUTH):**
{few_shot_examples}

**CALIBRATION INSTRUCTION (CRITICAL):**
The examples above represent the **Gold Standard** for this task.
1. **Analyze the patterns:** Observe how the human annotators defined "Sensationalism," "Stance," and "News Topic" in the examples above.
2. **Mimic the Logic:** You must calibrate your internal thresholds to match these examples.
   - Example: If the human labels a slightly critical text as "Neutral," you must also label similar texts as "Neutral."
   - Example: If the human labels a specific source as "High Reputation," align your judgment with that standard.
3. **Prioritize Precedent:** Your final labels must be consistent with the decision boundaries established in the Ground Truth examples provided above.

Your goal is to provide a hybrid analysis.

**Here is the article:**

---
{article_to_analyze}
---

**You have access to two tools:**
1. `analyze_complete_article` – runs predictive models.
2. `serpapi_search` – performs web search to verify claims.

**STRICT EXECUTION PROTOCOL (Follow Order):**

1.  **GATHER DATA FIRST:**
    * Call `analyze_complete_article` immediately.
    * Review the article for specific claims and call `serpapi_search` to verify facts, dates, or events.
    * **CRITICAL:** Do NOT start writing the final report yet. Just collect your tool outputs.

2.  **GENERATE REPORT SECOND:**
    * **ONLY** after you have received all tool outputs, generate the **COMPLETE** Markdown dashboard below.

---

**Output Format Requirements:**

Your final response MUST be a clean Markdown dashboard with the following two main sections:

## Phase 1: Model Predictions
**CRITICAL INSTRUCTION: STRICT TRANSCRIPTION ONLY**
- In this section, you are a **mechanical transcriber**.
* **Action:** You MUST call `analyze_complete_article` immediately using the article's title and body.
* **Wait:** Do not proceed to Phase 2 until you receive the JSON output from this tool.
- You must output the **EXACT** label provided by the `analyze_complete_article` tool.
- **DO NOT** correct, interpret, or change the tool's labels, even if you think they are wrong.

**Required Output List:**
  1. news_topic
  2. intent
  3. sensationalism
  4. sentiment
  5. reputation
  6. stance
  
- Do NOT show raw JSON.

## Phase 2: Qualitative Analysis
- Follow the factor-by-factor structure below.
- You MUST provide ALL SIX sections in this order:
     1. News Topic
     2. Sensationalism
     3. Stance
     4. Title vs. Body
     5. Context Veracity
     6. Location / Geography

---

**CONFIDENCE SCORE RUBRIC (0–100%):**

**TRACK A: For "Context Veracity" & "Location" (Evidence-Based)**
* **90–100%:** Direct, quote-level evidence from search explicitly confirms the claim.
* **75–89%:** Strong supporting evidence found, but minor details differ.
* **50–74%:** Related context found, but no direct confirmation.
* **25–49%:** No strong evidence; weak or conflicting.
* **0–24%:** Evidence strongly contradicts the article.

**TRACK B: For Topic, Stance, Title vs. Body & Sensationalism (Analysis-Based)**
* **90–100%:** Explicit, unambiguous language supports your label.
* **75–89%:** Trend is clear and consistent.
* **50–74%:** Text is mixed, ambiguous, or open to interpretation.
* **25–49%:** Text is too short or vague.
* **0–24%:** Cannot meaningfully determine.

---

**Analysis Sections (Include these in your final output):**

**1. News Topic**
* Recipe: What kind of news is covered in this article? Determine the type of news: local, global, opinion, etc. Check if similar events receive similar coverage. Compare coverage angle with other reputable sources.
* **Output:** [Label]
* **Confidence:** [0–100]%
* **Reasoning:** [Brief explanation]

**2. Sensationalism**
* Recipe: Is the text using sensationalist words and phrases designed to attract attention or manipulate? Examine text for overly dramatic or exaggerated claims. Compare the emotional tone of the headline vs. the content.
* **Output:** [sensational or neutral]
* **Confidence:** [0–100]%
* **Reasoning:** [Brief explanation]

**3. Stance**
* Recipe: What is the author's opinion about the news? Analyze if content supports, denies, or is neutral towards claims. Evaluate consistency in stance throughout the content. Determine if shifts in stance are supported by factual developments.
* **Output:** [support, deny, or neutral]
* **Confidence:** [0–100]%
* **Reasoning:** [Brief explanation]

**4. Title vs. Body**
* Recipe: Analyze the relationship between the title and the body text: "Does the title agree with, discuss, is unrelated to, or negate the body of the text?" First, identify the main claim of the title. Second, summarize the main arguments of the full article text. Third, explain your verdict.
* **Output:** ["Agree", "Discuss", "Negate", "Unrelated"]
* **Confidence:** [0–100]%
* **Reasoning:** [Brief explanation]

**5. Context Veracity (Cite your search results here)**
* Recipe: Evaluate truthfulness on two levels: 1) Internal Check for consistency, 2) External Verification using `serpapi_search` to verify core claims. Synthesis: Weigh internal logic against external evidence.
* **Output:** [Accurate, Inaccurate, Misleading, Unverified]
* **Confidence:** [0–100]%
* **Reasoning:**
    * **Internal:** Check for consistency.
    * **External:** Explicitly cite what you found using `serpapi_search`.

**6. Location / Geography (Cite your search results here)**
* Recipe: Where is the text about? What are the geographic elements? Validate the accuracy of geographic details using `serpapi_search`.
* **Output:** [e.g., "Global," "US-centric," "Specific"]
* **Confidence:** [0–100]%
* **Reasoning:** Validate geographic details.

---

Start by calling `analyze_complete_article`.
"""

client = genai.Client(api_key=GOOGLE_API_KEY)

def run_agent(title, body):
    # Load few-shot data
    try:
        train_articles = load_data(DATA_PATH + "train_article.json")
        few_shot_context = format_few_shot_context(train_articles)
    except Exception:
        few_shot_context = ""

    tools = [analyze_complete_article, serpapi_search]
    system_instruction = (
        "You are a news-analysis assistant. "
        "You must obey the user’s instructions about phases and output format. "
        "Always use the available tools when appropriate. "
        "Do not output raw tool JSON in your final answer."
    )
    
    article_content = f"Title: {title}\\nBody: {body}"
    final_prompt = PROMPT_COT_GOOGLE.format(
        few_shot_examples=few_shot_context,
        article_to_analyze=article_content
    )
    
    config = types.GenerateContentConfig(
        tools=tools,
        temperature=0.1,
        system_instruction=system_instruction,
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(mode="AUTO")
        ),
    )
    
    with st.spinner("Agent is thinking... (This may take a minute)"):
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=final_prompt,
            config=config,
        )
        return response.text

# --- UI ---

st.markdown("---")

with st.form("article_form"):
    col_input, col_help = st.columns([3, 1])
    
    with col_input:
        article_title = st.text_input("Article Headline", placeholder="Enter the headline here...")
        article_body = st.text_area("Article Content", height=400, placeholder="Paste the full article text here...")
    
    with col_help:
        st.markdown("### ℹ️ How it works")
        st.info(
            """
            **1. Input Data**
            Paste the headline and body text of the article you want to analyze.

            **2. Run Analysis**
            Our AI agent will run 6 predictive models and cross-reference claims with Google Search.

            **3. Review Report**
            Get a comprehensive dashboard with fact-checking and context.
            """
        )
        st.write("") # Spacer
        st.write("") # Spacer
        submitted = st.form_submit_button("🔍 Start Analysis", type="primary")

if submitted:
    if not article_title or not article_body:
        st.error("⚠️ Please provide both a headline and the body text.")
    else:
        try:
            result_text = run_agent(article_title, article_body)
            st.markdown("---")
            st.markdown("<div class='report-container'>", unsafe_allow_html=True)
            st.markdown(result_text)
            st.markdown("</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"An error occurred: {e}")