# %%
# pip install pandas numpy scikit-learn torch transformers nrclex vaderSentiment
import nltk
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# %%
# Core imports
import pandas as pd
import numpy as np
import re
import csv
import torch
import time
import warnings
warnings.filterwarnings("ignore")

# ML and NLP imports
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import SVC
from scipy.sparse import hstack, csr_matrix
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize, StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import LatentDirichletAllocation
from sentence_transformers import SentenceTransformer

# Transformers
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast, AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

# Sentiment and emotion analysis
from nrclex import NRCLex
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import os
from google import genai
from google.genai import types
import json
import requests
import uuid
from collections import Counter

print("All imports loaded successfully!")

# %%
from google.adk.agents import Agent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import AgentTool, FunctionTool, google_search

from google.adk.a2a.utils.agent_to_a2a import to_a2a

print("✅ ADK components imported successfully.")

# %%
# Data loading and preprocessing functions
COLS = ["id","label","statement","subjects","speaker","job_title",
        "state_info","party_affiliation","barely_true_cnt","false_cnt",
        "half_true_cnt","mostly_true_cnt","pants_on_fire_cnt","context","justification"]

data_path = './data/'

def read_tsv(path):
    """Load TSV data with proper handling of quotes and escape characters"""
    return pd.read_csv(path, sep="\t", header=None, names=COLS,
                       engine="python", quoting=csv.QUOTE_NONE, escapechar="\\",
                       on_bad_lines="skip")

def text_of(r):
    """Combine statement, context, and justification into single text"""
    return " ".join([str(r.get("statement","")), str(r.get("context","")), str(r.get("justification",""))]).strip()

def first_subject(s):
    """Extract first subject from subjects field"""
    parts = re.split(r"[;,]", s) if isinstance(s,str) else []
    return parts[0].strip().lower() if parts and parts[0].strip() else "unknown"

print("Data loading functions defined!")

# %%
# Model 1: News Coverage Classification
print("Loading News Coverage Model...")

# Load training data
df_tr = read_tsv(data_path + "train2.tsv")
df_va = read_tsv(data_path + "val2.tsv")

X_tr = df_tr.apply(text_of, axis=1)
X_va = df_va.apply(text_of, axis=1)

y_tr = df_tr["subjects"].apply(first_subject)
y_va = df_va["subjects"].apply(first_subject)

keep = y_tr.ne("unknown")
X_tr, y_tr = X_tr[keep], y_tr[keep]

classes = np.unique(y_tr)
weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_tr)
wmap = {c:w for c,w in zip(classes, weights)}

# Train news coverage pipeline
news_coverage_pipe = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, strip_accents="unicode",
                              analyzer="word", ngram_range=(1,2),
                              min_df=2, max_df=0.9, sublinear_tf=True)),
    ("clf", LinearSVC(class_weight=wmap, random_state=42))
])

news_coverage_pipe.fit(X_tr, y_tr)
pred = news_coverage_pipe.predict(X_va)
print(f"News Coverage Model - Accuracy: {accuracy_score(y_va, pred):.4f}")

def predict_news_coverage(text):
    """Predict news coverage topic for given text"""
    prediction = news_coverage_pipe.predict([text])[0]
    return {"topic": str(prediction)}

# %%
# Model 2: Intent Classification
print("Loading Intent Classification Model...")

# Load and prepare data
df = read_tsv(data_path + "train2.tsv")
for c in ["statement","context","justification"]:
    df[c] = df[c].fillna("").astype(str).str.strip()

texts = df.apply(text_of, axis=1).tolist()

# Train TF-IDF vectorizer
intent_tfidf = TfidfVectorizer(
    lowercase=True, strip_accents="unicode",
    analyzer="word", ngram_range=(1,2),
    min_df=3, max_df=0.95, sublinear_tf=True
)
X = intent_tfidf.fit_transform(texts)

# Define prototypes for each intent
PROTOS = {
  "inform":   ["Officials said the department released a report with data and timelines."],
  "persuade": ["We should support this policy because it will improve outcomes."],
  "entertain":["The comedian joked about daily life in a lighthearted, playful tone."],
  "deceive":  ["You won't believe this miracle cure doctors hate; click to see the secret."]
}
CLASS_NAMES = ["inform","persuade","entertain","deceive"]

# Create prototype matrix
proto_rows = []
for name in CLASS_NAMES:
    pv = intent_tfidf.transform(PROTOS[name])
    pv_mean = np.asarray(pv.mean(axis=0))
    pv_norm = normalize(pv_mean, norm="l2", axis=1)
    proto_rows.append(pv_norm.ravel())
PROTO_MAT = np.vstack(proto_rows)

def predict_intent(title="", body=""):
    """Predict intent using prototype matching"""
    text = (" ".join([str(title or ""), str(body or "")])).strip()
    z = intent_tfidf.transform([text])
    zn = normalize(z, norm="l2", axis=1)
    scores = (zn @ PROTO_MAT.T).ravel()
    by_label = {CLASS_NAMES[i]: float(scores[i]) for i in range(4)}
    top_label = max(by_label, key=by_label.get)
    
    return {
        "primary_intent": top_label,
        "confidence_scores": by_label
    }

print("Intent Classification Model loaded!")

# %%
# Model 3: Sensationalism Detection
print("Loading Sensationalism Model...")

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

tfidf = TfidfVectorizer(max_features=3000, stop_words='english', ngram_range=(1,2))
train_text = df_tr['statement'].astype(str) + " " + df_tr['context'].astype(str)

X_train_tfidf = tfidf.fit_transform(train_text)

X_train_ev = df_tr['statement'].apply(evidence_anchors).values.reshape(-1, 1)

X_train_final = hstack([X_train_tfidf, X_train_ev])

SCORE_MAP = {
    "pants-fire": 5, 
    "false": 4, 
    "barely-true": 3, 
    "half-true": 2, 
    "mostly-true": 1, 
    "true": 0
}

# Map labels to scores
train_scores = df_tr['label'].map(SCORE_MAP).fillna(0)

# THRESHOLD = 2.5 means:
# Scores 0, 1, 2 (True, Mostly-True, Half-True) -> 0 (Neutral)
# Scores 3, 4, 5 (Barely-True, False, Pants-Fire) -> 1 (Sensational)
THRESHOLD = 2.5 

y_train_binary = train_scores.apply(lambda x: 1 if x >= THRESHOLD else 0)

rf_model = SVC(kernel="linear", C=0.025, class_weight="balanced", probability=True)
rf_model.fit(X_train_final, y_train_binary)

def _features_for_inference(statement: str, context: str = "") -> np.ndarray:
    # Combine Text (Statement + Context)
    full_text = str(statement) + " " + str(context)
    
    # Vectorize
    tfidf_vec = tfidf.transform([full_text])
    
    # Calculate Evidence Score
    ev_score = evidence_anchors(statement)
    ev_vec = np.array([[ev_score]]) # Reshape to (1, 1) to match sparse matrix format
    
    return hstack([tfidf_vec, ev_vec])

def predict_sensationalism(statement: str, justification: str = ""):
    """
    Predicts if a statement is Sensational/False based on the trained model.
    """
    # Get features
    f_vec = _features_for_inference(statement, justification)
    
    # Predict Probability
    p_sensational = float(rf_model.predict_proba(f_vec)[0, 1])
    
    # Calculate Score (0 to 10)
    # High probability of False = High Sensationalism Score
    score_0_10 = float(np.clip(10.0 * p_sensational, 0.0, 10.0))
    
    # Determine Label
    label = "sensational" if p_sensational >= 0.45 else "neutral"
    
    # Calculate Confidence
    confidence = float(max(p_sensational, 1 - p_sensational))
    
    # Get Evidence Subscore
    evidence_val = float(evidence_anchors(statement))

    return {
        "factor": "sensationalism",
        "score": round(score_0_10, 3),
        "confidence": round(confidence, 3),
        "label": label,
        "subscores": {
            "evidence_density": round(evidence_val, 3),
            "probability_fake": round(p_sensational, 3)
        },
    }

print("Sensationalism Model loaded!")

# %%
# Model 6: Stance Classification
print("Loading Stance Model...")

# Ensure imports are available
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

try:
    stance_model_path = "./preditive_models/stance_model"
    stance_tokenizer = AutoTokenizer.from_pretrained(stance_model_path)
    stance_model = AutoModelForSequenceClassification.from_pretrained(stance_model_path)
    stance_model.to(device)
    
    # Stance mapping
    id2stance = {0: "support", 1: "deny", 2: "neutral"}
    
    print(" Stance Model loaded successfully!")
    
except Exception as e:
    print(f" Stance model not found: {e}")
    print("Using fallback stance prediction...")
    
    # Fallback stance model
    stance_model = None
    stance_tokenizer = None
    id2stance = None

# %%
def predict_article_stance(article_text=None, sentences=None):
    """
    Predicts stance for each sentence and aggregates results with majority vote.
    Can accept either full article text or pre-split sentences.
    """
    # Check if model is loaded
    if stance_model is None or stance_tokenizer is None:
        return {"final_label": "neutral", "counts": {"neutral": 1}}
    
    if sentences is None:
        if article_text is None:
            return {"final_label": "neutral", "counts": {"neutral": 1}}
        # Split article into sentences if not provided
        sentences = nltk.sent_tokenize(article_text)
        sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return {"final_label": "neutral", "counts": {"neutral": 1}}

    results = []
    for sent in sentences:
        # Tokenize each sentence separately
        inputs = stance_tokenizer(
            sent,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)

        # Forward pass
        with torch.no_grad():
            outputs = stance_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            pred_id = torch.argmax(probs, dim=-1).item()
            label = stance_model.config.id2label[pred_id]
            score = probs[0][pred_id].item()

        results.append((sent, label, score))

    # --- Majority vote across sentences ---
    labels = [label for _, label, _ in results]
    label_counts = Counter(labels)
    majority_label = label_counts.most_common(1)[0][0]

    return {
        "final_label": majority_label,
        "counts": dict(label_counts),
    }

# %%
def get_sentences(text):
    """Helper to ensure consistent sentence tokenization across all tools."""
    try:
        sentences = nltk.sent_tokenize(text or "")
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        
        if not sentences and text.strip():
            sentences = [text.strip()]
            
        return sentences if sentences else []
    except Exception:
        return [text.strip()] if text else []

# 1. News Coverage Tool
def tool_news_topic(article_text: str) -> dict:
    """Classifies the primary news topic of the article."""
    sentences = get_sentences(article_text)
    votes = []
    for s in sentences:
        try:
            votes.append(predict_news_coverage(s)["topic"])
        except: continue
    
    topic = Counter(votes).most_common(1)[0][0] if votes else "unknown"
    return {"news_coverage": topic}

# 2. Intent Tool
def tool_intent(article_text: str) -> dict:
    """Identifies the primary communication intent (inform, persuade, etc)."""
    sentences = get_sentences(article_text)
    votes = []
    for s in sentences:
        try:
            votes.append(predict_intent(title="", body=s)["primary_intent"])
        except: continue
    
    intent = Counter(votes).most_common(1)[0][0] if votes else "unknown"
    return {"intent": intent}

# 3. Sensationalism Tool
def tool_sensationalism(article_text: str) -> dict:
    """Detects if the article uses sensationalist or neutral framing."""
    sentences = get_sentences(article_text)
    votes = []
    for s in sentences:
        try:
            votes.append(predict_sensationalism(s)["label"])
        except: continue
    
    label = Counter(votes).most_common(1)[0][0] if votes else "neutral"
    return {"sensationalism": label}


# 6. Stance Tool
def tool_stance(article_text: str) -> dict:
    """Determines the political stance (support, deny, or neutral)."""
    sentences = get_sentences(article_text)
    try:
        result = predict_article_stance(sentences=sentences)
        return {"stance": result["final_label"]}
    except:
        return {"stance": "neutral"}
# %%
os.environ["GOOGLE_API_KEY"] = "AIzaSyB1G69vygim6hTtn8KxYXP-3wlTfNNGNIU"
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"

retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504], # Retry on these HTTP errors
)

# %%
sensationalism_agent = Agent(
    name="Sensationalism_Analyst",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction=(
        "You are a senior linguistic editor specializing in media bias. You follow a strict two-phase analysis protocol:\n\n"
        
        "PHASE 1: PREDICTIVE DATA GATHERING\n"
        "- Call the 'tool_sensationalism' tool immediately to get the initial model label.\n"
        "- - Do NOT provide a final answer until you have the results of this tool, and critically evaluate the model's label.\n\n"
        
        "PHASE 2: QUALITATIVE SYNTHESIS\n"
        "- Review the article's text independently for sensationalist phrasing and dramatic claims.\n"
        "- Compare the emotional tone of the headline vs. the content.\n"
        "- Identify 'loaded' language, superlatives, and clickbait structures designed to provoke emotion.\n"
        "- Determine if the content prioritizes shock value and narrative over verifiable facts.\n"
        "- Based on your analysis, label the sensationalism for this article. If your independent analysis of "
        "linguistic patterns (e.g., use of exclamation, intense adjectives) contradicts the phase 1 model, "
        "provide clear reasoning for the discrepancy in one sentence.\n\n"
        
        "**CONFIDENCE SCORE RUBRIC (0–100%):**\n"
        "90–100%: Definitive evidence. Multiple linguistic markers (hyperbole, clickbait, emotional appeals) align perfectly with tool results.\n"
        "75–89%: High Probability. Clear patterns of sensationalist or neutral language are present throughout the text.\n"
        "50–74%: Moderate Certainty. Text uses occasional colorful language that may be stylistic rather than manipulative, or your analysis deviates from the tool.\n"
        "25–49%: Ambiguous Intent. The tone is dry but the subject matter is inherently dramatic, making intent difficult to isolate.\n"
        "0–24%: Insufficient Data. Text lacks enough descriptive language or context to determine a rhetorical strategy.\n\n"

        "OUTPUT FORMAT:\n"
        "**Sensationalism**\n"
        "* **Final Output:** [sensational or neutral] (Your final verdict)\n"
        "* **Confidence:** [0–100]%\n"
        "* **Reasoning:** [Explain how you synthesized the model's prediction with your own analysis in 1 bullet point.]"
    ),
    tools=[tool_sensationalism],
    output_key="sensationalism_report"
)

# %%
stance_agent = Agent(
    name="Stance_Analyst",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction=(
        "You are a senior linguistic editor. You follow a strict two-phase analysis protocol:\n\n"
        
        "PHASE 1: PREDICTIVE DATA GATHERING\n"
        "- Call the 'tool_stance' tool immediately to get the initial model label.\n"
        "- Do NOT provide a final answer until you have the results of this tool, and critically evaluate the model's label.\n\n"

        "PHASE 2: QUALITATIVE SYNTHESIS\n"
        "- Review the article's text independently to understand the author's opinion about the news.\n"
        "- Analyze if content supports, denies, or is neutral towards claims.\n"
        "- Evaluate consistency in stance throughout the content.\n"
        "- Determine if shifts in stance are supported by factual developments.\n"
        "- Label the article with your analysis result and compare with phase 1 model label. "
        "If your independent analysis of the arguments in the article "
        "contradicts the model, provide your reasoning for the discrepancy.\n\n"
        
        "**CONFIDENCE SCORE RUBRIC (0–100%):**\n"
        "90–100%: High Alignment. Qualitative analysis strongly matches tool output with explicit, unambiguous evidence.\n"
        "75–89%: High Certainty. Clear rhetorical trend; minor stylistic nuance doesn't obscure the primary stance.\n"
        "50–74%: Moderate Uncertainty. Content is balanced, ambiguous, or your analysis deviates from the tool's label.\n"
        "25–49%: Low Reliability. Evidence is contradictory or the text relies on heavy sarcasm/subtext that is difficult to verify.\n"
        "0–24%: Non-Determinate. Text lacks sufficient linguistic markers to assign a stance meaningfully.\n\n"
         
        "OUTPUT FORMAT:\n"
        "**Stance**\n"
        "**Final Output:** [support, deny, or neutral] (Your final verdict)\n"
        "**Confidence:** [0–100]%\n"
        "**Reasoning:** [Explain how you synthesized the model's prediction with your own analysis in 1 bullet point]"
    ),
    tools=[tool_stance],
)

# %%
instruction_text = """
You are a senior investigative fact-checker specialized in analyzing **Context Veracity**.
Your task is to evaluate the truthfulness and reliability of the article below based strictly on:
1. **Contextual Coherence**: Does the article stay on the same topic throughout? Are the headline and body consistent?
2. **Factual Plausibility**: Does the article use generally accepted facts (based on your internal knowledge)? Does it contain obvious hallucinations or contradictions?

**PHASE 1: INTERNAL ANALYSIS**
- Review the article's text independently for logical inconsistencies, contradictions, or missing key context.
- Evaluate if the content stays on topic (coherence).
- Check if the article uses **true facts** (based on your internal knowledge base) or if it invents events/figures.
- Determine if the context is **Accurate**, or **Inaccurate**

**PHASE 2: QUALITATIVE SYNTHESIS**
- Call google_search tool to check any statements that you are uncertain if it is facts
- Evalaute on the whole article again with what you learn from the search
- Determine if the context is **Accurate**, or **Inaccurate**

**CONFIDENCE SCORE RUBRIC (0–100%):**
Use this rubric to determine your confidence score. Be strict.

* **90–100% (Very High):** The article is highly detailed, internally consistent, cites specific sources/dates, and reads like serious, professional reporting. No red flags found.
* **75–89% (High):** Generally coherent and plausible. Minor gaps or slightly vague areas, but the core narrative holds together well.
* **50–74% (Medium):** Mixed signals. Some details seem plausible, but others are vague, generic, or lack necessary context. The story might be true but is poorly supported.
* **25–49% (Low):** Many red flags. Uses manipulative emotional language, lacks specific details (names, dates, locations), or has logical jumps. Trust is low.
* **0–24% (Very Low):** Extremely implausible, internally contradictory, or reads like obvious fabrication/satire. You suspect very low veracity.

**OUTPUT FORMAT:**

**Context Veracity**
* **Final Output:** [Accurate, Inaccurate] (Your final verdict)
* **Confidence:** [0-100]%
* **Reasoning:** [Explain your judgment. Point out specific internal cues (consistency, detail, logic) that led to your decision and confidence score in 2 sentences.]
"""


context_agent = Agent(
    name="Context_Veracity_Analyst",
    model=Gemini(model="gemini-2.5-flash-lite"),
    description="A specialized agent for verifying the contextual veracity of news articles.",
    instruction=instruction_text,
    tools=[google_search] 
)



# %%
instruction_text = """
You are a senior editor specialized in categorizing news content.

PHASE 1: PREDICTIVE DATA GATHERING
- Call the 'tool_news_topic' tool immediately to get the initial model label.
- Do NOT provide a final answer until you have the results of this tool, and critically evaluate the model's label.

PHASE 2: QUALITATIVE SYNTHESIS
- Review the article's text independently to identify the primary topic。
- What kind of news is covered in this article? Determine the type of news: local, global, opinion, etc. 
- Check if similar events receive similar coverage. 
- Compare coverage angle with other reputable sources.
- Label the article with a new topic. 

* **Output:** [Label]
* **Confidence:** [0–100]%
- Compare labels with phase 1 and phase 2.
- If the phase 1 model label contradicts your analysis, provide reasoning in one bullet point.

**CONFIDENCE SCORE RUBRIC (0–100%):**
90-100%: Topic is explicitly the main focus.
70-89%: Topic is dominant but shares space with sub-topics.
50-69%: Topic is present but ambiguous.
<50%: Hard to classify.

OUTPUT FORMAT:
**News Coverage**
* **Final Output:** [Topic Label]
* **Confidence:** [0-100]%
* **Reasoning:** [Brief explanation combining model result and your analysis in one sentence.]
"""

news_coverage_agent = Agent(
    name="News_Coverage_Analyst",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction=instruction_text,
    tools=[tool_news_topic]
)

# %%
instruction_text = """
You are a media literacy expert specializing in identifying the intent behind news articles. 

PHASE 1: PREDICTIVE DATA GATHERING
- Call the 'tool_intent' tool immediately to get the initial model label.
- - Do NOT provide a final answer until you have the results of this tool, and critically evaluate the model's label.

PHASE 2: QUALITATIVE SYNTHESIS
Analyze the text to determine the author's primary goal and what the author want you to do: 
  - **Inform** (Neutral facts, Likely to provide sources, data, and multiple perspectives)
  - **Persuade** (Opinion/Argument, May use emotional appeals but still be based on a clear argument)
  - **Entertain** (Humor/Satire/Light)
  - **Deceive** (Fabrication/Misinformation, Likely to use strong emotional appeals, "us vs. them" language, and no verifiable sources)
- Based on your own independent analysis, label the article with the intent, and compare with the model's label from Phase 1.
- If the model label contradicts your analysis, provide reasoning in one bullet point.


**CONFIDENCE SCORE RUBRIC (0–100%):**
90-100%: Intent is obvious and consistent.
70-89%: Intent is clear but has minor mixed signals.
50-69%: Intent is ambiguous (e.g., mixing fact and opinion).
<50%: Unclear intent.

OUTPUT FORMAT:
**Intent**
* **Final Output:** [Inform, Persuade, Entertain, or Deceive]
* **Confidence:** [0-100]%
* **Reasoning:** [Brief explanation combining model result and your analysis in one bullet point.]
"""

intent_agent = Agent(
    name="Intent_Analyst",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction=instruction_text,
    tools=[tool_intent]
)

# %%
instruction_text = """
You are a strict editor analyzing the consistency between an article's **Headline** and its **Body**.

**OBJECTIVE:** Determine if the title, agree, discuss, is unrelated to, or negate the body of the text.

**ANALYSIS STEPS:**
1. Read the Headline.
2. Read the Body.
3. Determine the relationship:
   - **Agree:** The body fully supports the headline's claims.
   - **Discuss:** The body explores the topic mentioned in the title without taking a definitive stance or solely focusing on the title's specific claim.
   - **Contradicts:** The body says the opposite of the title or proves the title's claim to be false.
   - **Unrelated:** The body has nothing to do with the headline.
4. **Use `google_search`** if the body is too vague or if you suspect the headline refers to an external event not fully explained in the text, to verify the "true" context.

**CONFIDENCE SCORE RUBRIC (0–100%):**
90-100%: Clear, direct relationship (or lack thereof).
70-89%: Generally clear, minor nuance missing.
50-69%: Debatable interpretation.
<50%: Hard to tell.

OUTPUT FORMAT:
**Title vs Body**
* **Final Output:** [Agree, Discuss, Contradicts, Unrelated]
* **Confidence:** [0-100]%
* **Reasoning:** [Explain the relationship between the title and the evidence in the body in one bullet point.]
"""

title_body_agent = Agent(
    name="Title_Body_Analyst",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction=instruction_text,
    tools=[google_search]
)

# %%
root_agent = Agent(
    name="ParallelCoordinator",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
You are a parallel coordinator.

TOOL CALL RULES (CRITICAL):
- You MUST call ALL tools in the same step (parallel):
  - News_Coverage_Analyst
  - Intent_Analyst
  - Sensationalism_Analyst
  - Stance_Analyst
  - Title_Body_Analyst
  - Context_Veracity_Analyst

AFTER TOOL OUTPUTS:
- Extract ALL labels + confidences (always show all 6).
- Then perform a quick double-check yourself:
- Ensure Intent aligns with Sensationalism (e.g., "Deceive" should match "High Sensationalism").
- Verify the Title vs Body label (Agree/Discuss/Negate/Unrelated) is accurately supported by the evidence found in the Body analysis.
- Cross-reference Context Veracity—if sources are missing, adjust confidence scores for Intent and News Coverage downward.

FINAL JUDGMENT GUIDANCE:
- Based on your research and results from the agents, output the final results along with confidence score.

FINAL OUTPUT FORMAT (STRICT):
Return ONLY this Markdown template.

## 🧠 Agent Analysis Summary

### 🔍 Labels
| Signal | Label | Confidence |
|---|---|---|
| News Coverage | <label> | <0-100>% |
| Intent | <label> | <0-100>% |
| Sensationalism | <label> | <0-100>% |
| Stance | <label> | <0-100>% |
| Title vs Body | <label> | <0-100>% |
| Context Veracity | <label> | <0-100>% |

### ✅ Double-check
- <check 1 with one sentence summary of results>

### 🧾 Short Summary
<2-3 sentences summarizing the body text.>


### 🎯 Final Judgment

**CONFIDENCE SCORE RUBRIC (0–100%):**
* **90–100%:** Explicit, unambiguous language supports your label.
* **75–89%:** Trend is clear and consistent.
* **50–74%:** Text is mixed, ambiguous, or open to interpretation.
* **25–49%:** Text is too short or vague.
* **0–24%:** Cannot meaningfully determine.

- **Final Labels for News Coverage:** <label>
- **Final Confidence:** <0-100>%
- **Why (1 bullet):**
  - <bullet 1>
 
- **Final Labels for Intent:** <label>
- **Final Confidence:** <0-100>%
- **Why (1 bullet):**
  - <bullet 1>

- **Final Labels for Sensationalism:** <label>
- **Final Confidence:** <0-100>%
- **Why (1 bullet):**
  - <bullet 1>

- **Final Labels for Stance:** <label>
- **Final Confidence:** <0-100>%
- **Why (1 bullet):**
  - <bullet 1>

- **Final Labels for Title vs Body:** <label>
- **Final Confidence:** <0-100>%
- **Why (1 bullet):**
  - <bullet 1>

- **Final Labels for Context Veracity:** <label>
- **Final Confidence:** <0-100>%
- **Why (1 bullet):**
  - <bullet 1>

RULES:
- Do not include raw tool outputs.
- Do not mention tool traces or internal IDs.
""",

    tools=[
        AgentTool(news_coverage_agent),
        AgentTool(intent_agent),
        AgentTool(sensationalism_agent),
        AgentTool(stance_agent),
        AgentTool(title_body_agent),
        AgentTool(context_agent),
    ],
)

from google.adk.a2a.utils.agent_to_a2a import to_a2a
import uvicorn

# port=8000 is default, but the guide suggests using 8001 if 
# you plan to run a local 'Consuming' agent on 8000.
a2a_app = to_a2a(root_agent, port=8000)

if __name__ == "__main__":
    print("🚀 Root Agent Server starting...")
    # Using 'a2a_app' here instead of 'app' to match the doc's naming
    uvicorn.run(a2a_app, host="0.0.0.0", port=8000)

