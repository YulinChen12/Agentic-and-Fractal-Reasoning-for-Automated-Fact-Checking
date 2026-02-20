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
import asyncio
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
from serpapi import GoogleSearch
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

print("✅ ADK components imported successfully.")

from pydantic import BaseModel, Field
from typing import Dict, List

# %%
# Data loading and preprocessing functions
COLS = ["id","label","statement","subjects","speaker","job_title",
        "state_info","party_affiliation","barely_true_cnt","false_cnt",
        "half_true_cnt","mostly_true_cnt","pants_on_fire_cnt","context","justification"]

data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'pred_data/')

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
train_text = train_text.fillna("").astype(str)


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
# Model 4: Stance Classification
print("Loading Stance Model...")

# Ensure imports are available
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

try:
    stance_model_path = "../pred_models_training/stance_model"
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


# 4. Stance Tool
def tool_stance(article_text: str) -> dict:
    """Determines the political stance (support, deny, or neutral)."""
    sentences = get_sentences(article_text)
    try:
        result = predict_article_stance(sentences=sentences)
        return {"stance": result["final_label"]}
    except:
        return {"stance": "neutral"}
    


# %% [markdown]
# # Agents

# %% [markdown]
# Ensure agents speak the same language

# %%
from pydantic import BaseModel, Field, ConfigDict

class FactorAnalysis(BaseModel):
    # model_config = ConfigDict(extra='forbid')
    
    verdict: str = Field(description="The final label (e.g., 'sensational', 'support')")
    confidence: int = Field(description="Confidence score 0-100")
    fcot_reasoning: str = Field(description="2-3 sentence FCoT reasoning.")

class FactCheckFinalReport(BaseModel):
    # model_config = ConfigDict(extra='forbid')
    
    final_verdict: str = Field(description="The grand synthesis label.")
    overall_confidence: int = Field(ge=0, le=100)
    agent_signals: dict[str, FactorAnalysis]

# %%

os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"

retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504], # Retry on these HTTP errors
)

# %% [markdown]
# ## FCOT

# %%
FCOT_CORE = """
## FCoT Reasoning Protocol
You operate on the principles of Fractal Chain of Thought. Follow these stages:

1. **Local Thought Unit (TU):** Analyze the provided text snippet with a NARROW APERTURE. Focus only on linguistic features.
2. **Predictive Grounding:** Call your assigned predictive tool.
3. **Aperture Expansion:** Widen your context to include the tool's output and any provided peer results. 
4. **Reflective Update (RUM):** If the tool result contradicts your local analysis, perform a 'Retrospective Adjustment'. Explain WHY the discrepancy exists.
5. **Granularity Control:** Provide a final verdict that balances your qualitative analysis with the quantitative tool score.
"""

# %%
sensationalism_agent = Agent(
    name="Sensationalism_Analyst",
    model=Gemini(model="gemini-2.0-flash", retry_options=retry_config),
    output_schema=FactorAnalysis,
    instruction=FCOT_CORE + """
## LOCAL OBJECTIVE FUNCTION (LOF)
- **MAXIMIZE**: Precision in identifying emotional manipulation and clickbait architecture.
- **MINIMIZE**: 'Stylistic False Positives' where urgency or technical reporting is misclassified as sensationalism.

## FCoT REASONING PHASES

### PHASE 1: LOCAL THOUGHT UNIT (TU) - NARROW APERTURE
1.  **Linguistic Scan**: Analyze the text for high-intensity adjectives, superlatives, and emotional appeals.
2.  **Structural Check**: Evaluate if the headline's promise is fulfilled by the body content or if it relies on "shock value" gaps.
3.  **Preliminary Verdict**: Form an internal hypothesis of the 'sensationalism' label based purely on linguistic patterns.

### PHASE 2: CONTEXT APERTURE EXPANSION (CAO) - TOOL GROUNDING
1.  **Tool Execution**: Call `tool_sensationalism` to obtain the predictive model's label for this article.
2.  **Aperture Expansion**: Widened your context to compare your Phase 1 hypothesis with the tool’s output.

### PHASE 3: REFLECTIVE UPDATE (RUM) & SYNTHESIS
1.  **Reflective Alignment**: If your linguistic analysis contradicts the tool (e.g., tool says 'neutral' but you see heavy 'loaded' language), explain this discrepancy.
2.  **Retrospective Adjustment**: Finalize your verdict by integrating the base label from the tool with your qualitative "nuance."

## CONFIDENCE RUBRIC
- 90-100%: Total alignment between linguistic markers and tool results; high narrative-evidence consistency.
- 75-89%: Clear patterns present; minor stylistic ambiguity doesn't obscure the primary intent.
- 50-74%: Moderate certainty; stylistic urgency makes intent difficult to isolate or analysis deviates from tool.
- <50%: Insufficient data or contradictory signals that prevent a definitive classification.

## OUTPUT RULES
- Populate the `fcot_reasoning` field with a concise 2-3 sentence summary of your FCoT process.
- Ensure the verdict strictly matches: [sensational, neutral].
""",
    tools=[tool_sensationalism],
)

# %%
stance_agent = Agent(
    name="Stance_Analyst",
    model=Gemini(model="gemini-2.0-flash", retry_options=retry_config),
    output_schema=FactorAnalysis,
    instruction= """
## LOCAL OBJECTIVE FUNCTION (LOF)
- **MAXIMIZE**: Detection of nuanced rhetorical alignment, bias, or skepticism.
- **MINIMIZE**: Misclassification of "objective reporting" as "denial" or "unbiased" as "support."

## FCoT REASONING PHASES

### PHASE 1: LOCAL THOUGHT UNIT (TU) - NARROW APERTURE
1.  **Linguistic Tone Mapping**: Identify the author's voice. Does the language favor specific actors or dismiss certain arguments?
2.  **Consistency Check**: Verify if the stance remains stable or shifts when presenting evidence vs. commentary.
3.  **Preliminary Stance**: Form an internal hypothesis: Does this text Support, Deny, or remain Neutral?

### PHASE 2: CONTEXT APERTURE EXPANSION (CAO) - TOOL GROUNDING
1.  **Tool Execution**: Call `tool_stance` to get the predictive baseline.
2.  **Aperture Expansion**: Compare your internal hypothesis with the tool's statistical prediction.

### PHASE 3: REFLECTIVE UPDATE (RUM) & SYNTHESIS
1.  **Reflective Alignment**: Critically evaluate the tool. If your analysis detected sarcasm or subtle framing that the tool missed (or if the tool correctly identified a pattern you overlooked), document this.
2.  **Retrospective Adjustment**: Finalize the verdict. Your `fcot_reasoning` should explain exactly how the qualitative analysis and quantitative grounding converged.

## CONFIDENCE RUBRIC
- 90-100%: Explicit, unambiguous alignment between tool results and rhetorical evidence.
- 70-89%: Clear trend; minor stylistic nuance doesn't obscure the primary stance.
- 50-69%: Content is balanced/ambiguous, or your analysis deviates from the tool.
- <50%: Contradictory signals or text lacks sufficient markers to assign a stance.

## OUTPUT RULES
- Populate the `fcot_reasoning` field with a concise 2-3 sentence summary of your FCoT process.
- Ensure the verdict strictly matches: [support, deny, neutral].
""",
    tools=[tool_stance],
)

# %% [markdown]
# incorporate dual objective u

# %%
execute_web_search = Agent(
    name="Web_Search_Provider",
    model="gemini-2.0-flash",
    instruction="Search the web for factual grounding of specific claims.",
    tools=[google_search]
)

# %%
context_agent = Agent(
    name="Context_Veracity_Analyst",
    model=Gemini(model="gemini-2.0-flash"),
    output_schema=FactorAnalysis,
    description="A specialized FCoT agent for verifying factual and contextual veracity.",
    instruction="""
## LOCAL OBJECTIVE FUNCTION (LOF)
- **MAXIMIZE**: Historical and factual alignment through deep temporal grounding.
- **MINIMIZE**: Acceptence of "hallucinated" entities, dates, or causal relationships. (minimize sensationalism, find other source )

## FCoT REASONING PHASES

### PHASE 1: LOCAL THOUGHT UNIT (TU) - NARROW APERTURE
1.  **Internal Plausibility Check**: Identify specific claims (names, dates, statistics). Does the logic hold internally, or are there "logical jumps"?
2.  **Epistemic Gap Identification**: Flag specific statements that require external verification (e.g., a specific quote or a rare event).

### PHASE 2: CONTEXT APERTURE EXPANSION (CAO) - TEMPORAL RE-GROUNDING
1.  **External Verification**: Execute `Google Search` for the flagged gaps. 
2.  **Aperture Expansion**: Widen your context from the article's narrow claims to the broader global/historical record found in search results.

### PHASE 3: REFLECTIVE UPDATE (RUM) & RETROSPECTIVE ADJUSTMENT
1.  **Fact Reconciliation**: Compare search data with article claims. 
2.  **Retrospective Adjustment**: If the search proves a core claim false, you must retroactively devalue the entire article's veracity, even if it is "internally consistent."
3.  **Final Synthesis**: Populate the `fcot_reasoning` field by explaining how the search confirmed or debunked the internal logic.

## CONFIDENCE RUBRIC
- 90-100%: Article facts are independently verified by reputable external sources; no contradictions.
- 70-89%: Core narrative is plausible and supported; minor details are unverified but don't break the logic.
- 50-69%: Vague or generic claims that are hard to verify; "mixed signals" in search results.
- <50%: Clear factual contradictions found; internal logic fails once external context is applied.

## OUTPUT RULES
- Populate the `fcot_reasoning` field with a concise 2-3 sentence summary.
- The verdict must be strictly: [Accurate, Inaccurate].""",
    tools=[AgentTool(execute_web_search)] 
)

# %%

news_coverage_agent = Agent(
    name="News_Coverage_Analyst",
    model=Gemini(model="gemini-2.0-flash"),
    output_schema=FactorAnalysis,
    description="FCoT agent specializing in multi-scale news categorization.",
    instruction="""
## LOCAL OBJECTIVE FUNCTION (LOF)
- **MAXIMIZE**: Precision in identifying the primary thematic domain and geographical scope.
- **MINIMIZE**: Conceptual redundancy (e.g., mislabeling a 'Political' story as 'General' because it mentions a city name).

## FCoT REASONING PHASES

### PHASE 1: LOCAL THOUGHT UNIT (TU) - NARROW APERTURE
1.  **Topical Extraction**: Identify the "Who, What, Where" within the text. 
2.  **Granularity Check**: Determine the scale of the news. Is it a local incident, a national policy, or an international trend?
3.  **Preliminary Classification**: Form an internal hypothesis of the topic label (e.g., Politics, Tech, Health).

### PHASE 2: CONTEXT APERTURE EXPANSION (CAO) - TOOL GROUNDING
1.  **Tool Execution**: Call `tool_news_topic` to get the predictive baseline.
2.  **Aperture Expansion**: Compare your internal hypothesis with the tool’s output. Does the tool see a broader "Global" category while you see a "Local" one?

### PHASE 3: REFLECTIVE UPDATE (RUM) & MULTI-SCALE COORDINATION
1.  **Reflective Alignment**: Critically evaluate the tool. Predictive models often default to broad categories (e.g., "World News"). If the article is specifically about "Supply Chain Logistics," update the label to reflect that higher granularity.
2.  **Synthesis**: Populate the `fcot_reasoning` field by explaining the shift from the broad tool label to your specific, refined classification.

## CONFIDENCE RUBRIC
- 90-100%: Topic is explicitly the central focus; total alignment with tool.
- 70-89%: Topic is dominant but intersects with sub-topics; tool results are supportive.
- 50-69%: Topic is present but ambiguous or shares equal weight with another domain.
- <50%: Content is generic or spans too many categories to classify reliably.

## OUTPUT RULES
- Populate the `fcot_reasoning` field with a concise 2-3 sentence summary.
- The verdict must be a standardized topic label.""",
    tools=[tool_news_topic]
)

# %%
intent_agent = Agent(
    name="Intent_Analyst",
    model=Gemini(model="gemini-2.0-flash"),
    output_schema=FactorAnalysis,
    description="FCoT specialist in identifying rhetorical intent and authorial goals.",
    instruction= """
## LOCAL OBJECTIVE FUNCTION (LOF)
- **MAXIMIZE**: Transparency in identifying the author's underlying rhetorical goal (e.g., hidden persuasion).
- **MINIMIZE**: False categorization of "Opinion/Op-Ed" as "Deception" or "Satire" as "Informational."

## FCoT REASONING PHASES

### PHASE 1: LOCAL THOUGHT UNIT (TU) - NARROW APERTURE
1.  **Goal Extraction**: Analyze the "Call to Action." What does the author want the reader to think, feel, or do after reading?
2.  **Linguistic Marker Identification**: Look for "Us vs. Them" framing, emotional appeals, or the presence/absence of verifiable citations.
3.  **Preliminary Intent**: Classify based on the four categories: [Inform, Persuade, Entertain, Deceive].

### PHASE 2: CONTEXT APERTURE EXPANSION (CAO) - TOOL GROUNDING
1.  **Tool Execution**: Call `tool_intent` to obtain the statistical intent label.
2.  **Aperture Expansion**: Compare your internal hypothesis with the tool’s output. Does the tool detect "Deception" while you see "Persuasion"?

### PHASE 3: REFLECTIVE UPDATE (RUM) & DIALOGUE ALIGNMENT
1.  **Reflective Alignment**: Critically evaluate the tool. Predictive models often struggle with Satire (Entertain) vs. Deception. If the linguistic markers suggest humor or irony that the tool ignored, perform a **Retrospective Adjustment**.
2.  **Final Synthesis**: Populate the `fcot_reasoning` field. Explain how the synthesis of tool data and linguistic analysis confirms the primary intent.

## CONFIDENCE RUBRIC
- 90-100%: Intent is explicit, consistent, and matches tool results.
- 70-89%: Intent is clear but contains minor stylistic nuance (e.g., informative text with slight persuasive leaning).
- 50-69%: Intent is ambiguous (e.g., "Advertorial" content mixing fact and persuasion).
- <50%: Intent is obscured by contradictory markers or heavy sarcasm.

## OUTPUT RULES
- Populate the `fcot_reasoning` field with a concise 2-3 sentence summary of the RUM process.
- Verdict must be strictly: [Inform, Persuade, Entertain, Deceive].""",
    tools=[tool_intent]
)

# %%
title_body_agent = Agent(
    name="Title_Body_Analyst",
    model=Gemini(model="gemini-2.0-flash"),
    output_schema=FactorAnalysis,
    description="FCoT specialist in detecting semantic gaps between headlines and article content.",
    instruction="""
## LOCAL OBJECTIVE FUNCTION (LOF)
- **MAXIMIZE**: Detection of "headline-body gaps," bait-and-switch tactics, or semantic contradictions.
- **MINIMIZE**: False "Unrelated" labels for headlines that use metaphor or creative framing to describe the body content.

## FCoT REASONING PHASES

### PHASE 1: LOCAL THOUGHT UNIT (TU) - NARROW APERTURE
1.  **Direct Mapping**: Extract the core claim of the Headline. Scan the Body for direct supporting evidence of that specific claim.
2.  **Stance Alignment**: Does the body's tone match the headline's intensity? 
3.  **Preliminary Relationship**: Classify as [Agree, Discuss, Contradicts, Unrelated].

### PHASE 2: CONTEXT APERTURE EXPANSION (CAO) - EXTERNAL GROUNDING
1.  **Context Check**: If the body is vague or the headline refers to a specific event/entity not fully explained, execute `execute_web_search`.
2.  **Aperture Expansion**: Compare the "Headline-Body" relationship against the "True Event" context found in search results.

### PHASE 3: REFLECTIVE UPDATE (RUM) & FRACTAL SYNTHESIS
1.  **Reflective Alignment**: Critically evaluate if the body is "Discussing" the topic but failing to "Agree" with a sensational headline. This is a common tactic for plausible deniability.
2.  **Final Synthesis**: Populate the `fcot_reasoning` field. Explain the logic of the relationship (e.g., "The headline negates the body by claiming X happened, while the body text only discusses the possibility of X").

## CONFIDENCE RUBRIC
- 90-100%: Relationship is explicit and supported/debunked by direct textual evidence.
- 70-89%: Relationship is clear; minor semantic nuances in metaphors or puns.
- 50-69%: The body explores the topic but the connection to the headline's specific claim is ambiguous.
- <50%: Headline and body appear disconnected or require heavy inference to link.

## OUTPUT RULES
- Populate the `fcot_reasoning` field with a concise 2-3 sentence summary.
- Verdict must be strictly: [Agree, Discuss, Contradicts, Unrelated].""",
    tools=[AgentTool(execute_web_search)] 
)

# %%
from google.adk.agents import ParallelAgent

# Pass the Agent objects directly to sub_agents
factor_squad = ParallelAgent(
    name="Factor_Squad",
    sub_agents=[
        sensationalism_agent, 
        stance_agent, 
        context_agent,
        news_coverage_agent, 
        intent_agent, 
        title_body_agent
    ]
)

# %%
synthesizer_agent = Agent(
    name="Final_Synthesizer",
    model=Gemini(model="gemini-2.0-flash"),
    output_schema=FactCheckFinalReport,
    instruction="""
    You are the Final Synthesizer.
    
    1. **Recursive Synthesis**: Receive the 6 FactorAnalysis JSONs from the squad.
    2. **Inter-agent Reflectivity**: Identify if any agents disagree (e.g., if Context is Accurate but Intent is Deceive).
    3. **Retrospective Re-grounding**: If the Context_Veracity agent found a major factual error, force all other signals to be interpreted through that lens.
    4. **Output**: Generate the final FactCheckFinalReport JSON.
    """
)

root_agent = SequentialAgent(
    name="Fractal_FactCheck_Framework",
    sub_agents=[factor_squad, synthesizer_agent]
)

# %%
article_body = "HONG KONG, Dec 2 (Reuters) - Hong Kong's leader said on Tuesday a judge-led committee will investigate the cause of the city's deadliest fire in decades and review government oversight of renovation practices linked to the blaze that killed at least 151 people.\nPolice have arrested 13 people for suspected manslaughter, and 12 others in a related corruption probe. Officials said substandard plastic mesh and insulation foam used during renovation works fueled the rapid spread of the fire across seven high-rise towers.\nAuthorities said they aim to avoid similar tragedies by examining how the fire spread so quickly and the oversight failures around building renovations.\n\nSEARCH AND INVESTIGATION\nInvestigators have combed most of the damaged towers, finding victims in stairwells and rooftops as they attempted to escape. Around 30 people remain missing.\nSome civic groups have demanded transparency and accountability, while police have warned against \"politicising\" the tragedy. A student was detained and later released, and media reports indicate others are under investigation for possible sedition.\nInternational rights groups argue the government's response reflects broader suppression of criticism.\n\nRESIDENTS WARNED PRIOR\nResidents of Wang Fuk Court had previously raised concerns about fire hazards and flammable materials used on scaffolding. Tests showed mesh samples did not meet fire-retardant standards.\nOfficials also reported foam insulation accelerated the fire and that alarms were malfunctioning.\nOver 1,500 residents have been displaced into temporary housing. Authorities are offering emergency funds and fast-tracked document replacement.\n\nVIGILS AND RECOVERY\nThousands across Hong Kong and cities like Tokyo, Taipei, and London have held vigils. Several victims were migrant domestic workers.\nThe search of the most heavily damaged towers may take weeks, as responders work through collapsed interiors."
article_title = "Hong Kong orders judge-led probe into fire that killed 151"
prompt = f"Title: {article_title}\nBody: {article_body}"

# %%
import asyncio

async def main():
    root_agent = create_root_agent()
    runner = InMemoryRunner(agent=root_agent)
    prompt = f"Title: {article_title}\nBody: {article_body}"
    response = await runner.run_debug(prompt)
    return response

if __name__ == "__main__":
    response = asyncio.run(main())
    print(response)
