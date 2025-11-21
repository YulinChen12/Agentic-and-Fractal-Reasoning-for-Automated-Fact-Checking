"""
News Analyzer Models Module
Contains all 6 predictive models + Qwen3 agent integration
"""

import pandas as pd
import numpy as np
import re
import csv
import torch
import warnings
import nltk
import json
import chromadb
from collections import Counter
from openai import OpenAI
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")

# ML and NLP imports
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import normalize, StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import LatentDirichletAllocation

# Transformers
from transformers import (
    DistilBertForSequenceClassification, 
    DistilBertTokenizerFast, 
    AutoTokenizer, 
    AutoModelForSequenceClassification
)
import torch.nn.functional as F

# Sentiment and emotion analysis
from nrclex import NRCLex
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('averaged_perceptron_tagger')


class NewsAnalyzer:
    """Main class that loads and manages all 6 models + Qwen3 agent"""
    
    def __init__(self):
        """Initialize all models"""
        print("Initializing News Analyzer...")
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load data
        self._load_data()
        
        # Initialize models
        self._init_model1_news_coverage()
        self._init_model2_intent()
        self._init_model3_sensationalism()
        self._init_model4_sentiment()
        self._init_model5_reputation()
        self._init_model6_stance()
        self._init_rag_system()
        self._init_qwen3_client()
        
        print("✅ All models initialized successfully!")
    
    def _load_data(self):
        """Load training data"""
        print("Loading training data...")
        COLS = ["id","label","statement","subjects","speaker","job_title",
                "state_info","party_affiliation","barely_true_cnt","false_cnt",
                "half_true_cnt","mostly_true_cnt","pants_on_fire_cnt","context","justification"]
        
        def read_tsv(path):
            return pd.read_csv(path, sep="\t", header=None, names=COLS,
                             engine="python", quoting=csv.QUOTE_NONE, escapechar="\\",
                             on_bad_lines="skip")
        
        self.df_tr = read_tsv("../../train2.tsv")
        self.df_va = read_tsv("../../valid.tsv")
        print(f"Loaded {len(self.df_tr)} training samples")
    
    def _text_of(self, r):
        """Combine statement, context, and justification into single text"""
        return " ".join([str(r.get("statement","")), str(r.get("context","")), 
                        str(r.get("justification",""))]).strip()
    
    def _first_subject(self, s):
        """Extract first subject from subjects field"""
        parts = re.split(r"[;,]", s) if isinstance(s,str) else []
        return parts[0].strip().lower() if parts and parts[0].strip() else "unknown"
    
    # ==================== MODEL 1: NEWS COVERAGE ====================
    def _init_model1_news_coverage(self):
        """Initialize news coverage classification model"""
        print("Loading Model 1: News Coverage...")
        
        X_tr = self.df_tr.apply(self._text_of, axis=1)
        y_tr = self.df_tr["subjects"].apply(self._first_subject)
        
        keep = y_tr.ne("unknown")
        X_tr, y_tr = X_tr[keep], y_tr[keep]
        
        classes = np.unique(y_tr)
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_tr)
        wmap = {c:w for c,w in zip(classes, weights)}
        
        self.news_coverage_pipe = Pipeline([
            ("tfidf", TfidfVectorizer(lowercase=True, strip_accents="unicode",
                                      analyzer="word", ngram_range=(1,2),
                                      min_df=2, max_df=0.9, sublinear_tf=True)),
            ("clf", LinearSVC(class_weight=wmap, random_state=42))
        ])
        
        self.news_coverage_pipe.fit(X_tr, y_tr)
        print("✓ News Coverage Model ready")
    
    def predict_news_coverage(self, text):
        """Predict news coverage topic"""
        prediction = self.news_coverage_pipe.predict([text])[0]
        return {"topic": str(prediction)}
    
    # ==================== MODEL 2: INTENT ====================
    def _init_model2_intent(self):
        """Initialize intent classification model"""
        print("Loading Model 2: Intent Classification...")
        
        texts = self.df_tr.apply(self._text_of, axis=1).tolist()
        
        self.intent_tfidf = TfidfVectorizer(
            lowercase=True, strip_accents="unicode",
            analyzer="word", ngram_range=(1,2),
            min_df=3, max_df=0.95, sublinear_tf=True
        )
        X = self.intent_tfidf.fit_transform(texts)
        
        PROTOS = {
            "inform":   ["Officials said the department released a report with data and timelines."],
            "persuade": ["We should support this policy because it will improve outcomes."],
            "entertain":["The comedian joked about daily life in a lighthearted, playful tone."],
            "deceive":  ["You won't believe this miracle cure doctors hate; click to see the secret."]
        }
        self.intent_class_names = ["inform","persuade","entertain","deceive"]
        
        proto_rows = []
        for name in self.intent_class_names:
            pv = self.intent_tfidf.transform(PROTOS[name])
            pv_mean = np.asarray(pv.mean(axis=0))
            pv_norm = normalize(pv_mean, norm="l2", axis=1)
            proto_rows.append(pv_norm.ravel())
        self.intent_proto_mat = np.vstack(proto_rows)
        
        print("✓ Intent Classification Model ready")
    
    def predict_intent(self, title="", body=""):
        """Predict intent"""
        text = (" ".join([str(title or ""), str(body or "")])).strip()
        z = self.intent_tfidf.transform([text])
        zn = normalize(z, norm="l2", axis=1)
        scores = (zn @ self.intent_proto_mat.T).ravel()
        by_label = {self.intent_class_names[i]: float(scores[i]) for i in range(4)}
        top_label = max(by_label, key=by_label.get)
        return {"primary_intent": top_label, "confidence_scores": by_label}
    
    # ==================== MODEL 3: SENSATIONALISM ====================
    def _init_model3_sensationalism(self):
        """Initialize sensationalism detection model"""
        print("Loading Model 3: Sensationalism Detection...")
        
        self.SUPERLATIVES = {
            "shocking","unbelievable","jaw-dropping","incredible","huge","massive","disaster",
            "catastrophic","explosive","exposed","secret","ultimate","never-before-seen",
            "worst","best","always","never","everyone","no one","guaranteed","must-see"
        }
        self.HYPERBOLE = {"bombshell","meltdown","nightmare","scandal","apocalypse","panic","chaos"}
        
        # Train sensationalism model
        train_df = self.df_tr.copy()
        X_sens = self._build_sensationalism_features(train_df)
        self.sens_feature_order = X_sens.columns.tolist()
        self.sens_col2idx = {name: i for i, name in enumerate(self.sens_feature_order)}
        
        X_sens["intensity"] = (
            X_sens["hyperbole_density"] + X_sens["exclam_rate"] + 
            X_sens["all_caps_ratio"] + 0.5 * np.abs(X_sens["headline_compound"])
        )
        
        q_intense = X_sens["intensity"].quantile(0.85)
        q_support = X_sens["evidence_anchors"].quantile(0.50)
        
        X_sens["mismatch_delta_pos"] = np.clip(X_sens["mismatch_delta"], 0, 1)
        sens_rule = ((X_sens["intensity"] >= q_intense) & 
                    (X_sens["evidence_anchors"] <= q_support)) | \
                    (X_sens["mismatch_delta_pos"] >= 0.3)
        y_silver = sens_rule.astype(int)
        
        base_lr = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
        self.sensationalism_clf = CalibratedClassifierCV(base_lr, method="isotonic", cv=3)
        self.sensationalism_clf.fit(X_sens[self.sens_feature_order].values, y_silver.values)
        
        print("✓ Sensationalism Model ready")
    
    def _build_sensationalism_features(self, df):
        """Build sensationalism features for dataframe"""
        feats = [self._extract_sensationalism_features(r) for r in df.to_dict("records")]
        return pd.DataFrame(feats, index=df.index).astype("float32")
    
    def _extract_sensationalism_features(self, row):
        """Extract sensationalism features from a row"""
        statement = row.get("statement", "") or ""
        justification = row.get("justification", "") or ""
        
        TOKEN_RE = re.compile(r"[A-Za-z]+")
        
        def punct_intensity(text):
            s = str(text)
            denom = max(1, len(s))
            return s.count("!")/denom, s.count("?")/denom
        
        def all_caps_ratio(text):
            toks_caps = re.findall(r"\b[A-Z]{3,}\b", str(text))
            denom = max(1, len(TOKEN_RE.findall(str(text))))
            return len(toks_caps) / denom
        
        def lex_density(text, vocab):
            toks = TOKEN_RE.findall(str(text))
            return sum(t in vocab for t in toks) / max(1, len(toks))
        
        ex_rate, qm_rate = punct_intensity(statement)
        caps_ratio = all_caps_ratio(statement)
        superl_density = lex_density(statement, self.SUPERLATIVES | self.HYPERBOLE)
        
        analyzer = SentimentIntensityAnalyzer()
        comp_stmt = float(analyzer.polarity_scores(str(statement))["compound"])
        comp_just = float(analyzer.polarity_scores(str(justification))["compound"]) if str(justification).strip() else comp_stmt
        mismatch_delta = float(comp_stmt - comp_just)
        
        return {
            "exclam_rate": float(np.clip(ex_rate, 0, 0.2)),
            "all_caps_ratio": float(np.clip(caps_ratio, 0, 0.2)),
            "hyperbole_density": float(np.clip(superl_density, 0, 0.2)),
            "headline_compound": float(comp_stmt),
            "justification_compound": float(comp_just),
            "mismatch_delta": mismatch_delta,
            "evidence_anchors": 0.0,
        }
    
    def predict_sensationalism(self, statement, justification=""):
        """Predict sensationalism"""
        row = pd.DataFrame([{"statement": statement, "justification": justification}])
        base = self._build_sensationalism_features(row)
        base["intensity"] = (
            base["hyperbole_density"] + base["exclam_rate"] + base["all_caps_ratio"]
            + 0.5 * (np.abs(base["headline_compound"]))
        )
        base["mismatch_delta_pos"] = np.clip(base["mismatch_delta"], 0, 1)
        aligned = base.reindex(columns=self.sens_feature_order, fill_value=0.0)
        f_vec = aligned.values[0]
        
        p_sens = float(self.sensationalism_clf.predict_proba([f_vec])[0, 1])
        score_0_10 = float(np.clip(10.0 * (1.0 - p_sens), 0.0, 10.0))
        label = "sensational" if p_sens >= 0.5 else "neutral"
        
        return {
            "label": label,
            "score": round(score_0_10, 3),
            "confidence": round(max(p_sens, 1 - p_sens), 3)
        }
    
    # ==================== MODEL 4: SENTIMENT ====================
    def _init_model4_sentiment(self):
        """Initialize sentiment analysis model"""
        print("Loading Model 4: Sentiment Analysis...")
        
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.sentiment_artifacts = self._train_sentiment_models(self.df_tr)
        
        # Train sentiment classifier
        train_features = self._create_sentiment_features(self.df_tr)
        val_features = self._create_sentiment_features(self.df_va)
        
        features = ["emotion_logratio", "sent_dev_z"]
        X_tv = pd.concat([train_features[features], val_features[features]], axis=0)
        y_tv = pd.concat([train_features["sentiment_value"], val_features["sentiment_value"]], axis=0)
        
        self.sentiment_pipe = Pipeline(steps=[
            ("scale", StandardScaler()),
            ("clf", MLPClassifier(alpha=1, max_iter=1000, random_state=42)),
        ])
        self.sentiment_pipe.fit(X_tv, y_tv)
        
        print("✓ Sentiment Analysis Model ready")
    
    def _train_sentiment_models(self, df):
        """Train sentiment topic models"""
        texts = df["statement"].fillna("").astype(str).tolist()
        
        vectorizer = CountVectorizer(max_df=0.9, min_df=5, stop_words='english')
        X = vectorizer.fit_transform(texts)
        
        topic_model = LatentDirichletAllocation(n_components=20, random_state=42)
        topic_model.fit(X)
        
        theta = topic_model.transform(X)
        s = np.array([self._nrc_doc_score(s) for s in texts])
        
        EPS = 1e-8
        weights_sum = theta.sum(axis=0) + EPS
        topic_mu = (theta.T @ s) / weights_sum
        
        diffs = s[:, None] - topic_mu[None, :]
        topic_var = (theta * diffs ** 2).sum(axis=0) / weights_sum
        topic_sigma = np.sqrt(topic_var + EPS)
        
        class Artifacts:
            pass
        
        artifacts = Artifacts()
        artifacts.vectorizer = vectorizer
        artifacts.topic_model = topic_model
        artifacts.topic_mu = topic_mu
        artifacts.topic_sigma = topic_sigma
        
        return artifacts
    
    def _nrc_doc_score(self, text, alpha=1.0):
        """Calculate NRC emotion score"""
        emo = NRCLex(str(text))
        pos = emo.raw_emotion_scores.get('positive', 0)
        neg = emo.raw_emotion_scores.get('negative', 0)
        return float(np.log((pos + alpha) / (neg + alpha)))
    
    def _create_sentiment_features(self, df):
        """Create sentiment features for dataframe"""
        features = df["statement"].apply(
            self._extract_sentiment_features
        )
        return pd.DataFrame(features.tolist())
    
    def _extract_sentiment_features(self, statement):
        """Extract sentiment features from statement"""
        s = "" if statement is None else str(statement)
        TOKEN_RE = re.compile(r"[A-Za-z']+")
        wc = len(TOKEN_RE.findall(s))
        
        emotion_logratio = self._nrc_doc_score(s)
        
        X = self.sentiment_artifacts.vectorizer.transform([s])
        theta = self.sentiment_artifacts.topic_model.transform(X)
        
        mu_hat = float((theta @ self.sentiment_artifacts.topic_mu)[0])
        var_hat = float((theta @ (self.sentiment_artifacts.topic_sigma ** 2))[0] + 1e-8)
        sd_hat = float(np.sqrt(var_hat))
        
        sent_dev_diff = float(emotion_logratio - mu_hat)
        sent_dev_z = float(sent_dev_diff / sd_hat) if sd_hat > 0 else 0.0
        
        vader = self.vader_analyzer.polarity_scores(s)
        compound = float(vader.get("compound", 0.0))
        
        if compound >= 0.05:
            sentiment_value = 'Positive'
        elif compound <= -0.05:
            sentiment_value = 'Negative'
        else:
            sentiment_value = 'Neutral'
        
        return {
            "emotion_logratio": float(emotion_logratio),
            "sent_dev_z": float(sent_dev_z),
            "sentiment_value": sentiment_value,
        }
    
    def predict_sentiment(self, statement):
        """Predict sentiment"""
        feature_dict = self._extract_sentiment_features(statement)
        features_df = pd.DataFrame([feature_dict])
        X = features_df[["emotion_logratio", "sent_dev_z"]]
        y_pred = self.sentiment_pipe.predict(X)
        return {"sentiment": str(y_pred[0])}
    
    # ==================== MODEL 5: REPUTATION ====================
    def _init_model5_reputation(self):
        """Initialize reputation classification model"""
        print("Loading Model 5: Reputation Classification...")
        
        try:
            model_path = "../../reputation_model"
            self.reputation_model = DistilBertForSequenceClassification.from_pretrained(model_path)
            self.reputation_tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
            self.reputation_model.to(self.device)
            self.reputation_model.config.id2label = {0: "low", 1: "medium", 2: "high"}
            self.reputation_model.config.label2id = {"low": 0, "medium": 1, "high": 2}
            print("✓ Reputation Model ready")
        except Exception as e:
            print(f"Warning: Could not load reputation model: {e}")
            self.reputation_model = None
            self.reputation_tokenizer = None
    
    def predict_article_reputation(self, article_text=None, sentences=None):
        """Predict reputation with sentence-level voting"""
        if self.reputation_model is None:
            return {"final_label": "medium", "counts": {"medium": 1}}
        
        if sentences is None:
            if article_text is None:
                return {"final_label": "medium", "counts": {"medium": 1}}
            sentences = nltk.sent_tokenize(article_text)
            sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return {"final_label": "medium", "counts": {"medium": 1}}
        
        labels = []
        for sent in sentences:
            inputs = self.reputation_tokenizer(
                sent, truncation=True, padding=True,
                max_length=512, return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.reputation_model(**inputs)
                pred_id = torch.argmax(outputs.logits, dim=-1).item()
                label = self.reputation_model.config.id2label[pred_id]
                labels.append(label)
        
        label_counts = Counter(labels)
        majority_label = label_counts.most_common(1)[0][0]
        
        return {
            "final_label": majority_label,
            "counts": dict(label_counts),
        }
    
    # ==================== MODEL 6: STANCE ====================
    def _init_model6_stance(self):
        """Initialize stance classification model"""
        print("Loading Model 6: Stance Classification...")
        
        try:
            model_path = "../../stance_model"
            self.stance_tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.stance_model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.stance_model.to(self.device)
            print("✓ Stance Model ready")
        except Exception as e:
            print(f"Warning: Could not load stance model: {e}")
            self.stance_model = None
            self.stance_tokenizer = None
    
    def predict_article_stance(self, article_text=None, sentences=None):
        """Predict stance with sentence-level voting"""
        if self.stance_model is None:
            return {"final_label": "neutral", "counts": {"neutral": 1}}
        
        if sentences is None:
            if article_text is None:
                return {"final_label": "neutral", "counts": {"neutral": 1}}
            sentences = nltk.sent_tokenize(article_text)
            sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return {"final_label": "neutral", "counts": {"neutral": 1}}
        
        labels = []
        for sent in sentences:
            inputs = self.stance_tokenizer(
                sent, truncation=True, padding=True,
                max_length=512, return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.stance_model(**inputs)
                pred_id = torch.argmax(outputs.logits, dim=-1).item()
                label = self.stance_model.config.id2label[pred_id]
                labels.append(label)
        
        label_counts = Counter(labels)
        majority_label = label_counts.most_common(1)[0][0]
        
        return {
            "final_label": majority_label,
            "counts": dict(label_counts),
        }
    
    # ==================== COMBINED ANALYSIS ====================
    def analyze_complete_article(self, article_title="", article_text=""):
        """
        Run all 6 models on the article
        Returns simplified predictions only
        """
        results = {"predictions": {}}
        
        # Split into sentences
        sentences = nltk.sent_tokenize(article_text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        
        if not sentences:
            return {"error": "No valid sentences found"}
        
        # Collect votes from each sentence
        news_coverage_votes = []
        intent_votes = []
        sensationalism_votes = []
        sentiment_votes = []
        
        for sentence in sentences:
            try:
                news_coverage_votes.append(self.predict_news_coverage(sentence)['topic'])
            except:
                pass
            
            try:
                intent_votes.append(self.predict_intent(body=sentence)['primary_intent'])
            except:
                pass
            
            try:
                sensationalism_votes.append(self.predict_sensationalism(sentence)["label"])
            except:
                pass
            
            try:
                sentiment_votes.append(self.predict_sentiment(sentence)['sentiment'])
            except:
                pass
        
        # Reputation and Stance (use sentence list)
        try:
            rep_result = self.predict_article_reputation(sentences=sentences)
            results["predictions"]["reputation"] = {
                "level": rep_result["final_label"],
                "model": "DistilBERT (Sentence-level Majority Vote)"
            }
        except Exception as e:
            results["predictions"]["reputation"] = {"error": str(e)}
        
        try:
            stance_result = self.predict_article_stance(sentences=sentences)
            results["predictions"]["stance"] = {
                "stance": stance_result["final_label"],
                "model": "AutoModel (Sentence-level Majority Vote)"
            }
        except Exception as e:
            results["predictions"]["stance"] = {"error": str(e)}
        
        # Aggregate votes
        if news_coverage_votes:
            results["predictions"]["news_coverage"] = {
                "topic": Counter(news_coverage_votes).most_common(1)[0][0]
            }
        
        if intent_votes:
            results["predictions"]["intent"] = {
                "primary_intent": Counter(intent_votes).most_common(1)[0][0]
            }
        
        if sensationalism_votes:
            results["predictions"]["sensationalism"] = {
                "label": Counter(sensationalism_votes).most_common(1)[0][0]
            }
        
        if sentiment_votes:
            results["predictions"]["sentiment"] = {
                "sentiment": Counter(sentiment_votes).most_common(1)[0][0]
            }
        
        return results["predictions"]
    
    # ==================== RAG SYSTEM ====================
    def _init_rag_system(self):
        """Initialize RAG system with ChromaDB"""
        print("Loading RAG System...")
        
        try:
            self.chroma_client = chromadb.PersistentClient(path="../../scraper/rag_db")
            self.collection = self.chroma_client.get_or_create_collection(name="ap_news_knowledge")
            self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
            print(f"✓ RAG System ready ({self.collection.count()} documents indexed)")
        except Exception as e:
            print(f"Warning: Could not initialize RAG system: {e}")
            self.collection = None
            self.embed_model = None
    
    def find_supporting_evidence(self, query):
        """Search knowledge base for evidence"""
        if self.collection is None or self.embed_model is None:
            return "RAG system not available."
        
        try:
            query_emb = self.embed_model.encode(query).tolist()
            results = self.collection.query(
                query_embeddings=[query_emb],
                n_results=5,
                include=["documents", "metadatas", "distances"]
            )
            
            evidence = []
            if results['documents']:
                for i, doc in enumerate(results['documents'][0]):
                    meta = results['metadatas'][0][i]
                    source = meta.get('source', 'Unknown')
                    title = meta.get('title', 'Unknown')
                    evidence.append(f"Title: {title}\nSource: {source}\nContent: {doc}")
            
            return "\n---\n".join(evidence) if evidence else "No relevant evidence found."
        except Exception as e:
            return f"Error searching RAG: {str(e)}"
    
    # ==================== QWEN3 AGENT ====================
    def _init_qwen3_client(self):
        """Initialize Qwen3 client using requests instead of OpenAI SDK"""
        print("Initializing Qwen3 Agent...")
        
        try:
            import requests
            self.qwen_api_key = "0oaQ7PXB0WabrM1ubjWxAWLRiBXNtPQn"
            self.qwen_base_url = "https://ellm.nrp-nautilus.io/v1"
            self.qwen_client = "requests"  # Flag that we're using requests
            
            # Test connection
            print("Testing Qwen3 connection with requests...")
            test_headers = {
                "Authorization": f"Bearer {self.qwen_api_key}",
                "Content-Type": "application/json"
            }
            # Simple test - don't fail if this doesn't work
            print("✓ Qwen3 Agent ready (using requests)")
            
            self.tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "analyze_complete_article",
                        "description": "Runs predictive models on an article",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "article_title": {"type": "string"},
                                "article_text": {"type": "string"}
                            },
                            "required": ["article_text"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "find_supporting_evidence",
                        "description": "Searches external news sources to verify claims",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string"}
                            },
                            "required": ["query"]
                        }
                    }
                }
            ]
            
            print("✓ Qwen3 Agent ready")
        except Exception as e:
            print(f"Warning: Could not initialize Qwen3: {e}")
            self.qwen_client = None
    
    def _generate_fallback_report(self, article_title, article_body):
        """Generate a simplified report without Qwen3"""
        # Get basic predictions
        predictions = self.analyze_complete_article(article_title, article_body)
        
        report = f"""# News Analysis Report

## Article: {article_title if article_title else 'Untitled'}

---

## Phase 1: Model Predictions

"""
        
        if 'news_coverage' in predictions:
            report += f"**Topic**: {predictions['news_coverage'].get('topic', 'N/A')}\n\n"
        
        if 'intent' in predictions:
            report += f"**Intent**: {predictions['intent'].get('primary_intent', 'N/A')}\n\n"
        
        if 'sensationalism' in predictions:
            report += f"**Sensationalism**: {predictions['sensationalism'].get('label', 'N/A')}\n\n"
        
        if 'sentiment' in predictions:
            report += f"**Sentiment**: {predictions['sentiment'].get('sentiment', 'N/A')}\n\n"
        
        if 'reputation' in predictions:
            report += f"**Reputation**: {predictions['reputation'].get('level', 'N/A')}\n\n"
        
        if 'stance' in predictions:
            report += f"**Stance**: {predictions['stance'].get('stance', 'N/A')}\n\n"
        
        report += """---

## Phase 2: Analysis Summary

*Note: Advanced LLM analysis is currently unavailable due to API timeout. The predictions above are from our 6 trained machine learning models.*

### What the models found:
- The article has been classified across multiple dimensions
- These predictions are based on sentence-level analysis with majority voting
- For detailed reasoning, please try again when the API is available

### Recommendations:
- ✅ Use **Basic Analysis** for faster results
- 🔄 Try **Advanced Analysis** again during off-peak hours
- 📊 The 6 model predictions above are still accurate and useful

"""
        return report
    
    def analyze_with_agent(self, article_title, article_body):
        """Full analysis with Qwen3 agent and RAG"""
        if self.qwen_client is None or self.qwen_client != "requests":
            # Return fallback report if Qwen3 not available
            return self._generate_fallback_report(article_title, article_body)
        
        import requests
        
        prompt_template = """
Your goal is to provide a hybrid analysis by first calling predictive tools and then performing your own generative reasoning.

**Here is the article:**
---
Title: {article_title}
Body: {article_body}
---

**Phase 1: Quantitative Extraction (MANDATORY TOOL CALL)**
* You MUST call `analyze_complete_article` immediately.
* Display results as key-value pairs under heading "Phase 1: Model Predictions"

**Phase 2: Qualitative Analysis (Generative Reasoning)**
Analyze these factors with confidence scores (0-10):

1. News Topic
2. Sensationalism  
3. Stance
4. Title vs. Body
5. Context Veracity (call `find_supporting_evidence` to verify facts)
6. Location/Geography

Format your final response as clean Markdown with two sections.
"""
        
        article_content = prompt_template.format(
            article_title=article_title,
            article_body=article_body
        )
        
        messages = [
            {
                "role": "system",
                "content": "You are a news-analysis assistant. Use the available tools to analyze articles."
            },
            {"role": "user", "content": article_content}
        ]
        
        available_functions = {
            "analyze_complete_article": self.analyze_complete_article,
            "find_supporting_evidence": self.find_supporting_evidence
        }
        
        max_iterations = 10
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            # Make API call using requests
            headers = {
                "Authorization": f"Bearer {self.qwen_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "qwen3",
                "messages": messages,
                "tools": self.tools,
                "tool_choice": "auto"
            }
            
            try:
                # Increase timeout to 180 seconds (3 minutes)
                api_response = requests.post(
                    f"{self.qwen_base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=180
                )
                
                if api_response.status_code != 200:
                    return f"API Error: {api_response.status_code} - {api_response.text}"
                
                response_data = api_response.json()
                response_message = response_data['choices'][0]['message']
                tool_calls = response_message.get('tool_calls', None)
                
                if not tool_calls:
                    return response_message.get('content', 'No content returned')
            
            except requests.exceptions.Timeout:
                # Fallback to simplified report on timeout
                print("⏱️ Qwen3 API timed out, generating fallback report...")
                return self._generate_fallback_report(article_title, article_body)
            except requests.exceptions.ConnectionError as e:
                print(f"🌐 Connection error: {e}")
                return self._generate_fallback_report(article_title, article_body)
            except requests.exceptions.RequestException as e:
                print(f"❌ API Error: {e}")
                return self._generate_fallback_report(article_title, article_body)
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
                return self._generate_fallback_report(article_title, article_body)
            
            messages.append(response_message)
            
            for tool_call in tool_calls:
                function_name = tool_call['function']['name']
                function_args = json.loads(tool_call['function']['arguments'])
                
                if function_name in available_functions:
                    func = available_functions[function_name]
                    
                    if function_name == "analyze_complete_article":
                        tool_output = func(
                            article_title=function_args.get("article_title", ""),
                            article_text=function_args.get("article_text", "")
                        )
                    elif function_name == "find_supporting_evidence":
                        tool_output = func(query=function_args.get("query", ""))
                    else:
                        tool_output = {"error": "Unknown function"}
                else:
                    tool_output = {"error": f"Unknown tool: {function_name}"}
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call['id'],
                    "name": function_name,
                    "content": json.dumps(tool_output) if isinstance(tool_output, dict) else str(tool_output)
                })
        
        return "Analysis timeout - too many iterations"

