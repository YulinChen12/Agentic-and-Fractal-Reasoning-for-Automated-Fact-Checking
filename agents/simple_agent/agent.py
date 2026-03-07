import os
import sys
import warnings
from collections import Counter
import nltk
from dotenv import load_dotenv

# ADK Imports
from google.adk.agents import Agent
from google.adk.tools import AgentTool, google_search
from google.adk.models.google_llm import Gemini
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.genai import types
import uvicorn


from pathlib import Path
import sys

current_file = Path(__file__).resolve()
project_root = current_file.parents[2]  
predictors_dir = project_root / "pred_models_training"

sys.path.insert(0, str(predictors_dir))

from predictors import (
    predict_news_coverage,
    predict_intent,
    predict_sensationalism,
    predict_article_stance,
)

warnings.filterwarnings("ignore")

# Ensure NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# -------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------

def get_sentences(text: str):
    """Helper to split text into sentences for granular analysis."""
    try:
        sentences = nltk.sent_tokenize(text or "")
        # Filter out very short fragments
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        if not sentences and text.strip():
            sentences = [text.strip()]
        return sentences if sentences else []
    except Exception:
        return [text.strip()] if text else []

# -------------------------------------------------------------------------
# Tool Wrappers (Connecting ADK to Predictors.py)
# -------------------------------------------------------------------------

def tool_news_topic(article_text: str) -> dict:
    print("\n   [⚙️ MODEL EXECUTING] 🟢 'tool_news_topic' running...")
    sentences = get_sentences(article_text)
    votes = []
    for s in sentences:
        try:
            res = predict_news_coverage(s)
            # FIX: Ignore missing data labels ("None", "nan")
            if res and res.get("label") and str(res["label"]) not in ["None", "nan", "unknown"]:
                votes.append(res["label"])
        except Exception as e:
            print(f"      ❌ tool_news_topic error: {e}")
            continue
            
    topic = Counter(votes).most_common(1)[0][0] if votes else "unknown"
    
    result = {"news_coverage": topic}
    print(f"   [✅ MODEL OUTPUT] 🟢 'tool_news_topic' returned: {result}")
    return result

def tool_intent(article_text: str) -> dict:
    print("\n   [⚙️ MODEL EXECUTING] 🟢 'tool_intent' running...")
    sentences = get_sentences(article_text)
    votes = []
    for s in sentences:
        try:
            res = predict_intent(title="", body=s)
            if res and res.get("label"):
                votes.append(res["label"])
        except Exception as e:
            # FIX: Stop failing silently! Print the actual error.
            print(f"      ❌ tool_intent error: {e}")
            continue
            
    intent = Counter(votes).most_common(1)[0][0] if votes else "unknown"
    
    result = {"intent": intent}
    print(f"   [✅ MODEL OUTPUT] 🟢 'tool_intent' returned: {result}")
    return result

def tool_sensationalism(article_text: str) -> dict:
    print("\n   [⚙️ MODEL EXECUTING] 🟢 'tool_sensationalism' running...")
    sentences = get_sentences(article_text)
    votes = []
    for s in sentences:
        try:
            res = predict_sensationalism(statement=s)
            if res and res.get("label"):
                votes.append(str(res["label"]))
        except Exception as e:
            print(f"      ❌ tool_sensationalism error: {e}")
            continue
            
    final_label = Counter(votes).most_common(1)[0][0] if votes else "neutral"
    
    result = {"sensationalism": final_label}
    print(f"   [✅ MODEL OUTPUT] 🟢 'tool_sensationalism' returned: {result}")
    return result

def tool_stance(article_text: str) -> dict:
    print("\n   [⚙️ MODEL EXECUTING] 🟢 'tool_stance' running...")
    try:
        res = predict_article_stance(article_text=article_text)
        label = res.get("label", "neutral")
    except Exception as e:
        print(f"      ❌ tool_stance error: {e}")
        label = "neutral"
        
    result = {"stance": label}
    print(f"   [✅ MODEL OUTPUT] 🟢 'tool_stance' returned: {result}")
    return result

# -------------------------------------------------------------------------
# Agent Configurations
# -------------------------------------------------------------------------

# Load the .env file
load_dotenv() 

# Retrieve the key
api_key = os.getenv("GOOGLE_API_KEY")

# Check if it loaded correctly
if not api_key:
    raise ValueError("❌ GOOGLE_API_KEY not found! Make sure you created the .env file.")

# Agent Definitions

retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

# 1. Sensationalism Agent
sensationalism_agent = Agent(
    name="Sensationalism_Analyst",
    model=Gemini(model="gemini-3-flash-preview", retry_options=retry_config),
    instruction=(
        "Analyze the provided article for sensationalism."
        "Output your final verdict ('sensational' or 'neutral'), a confidence score (0-100), and a brief 2-3 sentence reasoning."
    )
)

# 2. Stance Agent
stance_agent = Agent(
    name="Stance_Analyst",
    model=Gemini(model="gemini-3-flash-preview", retry_options=retry_config),
    instruction=(
        "Determine the author's stance in the provided article."
        "Output your final verdict ('support', 'deny', or 'neutral'), a confidence score (0-100), and a brief 2-3 sentence reasoning."
    )
)

# 3. Context Veracity Agent
context_agent = Agent(
    name="Context_Veracity_Analyst",
    model=Gemini(model="gemini-3-flash-preview"),
    description="A specialized agent for verifying the contextual veracity of news articles.",
    instruction=(
        "Evaluate the factual accuracy and contextual coherence of the provided article."
        "Output your final verdict ('Accurate' or 'Inaccurate'), a confidence score (0-100), and a brief 2-3 sentence reasoning."
    )
)

# 4. News Coverage Agent
news_coverage_agent = Agent(
    name="News_Coverage_Analyst",
    model=Gemini(model="gemini-3-flash-preview", retry_options=retry_config),
    instruction=(
        "Categorize the primary topic of the provided news article."
        "Output your final topic label as the verdict, a confidence score (0-100), and a brief 2-3 sentence reasoning."
    )
)

# 5. Intent Agent
intent_agent = Agent(
    name="Intent_Analyst",
    model=Gemini(model="gemini-3-flash-preview", retry_options=retry_config),
    instruction=(
        "Identify the author's primary intent in writing the provided article."
        "Output your final verdict ('Inform', 'Persuade', 'Entertain', or 'Deceive'), a confidence score (0-100), and a brief 2-3 sentence reasoning."
    )
)

# 6. Title vs Body Agent
title_body_agent = Agent(
    name="Title_Body_Analyst",
    model=Gemini(model="gemini-3-flash-preview", retry_options=retry_config),
    instruction=(
        "Determine the relationship between the article's headline and its body text."
        "Output your final verdict ('Agree', 'Discuss', 'Contradicts', or 'Unrelated'), a confidence score (0-100), and a brief 2-3 sentence reasoning."
    )
)

# -------------------------------------------------------------------------
# Root Coordinator
# -------------------------------------------------------------------------

root_agent = Agent(
    name="ParallelCoordinator",
    model=Gemini(model="gemini-3-flash-preview", retry_options=retry_config),
    instruction="""
You are a parallel coordinator.

FINAL JUDGMENT GUIDANCE:
- Based on your research and results from the agents, output the final results along with the confidence score.

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
- **Why:**
 
- **Final Labels for Intent:** <label>
- **Final Confidence:** <0-100>%
- **Why:**

- **Final Labels for Sensationalism:** <label>
- **Final Confidence:** <0-100>%
- **Why:**

- **Final Labels for Stance:** <label>
- **Final Confidence:** <0-100>%
- **Why:**

- **Final Labels for Title vs Body:** <label>
- **Final Confidence:** <0-100>%
- **Why:**

- **Final Labels for Context Veracity:** <label>
- **Final Confidence:** <0-100>%
- **Why:**
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

# import asyncio
# from google.adk.runners import InMemoryRunner

# async def main():
#     runner = InMemoryRunner(agent=root_agent)
#     prompt = "Hello, how does this work?"
#     response = await runner.run_debug(prompt)
    
#     print(response)

# if __name__ == "__main__":
#     asyncio.run(main())

# -------------------------------------------------------------------------
# Server Execution
# -------------------------------------------------------------------------

'''
a2a_app = to_a2a(root_agent, port=8000)

if __name__ == "__main__":
    print("🚀 Root Agent Server starting...")
    # Using 'a2a_app' here instead of 'app' to match the doc's naming
    uvicorn.run(a2a_app, host="0.0.0.0", port=8000)
'''
