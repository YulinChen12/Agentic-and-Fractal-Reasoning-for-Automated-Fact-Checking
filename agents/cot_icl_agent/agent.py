# %%
import os
import sys
import warnings
from collections import Counter
import nltk
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime

# -------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------
# ADK Imports
from google.adk.agents import Agent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.tools import AgentTool, google_search
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.genai import types
import uvicorn

# Predictors API
from pathlib import Path
import sys

current_file = Path(__file__).resolve()
project_root = current_file.parents[2]

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from pred_models_training.predictors import (
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
    print("\n'tool_news_topic' running...")
    sentences = get_sentences(article_text)
    votes = []
    for s in sentences:
        try:
            res = predict_news_coverage(s)
            if res and res.get("label") and str(res["label"]) not in ["None", "nan", "unknown"]:
                votes.append(res["label"])
        except Exception as e:
            print(f"tool_news_topic error: {e}")
            continue
            
    topic = Counter(votes).most_common(1)[0][0] if votes else "unknown"
    
    result = {"news_coverage": topic}
    print(f"'tool_news_topic' returned: {result}")
    return result

def tool_intent(article_text: str) -> dict:
    print("\n 'tool_intent' running...")
    sentences = get_sentences(article_text)
    votes = []
    for s in sentences:
        try:
            res = predict_intent(title="", body=s)
            if res and res.get("label"):
                votes.append(res["label"])
        except Exception as e:
            print(f"tool_intent error: {e}")
            continue
            
    intent = Counter(votes).most_common(1)[0][0] if votes else "unknown"
    
    result = {"intent": intent}
    print(f"'tool_intent' returned: {result}")
    return result

def tool_sensationalism(article_text: str) -> dict:
    print("\n'tool_sensationalism' running...")
    sentences = get_sentences(article_text)
    votes = []
    for s in sentences:
        try:
            res = predict_sensationalism(statement=s)
            if res and res.get("label"):
                votes.append(str(res["label"]))
        except Exception as e:
            print(f"tool_sensationalism error: {e}")
            continue
            
    final_label = Counter(votes).most_common(1)[0][0] if votes else "neutral"
    
    result = {"sensationalism": final_label}
    print(f"'tool_sensationalism' returned: {result}")
    return result

def tool_stance(article_text: str) -> dict:
    print("\n'tool_stance' running...")
    try:
        res = predict_article_stance(article_text=article_text)
        label = res.get("label", "neutral")
    except Exception as e:
        print(f"tool_stance error: {e}")
        label = "neutral"
        
    result = {"stance": label}
    print(f"'tool_stance' returned: {result}")
    return result

# %%
import json
import asyncio

def load_train_articles(path = project_root / "data" / "gen_data" / "train_article.json"):
    if not os.path.exists(path):
        print(f"Error: File not found at {path}")
        return []
    
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    return data.get("articles", [])

# %%
training_data = load_train_articles()

# %%
execute_web_search = Agent(
    name="Web_Search_Provider",
    model="gemini-3-flash-preview",
    instruction="Search the web for factual grounding of specific claims.",
    tools=[google_search]
)

# %%

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

retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504], # Retry on these HTTP errors
)

# 1. Sensationalism Agent
sensationalism_agent = Agent(
    name="Sensationalism_Analyst",
    model=Gemini(
        model="gemini-3-flash-preview",
        retry_options=retry_config
    ),
    instruction=(
        "You are a senior linguistic editor specializing in media bias. You follow a strict two-phase analysis protocol:\n\n"

        "### REFERENCE LIBRARY (Human-Labeled Examples)\n"
        "Use these 7 examples to calibrate your judgment, focus on the sensationalism label to learn patterns to help you analyze.\n" f"{training_data}\n\n" 
        "CRTICITAL: Share one sentence with a pattern you learned from reading the training articles and label"
        
        "PHASE 1: PREDICTIVE DATA GATHERING\n"
        "- Call the 'tool_sensationalism' tool immediately to get the initial model label.\n"
        "- - Do NOT provide a final answer until you have the results of this tool, and critically evaluate the model's label.\n\n"
        
        "PHASE 2: QUALITATIVE SYNTHESIS\n"
        "- Review the article's text independently for sensationalist phrasing and dramatic claims.\n"
        "- Compare the emotional tone of the headline vs. the content.\n"
        "- Determine if the content prioritizes shock value and narrative over verifiable facts.\n"
        "- Based on your analysis, label the sensationalism for this article. If your independent analysis of "
        "linguistic patterns (e.g., use of exclamation, intense adjectives) contradicts the phase 1 model, "
        "provide clear reasoning for the discrepancy in three sentences.\n\n"
        
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
        "* **Reasoning:** [Explain how you synthesized the model's prediction with your own analysis in 3 bullet points.]"
    ),
    tools=[tool_sensationalism],
    output_key="sensationalism_report"
)

# 2. Stance Agent
stance_agent = Agent(
    name="Stance_Analyst",
    model=Gemini(model="gemini-3-flash-preview", retry_options=retry_config),
    instruction=(
        "You are a senior linguistic editor. You follow a strict two-phase analysis protocol:\n\n"

        "### REFERENCE LIBRARY (Human-Labeled Examples)\n"
        "Use these 7 examples to calibrate your judgment. Pay close attention to how "
        "sensationalism or tone influences the final 'Stance' label:\n"
        f"{training_data}\n\n" 
        "CRTICITAL: Share one sentence with a pattern you learned from reading the training articles and label"
        
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

# 3. Context Veracity Agent (Uses Google Search)
current_date = datetime.now().strftime("%B %d, 2026")

instruction_text = f"""
## TEMPORAL ANCHORING
- **CURRENT DATE**: {current_date}
- **CRITICAL RULE**: Do NOT rely on internal training data for events occurring. If a claim involves 2025 or 2026, you MUST treat results from execute_web_search tool as the primary source of truth, then cite your source in the explanation, and label which search terms you used in the reasoning.

You are a senior investigative fact-checker specialized in analyzing **Context Veracity**.
Your task is to evaluate the truthfulness and reliability of the article below based strictly on:
1. **Contextual Coherence**: Does the article stay on the same topic throughout? Are the headline and body consistent?
2. **Factual Plausibility**: Does the article use generally accepted facts (based on your internal knowledge)? Does it contain obvious hallucinations or contradictions?

### REFERENCE LIBRARY (Human-Labeled Examples)
Use these 7 examples to calibrate your judgment, focus on the context_veracity label to learn patterns to help you analyze. {training_data}
CRTICITAL: Share one sentence with a pattern you learned from reading the training articles and label.
        

**PHASE 1: INTERNAL ANALYSIS**
- Review the article's text independently for logical inconsistencies, contradictions, or missing key context.
- Evaluate if the content stays on topic (coherence).
- Check if the article uses **true facts** (based on your internal knowledge base) or if it invents events/figures.
- Determine if the context is **Accurate**, or **Inaccurate**

**PHASE 2: QUALITATIVE SYNTHESIS**
- Call execute_web_search tool to check any statements that you are uncertain if it is facts
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
    model=Gemini(model="gemini-3-flash-preview"),
    description="A specialized agent for verifying the contextual veracity of news articles.",
    instruction=instruction_text,
    tools=[AgentTool(execute_web_search)] 
)



# 4. News Coverage Agent
instruction_text = f"""
You are a senior editor specialized in categorizing news content.
### REFERENCE LIBRARY (Human-Labeled Examples)
Use these 7 examples to calibrate your judgment, focus on the coverage label to learn patterns to help you analyze. {training_data}
CRTICITAL:  Share one sentence with a pattern you learned from reading the training articles and label

PHASE 1: PREDICTIVE DATA GATHERING
- Call the 'tool_news_topic' tool immediately to get the initial model label.
- Do NOT provide a final answer until you have the results of this tool, and critically evaluate the model's label.

PHASE 2: QUALITATIVE SYNTHESIS
- Review the article's text independently to identify the primary topic
- What kind of news is covered in this article? Determine the type of news: local, global, opinion, etc. 
- Check if similar events receive similar coverage. 
- Compare coverage angle with other reputable sources.
- Label the article with a news topic. 

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
    model=Gemini(model="gemini-3-flash-preview", retry_options=retry_config),
    instruction=instruction_text,
    tools=[tool_news_topic]
)

# 5. Intent Agent
instruction_text = f"""
You are a media literacy expert specializing in identifying the intent behind news articles. 
### REFERENCE LIBRARY (Human-Labeled Examples)
Use these 7 examples to calibrate your judgment, focus on the intent label to learn patterns to help you analyze. {training_data}
CRTICITAL: Share one sentence with a pattern you learned from reading the training articles and label

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
    model=Gemini(model="gemini-3-flash-preview", retry_options=retry_config),
    instruction=instruction_text,
    tools=[tool_intent]
)

# 6. Title vs Body Agent (Uses Google Search)
instruction_text = f"""
You are a strict editor analyzing the consistency between an article's **Headline** and its **Body**.
### REFERENCE LIBRARY (Human-Labeled Examples)
Use these 7 examples to calibrate your judgment, focus on the title_vs_body label to learn patterns to help you analyze. {training_data}
Share one sentence with a pattern you learned from reading the training articles and label
  
**OBJECTIVE:** Determine if the title, agree, discuss, is unrelated to, or negate the body of the text.

**ANALYSIS STEPS:**
1. Read the Headline.
2. Read the Body.
3. Determine the relationship:
   - **Agree:** The body fully supports the headline's claims.
   - **Discuss:** The body explores the topic mentioned in the title without taking a definitive stance or solely focusing on the title's specific claim.
   - **Contradicts:** The body says the opposite of the title or proves the title's claim to be false.
   - **Unrelated:** The body has nothing to do with the headline.
4. **Use execute_web_search tool** if the body is too vague or if you suspect the headline refers to an external event not fully explained in the text, to verify the "true" context.

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
    model=Gemini(model="gemini-3-flash-preview", retry_options=retry_config),
    instruction=instruction_text,
    tools=[AgentTool(execute_web_search)]
)


# %%
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
    model=Gemini(model="gemini-3-flash-preview"),
    instruction="""
    You are the Final Judgment Lead. You will receive tool outputs from 6 different analysts.
    
    YOUR TASKS:
    1. Extract ALL labels and confidence scores (0-100%).
    2. Perform a Double-Check: 
       - Does Intent ("Deceive") align with Sensationalism ("High")?
       - Does Title vs Body (Agree/Negate) match the body evidence?
       - Adjust confidence downward if Context Veracity reports missing sources.
    3. Check the overall accuracy of article based on the all 6 of the signals and reasonings.
    
    4. OUTPUT FORMAT (STRICT):

    ## Agent Analysis Summary

    ### Labels
    | Signal | Label | Confidence |
    |---|---|---|
    | News Coverage | <label> | <0-100>% |
    | Intent | <label> | <0-100>% |
    | Sensationalism | <label> | <0-100>% |
    | Stance | <label> | <0-100>% |
    | Title vs Body | <label> | <0-100>% |
    | Context Veracity | <label> | <0-100>% |

    ### Double-check
    - <check 1 with one sentence summary of results>

    ### Short Summary
    <2-3 sentences summarizing the body text.>


    ### Final Judgment

    **CONFIDENCE SCORE RUBRIC (0–100%):**
    * **90–100%:** Explicit, unambiguous language supports your label.
    * **75–89%:** Trend is clear and consistent.
    * **50–74%:** Text is mixed, ambiguous, or open to interpretation.
    * **25–49%:** Text is too short or vague.
    * **0–24%:** Cannot meaningfully determine.
 
    - **Final Article Verdict:** <The definitive verdict (e.g., Verified Accurate, Misleading, Misinformation, Disinformation, etc.>
    - **Overall_confidence:** <0-100>%
    - **Reasoning explanation:** <A 1-3 sentence explanation synthesizing why this verdict was reached based on the factor analysis.>
    
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
    - **Why (3 bullet):**
    - <bullet 1>
    - <bullet 2>
    - <bullet 3>

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
    """
)


# %%
root_agent = SequentialAgent(
    name="COT_Framework",
    sub_agents=[factor_squad, synthesizer_agent]
)

# async def main():
#     runner = InMemoryRunner(agent=root_agent)
#     prompt = "Hello, how does this work?"
#     response = await runner.run_debug(prompt)
    
#     print(response)

# if __name__ == "__main__":
#     asyncio.run(main())



# %%
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


