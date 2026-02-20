import os
import sys
import warnings
from collections import Counter
import nltk
from dotenv import load_dotenv

# -------------------------------------------------------------------------
# PATH CONFIGURATION
# -------------------------------------------------------------------------
# Get the absolute path of the current file (agents/cot_agent.py)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (project_root)
parent_dir = os.path.dirname(current_dir)
# Construct path to sibling directory (pred_models_training)
predictors_dir = os.path.join(parent_dir, 'pred_models_training')

# Add to sys.path so Python can find predictors.py
if predictors_dir not in sys.path:
    sys.path.append(predictors_dir)

# -------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------
# ADK Imports
from google.adk.agents import Agent
from google.adk.tools import AgentTool, google_search
from google.adk.models.google_llm import Gemini
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.genai import types
import uvicorn

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

# --- Agent Definitions ---

sensationalism_agent = Agent(
    name="Sensationalism_Analyst",
    model=Gemini(
        model="gemini-3-flash-preview",
        retry_options=retry_config
    ),
    instruction=(
        "You are a senior linguistic editor specializing in media bias. You follow a strict two-phase analysis protocol:\n\n"
        
        "ANALYSIS PROTOCOL\n"
        "- Review the article's text independently for sensationalist phrasing and dramatic claims.\n"
        "- Compare the emotional tone of the headline vs. the content.\n"
        "- Identify 'loaded' language, superlatives, and clickbait structures designed to provoke emotion.\n"
        "- Determine if the content prioritizes shock value and narrative over verifiable facts.\n"
        "- Based on your analysis, label the sensationalism for this article. If your independent analysis of "
        "linguistic patterns (e.g., use of exclamation, intense adjectives) is contradictory, "
        "provide clear reasoning for the discrepancy in one sentence.\n\n"
        
        "**CONFIDENCE SCORE RUBRIC (0–100%):**\n"
        "90–100%: Definitive evidence. Multiple linguistic markers (hyperbole, clickbait, emotional appeals) are strongly supported by the text.\n"
        "75–89%: High Probability. Clear patterns of sensationalist or neutral language are present throughout the text.\n"
        "50–74%: Moderate Certainty. Text uses occasional colorful language that may be stylistic rather than manipulative, or your analysis is ambiguous.\n"
        "25–49%: Ambiguous Intent. The tone is dry but the subject matter is inherently dramatic, making intent difficult to isolate.\n"
        "0–24%: Insufficient Data. Text lacks enough descriptive language or context to determine a rhetorical strategy.\n\n"

        "OUTPUT FORMAT:\n"
        "**Sensationalism**\n"
        "* **Final Output:** [sensational or neutral] (Your final verdict)\n"
        "* **Confidence:** [0–100]%\n"
        "* **Reasoning:** [Explain how you reached this conclusion in 1 bullet point.]"
    ),
    output_key="sensationalism_report"
)

stance_agent = Agent(
    name="Stance_Analyst",
    model=Gemini(model="gemini-3-flash-preview", retry_options=retry_config),
    instruction=(
        "You are a senior linguistic editor. You follow a strict two-phase analysis protocol:\n\n"
        
        "ANALYSIS PROTOCOL\n"
        "- Review the article's text independently to understand the author's opinion about the news.\n"
        "- Analyze if content supports, denies, or is neutral towards claims.\n"
        "- Evaluate consistency in stance throughout the content.\n"
        "- Determine if shifts in stance are supported by factual developments.\n"
        "- Label the article with your analysis result and  "
        "If your independent analysis of the arguments in the article "
        "is contradictory, provide your reasoning for the discrepancy.\n\n"
        
        "**CONFIDENCE SCORE RUBRIC (0–100%):**\n"
        "90–100%: High Alignment. Qualitative analysis strongly is supported by the text with explicit, unambiguous evidence.\n"
        "75–89%: High Certainty. Clear rhetorical trend; minor stylistic nuance doesn't obscure the primary stance.\n"
        "50–74%: Moderate Uncertainty. Content is balanced, ambiguous, or your analysis is ambiguous.\n"
        "25–49%: Low Reliability. Evidence is contradictory or the text relies on heavy sarcasm/subtext that is difficult to verify.\n"
        "0–24%: Non-Determinate. Text lacks sufficient linguistic markers to assign a stance meaningfully.\n\n"
         
        "OUTPUT FORMAT:\n"
        "**Stance**\n"
        "**Final Output:** [support, deny, or neutral] (Your final verdict)\n"
        "**Confidence:** [0–100]%\n"
        "**Reasoning:** [Explain how you reached this conclusion in 1 bullet point]"
    ),
)

instruction_text_context = """
You are a senior investigative fact-checker specialized in analyzing **Context Veracity**.
Your task is to evaluate the truthfulness and reliability of the article below based strictly on:
1. **Contextual Coherence**: Does the article stay on the same topic throughout? Are the headline and body consistent?
2. **Factual Plausibility**: Does the article use generally accepted facts (based on your internal knowledge)? Does it contain obvious hallucinations or contradictions?

**PHASE 1: INTERNAL ANALYSIS**
- Review the article's text independently for logical inconsistencies, contradictions, or missing key context.
- Evaluate if the content stays on topic (coherence).
- Check if the article uses **true facts** (based on your internal knowledge base) o
r if it invents events/figures.
- Determine if the context is **Accurate**, or **Inaccurate**

**ANALYSIS PROTOCOL**
- Evaluate the whole article 
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
    instruction=instruction_text_context 
)

instruction_text_coverage = """
You are a senior editor specialized in categorizing news content.

ANALYSIS PROTOCOL
- Review the article's text independently to identify the primary topic
- What kind of news is covered in this article? Determine the type of news: local, global, opinion, etc. 
- A news topic is based on the topics that the article talked about. For example, if the article talked about movie reviews, then the topic would be entertainment.
- Compare coverage angle with other reputable sources.
- Label the article with a news topic. 

**CONFIDENCE SCORE RUBRIC (0–100%):**
90-100%: Topic is explicitly the main focus.
70-89%: Topic is dominant but shares space with sub-topics.
50-69%: Topic is present but ambiguous.
<50%: Hard to classify.

OUTPUT FORMAT:
**News Coverage**
* **Final Output:** [Topic Label]
* **Confidence:** [0-100]%
* **Reasoning:** [Brief explanation for your decision in one sentence.]
"""

news_coverage_agent = Agent(
    name="News_Coverage_Analyst",
    model=Gemini(model="gemini-3-flash-preview", retry_options=retry_config),
    instruction=instruction_text_coverage
)

instruction_text_intent = """
You are a media literacy expert specializing in identifying the intent behind news articles. 

ANALYSIS PROTOCOL
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
* **Reasoning:** [Brief explanation for your decision in one bullet point.]
"""

intent_agent = Agent(
    name="Intent_Analyst",
    model=Gemini(model="gemini-3-flash-preview", retry_options=retry_config),
    instruction=instruction_text_intent
)

instruction_text_title = """
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
    instruction=instruction_text_title
)

root_agent = Agent(
    name="ParallelCoordinator",
    model=Gemini(model="gemini-3-flash-preview", retry_options=retry_config),
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
)

# --- Execution Logic ---

def load_test_articles(path="gen_data/test_article_no_label.json"):
    if not os.path.exists(path):
        print(f"Error: File not found at {path}")
        return []
    
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    return data.get("articles", [])

async def process_batch(articles_batch, batch_name="Batch"):
    runner = InMemoryRunner(agent=root_agent)
    print(f"=== Processing {batch_name} ({len(articles_batch)} articles) ===")
    
    for i, art in enumerate(articles_batch):
        headline = art.get('headline', 'No Title')
        print(f"\n[{batch_name}] Article {i+1}: {headline}")
        
        # Prepare Prompt
        prompt = (
            f"Headline: {headline}\n"
            f"Source: {art.get('news_source', 'Unknown')}\n"
            f"Author: {art.get('author', 'Unknown')}\n"
            f"Date: {art.get('date', 'Unknown')}\n\n"
            f"Body:\n{art.get('text', '')}"
        )
        
        # Run Agent using run_debug
        try:
            response = await runner.run_debug(prompt)
            
            # Access output based on ADK response structure
            if hasattr(response, 'output'):
                print(response.output)
            else:
                print(response)
                
        except Exception as e:
            print(f"Error running agent: {e}")

        print("-" * 50)
        
        # Sleep slightly to help with rate limits even within batch
        await asyncio.sleep(2)

async def main():
    # Load all data
    all_articles = load_test_articles()
    print(f"Total articles loaded: {len(all_articles)}")
    
    # Process batches as requested in the notebook
    if len(all_articles) > 0:
        await process_batch(all_articles[0:5], "Batch 1")
    if len(all_articles) > 5:
        await process_batch(all_articles[5:10], "Batch 2")
    # Batch 3 overlap in notebook: 8:9 then 10:15
    if len(all_articles) > 8:
        await process_batch(all_articles[8:9], "Batch 3 (Single)")
    if len(all_articles) > 10:
        await process_batch(all_articles[10:15], "Batch 3")
    if len(all_articles) > 15:
        await process_batch(all_articles[15:], "Batch 4")

if __name__ == "__main__":
    asyncio.run(main())
