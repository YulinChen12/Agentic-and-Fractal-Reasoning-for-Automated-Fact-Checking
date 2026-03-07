# %%
# ADK Imports
from google.adk.agents import Agent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.tools import AgentTool, google_search
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.genai import types
from dotenv import load_dotenv
import os
from pathlib import Path


# %%
# -------------------------------------------------------------------------
# PATH CONFIGURATION
# -------------------------------------------------------------------------
try:
    current_dir = Path(__file__).resolve().parent         
except NameError:
    current_dir = Path.cwd()                               

parent_dir = current_dir.parent                            # project_root

# %%
from pydantic import BaseModel, Field, ConfigDict
from typing import List

# Ensure agents use the same langauge

class FactorAnalysis(BaseModel):
    
    verdict: str = Field(description="The final label (e.g., 'sensational', 'support')")
    confidence: int = Field(description="Confidence score 0-100")
    fcot_reasoning: str = Field(description="2-3 sentence FCoT reasoning.")

class FactCheckFinalReport(BaseModel):
    # 1. High-Level Summary
    final_verdict: str = Field(..., description="The definitive verdict (e.g., Verified Accurate, Misleading, Misinformation, Disinformation, etc.).")
    overall_confidence: int = Field(..., ge=0, le=100, description="Confidence score from 0-100.")
    
    # 2. Human-Centric Explanation (The 'Why')
    verdict_justification: str = Field(
        ..., 
        description="A 1-3 sentence explanation synthesizing why this verdict was reached based on the factor analysis."
    )

    # 3. Agent Metadata
    agents_involved: List[str] = Field(
        default=["Sensationalism_Analyst", "Stance_Analyst", "Context_Veracity_Analyst", 
                 "News_Coverage_Analyst", "Intent_Analyst", "Title_Body_Analyst"],
        description="List of specialized agents that contributed factor data."
    )

    # 4. Detailed Factor Signals
    # These contain the individual verdicts and FCoT reasoning for the audit trail
    sensationalism_signal: FactorAnalysis
    stance_signal: FactorAnalysis
    context_veracity_signal: FactorAnalysis
    news_coverage_signal: FactorAnalysis
    intent_signal: FactorAnalysis
    title_body_signal: FactorAnalysis
# -------------------------------------------------------------------------
# Agent Configurations
# -------------------------------------------------------------------------

# Load the .env file
load_dotenv() 

# Retrieve the key
api_key = os.getenv("GOOGLE_API_KEY")

# Check if it loaded correctly
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found! Make sure you created the .env file.")

retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504], # Retry on these HTTP errors
)

# %% [markdown]
# --- Agent Definitions ---

# %%
sensationalism_agent = Agent(
    name="Sensationalism_Analyst",
    model=Gemini(
        model="gemini-3-flash-preview",
        retry_options=retry_config
    ),
    instruction=(
        "You are a senior linguistic editor specializing in media bias.\n\n"
        
        "ANALYSIS PROTOCOL\n"
        "- Review the article's text independently for sensationalist phrasing and dramatic claims.\n"
        "- Compare the emotional tone of the headline vs. the content.\n"
        "- Identify 'loaded' language, superlatives, and clickbait structures designed to provoke emotion.\n"
        "- Determine if the content prioritizes shock value and narrative over verifiable facts.\n"
        "- Based on your analysis, label the sensationalism for this article."
        
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

# %%
stance_agent = Agent(
    name="Stance_Analyst",
    model=Gemini(model="gemini-3-flash-preview", retry_options=retry_config),
    instruction=(
        "You are a senior linguistic editor.\n\n"
        
        "ANALYSIS PROTOCOL\n"
        "- Review the article's text independently to understand the author's opinion about the news.\n"
        "- Analyze if content supports, denies, or is neutral towards claims.\n"
        "- Evaluate consistency in stance throughout the content.\n"
        "- Determine if shifts in stance are supported by factual developments.\n"
        "- Based on your own independent analysis using linguistics patterns, label the article with the stance."

        
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

# %%
instruction_text_context = """
You are a senior investigative fact-checker specialized in analyzing **Context Veracity**.
Your task is to evaluate the truthfulness and reliability of the article below based strictly on:
1. **Contextual Coherence**: Does the article stay on the same topic throughout? Are the headline and body consistent?
2. **Factual Plausibility**: Does the article use generally accepted facts (based on your internal knowledge)? Does it contain obvious hallucinations or contradictions?

- Review the article's text independently for logical inconsistencies, contradictions, or missing key context.
- Evaluate if the content stays on topic (coherence).
- Check if the article uses **true facts** (based on your internal knowledge base) or if it invents events/figures.
- If the event takes place in 2025 or 2026, then use your knowledge to determine if it is plausible to happen instead of relying on the internal training date you had.
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

# %%
context_agent = Agent(
    name="Context_Veracity_Analyst",
    model=Gemini(model="gemini-3-flash-preview"),
    description="A specialized agent for verifying the contextual veracity of news articles.",
    instruction=instruction_text_context 
)

# %%
instruction_text_coverage = """
You are a senior editor specialized in categorizing news content.

ANALYSIS PROTOCOL
- Review the article's text independently to identify the primary topic
- What kind of news is covered in this article? Determine the type of news: local, global, opinion, etc. 
- A news topic is based on the topics that the article talked about. For example, if the article talked about movie reviews, then the topic would be entertainment.
- Compare coverage angle with other reputable sources in your internal training datas.
- Based on your own independent analysis using linguistics patterns, label the article with the news topic.


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

# %%
news_coverage_agent = Agent(
    name="News_Coverage_Analyst",
    model=Gemini(model="gemini-3-flash-preview", retry_options=retry_config),
    instruction=instruction_text_coverage
)

# %%
instruction_text_intent = """
You are a media literacy expert specializing in identifying the intent behind news articles. 

ANALYSIS PROTOCOL
Analyze the text to determine the author's primary goal and what the author want you to do: 
  - **Inform** (Neutral facts, Likely to provide sources, data, and multiple perspectives)
  - **Persuade** (Opinion/Argument, May use emotional appeals but still be based on a clear argument)
  - **Entertain** (Humor/Satire/Light)
  - **Deceive** (Fabrication/Misinformation, Likely to use strong emotional appeals, "us vs. them" language, and no verifiable sources)
- Based on your own independent analysis using linguistics patterns, label the article with the intent.


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

# %%
intent_agent = Agent(
    name="Intent_Analyst",
    model=Gemini(model="gemini-3-flash-preview", retry_options=retry_config),
    instruction=instruction_text_intent
)

# %%
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

# %%
title_body_agent = Agent(
    name="Title_Body_Analyst",
    model=Gemini(model="gemini-3-flash-preview", retry_options=retry_config),
    instruction=instruction_text_title
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
    
    3. OUTPUT FORMAT (STRICT):
    Return ONLY the Markdown template provided below. 
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
    name="COT__No_Tool_Framework",
    sub_agents=[factor_squad, synthesizer_agent]
)

# %%
# import asyncio

# async def main():
#     runner = InMemoryRunner(agent=root_agent)
#     prompt = "Hello, how does this work?"
#     response = await runner.run_debug(prompt)
    
#     print(response)

# if __name__ == "__main__":
#     asyncio.run(main())