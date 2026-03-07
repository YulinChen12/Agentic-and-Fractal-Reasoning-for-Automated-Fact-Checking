# %%
import os
import sys
import warnings
from collections import Counter
import nltk
from dotenv import load_dotenv
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
try:
    from predictors import (
        predict_news_coverage,
        predict_intent,
        predict_sensationalism,
        predict_article_stance
    )
    print(f"Successfully imported predictors from {predictors_dir}")
except ImportError as e:
    print(f"Failed to import predictors: {e}")
    print(f"   Current sys.path: {sys.path}")
    raise e

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

def load_train_articles(path = os.path.join(parent_dir, 'gen_data/train_article.json')):
    if not os.path.exists(path):
        print(f"Error: File not found at {path}")
        return []
    
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    return data.get("articles", [])

# %%
training_data = load_train_articles()

# %% [markdown]
# # Agents

# %%
from pydantic import BaseModel, Field, ConfigDict
from typing import List

# Ensure agents use the same langauge

class FactorAnalysis(BaseModel):

    learned_pattern: str = Field(description="Pattern learned from training articles and human labels of the articles")
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
# ## FCOT

# %%
sensationalism_agent = Agent(
    name="Sensationalism_Analyst",
    model=Gemini(model="gemini-3-flash-preview", retry_options=retry_config),
    output_schema=FactorAnalysis,
    instruction=f"""
## LOCAL OBJECTIVE FUNCTION (LOF)
- **MAXIMIZE**: Precision in identifying emotional manipulation and clickbait architecture.
- **MINIMIZE**: 'Stylistic False Positives' where urgency or technical reporting is misclassified as sensationalism.

### REFERENCE LIBRARY (Human-Labeled Examples)
Use these 7 examples to calibrate your judgment, focus on the sensationalism label to learn patterns to help you analyze.{training_data}
CRTICITAL: Share one sentence with a pattern you learned from reading the training articles and label to fill in the `learned_pattern' field

## FCoT REASONING PHASES

### PHASE 1: LOCAL THOUGHT UNIT (TU) - NARROW APERTURE
1.  **Linguistic Scan**: Isolate high-intensity adjectives, superlatives, and emotional triggers (e.g., "Shocking," "Outrageous," "Final Warning").
2.  **Structural Integrity Check**: Does the Body of the text actually contain the "shocking" information promised by the Headline? Identify any "curiosity gaps."
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
    model=Gemini(model="gemini-3-flash-preview", retry_options=retry_config),
    output_schema=FactorAnalysis,
    instruction= f"""
## LOCAL OBJECTIVE FUNCTION (LOF)
- **MAXIMIZE**: Detection of nuanced rhetorical alignment, bias, or skepticism.
- **MINIMIZE**: Misclassification of "objective reporting" as "denial" or "unbiased" as "support."

### REFERENCE LIBRARY (Human-Labeled Examples)
Use these 7 examples to calibrate your judgment, focus on the stance label to learn patterns to help you analyze.{training_data} 
CRTICITAL: Share one sentence with a pattern you learned from reading the training articles and label to fill in the `learned_pattern' field

## FCoT REASONING PHASES

### PHASE 1: LOCAL THOUGHT UNIT (TU) - NARROW APERTURE
1.  **Linguistic Tone Mapping**: Identify the author's voice. Does the language favor specific actors or dismiss certain arguments?
2.  **Consistency Check**: Verify if the stance remains stable or shifts when presenting evidence vs. commentary.
3.  **Actor Alignment**: Identify which stakeholders are mentioned more or cited with the most authority. 
4.  **Omission Check**: Widen the aperture to look for what is *not* said. Does the text ignore standard counter-arguments?
5.  **Preliminary Stance**: Form an internal hypothesis: Does this text Support, Deny, or remain Neutral?

### PHASE 2: CONTEXT APERTURE EXPANSION (CAO) - TOOL GROUNDING
1.  **Tool Execution**: Call `tool_stance` to get the predictive baseline.
2.  **Aperture Expansion**: Compare your internal hypothesis with the tool's prediction.

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

# %%
execute_web_search = Agent(
    name="Web_Search_Provider",
    model="gemini-3-flash-preview",
    instruction="Search the web for factual grounding of specific claims.",
    tools=[google_search]
)

# %%
from datetime import datetime
current_date = datetime.now().strftime("%B %d, 2026")

context_agent = Agent(
    name="Context_Veracity_Analyst",
    model=Gemini(model="gemini-3-flash-preview"),
    output_schema=FactorAnalysis,
    description="A specialized FCoT agent for verifying factual and contextual veracity with 2026 real-time grounding.",
    instruction=f"""
## TEMPORAL ANCHORING
- **CURRENT DATE**: {current_date}
- **CRITICAL RULE**: Do NOT rely on internal training data for events occurring. If a claim involves 2025 or 2026, you MUST treat `execute_web_search` as the primary source of truth, then cite your source in the explanation, and label which search terms you used in the reasoning.

### REFERENCE LIBRARY (Human-Labeled Examples)
Use these 7 examples to calibrate your judgment, focus on the context_veracity label to learn patterns to help you analyze.{training_data} 
CRTICITAL: Share one sentence with a pattern you learned from reading the training articles and label to fill in the `learned_pattern' field

## LOCAL OBJECTIVE FUNCTION (LOF)
- **MAXIMIZE**: Historical and factual alignment using the 2026 web index.
- **MINIMIZE**: "False Hoax" flags caused by training data lag.

## FCoT REASONING PHASES

### PHASE 1: LOCAL THOUGHT UNIT (TU) - NARROW APERTURE
1.  **Date/Entity Extraction**: Identify all names, dates, and specific claims.
2.  **Age Assessment**: If the claim is dated after 2024, flag it as "High-Priority Search" to avoid internal bias.

### PHASE 2: CONTEXT APERTURE EXPANSION (CAO) - TEMPORAL RE-GROUNDING
1.  **Dynamic Search**: Use `execute_web_search`. You MUST append "2026" or "after:2025-01-01" to queries for current events.
2.  **Multi-Source Triangulation**: Look for at least two independent reports to confirm a 2026 event.

### PHASE 3: REFLECTIVE UPDATE (RUM) & RETROSPECTIVE ADJUSTMENT
1.  **Update Internal Model**: If search results confirm a new 2026 event, override your internal "hallucination" trigger. 
2.  **Synthesis**: In `fcot_reasoning`, explicitly state the terms you searched to confirm the facts

## CONFIDENCE RUBRIC
- 90-100%: Facts verified by 2026 search results; matches current news cycle.
- <50%: Direct contradiction found in 2026 search results (e.g., search proves the event did NOT happen).

## OUTPUT RULES
- Verdict must be strictly: [Accurate, Inaccurate].
""",
    tools=[AgentTool(execute_web_search)] 
)

# %%

news_coverage_agent = Agent(
    name="News_Coverage_Analyst",
    model=Gemini(model="gemini-3-flash-preview"),
    output_schema=FactorAnalysis,
    description="FCoT agent specializing in multi-scale news categorization.",
    instruction=f"""
## LOCAL OBJECTIVE FUNCTION (LOF)
- **MAXIMIZE**: Precision in identifying the primary thematic domain and geographical scope.
- **MINIMIZE**: Conceptual redundancy (e.g., mislabeling a 'Political' story as 'General' because it mentions a city name).

### REFERENCE LIBRARY (Human-Labeled Examples)
Use these 7 examples to calibrate your judgment, focus on the coverage label to learn patterns to help you analyze.{training_data} 
CRTICITAL: Share one sentence with a pattern you learned from reading the training articles and label to fill in the `learned_pattern' field

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
    model=Gemini(model="gemini-3-flash-preview"),
    output_schema=FactorAnalysis,
    description="FCoT specialist in identifying rhetorical intent and authorial goals.",
    instruction= f"""
## LOCAL OBJECTIVE FUNCTION (LOF)
- **MAXIMIZE**: Transparency in identifying the author's underlying rhetorical goal (e.g., hidden persuasion).
- **MINIMIZE**: False categorization of "Opinion/Op-Ed" as "Deception" or "Satire" as "Informational."

### REFERENCE LIBRARY (Human-Labeled Examples)
Use these 7 examples to calibrate your judgment, focus on the intent label to learn patterns to help you analyze.{training_data} 
CRTICITAL: Share one sentence with a pattern you learned from reading the training articles and label to fill in the `learned_pattern' field

## FCoT REASONING PHASES

### PHASE 1: LOCAL THOUGHT UNIT (TU) - NARROW APERTURE
1.  **Goal Extraction**: Analyze the "Call to Action." What does the author want the reader to think, feel, or do after reading?
2.  **Linguistic Marker Identification**: Look for "Us vs. Them" framing, emotional appeals, or the presence/absence of verifiable citations.
3.  **Preliminary Intent**: Classify based on the four categories: [Inform, Persuade, Entertain, Deceive].

### PHASE 2: CONTEXT APERTURE EXPANSION (CAO) - TOOL GROUNDING
1.  **Tool Execution**: Call `tool_intent` to obtain the statistical intent label.
2.  **Aperture Expansion**: Compare your internal hypothesis with the tool’s output. Does the tool detect "Deception" while you see "Persuasion"?

### PHASE 3: REFLECTIVE UPDATE (RUM) & DIALOGUE ALIGNMENT
1.  **Reflective Alignment**: Critically evaluate the tool. Predictive models often struggle with Satire (Entertain) vs. Deception.
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
    model=Gemini(model="gemini-3-flash-preview"),
    output_schema=FactorAnalysis,
    description="FCoT specialist in detecting semantic gaps between headlines and article content.",
    instruction=f"""
## LOCAL OBJECTIVE FUNCTION (LOF)
- **MAXIMIZE**: Detection of "headline-body gaps," bait-and-switch tactics, or semantic contradictions.
- **MINIMIZE**: False "Unrelated" labels for headlines that use metaphor or creative framing to describe the body content.

### REFERENCE LIBRARY (Human-Labeled Examples)
Use these 7 examples to calibrate your judgment, focus on the title_vs_body label to learn patterns to help you analyze.{training_data} 
CRTICITAL: Share one sentence with a pattern you learned from reading the training articles and label to fill in the `learned_pattern' field

## FCoT REASONING PHASES

### PHASE 1: LOCAL THOUGHT UNIT (TU) - NARROW APERTURE
1.  **Direct Mapping**: Extract the core claim of the Headline. Scan the Body for direct supporting evidence of that specific claim.
2.  **Stance Alignment**: Does the body's tone match the headline's intensity? 
3.  **Preliminary Relationship**: Classify as [Agree, Discuss, Contradicts, Unrelated].

### PHASE 2: CONTEXT APERTURE EXPANSION (CAO) - EXTERNAL GROUNDING
1.  **Context Check**: If the body is vague or the headline refers to a specific event/entity not fully explained, execute `execute_web_search` and label which search terms you used in the reasoning.
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
    output_schema=FactCheckFinalReport,
    instruction="""
    You are the Final Synthesizer.
    
    1. **Recursive Synthesis**: Receive the 6 FactorAnalysis JSONs from the squad.
    2. **Inter-agent Reflectivity**: Identify if any agents disagree (e.g., if Context is Accurate but Intent is Deceive).
    3. **Retrospective Re-grounding**: If the Context_Veracity agent found a major factual error, force all other signals to be interpreted through that lens.
    4. **Output**: Generate a 'Human_Report' field using the following Markdown structure:

    1. **Executive Summary**: A bold verdict and 2-sentence 'Bottom Line Up Front'.
    2. **Factors Analysis Table**: A table with columns: | Factor | Verdict | Confidence | Key Evidence |.
    
    RULES:
    - Do not include raw tool outputs.
    - Do not mention tool traces or internal IDs.
    """
)

root_agent = SequentialAgent(
    name="Fractal_FactCheck_Framework",
    sub_agents=[factor_squad, synthesizer_agent]
)

# %%
# import json
# import asyncio
# # %%
# async def main():
#     runner = InMemoryRunner(agent=root_agent)
#     prompt = "Hello, how does this work?"
#     response = await runner.run_debug(prompt)
    
#     print(response)

# if __name__ == "__main__":
#     asyncio.run(main())


