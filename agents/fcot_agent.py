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
        predict_article_stance,
        analyze_complete_article
    )
    print(f"✅ Successfully imported predictors from {predictors_dir}")
except ImportError as e:
    print(f"❌ Failed to import predictors: {e}")
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

# %% [markdown]
# # Agents

# %% [markdown]
# Ensure agents speak the same language
    
# %%
from pydantic import BaseModel, Field, ConfigDict

class FactorAnalysis(BaseModel):
    # Removed: model_config = ConfigDict(extra='forbid')
    
    verdict: str = Field(description="The final label (e.g., 'sensational', 'support')")
    confidence: int = Field(description="Confidence score 0-100")
    fcot_reasoning: str = Field(description="2-3 sentence FCoT reasoning.")

class FactCheckFinalReport(BaseModel):
    # Removed: model_config = ConfigDict(extra='forbid')
    
    final_verdict: str = Field(description="The grand synthesis label.")
    overall_confidence: int = Field(ge=0, le=100)
    # agent_signals: dict[str, FactorAnalysis]

    # Define explicitly instead of using a dict
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
    raise ValueError("❌ GOOGLE_API_KEY not found! Make sure you created the .env file.")

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
    model=Gemini(model="gemini-3-flash-preview", retry_options=retry_config),
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
    model=Gemini(model="gemini-3-flash-preview", retry_options=retry_config),
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
    model="gemini-3-flash-preview",
    instruction="Search the web for factual grounding of specific claims.",
    tools=[google_search]
)

# %%
context_agent = Agent(
    name="Context_Veracity_Analyst",
    model=Gemini(model="gemini-3-flash-preview"),
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
    model=Gemini(model="gemini-3-flash-preview"),
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
    model=Gemini(model="gemini-3-flash-preview"),
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
    model=Gemini(model="gemini-3-flash-preview"),
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
    model=Gemini(model="gemini-3-flash-preview"),
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
    runner = InMemoryRunner(agent=root_agent)
    prompt = f"Title: {article_title}\nBody: {article_body}"
    response = await runner.run_debug(prompt)
    return response

if __name__ == "__main__":
    response = asyncio.run(main())
    print(response)