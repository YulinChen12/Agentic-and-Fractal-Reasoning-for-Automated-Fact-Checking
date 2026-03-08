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
# %% [markdown]
# # Agents

# %%
from pydantic import BaseModel, Field, ConfigDict
from typing import List

# Ensure agents use the same langauge

class FactorAnalysis(BaseModel):
    
    verdict: str = Field(description="The final label (e.g., 'sensational', 'support')")
    confidence: int = Field(description="Confidence score 0-100")
    fcot_reasoning: str = Field(description="2-3 sentence FCoT reasoning.")
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
    model=Gemini(model="gemini-3-flash-preview"),
    output_schema=FactorAnalysis,
    instruction="""
## LOCAL OBJECTIVE FUNCTION (LOF)
- **MAXIMIZE**: Precision in identifying emotional manipulation and clickbait architecture.
- **MINIMIZE**: 'Stylistic False Positives' where urgency or technical reporting is misclassified as sensationalism.

## FCoT REASONING PHASES

### PHASE 1: LOCAL THOUGHT UNIT (TU) - NARROW APERTURE
1.  **Linguistic Scan**: Isolate high-intensity adjectives, superlatives, and emotional triggers (e.g., "Shocking," "Outrageous," "Final Warning").
2.  **Structural Integrity Check**: Does the Body of the text actually contain the "shocking" information promised by the Headline? Identify any "curiosity gaps."

### PHASE 2: CONTEXT APERTURE EXPANSION (CAO) - RHETORICAL DECONSTRUCTION
1.  **Inverse Perspective**: Re-read the article while stripping away all adjectives. If the core facts remain significant, it is 'Neutral'. If the article collapses into triviality without the "fluff," it is 'Sensational'.
2.  **Rhetorical Intent**: Determine if the author is informing the reader or attempting to trigger a specific physiological stress response.

### PHASE 3: REFLECTIVE UPDATE (RUM) & RETROSPECTIVE ADJUSTMENT
1.  **Consistency Check**: Compare your findings from Phase 1 and Phase 2. If the language is urgent but the facts are heavy, adjust the label toward 'Neutral'. 
2.  **Final Synthesis**: Populate the `fcot_reasoning` field by explaining the logic behind your final classification.

## CONFIDENCE RUBRIC
- 90-100%: Explicit emotional triggers and clear headline-body baiting.
- 75-89%: Pattern of hyperbolic language is consistent throughout.
- 50-74%: Urgency exists, but may be justified by the gravity of the news topic.
- <50%: Signals are too mixed to provide a reliable classification.

## OUTPUT RULES
- Populate the `fcot_reasoning` field with a concise 2-3 sentence summary.
- The verdict must be strictly: [sensational, neutral].
"""
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
1.  **Linguistic Tone Mapping**: Identify the author's voice. Look for "loaded" verbs (e.g., 'claimed' vs 'demonstrated') and specific actor framing.
2.  **Preliminary Stance**: Form an internal hypothesis: Does the narrative arc support the primary claim, attempt to debunk it (Deny), or provide balanced reporting (Neutral)?

### PHASE 2: CONTEXT APERTURE EXPANSION (CAO) - RHETORICAL DECONSTRUCTION
1.  **Actor Alignment**: Identify which stakeholders are mentioned more or cited with the most authority. 
2.  **Omission Check**: Widen the aperture to look for what is *not* said. Does the text ignore standard counter-arguments?
3.  **Inverse Stance Test**: If you re-read the article from the perspective of an opponent, does it feel like an attack, or an fair representation?

### PHASE 3: REFLECTIVE UPDATE (RUM) & SYNTHESIS
1.  **Reflective Alignment**: Compare your findings from Phase 1 and 2. If the tone is neutral but the actor alignment is biased, adjust the verdict toward 'Support'.
2.  **Final Synthesis**: Populate the `fcot_reasoning` field by explaining the specific rhetorical markers that led to your final classification.

## CONFIDENCE RUBRIC
- 90-100%: Explicit, unambiguous alignment of tone and actor framing.
- 70-89%: Clear trend; minor stylistic nuance doesn't obscure the primary stance.
- 50-69%: Content is balanced or uses "both-sidesism" to obscure a subtle bias.
- <50%: Contradictory signals or text lacks sufficient markers to assign a stance.

## OUTPUT RULES
- Populate the `fcot_reasoning` field with a concise 2-3 sentence summary.
- Ensure the verdict strictly matches: [support, deny, neutral].
"""
)

# %%
context_agent = Agent(
    name="Context_Veracity_Analyst",
    model=Gemini(
        model="gemini-3-flash-preview",
        thinking_level="HIGH"
    ),
    output_schema=FactorAnalysis,
    description="Tool-free FCoT specialist for internal factual consistency and plausibility.",
    instruction=f"""
## LOCAL OBJECTIVE FUNCTION (LOF)
- **MAXIMIZE**: Detection of internal logical fallacies, chronological impossibilities, and source-validity gaps.
- **MINIMIZE**: False positives on breaking 2026 news that may seem "implausible" but is actually current events.

## FCoT REASONING PHASES

### PHASE 1: LOCAL THOUGHT UNIT (TU) - NARROW APERTURE
1.  **Date/Entity Extraction**: Identify all names, dates, and specific claims.
2.  **Structural Audit**: Check for claims that rely on sources that are not named or are vaguely defined (e.g., "experts say").

### PHASE 2: CONTEXT APERTURE EXPANSION (CAO)
1.  **Plausibility Stress-Test**: Evaluate the claims against the known laws of sciences, economics, and historical precedent. 
2.  **Entity Consistency**: Does a named organization actually have the jurisdiction or capability to do what the article claims? (e.g., The WHO declaring a law in a specific US city).
3.  **Temporal Conflict Check**: Do the dates provided (especially 2025-2026) align with the sequence of events described?

### PHASE 3: REFLECTIVE UPDATE (RUM) & SYNTHESIS
1.  **Reflective Alignment**: If the story is internally consistent but relies on extreme "Shock Value" without specific evidence, adjust confidence downward.
2.  **Final Synthesis**: In `fcot_reasoning`, explain the logical "fail points" that determined the veracity.

## CONFIDENCE RUBRIC
- 90-100%: Claims are internally consistent, logical, and follow a verifiable narrative structure.
- 70-89%: Generally plausible; minor vague areas but no logical "deal-breakers."
- 50-69%: "Red Flag" territory; relies on high-emotion claims with low logical backing.
- <50%: Contains a clear logical or chronological impossibility (e.g., an event happening before its cause).

## OUTPUT RULES
- Populate the `fcot_reasoning` field with a concise 2-3 sentence summary.
- The verdict must be strictly: [Accurate, Inaccurate].
"""
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

### PHASE 2: CONTEXT APERTURE EXPANSION (CAO)
1.  **Vertical Integration**: Look beyond the immediate story. If the story is about a local hospital (Local), does it represent a trend in National Healthcare Policy (Global)?
2.  **Category Refinement**: Choose the label that represents the highest "Level of Significance" in the article.

### PHASE 3: REFLECTIVE UPDATE (RUM)
1.  **Reflective Alignment**: Challenge your Phase 1 hypothesis and label the article with the correct topic.
2.  **Final Synthesis**: Populate the `fcot_reasoning` field by explaining the logic of the chosen scale and domain.

## CONFIDENCE RUBRIC
- 90-100%: Topic is explicitly the central focus; total alignment with tool.
- 70-89%: Topic is dominant but intersects with sub-topics; tool results are supportive.
- 50-69%: Topic is present but ambiguous or shares equal weight with another domain.
- <50%: Content is generic or spans too many categories to classify reliably.

## OUTPUT RULES
- Populate the `fcot_reasoning` field with a concise 2-3 sentence summary.
- The verdict must be a standardized topic label.
"""
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
1.  **Stakeholder Analysis**: Who benefits from this narrative? Does the text promote a specific commercial, political, or ideological interest?
2.  **Incentive Check**: If the article is factually weak but emotionally high-intensity, determine if the intent is to drive "clicks" (Entertain/Persuade) or to manipulate public perception (Deceive).

### PHASE 3: REFLECTIVE UPDATE (RUM) & DIALOGUE ALIGNMENT
1.  **Reflective Alignment**: Critically evaluate the article and determine if the linguistic markers suggest humor or irony that the tool ignored, and label the article with the correct intent.
2.  **Final Synthesis**: Populate the `fcot_reasoning` field. Explain how the synthesis of analysis confirms the primary intent.

## CONFIDENCE RUBRIC
- 90-100%: Intent is explicit, consistent, and matches tool results.
- 70-89%: Intent is clear but contains minor stylistic nuance (e.g., informative text with slight persuasive leaning).
- 50-69%: Intent is ambiguous (e.g., "Advertorial" content mixing fact and persuasion).
- <50%: Intent is obscured by contradictory markers or heavy sarcasm.

## OUTPUT RULES
- Populate the `fcot_reasoning` field with a concise 2-3 sentence summary of the RUM process.
- Verdict must be strictly: [Inform, Persuade, Entertain, Deceive]."""
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

### PHASE 2: CONTEXT APERTURE EXPANSION (CAO)
1.  **Bait-and-Switch Audit**: Determine if the headline uses a "Curiosity Gap" that the body fails to resolve.
2.  **Logical Flow Test**: If you only read the body, would you have written that specific headline? If not, identify the point of divergence.

### PHASE 3: REFLECTIVE UPDATE (RUM)
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
    instruction="""
    You are the Final Judgment Lead. You will receive outputs from 6 different analysts.

    YOUR TASKS:
    1. **Recursive Synthesis**: Receive the 6 FactorAnalysis JSONs from the squad. 
    2. **Inter-agent Reflectivity**: Identify if any agents disagree (e.g., if Context is Accurate but Intent is Deceive). 
    3. **Retrospective Re-grounding**: If the Context_Veracity agent found a major factual error, force all other signals to be interpreted through that lens. 
    4. **Output**: Generate a report using the following Markdown structure:
    
    OUTPUT FORMAT (STRICT):

    # Fact-Check Final Report
    ## Executive Summary
    **Verdict:** <final verdict>

    <2-sentence bottom line up front>

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
    - <one sentence on whether the signals support each other or conflict>
    - <one sentence on whether Context Veracity changes how the article should be interpreted>

    ### Short Summary
    <2-3 sentences summarizing the article body and the main issue being evaluated>

    ## Final Judgment

    **CONFIDENCE SCORE RUBRIC (0–100%):**
    * **90–100%:** Explicit, unambiguous evidence supports the label.
    * **75–89%:** Strong trend with mostly consistent evidence.
    * **50–74%:** Mixed, partial, or somewhat ambiguous evidence.
    * **25–49%:** Weak, vague, or limited evidence.
    * **0–24%:** Cannot meaningfully determine.

    - **Final Article Verdict:** <Verified Accurate, Misleading, Misinformation, Disinformation, etc.>
    - **Overall Confidence:** <0-100>%
    - **Reasoning Explanation:** <1-3 sentences explaining why this final verdict was reached>

    ### Final Signal Judgments

    - **Final Label for News Coverage:** <label>
    - **Final Confidence:** <0-100>%
    - **Why (1 bullet):**
      - <bullet 1>

    - **Final Label for Intent:** <label>
    - **Final Confidence:** <0-100>%
    - **Why (1 bullet):**
      - <bullet 1>

    - **Final Label for Sensationalism:** <label>
    - **Final Confidence:** <0-100>%
    - **Why (up to 3 bullets):**
      - <bullet 1>
      - <bullet 2>
      - <bullet 3>

    - **Final Label for Stance:** <label>
    - **Final Confidence:** <0-100>%
    - **Why (1 bullet):**
      - <bullet 1>

    - **Final Label for Title vs Body:** <label>
    - **Final Confidence:** <0-100>%
    - **Why (1 bullet):**
      - <bullet 1>

    - **Final Label for Context Veracity:** <label>
    - **Final Confidence:** <0-100>%
    - **Why (1 bullet):**
      - <bullet 1>

    RULES:
    - Do not include raw tool outputs.
    - Do not mention tool traces or internal IDs.
    - Be consistent across labels, confidence scores, and explanations.
    - If signals conflict, explicitly mention that in the Double-check or Reasoning Explanation.
    - Keep the report readable and concise.
    """)

root_agent = SequentialAgent(
    name="Fractal_FactCheck_Framework",
    sub_agents=[factor_squad, synthesizer_agent]
)

# import asyncio

# async def main():
#     runner = InMemoryRunner(agent=root_agent)
#     prompt = "Hello, how does this work?"
#     response = await runner.run_debug(prompt)
    
#     print(response)

# if __name__ == "__main__":
#     asyncio.run(main())
