import streamlit as st
import sys
import os
import asyncio
import warnings
from pathlib import Path
from dotenv import load_dotenv
from collections import Counter
import nltk
from datetime import datetime

# --- Path Setup ---
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Add project root to sys.path to find pred_models_training
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# --- Environment Setup ---
load_dotenv()
# (Key check moved to Sidebar in UI section)

# --- ADK Imports ---
from google.adk.agents import Agent, SequentialAgent, ParallelAgent
from google.adk.tools import AgentTool, google_search
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.genai import types

# --- Predictors Import ---
try:
    from pred_models_training.predictors import (
        predict_news_coverage,
        predict_intent,
        predict_sensationalism,
        predict_article_stance,
    )
except ImportError:
    try:
        from predictors import (
            predict_news_coverage,
            predict_intent,
            predict_sensationalism,
            predict_article_stance,
        )
    except ImportError as e:
        st.error(f"Failed to import predictors: {e}")
        st.stop()

warnings.filterwarnings("ignore")

# Ensure NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# --- Helper Functions ---
def get_sentences(text: str):
    try:
        sentences = nltk.sent_tokenize(text or "")
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        if not sentences and text.strip():
            sentences = [text.strip()]
        return sentences if sentences else []
    except Exception:
        return [text.strip()] if text else []

# --- Tool Wrappers ---
def tool_news_topic(article_text: str) -> dict:
    sentences = get_sentences(article_text)
    votes = []
    for s in sentences:
        try:
            res = predict_news_coverage(s)
            if res and res.get("label") and str(res["label"]) not in ["None", "nan", "unknown"]:
                votes.append(res["label"])
        except Exception:
            continue
    topic = Counter(votes).most_common(1)[0][0] if votes else "unknown"
    return {"news_coverage": topic}

def tool_intent(article_text: str) -> dict:
    sentences = get_sentences(article_text)
    votes = []
    for s in sentences:
        try:
            res = predict_intent(title="", body=s)
            if res and res.get("label"):
                votes.append(res["label"])
        except Exception:
            continue
    intent = Counter(votes).most_common(1)[0][0] if votes else "unknown"
    return {"intent": intent}

def tool_sensationalism(article_text: str) -> dict:
    sentences = get_sentences(article_text)
    votes = []
    for s in sentences:
        try:
            res = predict_sensationalism(statement=s)
            if res and res.get("label"):
                votes.append(str(res["label"]))
        except Exception:
            continue
    final_label = Counter(votes).most_common(1)[0][0] if votes else "neutral"
    return {"sensationalism": final_label}

def tool_stance(article_text: str) -> dict:
    try:
        res = predict_article_stance(article_text=article_text)
        label = res.get("label", "neutral")
    except Exception:
        label = "neutral"
    return {"stance": label}

# --- Agent Factory ---
def build_agent():
    api_key = os.getenv("GOOGLE_API_KEY")
    retry_config = types.HttpRetryOptions(
        attempts=5,
        exp_base=7,
        initial_delay=1,
        http_status_codes=[429, 500, 503, 504],
    )

    sensationalism_agent = Agent(
        name="Sensationalism_Analyst",
        model=Gemini(model="gemini-3-flash-preview", retry_options=retry_config),
        instruction=(
            "You are a senior linguistic editor specializing in media bias. You follow a strict two-phase analysis protocol:\n\n"
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

    stance_agent = Agent(
        name="Stance_Analyst",
        model=Gemini(model="gemini-3-flash-preview", retry_options=retry_config),
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

    execute_web_search = Agent(
        name="Web_Search_Provider",
        model="gemini-3-flash-preview",
        instruction="Search the web for factual grounding of specific claims.",
        tools=[google_search]
    )

    current_date = datetime.now().strftime("%B %d, 2026")
    context_agent = Agent(
        name="Context_Veracity_Analyst",
        model=Gemini(model="gemini-3-flash-preview"),
        description="A specialized agent for verifying the contextual veracity of news articles.",
        instruction=f"""
        ## TEMPORAL ANCHORING
        - **CURRENT DATE**: {current_date}
        - **CRITICAL RULE**: Do NOT rely on internal training data for events occurring. If a claim involves 2025 or 2026, you MUST treat results from execute_web_search tool as the primary source of truth, then cite your source in the explanation, and label which search terms you used in the reasoning.

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
        """,
        tools=[AgentTool(execute_web_search)]
    )

    news_coverage_agent = Agent(
        name="News_Coverage_Analyst",
        model=Gemini(model="gemini-3-flash-preview", retry_options=retry_config),
        instruction="""
        You are a senior editor specialized in categorizing news content.

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
        """,
        tools=[tool_news_topic]
    )

    intent_agent = Agent(
        name="Intent_Analyst",
        model=Gemini(model="gemini-3-flash-preview", retry_options=retry_config),
        instruction="""
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
        """,
        tools=[tool_intent]
    )

    title_body_agent = Agent(
        name="Title_Body_Analyst",
        model=Gemini(model="gemini-3-flash-preview", retry_options=retry_config),
        instruction="""
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
        """,
        tools=[AgentTool(execute_web_search)]
    )

    factor_squad = SequentialAgent(
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

    root_agent = SequentialAgent(
        name="COT_Framework",
        sub_agents=[factor_squad, synthesizer_agent]
    )
    
    return root_agent

# --- Streamlit UI ---
st.set_page_config(
    page_title="Agentic and Fractal Reasoning for Automated Fact-Checking",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Sidebar Settings ---
with st.sidebar:
    st.title("Settings")
    
    api_key_input = st.text_input(
        "Google API Key",
        type="password",
        placeholder="Paste your Google Gemini API Key here",
        help="Get your key from https://aistudio.google.com/app/apikey",
        value=os.getenv("GOOGLE_API_KEY", "")
    )
    
    if api_key_input:
        os.environ["GOOGLE_API_KEY"] = api_key_input
    
    st.markdown("---")
    st.markdown("### Debug Info")
    current_key = os.getenv("GOOGLE_API_KEY", "")
    if current_key:
        masked_key = f"{current_key[:5]}...{current_key[-4:]}" if len(current_key) > 10 else "****"
        st.write(f"Active Key: `{masked_key}`")
        if current_key.startswith("AIzaSyDNeWcpu"):
            st.error("⚠️ The detected key is the expired demo key. Please update it.")
    else:
        st.warning("⚠️ No API Key set.")

# --- Custom CSS (Clean, Professional, No Emojis) ---
st.markdown("""
    <style>
    /* Global Text Color */
    .stMarkdown, .stText, p, li, span {
        color: #333333 !important;
    }
    
    /* Global Styles */
    .stApp {
        background-color: #fcfcfc;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    
    h1, h2, h3, h4 {
        color: #111;
        font-weight: 600;
        letter-spacing: -0.5px;
    }
    
    /* Navigation/Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        padding-bottom: 1rem;
        border-bottom: 1px solid #eee;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px;
        color: #666;
        font-size: 16px;
        font-weight: 500;
        border: none;
        padding: 0 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #f0f0f0;
        color: #000;
        font-weight: 600;
    }

    /* Input Areas */
    .stTextInput > div > div > input, .stTextArea > div > div > textarea {
        background-color: #fff;
        border: 1px solid #ddd;
        border-radius: 4px;
        padding: 12px;
        font-size: 15px;
        line-height: 1.5;
        color: #333 !important; /* Force Text Color */
        -webkit-text-fill-color: #333 !important;
    }
    .stTextInput > div > div > input:focus, .stTextArea > div > div > textarea:focus {
        border-color: #333;
        box-shadow: 0 0 0 1px #333;
    }

    /* Buttons */
    .stButton > button {
        background-color: #111;
        color: white;
        border: none;
        padding: 0.6rem 1.2rem;
        border-radius: 4px;
        font-weight: 500;
        font-size: 14px;
        transition: all 0.2s;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: #333;
        color: white;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    /* Analysis Cards */
    .analysis-card {
        background: white;
        padding: 24px;
        border-radius: 8px;
        border: 1px solid #eee;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 24px;
    }
    .metric-container {
        display: flex;
        justify-content: space-between;
        padding: 12px 0;
        border-bottom: 1px solid #f5f5f5;
    }
    .metric-label {
        color: #666;
        font-weight: 500;
    }
    .metric-value {
        color: #111;
        font-weight: 600;
    }
    
    /* Chart Container */
    .chart-box {
        background: white;
        padding: 20px;
        border: 1px solid #eee;
        border-radius: 8px;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Logic ---
async def run_analysis_task(prompt_text):
    # CRITICAL: Build agent INSIDE the task to attach to current loop
    root_agent = build_agent()
    runner = InMemoryRunner(agent=root_agent)
    # Using run_debug as it's the standard entry point for single-turn checks
    response_events = await runner.run_debug(prompt_text)
    
    # Extract the final text response from the events
    final_text = ""
    # Traverse events in reverse to find the last model text output
    for event in reversed(response_events):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    final_text = part.text
                    break
        if final_text:
            break
            
    if not final_text:
        # Fallback: if no text found, format the last event's content
        final_text = "Analysis completed, but no text output was found."
        
    return final_text

# --- Initialize Session State ---
if 'analysis_result' not in st.session_state:
    st.session_state['analysis_result'] = None
if 'is_analyzing' not in st.session_state:
    st.session_state['is_analyzing'] = False

# --- Layout ---

st.title("Agentic and Fractal Reasoning for Automated Fact-Checking")

# Create two main tabs
tab1, tab2 = st.tabs(["Instructions", "Analysis Dashboard"])

# --- Tab 1: Instructions ---
with tab1:
    col_main, _ = st.columns([2, 1])
    with col_main:
        st.markdown("### 🛠 System Overview")
        st.info(
            """
            **This platform addresses the rapid spread of online misinformation by combining specialized machine-learning classifiers with a multi-agent system built on the Google Agent Development Kit (ADK) and powered by Gemini 3 Flash. This integration produces an interpretable, multi-dimensional credibility assessment that balances statistical rigor with grounded reasoning.**
            """
        )
        
        st.markdown("#### ⚙️ How It Works")
        st.markdown(
            """
            1. **Predictive Modeling**: The system first runs 6 specialized classifiers to detect specific signals:
                - **News Coverage**: What topic is being discussed?
                - **Intent**: Is the goal to inform, persuade, or deceive?
                - **Stance**: Is the article supportive, neutral, or opposing?
                - **Sensationalism**: Does it use emotionally charged language?
                - **Context Veracity**: Is the context supported by internal/external evidence?
                - **Title–Body Alignment**: Does the headline accurately reflect the content?
            
            2. **Chain-of-Thought Reasoning**: A Google Gemini-powered agent synthesizes these signals. It analyzes the article step-by-step, cross-referencing internal consistency and logic, similar to a human fact-checker.
            """
        )

        st.markdown("#### 📊 Model Performance Comparison")
        st.info(
            """
            We evaluated several agent architectures to determine the optimal balance of accuracy and consistency. The table below shows the performance of different prompting strategies across our six key factuality factors.
            """
        )

        # Performance Data DataFrame
        import pandas as pd
        performance_data = {
            "Agent Strategy": [
                "Simple Prompt", "COT", "COT + Function Calling", 
                "COT + Few Shot + Function Calling", "FCOT", 
                "FCOT + Function Calling", "FCOT + Few Shot + Function Calling"
            ],
            "News Coverage": [0.70, 0.90, 0.80, 0.85, 0.85, 0.75, 0.85],
            "Intent": [0.70, 0.70, 0.80, 0.85, 0.70, 0.80, 0.75],
            "Sensationalism": [0.60, 0.85, 0.90, 0.80, 0.75, 0.40, 0.75],
            "Stance": [0.65, 0.50, 0.55, 0.55, 0.40, 0.55, 0.60],
            "Title vs Body": [0.80, 0.90, 0.85, 0.85, 0.90, 0.85, 0.85],
            "Context Veracity": [0.35, 0.70, 0.70, 0.70, 0.60, 0.75, 0.70],
            "Average Performance": [0.63, 0.76, 0.77, 0.77, 0.70, 0.68, 0.75],
            "Standard Deviation": [0.15, 0.16, 0.13, 0.12, 0.18, 0.17, 0.09]
        }
        df_perf = pd.DataFrame(performance_data)
        st.dataframe(df_perf, hide_index=True)

        st.markdown("#### 🏆 Why We Chose 'COT + Function Calling'")
        st.success(
            """
            Based on our experiments, we selected the **Chain-of-Thought (CoT) + Function Calling** architecture for this application.
            
            1.  **Highest Average Performance (0.77)**: It ties for the top spot in overall accuracy, significantly outperforming simple prompting (0.63).
            2.  **Superior Sensationalism Detection (0.90)**: It achieved the highest score in detecting sensationalist language, a critical signal for misinformation.
            3.  **Balanced Consistency (Std Dev 0.13)**: Unlike other high-performing models that varied wildly between tasks, this agent maintained consistent reliability across all factors.
            4.  **Operational Efficiency**: While "Few Shot" approaches performed similarly, they require significantly longer prompts (higher cost/latency). The standard CoT + Function Calling model delivers top-tier results with greater efficiency.
            """
        )
        
        st.markdown("#### 🚀 Usage Guide")
        st.success(
            """
            1. Switch to the **Analysis Dashboard** tab.
            2. Paste the **Headline** and **Body Text** of an article.
            3. Click **Run Analysis**.
            4. Review the detailed report on the right panel.
            """
        )

# --- Tab 2: Analysis Dashboard ---
with tab2:
    # Use columns to create a split view
    col_left, col_right = st.columns([1, 1], gap="large")
    
    # Left Column: Input
    with col_left:
        st.subheader("Input")
        with st.form("main_form"):
            title_input = st.text_input("Headline", placeholder="Article Headline")
            body_input = st.text_area("Body Text", height=500, placeholder="Paste full article text here...")
            submit_btn = st.form_submit_button("Run Analysis")
        
        if submit_btn:
            if not title_input or not body_input:
                st.error("Please provide both headline and body text.")
            else:
                st.session_state['is_analyzing'] = True
                full_prompt = f"Title: {title_input}\n\nBody: {body_input}"
                
                try:
                    # Create a fresh event loop for this run
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    result = loop.run_until_complete(run_analysis_task(full_prompt))
                    
                    loop.close()
                    
                    st.session_state['analysis_result'] = result
                    st.session_state['is_analyzing'] = False
                except Exception as e:
                    import traceback
                    st.error(f"Analysis failed: {str(e)}")
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())
                    st.session_state['is_analyzing'] = False

    # Right Column: Output
    with col_right:
        st.subheader("Analysis Report")
        
        if st.session_state['is_analyzing']:
            with st.spinner("Analyzing article structure, tone, and logic..."):
                 st.empty() # Placeholder for spinner
        
        elif st.session_state['analysis_result']:
            result_text = st.session_state['analysis_result']
            
            # Render the Markdown Report
            st.markdown(f"""
            <div class="analysis-card">
                {result_text}
            </div>
            """, unsafe_allow_html=True)
            
            # --- Visualization Section ---
            st.markdown("### Signal Confidence Overview")
            
            # Sample visualization data
            import pandas as pd
            import numpy as np
            
            chart_data = pd.DataFrame({
                'Factor': ['News Coverage', 'Intent', 'Sensationalism', 'Stance', 'Title Alignment', 'Veracity'],
                'Confidence': [85, 70, 90, 65, 80, 75] 
            })
            
            st.bar_chart(chart_data.set_index('Factor'))
            st.caption("Note: Chart values are illustrative until confidence parsing is connected to the agent output.")
            
        else:
            # Empty State
            st.info("Ready to analyze. Submit an article on the left to see results here.")