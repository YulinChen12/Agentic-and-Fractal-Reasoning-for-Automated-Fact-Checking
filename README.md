# Agentic and Fractal Reasoning for Automated Fact-Checking
  
## Project Overview
Online misinformation spreads rapidly and can influence public opinion, policy decisions, and social discourse. Detecting misleading or biased information in news articles is challenging because credibility depends not only on factual accuracy but also on factors such as tone, intent, framing, and contextual alignment between headlines and content.

This project builds a **hybrid misinformation detection system** that combines predictive machine learning models with large language model (LLM) reasoning agents. The system extracts structured credibility signals from an article—such as topic coverage, intent, stance, and sensationalism—and integrates them into a multi-step reasoning pipeline to assess the article's credibility.

Our framework evaluates news articles across **six factuality factors**:

- **News Coverage** – the primary topic or domain of the article  
- **Intent** – whether the article is informational, persuasive, or opinion-driven  
- **Stance** – the article’s position toward a claim or topic  
- **Sensationalism** – whether emotionally exaggerated or misleading language is used  
- **Context Veracity** – whether supporting context appears credible or misleading  
- **Title vs Body Alignment** – whether the headline accurately reflects the article content  

To perform this analysis, the system combines **predictive models** trained on labeled datasets with **agent-based reasoning strategies** such as Chain-of-Thought (CoT) and Fractal Chain-of-Thought (FCoT). These agents interpret predictive signals, reason over article content, and produce a structured analysis that helps identify potential misinformation or bias.

## Repository Structure

```
DSC180A-GroupNull/
│
├── agents/                     # LLM-based reasoning agents
│   ├── simple_agent/           # Baseline agent
│   ├── cot_agent/              # Chain-of-Thought (CoT) agent
│   ├── cot_icl_agent/          # CoT with In-Context Learning
│   ├── cot_no_function_calling_agent/
│   ├── fcot_agent/             # Fractal CoT agent
│   ├── fcot_icl_agent/         # Fractal CoT with ICL
│   └── fcot_no_function_calling_agent/
│
├── data/                       # Datasets
│   ├── gen_data/               # JSON data for agent training/eval
│   └── pred_data/              # CSV/TSV data for predictive models
│
├── pred_models_training/       # Predictive Model Pipeline
│   ├── artifacts/              # Trained joblib artifacts (LinearSVC, etc.)
│   ├── stance_model/           # Fine-tuned BERT/RoBERTa model files
│   ├── predictors.py           # Python interface for all predictive models
│   └── train_all.py            # Script to retrain all models
│
├── results/                    # Evaluation Results (CSVs)
│   ├── Simple_Agent/
│   ├── COT_Function_Calling/
│   ├── COT_ICL/
│   ├── COT_No_Function_Calling/
│   ├── FCOT_Function_Calling/
│   ├── FCOT_ICL/
│   └── FCOT_No_Function_Calling/
│
├── streamlit_app/              # Interactive Web Application
│   └── app.py                  # Main Streamlit application entry point
│
├── website/                    # Project Website (React/Vite)
│   ├── src/
│   └── public/
│
├── pred_article.py             # CLI script for quick article analysis
├── start_streamlit.sh          # Helper script to launch the app
├── environment.yml             # Conda environment definition
├── requirements.txt            # Pip requirements
└── README.md
```

### Setup Steps
   
1. **Clone the repository**
```bash
git clone https://github.com/YulinChen12/DSC180A-GroupNull.git
cd DSC180A-GroupNull
```

2. **Create Environment**
```
conda env create -f environment.yml
conda activate dsc180
```
OR

```
pip install -r requirements.txt
```

3. **Download Pretrained Artifacts**
```
https://drive.google.com/file/d/1sZnWKuClTbTjlG2rgRoGtpJKNq4_og-t/view?usp=sharing
```
After downloading, place them in: stance_model/
Expected structure:
```
pred_models_training/      
└── stance_model
      └── model.safesensors  
```
OR
**Ensure Git LFS is installed and pull large files**
```
git lfs install
git lfs pull
```
Large trained model files are NOT uploaded due to GitHub size limits. You must generate them locally or download from Google drive.

Generate Predictive Model Artifacts
From the repo root, run:
```
python -m pred_models_training.train_all
```

### Dataset Overview
This project uses multiple datasets for different components of the pipeline. Below is a clear explanation of each key file found inside the ```pred_data/``` folder

- ```pl_train.csv```, ```pl_val.csv```, ```pl_test.csv```
  
  These files come from the LIAR-PLUS stance dataset, where each sentence is labeled with:
  - support
  - neutral
  - deny
  
  These labels are used exclusively for training our stance prediction model located in:```pred_model_training/stance_model/```
  
  The stance model learns to classify how a sentence relates to a claim (agree, neutral, or disagree).
 
- ```train2.tsv```, ```test2.tsv```, ```val2.tsv```
  - These files are used for training and evaluating the predictive components (e.g., sensationalism, sentiment, coverage, reputation).

- ```train_article.json```, ```test_article.json```, ```test_article_no_label.json```
  - This dataset contains full news articles with labels that we use to train and evaluate the generative agent.

  
## 1. Running Predictive Pipeline Only
After training the predictive models, you can run `pred_article.py` to get prediction results for a specific news article. In `pred_article.py`, replace the `article_title` and `article_text` fields with the title and full text of the article you want to analyze, then run:
```
python pred_article.py
```
It will output the following:
```
=== Predictions ===
factor           label         conf
--------------------------------------
news_coverage    ...
intent           ...
sensationalism   ...
stance           ...

```

## 2. Agent Framework

Make sure you have a **Google AI Studio API key** to run the agents.

Set your API key in a `.env` file:
```
GOOGLE_API_KEY = "YOUR_API_KEY"
```
This project includes **7 reasoning agents**. All agents are located in the `agents/` directory, with each agent implemented in its own folder.

### Available Agents

- **Simple Agent**  
  Uses direct prompting without structured reasoning. Serves as a baseline for comparison.

- **CoT Agent (Chain-of-Thought)**  
  Uses step-by-step reasoning to analyze articles before producing predictions.

- **CoT + In-Context Learning Agent**  
  Combines Chain-of-Thought reasoning with 9 human labeled articles prompts to guide the model.

- **CoT (No Function Calling) Agent**  
  Uses Chain-of-Thought reasoning without structured function calls.

- **FCoT Agent (Fractal Chain-of-Thought)**  
  Implements a structured reasoning framework that includes:
  - **Local Thought Units** – decomposes reasoning into smaller reasoning modules  
  - **Predictive Grounding** – connects reasoning steps to predictive model outputs  
  - **Aperture Expansion** – progressively expands the reasoning context  
  - **Reflective Update** – revises reasoning based on intermediate results  
  - **Granularity Control** – adjusts the level of reasoning detail

- **FCoT + In-Context Learning Agent**  
  Combines Fractal Chain-of-Thought reasoning above with 9 human labeled articles prompts.

- **FCoT (No Function Calling) Agent**  
  A variant of FCoT that performs structured reasoning without explicit function execution and predictive grounding.

### Running the Agent Interface

You can interact with the agents using the ADK web interface.

From the repository root:
```
cd agents
adk web
```

This launches a local interface where you can paste a news article into the chat box.

The agents will analyze the article and return results across **six factuality factors**:

- **News Coverage** – topic classification of the article  
- **Intent** – the communicative goal of the article  
- **Stance** – the article’s position toward the topic  
- **Sensationalism** – whether exaggerated or emotionally charged language is used  
- **Context Veracity** – whether contextual information appears credible or misleading  
- **Title vs Body Alignment** – whether the headline accurately reflects the article content

The output provides a structured analysis across these factors to help evaluate potential misinformation or bias in news articles.

### Simplified General Output Template

# Fact-Check Final Report

## Agent Analysis Summary

### Labels
| Signal | Label | Confidence |
|---|---|---|
| News Coverage | Biased | 82% |
| Intent | Mislead | 76% |
| Sensationalism | High | 88% |
| Stance | Against | 80% |
| Title vs Body | Exaggerates | 84% |
| Context Veracity | Missing Context | 79% |

### Double-check
- Intent and sensationalism are aligned, since the article uses exaggerated language that appears designed to mislead.
- Context veracity lowers overall confidence because the article leaves out important background information.

### Short Summary
The article presents a real event, but it frames the story in a dramatic and one-sided way. While some details are accurate, the lack of context and exaggerated title make the overall article misleading.

## Final Judgment
- **Final Article Verdict:** Misleading
- **Overall Confidence:** 83%
- **Reasoning Explanation:** The article contains a mix of truth and distortion. Its strongest issues are exaggerated framing, missing context, and signals that suggest an intent to shape reader perception rather than inform clearly.

## Evaluation and Experiment Results
### Evaluation Metrics

To evaluate the performance of the predictive models and reasoning agents, we compute accuracy-based metrics for each of the six factuality factors analyzed in this project:

- **News Coverage** – topic classification of the article  
- **Intent** – the communicative goal of the article  
- **Stance** – the article’s position toward the topic  
- **Sensationalism** – whether exaggerated or emotionally charged language is used  
- **Context Veracity** – whether contextual information appears credible or misleading  
- **Title vs Body Alignment** – whether the headline accurately reflects the article content

For each agent configuration, predictions are compared against labeled evaluation data and the accuracy for each factor is recorded. These metrics allow us to compare the effectiveness of different reasoning strategies such as:

- Simple Prompting
- Chain-of-Thought (CoT)
- Fractal Chain-of-Thought (FCoT)

---

### Experiment Execution

Experiments were conducted using batch evaluation notebooks located in the `experiment_notebooks/` directory. Each notebook evaluates a specific agent configuration on a set of test articles and records the outputs for the six factuality factors.

Example experiment notebooks:

```
- experiment_notebooks/cot_agent.ipynb
- experiment_notebooks/cot_icl.ipynb
- experiment_notebooks/fcot_agent.ipynb
```

These notebooks run the agents on batches of test articles and export the results for further analysis.

---

### Saved Results

All experiment outputs are exported and stored in the `results/` directory. Each agent has its own folder containing CSV files corresponding to the six factuality factors.
Each experiment folder contains result files such as:

- News_Coverage.csv

- Intent.csv

- Stance.csv

- Sensationalism.csv

- Context_Veracity.csv

- Title_vs_Body.csv

These CSV files contain the predictions generated by each agent during batch evaluation.

Reproducing Results
To reproduce predictive model outputs, replace the article text, then run the predictive pipeline:
```
python -m pred_models_training.train_all
python pred_article.py
```

To evaluate the reasoning agents interactively, launch the ADK interface:
```
cd agents
adk web
```

### 3. Running the Streamlit App
The Streamlit web interface is located in ```streamlit_app/app.py```

Start Using Provided Script in your terminal
```
./start_streamlit.sh
```

This launches a live UI where users can paste article text to 
- Visualize stance, sentiment, sensationalism, reputation, news coverage predictions
- Get additional feature labels: title vs body alignment, context veracity, location
- Run the agentic Gemini-based credibility analysis
