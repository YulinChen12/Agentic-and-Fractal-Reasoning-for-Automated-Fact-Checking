# DSC180A-GroupNull: Misinformation Detection System
  
## Project Overview

A hybrid misinformation detection framework that combines:
- Predictive ML models (stance, sensationalism, topic, intent)
- Agent-based reasoning using Gemini (Chain-of-Thought(CoT) & Fractal CoT)
- Optional web-grounded verification
- Streamlit interface for interactive testing

The system extracts structured credibility signals from an article and integrates them into a multi-step reasoning pipeline for final credibility analysis.

## Repository Structure

```
DSC180A-GroupNull/
│
├── agents/                     # All agent implementations
│   ├── simple_agent/
│   ├── cot_agent/
│   ├── cot_icl_agent/
│   ├── cot_no_function_calling_agent/
│   ├── fcot_agent/
│   ├── fcot_icl_agent/
│   └── fcot_no_function_calling_agent/
│
├── data/                       # Dataset storage
│   ├── gen_data/
│   └── pred_data/
│
├── pred_models_training/       # Training scripts for predictive models
│   ├── stance_model # pretrained artifacts
│   ├── artifacts/    # local predictive model artifacts
│   ├── predictors.py
│   ├── train_all.py
│
│
├── streamlit_app/
│   ├── app.py
│
├── pred_article.py             # Script to run predictions on an article

├── environment.yml
├── requirements.txt
└── .gitignore
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

