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
After training, you can place a news article into this file to test: 
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
sentiment        ...
reputation       ...
stance           ...

```

## 2.Agent Framework
Make sure you have a AI Studio API key to run the agents!

Set them in the py files:
```
os.environ["GOOGLE_API_KEY"] =YOUR_KEY
```
This project includes two reasoning agents:

### 1. CoT Agent (```cot_agent.py```)
- Standard Chain-of-Thought reasoning
- Can be exposed on a shared IP
- Designed for live API-style usage
  
To run:
```
python agents/cot_agent.py
```

Then test using:
```
python client.py
```

All machines using the same IP can call this agent once it's running.

### 2. FCoT Agent (```fcot_agent.py```)
- Fractal Chain-of-Thought reasoning
- Multi-agent architecture
- Parallel factor analysts
-  Context-grounded verification
  
To run:
```
python agents/fcot_agent.py
```

Then test using:
```
python client.py
```

All machines using the same IP can call this agent once it's running.


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

