# DSC180A-GroupNull: Misinformation Detection System

## Project Overview

This repository contains the full implementation of our DSC 180A capstone project: a hybrid  misinformation detection pipeline combining predictive models with a grounded LLM (Gemini 2.5 Flash). The system extracts stance, sentiment, sensationalism, reputation, and news-coverage features from input articles and integrates them into a multi-step reasoning agent for credibility analysis.

## Repository Structure

```
DSC180A-GroupNull/
│
├── data/
│   ├── pl_train.csv
│   ├── pl_val.csv
│   ├── pl_test.csv
│   ├── train_article.json
│   ├── test_article.json
│   ├── train2.tsv
│   ├── test2.csv
│   └── val2.tsv
│
├── predictive_models/
│   ├── reputation_model/
│   ├── stance_model/
│   ├── Intent_Classification_Model.ipynb
│   ├── News_Coverage_Model.ipynb
│   ├── reputation_model.ipynb
│   ├── Sensationalism.ipynb
│   ├── Sentiment.ipynb
│   └── stance_model.ipynb
│
├── streamlit_app/
│   ├── app.py
│   ├── .streamlit/
│   └── requirements.txt
│
├── combined_pred_model.ipynb
├── final_model.ipynb
├── requirements.yml
└── start_streamlit.sh

```
### Setup Steps

1. **Log into DSMLP and launch the required GPU environment**
All model training, evaluation, and experiments should be run inside the following DSMLP container:
```
launch-scipy-ml.sh -W DSC180A_FA25_A00 -c 8 -m 32 -g 1
```
   
2. **Clone the repository**
```bash
git clone https://github.com/YulinChen12/DSC180A-GroupNull.git
cd DSC180A-GroupNull
```

3. **Install required packages using Conda**
```
conda env create -f requirements.yml
conda activate groupnull
```
4. **Ensure Git LFS is installed and pull large files**
```
git lfs install
git lfs pull
```

### Dataset Overview
This project uses multiple datasets for different components of the pipeline. Below is a clear explanation of each key file found inside the ```data/``` folder

- ```pl_train.csv```, ```pl_val.csv```, ```pl_test.csv```
  
  These files come from the LIAR-PLUS stance dataset, where each sentence is labeled with:
  - support
  - neutral
  - deny
  
  These labels are used exclusively for training our stance prediction model located in:```predictive_models/stance_model/```
  
  The stance model learns to classify how a sentence relates to a claim (agree, neutral, or disagree).

- ```train_article.json```, ```train_article.json```, ```train_article.json```
  - This dataset contains full news articles with labels that we use to train and evaluate the generative agent.
 
- ```train2.tsv```, ```test2.tsv```, ```val2.tsv```
  - These files are used for training and evaluating the predictive components (e.g., sensationalism, sentiment, coverage, reputation).

- ```scraper/``` Folder
  - This folder contains additional article data collected through our custom scraping scripts.
  - These articles are not currently part of the training pipeline, but they are included in the repository because:
      - They can be used to augment model training in the future
      - They provide more diverse real-world samples for testing


  
### 1. Running Individual Predictive Models
Each predictive model is implemented in a standalone Jupyter notebook inside ```predictive_models/```.

To train or evaluate a model:
1. Open the corresponding notebook
2. Run all cells
3. The notebook will:
   - load LIAR-PLUS or processed data
   - fine-tune a classifier
   - evaluate (accuracy, F1, confusion matrix)
  

### 2. Combined Predictive Feature Pipeline
The notebook ```combined_pred_model.ipynb``` runs an input article through all predictive models and produces:

```
{
  "stance": "...",
  "reputation": "...",
  "sensationalism": "...",
  "sentiment": "...",
  "coverage": "..."
}
```

This structured output becomes the input to the agentic LLM reasoning layer.


### 3. Final Hybrid Agentic Model
The notebook ```final_model.ipynb``` is the experimental notebook where we combine the final predictive feature pipeline (stance, sensationalism, sentiment, reputation, coverage, intent) and the Gemini Flash generative model to evaluate how different prompting strategies affect misinformation classification quality.

This notebook runs three structured experiments:

- **Experiment 1 — Zero-Shot Prompting**

    - Gemini Flash receives only the article text and predictive model labels and is asked to produce a credibility assessment without any examples or reasoning demonstrations. This serves as our baseline agent performance.

- **Experiment 2 — Few-Shot + Chain-of-Thought Prompting**

  - Here we build onto the baseline agent, and add 7 labeled examples with a recipe for reasoning. This experiment tests whether explicit examples and step-by-step reasoning improve the model’s accuracy and explanation quality.

- **Experiment 3 — Few-Shot Chain-of-Thought + Web Search (SERP API)**

  - The third experiment extends Experiment 2 by adding live web search retrieval via SERP API.


### 5. Running the Streamlit App
The Streamlit web interface is located in ```streamlit_app/app.py```

Start Using Provided Script in your terminal
```
./start_streamlit.sh
```

This launches a live UI where users can paste article text to 
- Visualize stance, sentiment, sensationalism, reputation, news coverage predictions
- Get additional feature labels: title vs body alignment, context veracity, location
- Run the agentic Gemini-based credibility analysis

