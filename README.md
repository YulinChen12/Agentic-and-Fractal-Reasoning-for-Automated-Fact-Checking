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

1. **Clone the repository**
```bash
git clone https://github.com/YulinChen12/DSC180A-GroupNull.git
cd DSC180A-GroupNull
```

2. **Install required packages**
```
conda env create -f requirements.yml
conda activate groupnull
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




