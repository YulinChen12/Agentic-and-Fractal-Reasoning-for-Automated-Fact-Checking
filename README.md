# DSC180A-GroupNull

## Overview
This repository contains machine learning models and notebooks for news analysis including stance detection, reputation analysis, and intent classification.

## Large Model Files

Due to file size limitations on GitHub, some model files are not included in this repository. Please download them from the following Google Drive links:

### Model Files - Part 1 (Reputation Model)
📥 [Download Reputation Model Files](https://drive.google.com/drive/folders/1XopQiNOQVu6t09uocvKyrS0BinZOU-QU?usp=drive_link)

Required file:
- `reputation_model/model.safetensors`

### Model Files - Part 2 (Stance Model and Artifacts)
📥 [Download Stance Model and Artifact Files](https://drive.google.com/drive/folders/1uSrrbrkRLXkZimxUv_yDoOnm4f4E4MFj?usp=drive_link)

Required files:
- `stance_model/model.safetensors`
- `artifacts/news_subject_linear_svc.joblib`
- `artifacts/news_subject_lr.joblib`
- `artifacts/news_subject_improved.joblib`
- `artifacts/intent_supervised.joblib`
- `artifacts/intent_tfidf.joblib`
- `artifacts/news_subject_tfidf.joblib`

### Installation Instructions

1. Clone this repository
2. Download the model files from the Google Drive links above
3. Place the downloaded files in their respective directories as indicated in the file paths
4. Run the Jupyter notebooks

## Repository Structure

- `artifacts/` - Trained model artifacts and predictions
- `reputation_model/` - Reputation analysis model files
- `stance_model/` - Stance detection model files
- `Combined_Model.ipynb` - Combined model notebook
- `Unified_Model-checkpoint.ipynb` - Unified model checkpoint
- `train.tsv`, `test.tsv`, `valid.tsv` - Training, testing, and validation datasets
- `train2.tsv`, `test2.tsv`, `val2.tsv` - Additional datasets

## Usage

After downloading the required model files from Google Drive and placing them in the correct directories, you can run the Jupyter notebooks to perform model inference and analysis.
