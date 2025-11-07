# DSC180A-GroupNull

## Project Overview

This project is a unified article analysis system that integrates 6 machine learning models to comprehensively analyze news articles and text content. The system provides multi-dimensional analysis including topic classification, intent detection, sensationalism scoring, sentiment analysis, reputation assessment, and stance detection.

Developed as part of DSC180A at UC San Diego.

## Combined_Model.ipynb

The main notebook (`Combined_Model.ipynb`) integrates all 6 models into a unified system that analyzes articles across multiple dimensions:

### 6 Integrated Models:

1. **News Coverage Classification** - Identifies the topic/subject of the article
2. **Intent Classification** - Determines communication intent (inform, persuade, entertain, deceive)
3. **Sensationalism Detection** - Classifies whether content is sensational or neutral
4. **Sentiment Analysis** - Analyzes emotional sentiment (positive, negative, neutral)
5. **Reputation Classification** - Assesses reputation level (low, medium, high)
6. **Stance Classification** - Detects political stance (against, neutral, favor)

### Key Features:
- Sentence-level analysis with voting mechanism for final predictions
- Complete end-to-end pipeline from text input to multi-model predictions
- Integrated prediction function: `analyze_complete_article(article_text)`
- Detailed results output with confidence scores and sentence-level breakdowns

## Installation

### Prerequisites
```
Python 3.8+
PyTorch
Transformers (Hugging Face)
scikit-learn
pandas
numpy
vaderSentiment
nrclex
```

### Setup Steps

1. **Clone the repository**
```bash
git clone https://github.com/YulinChen12/DSC180A-GroupNull.git
cd DSC180A-GroupNull
```

2. **Install required packages**
```bash
pip install torch transformers scikit-learn pandas numpy vaderSentiment nrclex joblib
```

3. **Download large model files**

Due to GitHub file size limitations, download the following files from Google Drive:

**Reputation Model Weights:**
- 📥 [Download Link](https://drive.google.com/drive/folders/1XopQiNOQVu6t09uocvKyrS0BinZOU-QU?usp=drive_link)
- Place `model.safetensors` in `reputation_model/`

**Stance Model Weights & Artifacts:**
- 📥 [Download Link](https://drive.google.com/drive/folders/1uSrrbrkRLXkZimxUv_yDoOnm4f4E4MFj?usp=drive_link)
- Place `model.safetensors` in `stance_model/`
- Place all `.joblib` files in `artifacts/`

4. **Run the notebook**
```bash
jupyter notebook Combined_Model.ipynb
```

## Repository Structure

```
DSC180A-GroupNull/
├── Combined_Model.ipynb          # Main unified analysis notebook
├── reputation_model/             # Reputation analysis model files
├── stance_model/                 # Stance detection model files
├── artifacts/                    # Trained model artifacts (.joblib files)
├── train.tsv, test.tsv, valid.tsv   # Datasets
└── per_class_metrics.csv         # Model performance metrics
```

## Usage

Open `Combined_Model.ipynb` in Jupyter Notebook and run all cells. The notebook will load all 6 models and provide the `analyze_complete_article()` function for analyzing any article text.

## License

This project is developed for educational purposes as part of UC San Diego's Data Science curriculum.
