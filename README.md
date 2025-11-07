# DSC180A-GroupNull

## Project Overview

This repository contains a comprehensive news analysis system that combines multiple machine learning models for analyzing news articles and social media content. The system performs three main tasks:

1. **Stance Detection** - Classifies the stance of text toward a given topic (support, refute, discuss, or unrelated)
2. **Reputation Analysis** - Analyzes the reputation score of entities mentioned in news articles
3. **Intent Classification** - Determines the intent behind news coverage (informative, persuasive, sensational, etc.)

This project was developed as part of DSC180A at UC San Diego.

## Features

- 🎯 **Multi-task Learning**: Combines stance detection, reputation analysis, and intent classification
- 🤖 **Transformer-based Models**: Utilizes state-of-the-art BERT-based models for NLP tasks
- 📊 **Traditional ML Models**: Includes TF-IDF with Linear SVC and Logistic Regression for subject classification
- 🔄 **End-to-End Pipeline**: Complete workflow from data loading to prediction
- 📈 **Performance Metrics**: Comprehensive evaluation with per-class metrics

## Models Included

### 1. Stance Detection Model (`stance_model/`)
Fine-tuned transformer model for detecting the stance of text toward a target topic.
- **Architecture**: RoBERTa-based
- **Classes**: Support, Refute, Discuss, Unrelated
- **Use Case**: Analyzing how news articles position themselves relative to specific claims or topics

### 2. Reputation Analysis Model (`reputation_model/`)
BERT-based model for analyzing entity reputation in news coverage.
- **Architecture**: BERT-based
- **Output**: Reputation scores and sentiment analysis
- **Use Case**: Understanding how news coverage affects public perception of entities

### 3. Intent Classification Models (`artifacts/`)
Traditional ML models for classifying news article subjects and intent.
- **Models**: Linear SVC, Logistic Regression, TF-IDF based classifiers
- **Features**: TF-IDF vectorization with supervised learning
- **Use Case**: Categorizing news articles by subject and identifying authorial intent

## Repository Structure

```
DSC180A-GroupNull/
├── stance_model/              # Stance detection model files
│   ├── config.json           # Model configuration
│   ├── tokenizer.json        # Tokenizer configuration
│   ├── vocab.json            # Vocabulary
│   └── model.safetensors     # Model weights (download required)
│
├── reputation_model/          # Reputation analysis model files
│   ├── config.json           # Model configuration
│   ├── tokenizer.json        # Tokenizer configuration
│   ├── vocab.txt             # Vocabulary
│   └── model.safetensors     # Model weights (download required)
│
├── artifacts/                 # Trained model artifacts
│   ├── test_predictions.csv  # Test set predictions
│   ├── valid_predictions.csv # Validation set predictions
│   └── *.joblib              # Scikit-learn models (download required)
│
├── Combined_Model.ipynb       # Notebook integrating all models
├── Prediction_Model.ipynb     # Prediction and evaluation notebook
├── per_class_metrics.csv      # Detailed performance metrics
│
└── Data files:
    ├── train.tsv, train2.tsv # Training datasets
    ├── test.tsv, test2.tsv   # Testing datasets
    └── valid.tsv, val2.tsv   # Validation datasets
```

## Installation

### Prerequisites
- Python 3.8+
- Jupyter Notebook
- PyTorch
- Transformers (Hugging Face)
- Scikit-learn
- Pandas, NumPy

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/YulinChen12/DSC180A-GroupNull.git
cd DSC180A-GroupNull
```

2. **Install dependencies**
```bash
pip install torch transformers scikit-learn pandas numpy jupyter joblib
```

3. **Download large model files**

Due to GitHub file size limitations, some model weights need to be downloaded separately:

**Reputation Model Files:**
- 📥 [Download Link](https://drive.google.com/drive/folders/1XopQiNOQVu6t09uocvKyrS0BinZOU-QU?usp=drive_link)
- Required: `reputation_model/model.safetensors`

**Stance Model & Artifacts:**
- 📥 [Download Link](https://drive.google.com/drive/folders/1uSrrbrkRLXkZimxUv_yDoOnm4f4E4MFj?usp=drive_link)
- Required files:
  - `stance_model/model.safetensors`
  - `artifacts/news_subject_linear_svc.joblib`
  - `artifacts/news_subject_lr.joblib`
  - `artifacts/news_subject_improved.joblib`
  - `artifacts/intent_supervised.joblib`
  - `artifacts/intent_tfidf.joblib`
  - `artifacts/news_subject_tfidf.joblib`

4. **Place downloaded files** in their respective directories as shown in the structure above.

## Usage

### Running the Combined Model

Open and run the `Combined_Model.ipynb` notebook to see all models working together:

```bash
jupyter notebook Combined_Model.ipynb
```

This notebook demonstrates:
- Loading all three models
- Preprocessing input data
- Making predictions with each model
- Combining results for comprehensive analysis

### Making Predictions

Use the `Prediction_Model.ipynb` notebook for inference on new data:

```bash
jupyter notebook Prediction_Model.ipynb
```

### Example Code

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load stance detection model
tokenizer = AutoTokenizer.from_pretrained("./stance_model")
model = AutoModelForSequenceClassification.from_pretrained("./stance_model")

# Make prediction
text = "Climate change is a serious threat to our planet"
target = "climate change"
inputs = tokenizer(text, target, return_tensors="pt", truncation=True)
outputs = model(**inputs)
prediction = torch.argmax(outputs.logits, dim=1)
```

## Dataset Information

The project uses TSV-formatted datasets for training and evaluation:
- **Training sets**: `train.tsv`, `train2.tsv`
- **Validation sets**: `valid.tsv`, `val2.tsv`
- **Test sets**: `test.tsv`, `test2.tsv`

Each dataset contains labeled examples for stance detection, reputation analysis, and intent classification tasks.

## Performance

Detailed per-class performance metrics are available in `per_class_metrics.csv`, including:
- Precision, Recall, F1-Score for each class
- Confusion matrices
- Overall accuracy metrics

## Contributing

This is an academic project for DSC180A. For questions or suggestions, please open an issue.

## License

This project is developed for educational purposes as part of UC San Diego's Data Science curriculum.

## Acknowledgments

- UC San Diego DSC180A Course
- Hugging Face Transformers Library
- PyTorch Team
