# DSC180A-GroupNull

## Project Overview

This project is a unified article analysis system that integrates 6 machine learning models to comprehensively analyze news articles and text content. The system provides multi-dimensional analysis including topic classification, intent detection, sensationalism scoring, sentiment analysis, reputation assessment, and stance detection.

Developed as part of DSC180A at UC San Diego.

## Notebooks

### Combined_Model.ipynb

The main notebook that integrates all 6 models into a unified system for comprehensive article analysis:

**6 Integrated Models:**
1. **News Coverage Classification** - Identifies the topic/subject of the article
2. **Intent Classification** - Determines communication intent (inform, persuade, entertain, deceive)
3. **Sensationalism Detection** - Classifies whether content is sensational or neutral
4. **Sentiment Analysis** - Analyzes emotional sentiment (positive, negative, neutral)
5. **Reputation Classification** - Assesses reputation level (low, medium, high)
6. **Stance Classification** - Detects political stance (against, neutral, favor)

**Key Features:**
- Sentence-level analysis with voting mechanism for final predictions
- Complete end-to-end pipeline from text input to multi-model predictions
- Integrated prediction function: `analyze_complete_article(article_text)`
- Detailed results output with confidence scores and sentence-level breakdowns

### RAG Practice.ipynb

A Retrieval-Augmented Generation (RAG) system implementation for article analysis:

**Components:**
1. **Text Chunking** - Uses NLTK to segment articles into manageable chunks
2. **Embedding Generation** - Creates vector embeddings using DistilBERT
3. **Vector Storage** - Local storage of text chunks and their embeddings
4. **Semantic Retrieval** - Cosine similarity-based retrieval for relevant context

**Key Features:**
- Query-based semantic search over article content
- DistilBERT-based text embeddings for semantic understanding
- Efficient retrieval of relevant text chunks for question answering

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
nltk
chromadb (optional, for RAG)
sentence-transformers (optional, for RAG)
```

### Setup Steps

1. **Clone the repository**
```bash
git clone https://github.com/YulinChen12/DSC180A-GroupNull.git
cd DSC180A-GroupNull
```

2. **Install required packages**

For Combined_Model.ipynb:
```bash
pip install torch transformers scikit-learn pandas numpy vaderSentiment nrclex joblib
```

For RAG Practice.ipynb (additional):
```bash
pip install nltk chromadb sentence-transformers
```

3. **Download large model files**

Due to GitHub file size limitations, download the following model weights from Google Drive:

**Reputation Model Weights:**
- 📥 [Download Link](https://drive.google.com/drive/folders/1XopQiNOQVu6t09uocvKyrS0BinZOU-QU?usp=drive_link)
- Place `model.safetensors` in `reputation_model/`

**Stance Model Weights:**
- 📥 [Download Link](https://drive.google.com/drive/folders/1uSrrbrkRLXkZimxUv_yDoOnm4f4E4MFj?usp=drive_link)
- Place `model.safetensors` in `stance_model/`

4. **Run the notebooks**
```bash
jupyter notebook Combined_Model.ipynb
# or
jupyter notebook "RAG Practice.ipynb"
```

## Repository Structure

```
DSC180A-GroupNull/
├── Combined_Model.ipynb          # Main unified analysis notebook (6 models)
├── RAG Practice.ipynb            # RAG system for semantic retrieval
├── Prediction_Model.ipynb        # Model prediction and evaluation
├── reputation_model/             # Reputation analysis model files
├── stance_model/                 # Stance detection model files
├── train.tsv, test.tsv, valid.tsv, train2.tsv, test2.tsv, val2.tsv  # Datasets
└── per_class_metrics.csv         # Model performance metrics
```

## Usage

**For Combined Model Analysis:**
Open `Combined_Model.ipynb` in Jupyter Notebook and run all cells. The notebook will load all 6 models and provide the `analyze_complete_article()` function for analyzing any article text.

**For RAG System:**
Open `RAG Practice.ipynb` to explore semantic retrieval and question-answering over article content using embeddings and similarity search.

## License

This project is developed for educational purposes as part of UC San Diego's Data Science curriculum.
