# 📰 News Analyzer Web Application

A full-stack web application that analyzes news articles using 6 machine learning models + Qwen3 LLM agent with RAG (Retrieval-Augmented Generation).

## 🎯 Features

### Basic Analysis (Fast)
Uses 6 trained ML models to predict:
1. **News Coverage** - Topic classification
2. **Intent** - Communication intent (inform, persuade, entertain, deceive)
3. **Sensationalism** - Sensational vs neutral content detection
4. **Sentiment** - Emotional sentiment (positive, negative, neutral)
5. **Reputation** - Reputation level (low, medium, high)
6. **Stance** - Political stance (support, deny, neutral)

### Advanced Analysis (Comprehensive)
Includes everything from Basic Analysis PLUS:
- **Qwen3 LLM Agent** - Advanced reasoning and interpretation
- **RAG System** - Fact-checking against external news database
- **3 Additional Factors:**
  - Title vs. Body relationship
  - Context Veracity (with fact verification)
  - Location/Geography analysis

## 🏗️ Architecture

```
news-analyzer-app/
├── backend/
│   ├── app.py              # Flask API server
│   ├── models.py           # All 6 models + Qwen3 agent
│   └── requirements.txt    # Python dependencies
│
└── frontend/
    ├── src/
    │   ├── App.js          # Main React component
    │   ├── App.css         # Styles
    │   ├── index.js        # React entry point
    │   └── index.css       # Global styles
    ├── public/
    │   └── index.html      # HTML template
    └── package.json        # Node dependencies
```

## 📋 Prerequisites

### Backend
- Python 3.8+
- pip

### Frontend
- Node.js 16+
- npm or yarn

### Required Files (from parent directory)
The backend expects these files in the parent directory:
- `train2.tsv`, `valid.tsv` - Training data
- `reputation_model/` - Reputation model files
- `stance_model/` - Stance model files
- `scraper/rag_db/` - ChromaDB database for RAG

## 🚀 Installation

### 1. Backend Setup

Navigate to the backend directory:
```bash
cd backend
```

Install Python dependencies:
```bash
pip install -r requirements.txt
```

**Important:** Make sure the following files exist in the parent directory:
- `../../train2.tsv`
- `../../valid.tsv`
- `../../reputation_model/`
- `../../stance_model/`
- `../../scraper/rag_db/`

### 2. Frontend Setup

Navigate to the frontend directory:
```bash
cd frontend
```

Install Node.js dependencies:
```bash
npm install
```

## 🎮 Running the Application

### Start Backend Server

In the `backend/` directory:
```bash
python app.py
```

The backend API will start on `http://localhost:5001`

**Note:** The first time you run this, it will take 1-2 minutes to load all models. You'll see progress messages in the console.

### Start Frontend Server

In a **new terminal**, navigate to `frontend/` directory:
```bash
npm start
```

The React app will start on `http://localhost:3000` and automatically open in your browser.

## 💻 Usage

1. **Open the web app** at `http://localhost:3000`

2. **Enter an article:**
   - Paste the article title (optional)
   - Paste the article body (required)
   - Or click "Load Example Article" to test with sample content

3. **Choose analysis type:**
   - **Basic Analysis**: Fast prediction from 6 ML models (~5-10 seconds)
   - **Advanced Analysis**: Full Qwen3 agent analysis with RAG (~30-60 seconds)

4. **Click "Analyze Article"**

5. **View results:**
   - Basic: See predictions from all 6 models in card format
   - Advanced: See comprehensive markdown report with reasoning and confidence scores

## 🔧 API Endpoints

### `GET /health`
Health check endpoint
- Returns: `{"status": "healthy"}`

### `POST /analyze`
Basic analysis with 6 models
- Request body:
  ```json
  {
    "title": "Article title (optional)",
    "body": "Article content (required)"
  }
  ```
- Returns: Model predictions

### `POST /analyze-with-agent`
Advanced analysis with Qwen3 + RAG
- Request body:
  ```json
  {
    "title": "Article title (optional)",
    "body": "Article content (required)"
  }
  ```
- Returns: Full markdown analysis report

## 🎨 Features

### Frontend
- ✨ Beautiful gradient UI design
- 📱 Responsive layout (mobile-friendly)
- 🔄 Loading states with spinner animation
- ❌ Error handling and display
- 📝 Example article loader
- 🎯 Two analysis modes (basic/advanced)
- 📊 Card-based results display
- 📄 Markdown rendering for advanced reports

### Backend
- 🚀 Flask REST API
- 🤖 6 trained ML models integrated
- 🧠 Qwen3 LLM agent with tool calling
- 🔍 RAG system with ChromaDB
- 🔄 Sentence-level voting for long articles
- ⚡ Efficient model loading and caching

## 🐛 Troubleshooting

### Backend won't start
- Check that all required data files exist in parent directory
- Make sure all Python packages are installed: `pip install -r requirements.txt`
- Download NLTK data if missing (will auto-download on first run)
- If port 5001 is in use, change port in `app.py` (last line)

### Frontend can't connect to backend
- Verify backend is running on port 5001
- Check CORS is enabled (should be by default)
- Look for errors in browser console (F12)
- Make sure API_URL in App.js matches backend port

### Models not loading
- Ensure `reputation_model/` and `stance_model/` directories exist
- Check that model files (`model.safetensors`, config files) are present
- Review backend console for specific error messages

### Advanced analysis times out
- Qwen3 API might be slow - this is normal
- Check internet connection (required for Qwen3 API)
- Try basic analysis first to verify models work

## 📊 Performance

- **Basic Analysis:** 5-15 seconds (depending on article length)
- **Advanced Analysis:** 30-90 seconds (includes LLM reasoning + RAG lookups)
- **Memory Usage:** ~2-3GB RAM for all models loaded
- **First Load:** Takes 1-2 minutes to initialize all models

## 🔐 Security Notes

- The Qwen3 API key is currently hardcoded (for development only)
- In production, move API keys to environment variables
- Consider adding rate limiting for API endpoints
- Validate and sanitize all user inputs

## 📝 License

This project is developed for educational purposes as part of UC San Diego's DSC180A course.

## 👥 Credits

**DSC180A - GroupNull**  
UC San Diego  
Data Science Capstone Project

## 🚀 Future Enhancements

- [ ] Add user authentication
- [ ] Save analysis history
- [ ] Export results to PDF
- [ ] Compare multiple articles
- [ ] Batch analysis support
- [ ] Real-time streaming for long analyses
- [ ] Model performance metrics dashboard
- [ ] Article scraping from URL

