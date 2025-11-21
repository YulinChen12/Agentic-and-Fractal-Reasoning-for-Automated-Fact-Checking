# 🚀 Quick Start Guide

## Step-by-Step Instructions

### 1️⃣ Install Backend Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2️⃣ Install Frontend Dependencies

```bash
cd ../frontend
npm install
```

### 3️⃣ Start Backend Server

Open a terminal and run:
```bash
cd backend
python app.py
```

✅ Wait for "All models loaded successfully!" message
🌐 Backend running at: `http://localhost:5001`

### 4️⃣ Start Frontend Server

Open a **NEW terminal** and run:
```bash
cd frontend
npm start
```

✅ Browser will automatically open to `http://localhost:3000`

## 🎉 You're Ready!

1. The web app should now be open in your browser
2. Click "Load Example Article" to test
3. Choose "Basic Analysis" for quick test
4. Click "Analyze Article"

## ⚠️ Troubleshooting

### Backend Error: "No such file or directory: '../../train2.tsv'"
**Solution:** The backend needs to be run from the `backend/` directory, and training data must exist in the parent directory.

Make sure your folder structure looks like this:
```
DSC180A-GroupNull/
├── train2.tsv
├── valid.tsv
├── reputation_model/
├── stance_model/
├── scraper/rag_db/
└── news-analyzer-app/
    ├── backend/
    └── frontend/
```

### Frontend Error: "Cannot connect to server"
**Solution:** Make sure backend is running on port 5000 first.

### Port Already in Use
**Backend (5000):** Stop other Flask apps or change port in `app.py`
**Frontend (3000):** React will offer to use a different port automatically

## 📱 First Time Usage

1. **Load Example:** Click "Load Example Article" button
2. **Select Mode:** Choose "Basic Analysis" (faster)
3. **Analyze:** Click "Analyze Article" button
4. **Wait:** Basic analysis takes 5-15 seconds
5. **View Results:** See predictions from all 6 models

## 🧪 Testing Advanced Analysis

1. Use the loaded example article
2. Select "Advanced Analysis (Qwen3 Agent + RAG)"
3. Click "Analyze Article"
4. Wait 30-90 seconds for comprehensive report
5. View detailed markdown analysis with reasoning

## 💡 Tips

- ✅ Start with Basic Analysis to verify models work
- ✅ Advanced Analysis is slower but more comprehensive
- ✅ Keep both terminals open while using the app
- ✅ Check terminal output for debugging info
- ✅ First load takes 1-2 minutes to initialize models

## Need Help?

Check the full [README.md](README.md) for detailed documentation.

