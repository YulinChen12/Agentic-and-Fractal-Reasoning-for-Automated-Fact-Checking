#!/bin/bash

echo "=================================================="
echo "🚀 Starting News Analyzer Application"
echo "=================================================="
echo ""

# Check if we're in the right directory
if [ ! -d "backend" ] || [ ! -d "frontend" ]; then
    echo "❌ Error: Please run this script from the news-analyzer-app directory"
    exit 1
fi

# Start backend
echo "📊 Starting Backend Server..."
cd backend
python app.py &
BACKEND_PID=$!
cd ..

echo "✅ Backend starting (PID: $BACKEND_PID)"
echo "⏳ Loading models... (this takes 1-2 minutes on first run)"
echo ""

# Wait a bit before starting frontend
sleep 3

# Start frontend
echo "🎨 Starting Frontend Server..."
cd frontend
BROWSER=none npm start &
FRONTEND_PID=$!
cd ..

echo "✅ Frontend starting (PID: $FRONTEND_PID)"
echo ""
echo "=================================================="
echo "✨ Application Starting!"
echo "=================================================="
echo ""
echo "📍 Backend API:  http://localhost:5001"
echo "🌐 Frontend App: http://localhost:3000"
echo ""
echo "⏰ Please wait 1-2 minutes for all models to load"
echo "🌐 Open http://localhost:3000 in your browser"
echo ""
echo "To stop the servers, press Ctrl+C"
echo ""

# Keep script running
wait

