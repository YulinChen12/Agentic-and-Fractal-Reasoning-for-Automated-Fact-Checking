#!/bin/bash

echo "=================================================="
echo "🔍 News Analyzer Status Check"
echo "=================================================="
echo ""

# Check backend process
BACKEND_PID=$(ps aux | grep "[p]ython app.py" | awk '{print $2}' | head -1)
if [ -n "$BACKEND_PID" ]; then
    echo "✅ Backend Process: Running (PID: $BACKEND_PID)"
    CPU=$(ps aux | grep "[p]ython app.py" | awk '{print $3}' | head -1)
    echo "   CPU Usage: $CPU%"
else
    echo "❌ Backend Process: Not running"
fi

echo ""

# Check if port 5001 is listening
if lsof -i :5001 >/dev/null 2>&1; then
    echo "✅ Backend API (Port 5001): Listening"
    
    # Try to connect
    HEALTH=$(curl -s http://localhost:5001/health 2>&1)
    if [ $? -eq 0 ]; then
        echo "✅ Backend Health Check: $HEALTH"
    else
        echo "⏳ Backend Health Check: Still initializing..."
    fi
else
    echo "⏳ Backend API (Port 5001): Not ready yet (models loading...)"
fi

echo ""

# Check frontend
if lsof -i :3000 >/dev/null 2>&1; then
    echo "✅ Frontend (Port 3000): Running"
else
    echo "❌ Frontend (Port 3000): Not running"
fi

echo ""
echo "=================================================="
echo "📝 Notes:"
echo "   - Backend model loading takes 2-3 minutes"
echo "   - Check again in a minute if backend not ready"
echo "   - Open http://localhost:3000 when both ready"
echo "=================================================="

