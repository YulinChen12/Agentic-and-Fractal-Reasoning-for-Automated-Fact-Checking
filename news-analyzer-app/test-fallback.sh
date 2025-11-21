#!/bin/bash

echo "=================================================="
echo "🧪 Testing Advanced Analysis with Fallback"
echo "=================================================="
echo ""

TITLE="Test Article About Technology"
BODY="This is a test article discussing artificial intelligence and its impact on society. The technology sector continues to evolve rapidly with new innovations emerging daily."

echo "📝 Test Article:"
echo "   Title: $TITLE"
echo "   Length: $(echo "$BODY" | wc -w) words"
echo ""
echo "🚀 Sending to Advanced Analysis endpoint..."
echo "   (This will use fallback if Qwen3 times out)"
echo ""

START=$(date +%s)

RESPONSE=$(curl -s -X POST http://localhost:5001/analyze-with-agent \
  -H "Content-Type: application/json" \
  -d "{\"title\": \"$TITLE\", \"body\": \"$BODY\"}" \
  --max-time 200)

END=$(date +%s)
DURATION=$((END - START))

echo "⏱️  Response time: ${DURATION} seconds"
echo ""

if echo "$RESPONSE" | grep -q "success"; then
    echo "✅ Request successful!"
    echo ""
    
    # Check if it's a fallback response
    if echo "$RESPONSE" | grep -q "Advanced LLM analysis is currently unavailable"; then
        echo "🔄 FALLBACK MODE ACTIVATED"
        echo "   → Qwen3 timed out or failed"
        echo "   → Returning 6 model predictions instead"
        echo ""
    elif echo "$RESPONSE" | grep -q "Phase 2: Qualitative Analysis"; then
        echo "🎉 FULL ANALYSIS COMPLETED"
        echo "   → Qwen3 API responded successfully"
        echo "   → Got Phase 1 + Phase 2"
        echo ""
    fi
    
    echo "📊 Report Preview:"
    echo "$RESPONSE" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('report', 'No report')[:500])" 2>/dev/null || echo "$RESPONSE" | head -c 500
    echo ""
    echo "..."
    echo ""
else
    echo "❌ Request failed"
    echo ""
    echo "Response:"
    echo "$RESPONSE"
fi

echo "=================================================="
echo ""
echo "💡 Interpretation:"
echo ""
if echo "$RESPONSE" | grep -q "Advanced LLM analysis is currently unavailable"; then
    echo "   The fallback system is working correctly!"
    echo "   - You still got 6 model predictions"
    echo "   - No error message shown to user"
    echo "   - Suggestions provided to try again"
    echo ""
    echo "   ✅ This is the expected behavior when Qwen3 is slow"
elif echo "$RESPONSE" | grep -q "Phase 2: Qualitative Analysis"; then
    echo "   Advanced Analysis completed successfully!"
    echo "   - Qwen3 API is currently responsive"
    echo "   - Full LLM analysis available"
    echo ""
    echo "   ✅ Try using the web interface for best experience"
else
    echo "   Unexpected response. Check backend logs."
fi

echo "=================================================="

