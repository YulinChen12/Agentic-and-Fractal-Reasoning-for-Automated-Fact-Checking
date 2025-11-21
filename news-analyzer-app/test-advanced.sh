#!/bin/bash

echo "🧪 Testing Advanced Analysis Endpoint"
echo "======================================"
echo ""

# Test article
TITLE="Test Article"
BODY="This is a test article about technology and artificial intelligence. It discusses the impact of AI on society."

# Make API call
echo "Sending request to http://localhost:5001/analyze-with-agent..."
echo ""

RESPONSE=$(curl -s -X POST http://localhost:5001/analyze-with-agent \
  -H "Content-Type: application/json" \
  -d "{\"title\": \"$TITLE\", \"body\": \"$BODY\"}")

# Check response
if echo "$RESPONSE" | grep -q "success"; then
    echo "✅ Request successful!"
    echo ""
    echo "Response:"
    echo "$RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE"
else
    echo "❌ Request failed or returned error"
    echo ""
    echo "Response:"
    echo "$RESPONSE"
fi

echo ""
echo "======================================"

