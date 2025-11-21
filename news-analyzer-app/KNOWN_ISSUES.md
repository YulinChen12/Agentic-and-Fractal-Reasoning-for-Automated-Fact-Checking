# Known Issues and Workarounds

## ⏱️ Advanced Analysis Timeout Issue

### Problem
Advanced Analysis (with Qwen3 LLM) may timeout after 3 minutes due to:
- High load on the Qwen3 API server (`ellm.nrp-nautilus.io`)
- Network latency
- Large article size requiring extensive processing

### Symptoms
- Loading spinner runs for 1-3 minutes
- Returns fallback report with message: "Advanced LLM analysis is currently unavailable due to API timeout"
- Backend logs show: `Read timed out. (read timeout=180)`

### Current Solutions

#### ✅ Automatic Fallback (Implemented)
When Qwen3 times out, the system automatically:
1. Returns the 6 machine learning model predictions
2. Generates a simplified analysis report
3. Provides recommendations to try again later

#### ✅ What You Still Get
Even with timeout, you still receive:
- ✅ All 6 model predictions (Topic, Intent, Sensationalism, Sentiment, Reputation, Stance)
- ✅ Sentence-level analysis with majority voting
- ✅ Accurate classification results
- ❌ LLM reasoning and detailed explanations (not available)
- ❌ RAG-based fact verification (not available)

### Recommended Workarounds

1. **Use Basic Analysis** (Recommended)
   - ✅ Fast: 5-15 seconds
   - ✅ Reliable: No API dependency
   - ✅ Accurate: Same 6 models
   - Perfect for most use cases

2. **Retry During Off-Peak Hours**
   - Try early morning or late evening
   - Weekends may have less load

3. **Shorter Articles**
   - Qwen3 works better with shorter texts (< 500 words)
   - Consider summarizing very long articles first

4. **Wait and Retry**
   - The fallback report will suggest trying again
   - API performance varies throughout the day

### Technical Details

**Current Timeout Settings:**
- Connection timeout: 180 seconds (3 minutes)
- Read timeout: 180 seconds (3 minutes)

**Why Not Increase Further?**
- Web browsers have their own timeouts
- User experience degrades with very long waits
- The fallback provides value immediately

**API Endpoint:**
- Base URL: `https://ellm.nrp-nautilus.io/v1`
- Model: `qwen3`
- Status: External service (not under our control)

### Future Improvements

Potential solutions being considered:
1. ⏳ Add streaming support for progress updates
2. 📊 Implement local LLM fallback (smaller model)
3. 💾 Cache common analyses
4. 🔄 Add queue system for batch processing
5. ⚡ Use async processing with webhooks

### Error Messages You Might See

**Timeout:**
```
⏱️ Analysis timed out. The API server is responding slowly.
```

**Connection Error:**
```
🌐 Connection error: Unable to reach Qwen3 API server.
```

**Fallback Report:**
```
*Note: Advanced LLM analysis is currently unavailable due to API timeout.
The predictions above are from our 6 trained machine learning models.*
```

### How to Report Issues

If you experience consistent timeouts:
1. Note the time of day
2. Check your internet speed
3. Try Basic Analysis to verify backend is working
4. Report with article length (word count)

### Bottom Line

**Advanced Analysis is still valuable when it works**, but:
- ✅ **Basic Analysis is more reliable** for production use
- ✅ **Fallback mode ensures you always get results**
- ✅ **6 model predictions are still highly accurate**

**Use Basic Analysis for:**
- Production workflows
- Batch processing
- Time-sensitive analysis

**Use Advanced Analysis for:**
- Research and exploration
- When you need detailed reasoning
- When you have time to wait
- When API is responsive (test with short article first)

---

*Last updated: 2025-11-20*

