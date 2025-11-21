import React, { useState } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import './App.css';

const API_URL = 'http://localhost:5001';

function App() {
  const [title, setTitle] = useState('');
  const [body, setBody] = useState('');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [agentReport, setAgentReport] = useState(null);
  const [error, setError] = useState(null);
  const [analysisType, setAnalysisType] = useState('basic'); // 'basic' or 'advanced'

  const handleBasicAnalysis = async () => {
    if (!body.trim()) {
      setError('Please enter article body');
      return;
    }

    setLoading(true);
    setError(null);
    setResults(null);
    setAgentReport(null);

    try {
      const response = await axios.post(`${API_URL}/analyze`, {
        title: title,
        body: body
      });

      if (response.data.success) {
        setResults(response.data.results);
      } else {
        setError(response.data.error || 'Analysis failed');
      }
    } catch (err) {
      setError(err.response?.data?.error || err.message || 'Failed to connect to server');
    } finally {
      setLoading(false);
    }
  };

  const handleAdvancedAnalysis = async () => {
    if (!body.trim()) {
      setError('Please enter article body');
      return;
    }

    setLoading(true);
    setError(null);
    setResults(null);
    setAgentReport(null);

    try {
      const response = await axios.post(`${API_URL}/analyze-with-agent`, {
        title: title,
        body: body
      });

      if (response.data.success) {
        setAgentReport(response.data.report);
      } else {
        setError(response.data.error || 'Analysis failed');
      }
    } catch (err) {
      setError(err.response?.data?.error || err.message || 'Failed to connect to server');
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (analysisType === 'basic') {
      handleBasicAnalysis();
    } else {
      handleAdvancedAnalysis();
    }
  };

  const loadExampleArticle = () => {
    setTitle("You thought Monday's internet outage was bad? Just wait");
    setBody(`Monday's Amazon Web Services outage — and the global disruption it caused — underscored just how reliant the internet has become on a small number of core infrastructure providers.

The ramifications of such outages could only get worse if artificial intelligence becomes as central to work and daily life as tech giants suggest it will in the coming years.

Monday's outage briefly blocked some people from scheduling doctor's appointments and accessing banking services. But what if an outage took down the AI tools that doctors were using to help diagnose patients, or that companies used to help facilitate financial transactions?

It may be a hypothetical scenario today, but the tech industry is promising a rapid shift toward AI "agents" doing more work on behalf of humans in the near future – and that could make businesses, schools, hospitals and financial institutions even more reliant on cloud-based services.`);
  };

  const clearForm = () => {
    setTitle('');
    setBody('');
    setResults(null);
    setAgentReport(null);
    setError(null);
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>📰 News Article Analyzer</h1>
        <p>Analyze news articles for credibility, sentiment, stance, and more</p>
      </header>

      <div className="container">
        <div className="input-section">
          <h2>Enter Article</h2>
          
          <div className="example-buttons">
            <button onClick={loadExampleArticle} className="btn-secondary">
              Load Example Article
            </button>
            <button onClick={clearForm} className="btn-secondary">
              Clear Form
            </button>
          </div>

          <form onSubmit={handleSubmit}>
            <div className="form-group">
              <label htmlFor="title">Article Title (Optional)</label>
              <input
                type="text"
                id="title"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                placeholder="Enter article title..."
                className="input-field"
              />
            </div>

            <div className="form-group">
              <label htmlFor="body">Article Body *</label>
              <textarea
                id="body"
                value={body}
                onChange={(e) => setBody(e.target.value)}
                placeholder="Paste or type the article content here..."
                rows="15"
                className="input-field textarea"
                required
              />
            </div>

            <div className="form-group">
              <label>Analysis Type</label>
              <div className="radio-group">
                <label className="radio-label">
                  <input
                    type="radio"
                    value="basic"
                    checked={analysisType === 'basic'}
                    onChange={(e) => setAnalysisType(e.target.value)}
                  />
                  <span>Basic Analysis (6 Models Only)</span>
                  <small>Fast - Returns predictions from all 6 ML models</small>
                </label>
                <label className="radio-label">
                  <input
                    type="radio"
                    value="advanced"
                    checked={analysisType === 'advanced'}
                    onChange={(e) => setAnalysisType(e.target.value)}
                  />
                  <span>Advanced Analysis (Qwen3 Agent + RAG)</span>
                  <small>Slow - Includes LLM reasoning and fact-checking</small>
                </label>
              </div>
            </div>

            <button 
              type="submit" 
              className="btn-primary"
              disabled={loading}
            >
              {loading ? 'Analyzing...' : 'Analyze Article'}
            </button>
          </form>
        </div>

        {error && (
          <div className="error-box">
            <h3>❌ Error</h3>
            <p>{error}</p>
          </div>
        )}

        {loading && (
          <div className="loading-box">
            <div className="spinner"></div>
            <p>
              {analysisType === 'basic' 
                ? 'Analyzing article... This may take 10-20 seconds.' 
                : 'Running Advanced Analysis... This may take 1-3 minutes. Please be patient.'}
            </p>
            {analysisType === 'advanced' && (
              <p style={{marginTop: '10px', fontSize: '0.9em', color: '#666'}}>
                💡 Tip: If this takes too long, try Basic Analysis for faster results.
              </p>
            )}
          </div>
        )}

        {results && !loading && (
          <div className="results-section">
            <h2>📊 Analysis Results</h2>
            
            <div className="results-grid">
              {results.news_coverage && (
                <div className="result-card">
                  <h3>📌 News Coverage</h3>
                  <p className="result-value">{results.news_coverage.topic}</p>
                </div>
              )}

              {results.intent && (
                <div className="result-card">
                  <h3>🎯 Intent</h3>
                  <p className="result-value">{results.intent.primary_intent}</p>
                </div>
              )}

              {results.sensationalism && (
                <div className="result-card">
                  <h3>⚡ Sensationalism</h3>
                  <p className="result-value">{results.sensationalism.label}</p>
                </div>
              )}

              {results.sentiment && (
                <div className="result-card">
                  <h3>😊 Sentiment</h3>
                  <p className="result-value">{results.sentiment.sentiment}</p>
                </div>
              )}

              {results.reputation && (
                <div className="result-card">
                  <h3>⭐ Reputation</h3>
                  <p className="result-value">{results.reputation.level}</p>
                  <p className="result-model">{results.reputation.model}</p>
                </div>
              )}

              {results.stance && (
                <div className="result-card">
                  <h3>🗳️ Stance</h3>
                  <p className="result-value">{results.stance.stance}</p>
                  <p className="result-model">{results.stance.model}</p>
                </div>
              )}
            </div>
          </div>
        )}

        {agentReport && !loading && (
          <div className="agent-report-section">
            <h2>🤖 Advanced Analysis Report</h2>
            <div className="markdown-content">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {agentReport}
              </ReactMarkdown>
            </div>
          </div>
        )}
      </div>

      <footer className="App-footer">
        <p>DSC180A - GroupNull | UC San Diego</p>
      </footer>
    </div>
  );
}

export default App;

