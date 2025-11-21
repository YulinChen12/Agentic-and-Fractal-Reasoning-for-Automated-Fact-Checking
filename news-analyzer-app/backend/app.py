from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os

# Add parent directory to path to import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import NewsAnalyzer

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Initialize the analyzer (loads all models)
print("Loading all models... This may take a minute...")
analyzer = NewsAnalyzer()
print("All models loaded successfully!")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "message": "News Analyzer API is running"})

@app.route('/analyze', methods=['POST'])
def analyze_article():
    """
    Main endpoint to analyze an article
    Expects JSON: { "title": "...", "body": "..." }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        article_title = data.get('title', '')
        article_body = data.get('body', '')
        
        if not article_body:
            return jsonify({"error": "Article body is required"}), 400
        
        # Run analysis
        print(f"Analyzing article: {article_title[:50]}...")
        results = analyzer.analyze_complete_article(article_title, article_body)
        
        return jsonify({
            "success": True,
            "results": results
        })
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/analyze-with-agent', methods=['POST'])
def analyze_with_agent():
    """
    Endpoint for full Qwen3 agent analysis with RAG
    Expects JSON: { "title": "...", "body": "..." }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        article_title = data.get('title', '')
        article_body = data.get('body', '')
        
        if not article_body:
            return jsonify({"error": "Article body is required"}), 400
        
        # Run full agent analysis with RAG
        print(f"Running Qwen3 agent analysis for: {article_title[:50]}...")
        report = analyzer.analyze_with_agent(article_title, article_body)
        
        return jsonify({
            "success": True,
            "report": report
        })
        
    except Exception as e:
        print(f"Error during agent analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 News Analyzer API Server Starting...")
    print("="*50 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5001)

