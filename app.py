
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, render_template, request, jsonify
from model.sentiment_analyzer import SentimentAnalyzer
from model.context_analyzer import ContextAnalyzer

app = Flask(__name__)

print("Loading models — please wait...")
_sentiment = SentimentAnalyzer()
_context = ContextAnalyzer()
print("Ready!")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json(force=True)
        text = (data.get('text') or '').strip()
        if not text:
            return jsonify({'error': 'No text provided'}), 400

        sentiment = _sentiment.analyze(text[:512])
        context = _context.analyze(text)
        return jsonify({'sentiment': sentiment, 'context': context})
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model': 'DistilBERT SST-2 + VADER'})


if __name__ == '__main__':
    from config import PORT, DEBUG
    app.run(debug=DEBUG, host='0.0.0.0', port=PORT)
