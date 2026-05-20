import torch
import nltk
from transformers import pipeline
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from config import SENTIMENT_MODEL, MAX_TEXT_LENGTH

nltk.download('vader_lexicon', quiet=True)

LABEL_MAP = {'positive': 'positive', 'neutral': 'neutral', 'negative': 'negative'}


class SentimentAnalyzer:
    def __init__(self):
        device = 0 if torch.cuda.is_available() else -1
        self.bert = pipeline(
            'sentiment-analysis',
            model=SENTIMENT_MODEL,
            device=device,
            truncation=True,
            max_length=MAX_TEXT_LENGTH,
        )
        self.vader = SentimentIntensityAnalyzer()

    def analyze(self, text: str) -> dict:
        bert_result = self.bert(text[:MAX_TEXT_LENGTH])[0]
        label = bert_result['label'].lower()        # 'positive' | 'neutral' | 'negative'
        confidence = bert_result['score']

        vader_scores = self.vader.polarity_scores(text)
        compound = vader_scores['compound']

        return {
            'label': label,
            'confidence': round(confidence, 4),
            'compound': round(compound, 4),
            'intensity': self._intensity(compound),
            'breakdown': {
                'positive': round(vader_scores['pos'] * 100, 1),
                'negative': round(vader_scores['neg'] * 100, 1),
                'neutral':  round(vader_scores['neu'] * 100, 1),
            },
            'word_scores': self._word_scores(text),
        }

    def _intensity(self, compound: float) -> str:
        if compound >= 0.75:   return 'Very Positive'
        if compound >= 0.4:    return 'Positive'
        if compound >= 0.1:    return 'Slightly Positive'
        if compound <= -0.75:  return 'Very Negative'
        if compound <= -0.4:   return 'Negative'
        if compound <= -0.1:   return 'Slightly Negative'
        return 'Neutral'

    def _word_scores(self, text: str) -> list:
        result = []
        for word in text.split():
            clean = ''.join(c for c in word if c.isalpha())
            if not clean:
                result.append({'word': word, 'sentiment': 'neutral', 'score': 0})
                continue
            score = self.vader.polarity_scores(clean)['compound']
            sentiment = 'positive' if score > 0.1 else 'negative' if score < -0.1 else 'neutral'
            result.append({'word': word, 'sentiment': sentiment, 'score': round(score, 3)})
        return result
