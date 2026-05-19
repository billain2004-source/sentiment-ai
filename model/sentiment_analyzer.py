import torch
import nltk
from transformers import pipeline
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from config import SENTIMENT_MODEL, NEUTRAL_THRESHOLD, MAX_TEXT_LENGTH

nltk.download('vader_lexicon', quiet=True)


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
        bert_label = bert_result['label']
        bert_score = bert_result['score']

        vader_scores = self.vader.polarity_scores(text)
        compound = vader_scores['compound']

        if bert_label == 'POSITIVE' and bert_score >= NEUTRAL_THRESHOLD:
            label = 'positive'
        elif bert_label == 'NEGATIVE' and bert_score >= NEUTRAL_THRESHOLD:
            label = 'negative'
        else:
            label = 'neutral'

        intensity = self._intensity(compound)
        word_scores = self._word_scores(text)

        return {
            'label': label,
            'confidence': round(bert_score, 4),
            'compound': round(compound, 4),
            'intensity': intensity,
            'breakdown': {
                'positive': round(vader_scores['pos'] * 100, 1),
                'negative': round(vader_scores['neg'] * 100, 1),
                'neutral': round(vader_scores['neu'] * 100, 1),
            },
            'word_scores': word_scores,
        }

    def _intensity(self, compound: float) -> str:
        if compound >= 0.75:
            return 'Very Positive'
        if compound >= 0.4:
            return 'Positive'
        if compound >= 0.1:
            return 'Slightly Positive'
        if compound <= -0.75:
            return 'Very Negative'
        if compound <= -0.4:
            return 'Negative'
        if compound <= -0.1:
            return 'Slightly Negative'
        return 'Neutral'

    def _word_scores(self, text: str) -> list:
        result = []
        for word in text.split():
            clean = ''.join(c for c in word if c.isalpha())
            if not clean:
                result.append({'word': word, 'sentiment': 'neutral', 'score': 0})
                continue
            score = self.vader.polarity_scores(clean)['compound']
            if score > 0.1:
                sentiment = 'positive'
            elif score < -0.1:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            result.append({'word': word, 'sentiment': sentiment, 'score': round(score, 3)})
        return result
