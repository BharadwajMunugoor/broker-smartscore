import pandas as pd

def calculate_smart_scores(df, weights=None):
    if weights is None:
        weights = {
            'completeness_score': 0.25,
            'appetite_alignment': 0.25,
            'responsiveness_score': 0.20,
            'quote_to_bind': 0.30
        }

    df['responsiveness_score'] = 1 / (1 + df['responsiveness_days'])

    metrics = df.groupby('broker_id').agg({
        'completeness_score': 'mean',
        'appetite_alignment': 'mean',
        'responsiveness_score': 'mean',
        'quote_to_bind': 'mean'
    }).reset_index()

    metrics['SmartScore'] = (
        weights['completeness_score'] * metrics['completeness_score'] +
        weights['appetite_alignment'] * metrics['appetite_alignment'] +
        weights['responsiveness_score'] * metrics['responsiveness_score'] +
        weights['quote_to_bind'] * metrics['quote_to_bind']
    ) * 100

    def classify(score):
        if score >= 70:
            return "Prioritize"
        elif score >= 40:
            return "Neutral"
        else:
            return "Watchlist"

    metrics['SmartScore'] = metrics['SmartScore'].round(2)
    metrics['Recommendation'] = metrics['SmartScore'].apply(classify)

    return metrics
