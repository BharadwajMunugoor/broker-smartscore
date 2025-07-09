import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_score, accuracy_score

def train_and_score_model(df, thresholds=(70, 40)):
    """
    Trains a Random Forest model and returns broker-level SmartScores and recommendations.
    :param df: DataFrame of submission-level data
    :param thresholds: Tuple of (prioritize_cutoff, watchlist_cutoff)
    :return: broker_scores DataFrame, evaluation_metrics dict
    """

    # Derived responsiveness score
    df['responsiveness_score'] = 1 / (1 + df['responsiveness_days'])

    # Broker-level metrics
    broker_df = df.groupby('broker_id').agg({
        'completeness_score': 'mean',
        'appetite_alignment': 'mean',
        'responsiveness_score': 'mean',
        'quote_to_bind': 'mean'
    }).reset_index()

    # Target: high performers if quote_to_bind rate >= 0.5
    broker_df['high_performer'] = (broker_df['quote_to_bind'] >= 0.5).astype(int)

    # ML features and labels
    X = broker_df[['completeness_score', 'appetite_alignment', 'responsiveness_score']]
    y = broker_df['high_performer']

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    evaluation = {
        "AUC": roc_auc_score(y_test, y_proba),
        "Precision": precision_score(y_test, y_pred),
        "Accuracy": accuracy_score(y_test, y_pred)
    }

    # Predict SmartScores on all brokers
    broker_df['SmartScore'] = model.predict_proba(X_scaled)[:, 1] * 100

    def recommend(score):
        if score >= thresholds[0]:
            return "Prioritize"
        elif score >= thresholds[1]:
            return "Neutral"
        else:
            return "Monitor"

    broker_df['SmartScore'] = broker_df['SmartScore'].round(2)
    broker_df['Recommendation'] = broker_df['SmartScore'].apply(recommend)

    return broker_df, evaluation
