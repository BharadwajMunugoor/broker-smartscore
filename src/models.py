import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_score, accuracy_score

def train_and_score_model(df, threshold=0.5):
    # Derive responsiveness_score
    df['responsiveness_score'] = (df['responsiveness_days'].max() - df['responsiveness_days']) / (
        df['responsiveness_days'].max() - df['responsiveness_days'].min()
    )

    # Group by broker
    broker_df = df.groupby('broker_id').agg({
        'completeness_score': 'mean',
        'appetite_alignment': 'mean',
        'responsiveness_score': 'mean',
        'quote_to_bind': 'mean'
    }).reset_index()

    # Label high performers if quote_to_bind >= threshold
    broker_df['high_performer'] = (broker_df['quote_to_bind'] >= threshold).astype(int)

    # Features and labels
    X = broker_df[['completeness_score', 'appetite_alignment', 'responsiveness_score']]
    y = broker_df['high_performer']

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y if y.nunique() > 1 else None
    )

    # Check class presence
    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise ValueError("Both training and test sets must contain at least two classes for classification.")

    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Predict probabilities
    y_proba = model.predict_proba(X_test)
    if y_proba.shape[1] == 1:
        y_pred = y_proba[:, 0]
    else:
        y_pred = y_proba[:, 1]

    # Evaluation
    evaluation = {
        "Accuracy": round(accuracy_score(y_test, y_pred.round()), 2),
        "Precision": round(precision_score(y_test, y_pred.round(), zero_division=0), 2),
        "AUC": round(roc_auc_score(y_test, y_pred), 2) if y_test.nunique() > 1 else None
    }

    # Feature importance
    importances = model.feature_importances_
    feature_df = pd.DataFrame({
        'feature': X.columns,
        'importance': importances
    }).sort_values(by='importance', ascending=False)

    # Predict scores for all brokers
    broker_df['StartScore'] = model.predict_proba(X_scaled)[:, 1] if model.n_classes_ > 1 else model.predict_proba(X_scaled)[:, 0]

    # Recommendation logic
    def recommend(score):
        if score >= 0.7:
            return "Yes"
        elif score >= 0.4:
            return "Neutral"
        else:
            return "No"

    broker_df['Recommendation'] = broker_df['StartScore'].apply(recommend)

    return broker_df, evaluation, feature_df
