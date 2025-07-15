import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_score, accuracy_score

def train_and_score_model(df, threshold=0.6):
    # Derive responsiveness score
    df['responsiveness_score'] = 1 / (1 + df['responsiveness_days'])

    # Broker-level metrics
    broker_df = df.groupby('broker_id').agg({
        'completeness_score': 'mean',
        'appetite_alignment': 'mean',
        'responsiveness_score': 'mean',
        'quote_to_bind': 'mean'
    }).reset_index()

    # Target label
    broker_df['high_performer'] = (broker_df['quote_to_bind'] > threshold).astype(int)

    # Features and labels
    X = broker_df[['completeness_score', 'appetite_alignment', 'responsiveness_score']]
    y = broker_df['high_performer']

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )

    # Validation for class diversity
    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise ValueError("Both training and test sets must contain at least two classes.")

    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    # Handle single-column probas
    if y_prob.shape[1] == 2:
        pos_probs = y_prob[:, 1]
    else:
        pos_probs = np.zeros(len(y_prob))

    evaluation = {
        "Accuracy": round(accuracy_score(y_test, y_pred), 2),
        "Precision": round(precision_score(y_test, y_pred), 2),
        "AUC": round(roc_auc_score(y_test, pos_probs), 2)
    }

    # Feature importances
    importances = model.feature_importances_
    feature_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    # Predict scores for all
    broker_df['SmartScore'] = model.predict_proba(X_scaled)[:, 1] * 100

    # Recommendations
    def recommend(score):
        if score >= 70:
            return "Prioritize"
        elif score >= 40:
            return "Neutral"
        else:
            return "Monitor"

    broker_df['Recommendation'] = broker_df['SmartScore'].apply(recommend)

    return broker_df, evaluation, feature_df
