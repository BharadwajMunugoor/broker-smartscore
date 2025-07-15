import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_score, accuracy_score

def train_and_score_model(df, threshold=0.4):
    df = df.copy()
    
    # Derive responsiveness score
    df['responsiveness_score'] = 1 / (1 + df['responsiveness_days'])

    # Broker-level aggregation
    broker_df = df.groupby("broker_id").agg({
        'completeness_score': 'mean',
        'appetite_alignment': 'mean',
        'responsiveness_score': 'mean',
        'quote_to_bind': 'mean'
    }).reset_index()

    # Label as high performers
    broker_df["high_performer"] = (broker_df["quote_to_bind"] > threshold).astype(int)

    # Features and labels
    X = broker_df[["completeness_score", "appetite_alignment", "responsiveness_score"]]
    y = broker_df["high_performer"]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )

    # Double check for binary class presence in both sets
    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise ValueError("Both training and test sets must contain at least two classes.")

    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Predict safely
    y_prob_all = model.predict_proba(X_test)
    if y_prob_all.shape[1] < 2:
        raise ValueError("Model prediction failed: possibly only one class was found during training.")
    y_prob = y_prob_all[:, 1]

    # Evaluation metrics
    y_pred = model.predict(X_test)
    evaluation = {
        "Accuracy": round(accuracy_score(y_test, y_pred), 2),
        "Precision": round(precision_score(y_test, y_pred), 2),
        "AUC": round(roc_auc_score(y_test, y_prob), 2)
    }

    # Feature importance
    importances = model.feature_importances_
    feature_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    # Predict SmartScore for all brokers
    broker_df["SmartScore"] = model.predict_proba(X_scaled)[:, 1] * 100

    # Recommendation logic
    def recommend(score):
        if score >= 70:
            return "Partner"
        elif score >= 40:
            return "Neutral"
        else:
            return "Avoid"

    broker_df["Recommendation"] = broker_df["SmartScore"].apply(recommend)

    return broker_df, evaluation, feature_df
