import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.models import train_and_score_model  # Updated function import

st.set_page_config(page_title="Broker SmartScore - ML-Powered Dashboard", layout="wide")
st.title("🤖 Broker SmartScore - ML-Powered Dashboard")

# Upload or fallback to default
uploaded_file = st.file_uploader("📁 Upload your broker submission CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded.")
else:
    st.warning("⚠ No file uploaded. Loading demo data from 'data/synthetic_broker_submissions.csv'.")
    df = pd.read_csv("data/synthetic_broker_submissions.csv")

# ✅ Validate required columns
required_cols = {"broker_id", "completeness_score", "appetite_alignment", "responsiveness_days", "quote_to_bind"}
if not required_cols.issubset(df.columns):
    st.error("❌ Uploaded file is missing required columns:")
    st.text(f"Expected columns: {sorted(required_cols)}")
    st.text(f"Found columns: {sorted(df.columns.tolist())}")
    st.stop()

# Show raw data
st.subheader("📝 Submission Data (Demo or Uploaded)")
st.dataframe(df.head())

# Run model and get scores
broker_scores, eval_metrics, feature_df = train_and_score_model(df)

# Filter and display SmartScores
st.subheader("🎯 Broker SmartScores")
score_range = st.slider("Filter by SmartScore:", 0, 100, (20, 90))
filtered = broker_scores[broker_scores['SmartScore'].between(*score_range)]
st.dataframe(filtered)

# Download option
st.download_button("📥 Download Filtered Scores as CSV", filtered.to_csv(index=False), "broker_scores.csv")

# Evaluation Metrics
st.subheader("📊 Model Evaluation Metrics")
st.metric("✅ Accuracy", f"{eval_metrics['Accuracy']:.2f}")
st.metric("🧠 Precision", f"{eval_metrics['Precision']:.2f}")
st.metric("📉 AUC Score", f"{eval_metrics['AUC']:.2f}")

# Feature importance
st.subheader("🔍 Feature Importance")
st.dataframe(feature_df)

# Visualizations
st.subheader("📈 Visual Insights")
col1, col2, col3 = st.columns(3)

with col1:
    top_brokers = broker_scores.sort_values(by="SmartScore", ascending=False).head(10)
    fig1, ax1 = plt.subplots()
    sns.barplot(data=top_brokers, x="SmartScore", y="broker_id", palette="Blues_d", ax=ax1)
    ax1.set_title("Top 10 Brokers by SmartScore")
    st.pyplot(fig1)

with col2:
    rec_counts = broker_scores['Recommendation'].value_counts()
    fig2, ax2 = plt.subplots()
    ax2.pie(rec_counts, labels=rec_counts.index, autopct="%1.1f%%", startangle=90, colors=sns.color_palette("pastel"))
    ax2.set_title("Recommendation Breakdown")
    st.pyplot(fig2)

with col3:
    fig3, ax3 = plt.subplots()
    sns.histplot(broker_scores['SmartScore'], bins=10, kde=True, ax=ax3, color="skyblue")
    ax3.set_title("Distribution of SmartScores")
    ax3.set_xlabel("SmartScore")
    st.pyplot(fig3)
