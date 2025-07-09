import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.models import train_and_score_model  # ML-powered SmartScore function

st.set_page_config(page_title="Broker SmartScore ML Dashboard", layout="wide")
st.title("🤖 Broker SmartScore – ML-Powered Dashboard")

uploaded_file = st.file_uploader("📤 Upload your broker submissions CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Raw Submission Data")
    st.dataframe(df.head())

    # Train ML model and generate scores
    broker_scores, eval_metrics = train_and_score_model(df)

    st.subheader("🎯 Broker SmartScores")
    score_range = st.slider("Filter by SmartScore:", 0, 100, (20, 90))
    filtered = broker_scores[broker_scores['SmartScore'].between(*score_range)]
    st.dataframe(filtered)

    st.download_button("📥 Download Filtered Scores", filtered.to_csv(index=False), "broker_scores.csv")

    # 📊 Evaluation metrics
    st.subheader("📊 Model Evaluation Metrics")
    st.markdown(f"""
    - **Accuracy**: `{eval_metrics['Accuracy']:.2f}`
    - **Precision**: `{eval_metrics['Precision']:.2f}`
    - **AUC Score**: `{eval_metrics['AUC']:.2f}`
    """)

    # 📈 Visuals
    st.subheader("📈 Visual Insights")

    col1, col2 = st.columns(2)

    with col1:
        top_brokers = broker_scores.sort_values(by="SmartScore", ascending=False).head(10)
        fig1, ax1 = plt.subplots()
        sns.barplot(data=top_brokers, x="SmartScore", y="broker_id", ax=ax1, palette="Blues_d")
        ax1.set_title("Top 10 Brokers by SmartScore")
        st.pyplot(fig1)

    with col2:
        rec_counts = broker_scores['Recommendation'].value_counts()
        fig2, ax2 = plt.subplots()
        ax2.pie(rec_counts, labels=rec_counts.index, autopct="%1.1f%%", startangle=90, colors=sns.color_palette("pastel"))
        ax2.set_title("Recommendation Breakdown")
        st.pyplot(fig2)

    fig3, ax3 = plt.subplots()
    sns.histplot(broker_scores['SmartScore'], bins=10, kde=True, ax=ax3, color="skyblue")
    ax3.set_title("Distribution of SmartScores")
    ax3.set_xlabel("SmartScore")
    st.pyplot(fig3)

else:
    st.info("👆 Upload a CSV file to begin.")
