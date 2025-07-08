import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.smartscore import calculate_smart_scores

st.set_page_config(page_title="Broker SmartScore", layout="wide")
st.title("📊 Broker SmartScore Dashboard")

uploaded_file = st.file_uploader("Upload your broker submissions CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Raw Submission Data")
    st.dataframe(df.head())

    broker_scores = calculate_smart_scores(df)

    st.subheader("🎯 Broker Scores")
    score_range = st.slider("Filter by SmartScore:", 0, 100, (20, 90))
    filtered = broker_scores[broker_scores['SmartScore'].between(*score_range)]
    st.dataframe(filtered)

    # 📥 Download
    st.download_button("Download Filtered Scores as CSV", filtered.to_csv(index=False), "broker_scores.csv")

    # 📊 Visualization Section
    st.subheader("📈 Visual Insights")

    # Top 10 brokers by score
    top_brokers = broker_scores.sort_values(by="SmartScore", ascending=False).head(10)
    fig1, ax1 = plt.subplots()
    sns.barplot(data=top_brokers, x="SmartScore", y="broker_id", ax=ax1, palette="Blues_d")
    ax1.set_title("Top 10 Brokers by SmartScore")
    ax1.set_xlabel("SmartScore")
    ax1.set_ylabel("Broker ID")
    st.pyplot(fig1)

    # Recommendation breakdown pie chart
    rec_counts = broker_scores['Recommendation'].value_counts()
    fig2, ax2 = plt.subplots()
    ax2.pie(rec_counts, labels=rec_counts.index, autopct="%1.1f%%", startangle=90, colors=sns.color_palette("pastel"))
    ax2.set_title("Recommendation Breakdown")
    st.pyplot(fig2)

    # Histogram of SmartScores
    fig3, ax3 = plt.subplots()
    sns.histplot(broker_scores['SmartScore'], bins=10, kde=True, ax=ax3, color="skyblue")
    ax3.set_title("Distribution of SmartScores")
    ax3.set_xlabel("SmartScore")
    st.pyplot(fig3)

else:
    st.info("👆 Upload a CSV file to get started.")

