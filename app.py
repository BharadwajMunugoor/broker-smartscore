import streamlit as st
import pandas as pd
from src.smartscore import calculate_smart_scores

st.title("Broker SmartScore Dashboard")

uploaded_file = st.file_uploader("Upload your broker submissions CSV", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Raw Submission Data")
    st.dataframe(df.head())

    broker_scores = calculate_smart_scores(df)

    st.subheader("Broker Scores with SmartScore")
    score_range = st.slider("Filter by SmartScore:", 0, 100, (20, 90))
    filtered = broker_scores[broker_scores['SmartScore'].between(*score_range)]
    st.dataframe(filtered)

    st.download_button("Download as CSV", filtered.to_csv(index=False), "broker_scores.csv")
else:
    st.info("Upload a CSV file to begin.")
