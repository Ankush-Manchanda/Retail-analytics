import streamlit as st
import pandas as pd
import os

def run_dashboard():
    st.subheader("📊 Retail Store Analytics Dashboard")

    if st.button("🔄 Reset Analytics"):
        if os.path.exists("analytics/footfall.csv"):
            os.remove("analytics/footfall.csv")
            st.success("Analytics reset.")
            st.stop()

    path = "analytics/footfall.csv"
    if not os.path.exists(path):
        st.info("No analytics data available.")
        return

    df = pd.read_csv(path, parse_dates=["timestamp"])

    avg_occupancy = round(df["occupancy"].mean(), 2)
    peak_occupancy = df["occupancy"].max()
    peak_time = df.loc[df["occupancy"].idxmax(), "timestamp"]

    df["hour"] = df["timestamp"].dt.hour
    hourly_avg = df.groupby("hour")["occupancy"].mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("👥 Avg Occupancy", avg_occupancy)
    col2.metric("🔥 Peak Occupancy", peak_occupancy)
    col3.metric("⏰ Peak Time", peak_time.strftime("%H:%M:%S"))

    st.subheader("📈 Occupancy Over Time")
    st.line_chart(df.set_index("timestamp")["occupancy"])

    st.subheader("⏱️ Average Occupancy per Hour")
    st.bar_chart(hourly_avg)

    st.subheader("📋 Raw Data")
    st.dataframe(df)

    st.download_button(
        "📥 Download CSV",
        df.to_csv(index=False),
        "footfall.csv",
        "text/csv"
    )
