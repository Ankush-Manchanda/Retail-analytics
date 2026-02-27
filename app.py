import streamlit as st
from detect import run_detection
from dashboard import run_dashboard

st.set_page_config(
    page_title="Retail Store Analytics",
    layout="wide"
)

st.sidebar.title("🛒 Retail Analytics System")
st.sidebar.info(
    "YOLOv8-based customer footfall\n"
    "and occupancy analytics"
)

st.title("🛍️ Retail Store Analytics Using Computer Vision")

tab1, tab2 = st.tabs(
    ["📸 Object Detection", "📊 Analytics Dashboard"]
)

with tab1:
    run_detection()

with tab2:
    run_dashboard()
