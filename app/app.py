import streamlit as st
import requests

# Page settings
st.set_page_config(page_title="EVA — AI Ticket Classifier", layout="centered")

st.title("🎫 EVA — AI-Driven Ticket Classifier")
st.write("Enter an IT ticket description below to predict which assignment group should handle it.")

# Input field
text = st.text_area("📝 Ticket Description", height=200, placeholder="Example: VPN not connecting from home network")

# API endpoint (replace with your deployed API URL later)
API_URL = "https://your-api-url.onrender.com/predict"

if st.button("🔍 Predict Assignment"):
    if not text.strip():
        st.warning("Please enter a valid ticket description.")
    else:
        try:
            response = requests.post(API_URL, json={"text": text})
            if response.status_code == 200:
                result = response.json()
                st.success(f"Predicted Assignment Group: **{result.get('prediction')}**")
                st.subheader("Model Output Probabilities")
                st.json(result.get("probabilities"))
            else:
                st.error(f"API returned an error: {response.status_code}")
        except Exception as e:
            st.error(f"Failed to connect to API: {e}")

st.caption("Built as part of M.Tech Project — AI-Driven Automated IT Ticket Classification and Assignment using NLP")
