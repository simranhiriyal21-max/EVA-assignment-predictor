# inside app/app.py
import streamlit as st
import joblib
import os

@st.cache_resource
def load_artifacts():
    base_dir = os.path.dirname(__file__)  # this points to app/ when running app/app.py
    # If you run from repo root, adjust accordingly:
    # base_dir = os.path.join(os.path.dirname(__file__), '..')  # alternative
    model_dir = os.path.normpath(os.path.join(base_dir, '..', 'model'))  # moves up to repo root then to model/
    lgb_path = os.path.join(model_dir, 'model_lgb.joblib')
    tfidf_path = os.path.join(model_dir, 'tfidf_vectorizer.joblib')
    label_enc_path = os.path.join(model_dir, 'label_encoder.joblib')

    lgb = None
    tfv = None
    le = None
    if os.path.exists(lgb_path):
        lgb = joblib.load(lgb_path)
    if os.path.exists(tfidf_path):
        tfv = joblib.load(tfidf_path)
    if os.path.exists(label_enc_path):
        le = joblib.load(label_enc_path)

    return lgb, tfv, le

lgb_model, tfv, label_encoder = load_artifacts()

# rest of your streamlit app follows...
