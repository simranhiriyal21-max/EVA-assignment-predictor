# app/app.py
import streamlit as st
import joblib
import os
import traceback
import json
from pathlib import Path
import numpy as np

# Optional model libraries used for type-checking/prediction behavior
try:
    import xgboost as xgb
except Exception:
    xgb = None

try:
    import lightgbm as lgb
except Exception:
    lgb = None

# Page config
st.set_page_config(page_title="EVA Assignment Predictor", layout="wide")
st.title("🎯 EVA Assignment Predictor")
st.markdown("**Starting app...**")

# -------- Config --------
# Token for minimal protection (change this for your demo or remove)
API_TOKEN = "secret123"

# -------- Helpers --------
def repo_root_from_app():
    """
    Return repo root path given this file lives in app/ folder.
    """
    return os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))

@st.cache_resource
def load_artifacts():
    """
    Load model, vectorizer, optional label encoder and metadata.
    Returns (model, vectorizer, label_encoder_or_None, metadata_dict_or_None, error_message_or_None)
    """
    try:
        root = repo_root_from_app()
        model_dir = os.path.join(root, "model")
        assets_dir = os.path.join(root, "assets")

        # Expected filenames (adapt if yours differ)
        lgb_path = os.path.join(model_dir, "model_lgb.joblib")
        xgb_path = os.path.join(model_dir, "model_xgb.joblib")  # optional
        tfidf_path = os.path.join(model_dir, "tfidf_vectorizer.joblib")
        label_enc_path = os.path.join(model_dir, "label_encoder.joblib")
        metadata_path = os.path.join(model_dir, "model_metadata.json")
        roc_path = os.path.join(assets_dir, "roc.png")

        missing = []
        if not os.path.exists(tfidf_path):
            missing.append(tfidf_path)
        # prefer LightGBM model filename, but accept xgb or other joblib
        model_path = None
        if os.path.exists(lgb_path):
            model_path = lgb_path
        elif os.path.exists(xgb_path):
            model_path = xgb_path
        else:
            # look for any joblib in model dir (fallback)
            for f in os.listdir(model_dir) if os.path.exists(model_dir) else []:
                if f.endswith(".joblib"):
                    model_path = os.path.join(model_dir, f)
                    break
            if model_path is None:
                missing.append("(no joblib model found in model/ folder)")

        if missing:
            return None, None, None, None, f"Missing files: {missing}"

        # Load vectorizer and model
        vectorizer = joblib.load(tfidf_path)
        model = joblib.load(model_path)

        label_encoder = None
        if os.path.exists(label_enc_path):
            try:
                label_encoder = joblib.load(label_enc_path)
            except Exception:
                label_encoder = None

        metadata = None
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r") as fh:
                    metadata = json.load(fh)
            except Exception:
                metadata = None

        # roc image path (not loaded here; used later if exists)
        roc_exists = os.path.exists(roc_path)

        return {
            "model": model,
            "vectorizer": vectorizer,
            "label_encoder": label_encoder,
            "metadata": metadata,
            "roc_path": roc_path if roc_exists else None,
            "model_path": model_path
        }, None
    except Exception as e:
        tb = traceback.format_exc()
        return None, f"Exception while loading artifacts: {e}\n{tb}"

# Robust prediction extractor
def get_prediction_prob(model, vectorizer, text):
    """
    Return:
      - a float (probability for class 1) for binary classifiers
      - a dict {class_index: prob, ...} for multiclass outputs
    Raises runtime error with informative message when it cannot interpret result.
    """
    X = vectorizer.transform([text])

    # sklearn-like with predict_proba
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        proba = np.asarray(proba)
        if proba.ndim == 2:
            if proba.shape[1] == 2:
                return float(proba[0, 1])
            else:
                # multi-class: return map
                return {int(i): float(p) for i, p in enumerate(proba[0])}
        # fallback
        return float(proba.ravel()[0])

    # XGBoost Booster object
    if xgb is not None and isinstance(model, xgb.Booster):
        dmat = xgb.DMatrix(X)
        pred = model.predict(dmat)
        pred = np.asarray(pred)
        if pred.ndim == 2:
            return {int(i): float(p) for i, p in enumerate(pred[0])}
        return float(pred.ravel()[0])

    # LightGBM Booster
    if lgb is not None and isinstance(model, lgb.basic.Booster):
        pred = model.predict(X)
        pred = np.asarray(pred)
        if pred.ndim == 2:
            return {int(i): float(p) for i, p in enumerate(pred[0])}
        return float(pred.ravel()[0])

    # generic predict fallback
    try:
        pred = model.predict(X)
        pred = np.asarray(pred)
        if pred.ndim == 0:
            return float(pred.item())
        if pred.ndim == 1:
            if pred.shape[0] == 1:
                return float(pred[0])
            # if more than 1, return dict for debugging
            return {int(i): float(p) for i, p in enumerate(pred)}
        if pred.ndim == 2:
            # try to get class 1 probability if present
            if pred.shape[1] >= 2:
                return float(pred[0, 1])
            return {int(i): float(p) for i, p in enumerate(pred[0])}
    except Exception as e:
        raise RuntimeError(f"Generic predict failed: {e}")

    raise RuntimeError("Unable to extract scalar prediction from model output.")

# -------- Load artifacts once ----------
artifacts, load_err = load_artifacts()
if load_err:
    st.error("Model load error:")
    st.code(load_err)
    # stop further UI (keeps the page visible but doesn't crash)
    st.stop()

model = artifacts["model"]
vectorizer = artifacts["vectorizer"]
label_encoder = artifacts.get("label_encoder")
metadata = artifacts.get("metadata")
roc_path = artifacts.get("roc_path")
model_path = artifacts.get("model_path")

st.success("Model and vectorizer loaded.")
st.markdown(f"**Model file:** `{model_path}`")

# Optional: display metadata if present
if metadata:
    st.subheader("Model metadata")
    try:
        st.json(metadata)
    except Exception:
        st.write(metadata)

# Optional: show ROC image if available
if roc_path:
    try:
        from PIL import Image
        img = Image.open(roc_path)
        st.image(img, caption="ROC Curve", use_column_width=True)
    except Exception:
        pass

# -------- UI: text input / manual predict ----------
st.header("Manual test")
text = st.text_area("Enter ticket text for prediction", height=160)

# Predict button uses robust function
if st.button("Predict"):
    if not text.strip():
        st.warning("Please enter ticket text.")
    else:
        try:
            result = get_prediction_prob(model, vectorizer, text)
            if isinstance(result, dict):
                st.write("Predicted class probabilities (per class):")
                st.json(result)
                # if you have label_encoder, map indices to labels
                if label_encoder is not None:
                    mapped = {label_encoder.inverse_transform([k])[0] if hasattr(label_encoder, 'inverse_transform') else k: v for k, v in result.items()}
                    st.write("Mapped labels (if label encoder available):")
                    st.json(mapped)
            else:
                st.write("Predicted probability (class=1):", float(result))
                st.write("Predicted label (0/1 using 0.5 threshold):", int(float(result) >= 0.5))
        except Exception as e:
            st.error("Error during prediction:")
            st.code(str(e))

st.markdown("---")

# -------- REST-style params handling (ServiceNow can call this) ----------
st.header("Incoming REST call (GET query params)")

params = st.experimental_get_query_params()
if params:
    # Show all incoming params for debugging
    st.write("Query params received:")
    st.json(params)

if "text" in params:
    incoming_text = params.get("text", [""])[0]
    ticket_id = params.get("ticket_id", [""])[0]
    token = params.get("token", [""])[0]

    st.write("**Incoming REST call**")
    st.write("Ticket ID:", ticket_id)
    st.write("Text:", incoming_text)

    # Simple token check (optional). If you don't want a token, remove this block.
    if API_TOKEN and token and token != API_TOKEN:
        st.error("Unauthorized token provided.")
    else:
        if not incoming_text.strip():
            st.warning("No text provided in query param.")
        else:
            try:
                result = get_prediction_prob(model, vectorizer, incoming_text)
                if isinstance(result, dict):
                    st.write("Predicted class probabilities (per class):")
                    st.json(result)
                else:
                    st.write("Predicted probability (class=1):", float(result))
                    st.write("Predicted label (0/1 using 0.5 threshold):", int(float(result) >= 0.5))
            except Exception as e:
                st.error("Error during prediction for incoming REST call:")
                st.code(str(e))
else:
    st.info("No incoming REST 'text' param detected. To test, call the app like: `?text=printer%20jam&ticket_id=INC1&token=secret123`")

# -------- Footer / debugging info ----------
st.markdown("---")
st.caption(f"App running from: `{repo_root_from_app()}`")
st.caption("Tip: update code in GitHub and Streamlit will auto-redeploy the app.")
