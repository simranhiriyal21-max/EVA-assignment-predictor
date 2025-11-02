# app/app.py
import streamlit as st
import joblib
import os
import traceback
import json
import numpy as np

# Optional model libraries
try:
    import xgboost as xgb
except Exception:
    xgb = None

try:
    import lightgbm as lgb
except Exception:
    lgb = None

# -------------------------
# 🔧 PAGE CONFIG
# -------------------------
st.set_page_config(page_title="EVA Assignment Group Predictor", layout="wide")
st.title("🎯 EVA Assignment Group Predictor")

# -------------------------
# 🔐 TOKEN CONFIG (for ServiceNow)
# -------------------------
API_TOKEN = "secret123"  # Must match the token in your ServiceNow Business Rule

# -------------------------
# 📦 LOAD MODEL ARTIFACTS
# -------------------------
def repo_root_from_app():
    return os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))

@st.cache_resource
def load_artifacts():
    """Load model, vectorizer, label encoder, and metadata."""
    try:
        root = repo_root_from_app()
        model_dir = os.path.join(root, "model")
        assets_dir = os.path.join(root, "assets")

        model_path = None
        lgb_path = os.path.join(model_dir, "model_lgb.joblib")
        xgb_path = os.path.join(model_dir, "model_xgb.joblib")
        tfidf_path = os.path.join(model_dir, "tfidf_vectorizer.joblib")
        label_enc_path = os.path.join(model_dir, "label_encoder.joblib")
        metadata_path = os.path.join(model_dir, "model_metadata.json")

        # pick available model
        if os.path.exists(lgb_path):
            model_path = lgb_path
        elif os.path.exists(xgb_path):
            model_path = xgb_path
        else:
            for f in os.listdir(model_dir):
                if f.endswith(".joblib"):
                    model_path = os.path.join(model_dir, f)
                    break

        missing = []
        if not model_path:
            missing.append("No .joblib model found")
        if not os.path.exists(tfidf_path):
            missing.append("Missing TF-IDF vectorizer")
        if missing:
            return None, f"Missing: {missing}"

        # load files
        model = joblib.load(model_path)
        vectorizer = joblib.load(tfidf_path)
        label_encoder = joblib.load(label_enc_path) if os.path.exists(label_enc_path) else None
        metadata = None
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as fh:
                metadata = json.load(fh)
        return {
            "model": model,
            "vectorizer": vectorizer,
            "label_encoder": label_encoder,
            "metadata": metadata,
            "model_path": model_path
        }, None
    except Exception as e:
        tb = traceback.format_exc()
        return None, f"Error loading artifacts: {e}\n{tb}"

artifacts, err = load_artifacts()
if err:
    st.error(err)
    st.stop()

model = artifacts["model"]
vectorizer = artifacts["vectorizer"]
label_encoder = artifacts["label_encoder"]
metadata = artifacts["metadata"]
model_path = artifacts["model_path"]

st.success("✅ Model and vectorizer loaded successfully.")
st.caption(f"Loaded model: `{model_path}`")

if metadata:
    st.markdown("**Model metadata:**")
    st.json(metadata)

# -------------------------
# 🧠 PREDICTION LOGIC
# -------------------------
def predict_group_and_probs(model, vectorizer, label_encoder, text):
    """Return predicted assignment group and probabilities."""
    X = vectorizer.transform([text])

    # sklearn-style predict_proba
    if hasattr(model, "predict_proba"):
        probs = np.asarray(model.predict_proba(X))[0]
        pred_idx = int(np.argmax(probs))
        pred_label = (
            label_encoder.inverse_transform([pred_idx])[0]
            if label_encoder is not None
            else str(pred_idx)
        )
        pred_prob = float(probs[pred_idx])
        prob_dict = {
            label_encoder.inverse_transform([i])[0] if label_encoder else str(i): float(p)
            for i, p in enumerate(probs)
        }
        return pred_label, pred_prob, prob_dict

    # LightGBM / XGBoost booster style
    try:
        pred = np.asarray(model.predict(X))
        if pred.ndim == 2:
            probs = pred[0]
            pred_idx = int(np.argmax(probs))
            pred_label = (
                label_encoder.inverse_transform([pred_idx])[0]
                if label_encoder is not None
                else str(pred_idx)
            )
            pred_prob = float(probs[pred_idx])
            prob_dict = {
                label_encoder.inverse_transform([i])[0] if label_encoder else str(i): float(p)
                for i, p in enumerate(probs)
            }
            return pred_label, pred_prob, prob_dict
        else:
            pred_idx = int(pred.ravel()[0])
            pred_label = (
                label_encoder.inverse_transform([pred_idx])[0]
                if label_encoder is not None
                else str(pred_idx)
            )
            return pred_label, 1.0, {pred_label: 1.0}
    except Exception as e:
        raise RuntimeError(f"Prediction failed: {e}")

# -------------------------
# 🧩 RULE-BASED OVERRIDES (optional)
# -------------------------
RULES = {
    "not booting": "Hardware Support",
    "won't boot": "Hardware Support",
    "blue screen": "Hardware Support",
    "wifi": "Network Team",
    "network": "Network Team",
    "outlook": "Email Team",
    "email": "Email Team",
    "vpn": "Network Team",
    "password": "Application Team"
}

def apply_rules(text):
    t = text.lower()
    for k, grp in RULES.items():
        if k in t:
            return grp
    return None

# -------------------------
# 🖥️ MANUAL UI (test input)
# -------------------------
st.header("Manual Ticket Prediction")
text_input = st.text_area("Enter ticket description:", height=150)

if st.button("Predict Assignment Group"):
    if not text_input.strip():
        st.warning("Please enter ticket text.")
    else:
        # check rules first
        rule_group = apply_rules(text_input)
        if rule_group:
            st.success(f"🧭 Rule-based match: Assigned to **{rule_group}**")
        else:
            try:
                pred_label, pred_prob, prob_dict = predict_group_and_probs(
                    model, vectorizer, label_encoder, text_input
                )
                st.write("**Predicted Assignment Group:**", pred_label)
                st.write("Confidence:", f"{pred_prob:.3f}")
                st.write("All group probabilities:")
                st.json(prob_dict)
            except Exception as e:
                st.error("Error during prediction:")
                st.code(str(e))

st.markdown("---")

# -------------------------
# 🌐 REST API CALL HANDLER (ServiceNow)
# -------------------------
st.header("Incoming REST call (from ServiceNow)")

params = st.experimental_get_query_params()
if params:
    st.write("Query params received:")
    st.json(params)

if "text" in params:
    incoming_text = params.get("text", [""])[0]
    ticket_id = params.get("ticket_id", [""])[0]
    token = params.get("token", [""])[0]

    st.subheader("Incoming REST Request")
    st.write("Ticket ID:", ticket_id)
    st.write("Text:", incoming_text)

    # Token validation
    if API_TOKEN:
        if token != API_TOKEN:
            st.error("❌ Unauthorized token. Check your Business Rule token.")
            st.stop()

    if not incoming_text.strip():
        st.warning("No text provided.")
    else:
        rule_group = apply_rules(incoming_text)
        if rule_group:
            st.success(f"🧭 Rule-based match: Assigned to **{rule_group}**")
        else:
            try:
                pred_label, pred_prob, prob_dict = predict_group_and_probs(
                    model, vectorizer, label_encoder, incoming_text
                )
                st.write("**Predicted Assignment Group:**", pred_label)
                st.write("Confidence:", f"{pred_prob:.3f}")
                st.write("All group probabilities:")
                st.json(prob_dict)
            except Exception as e:
                st.error("Error during REST prediction:")
                st.code(str(e))
else:
    st.info(
        "No REST query params detected. Test via URL like: "
        "`?text=laptop%20not%20booting&ticket_id=INC0010001&token=secret123`"
    )

# -------------------------
# 📋 FOOTER
# -------------------------
st.markdown("---")
st.caption("EVA Assignment Group Predictor - Streamlit Integration Demo")
st.caption("Developed for M.Tech Project (AI-based Ticket Routing)")
