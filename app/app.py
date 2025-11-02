# app/app.py
"""
Full Streamlit app: EVA Assignment Group Predictor
- loads model artifacts from ../model/
- accepts manual text input
- accepts ServiceNow calls via query params (payload=... or text=...)
- predicts assignment group (using label encoder if present)
- optionally updates the incident's assignment_group back in ServiceNow
Make sure:
 - API_TOKEN matches the token used in your ServiceNow Business Rule
 - Model files are in repo model/ : model_lgb.joblib (or any .joblib), tfidf_vectorizer.joblib, label_encoder.joblib (optional), model_metadata.json (optional)
 - Streamlit secrets contain SERVICENOW_INSTANCE, SERVICENOW_USER, SERVICENOW_PWD for auto-update
"""

import os
import json
import traceback
from pathlib import Path
import joblib
import numpy as np
import streamlit as st

# Optional ML libs (XGBoost/LightGBM)
try:
    import xgboost as xgb
except Exception:
    xgb = None

try:
    import lightgbm as lgb
except Exception:
    lgb = None

# network requests for ServiceNow updates
import requests
from requests.auth import HTTPBasicAuth

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="EVA Assignment Group Predictor", layout="wide")
st.title("🎯 EVA Assignment Group Predictor")

# -------------------------
# Config / token
# -------------------------
# Minimal token check to prevent unauthorised calls
API_TOKEN = "secret123"  # Change to match Business Rule or change Business Rule to match this token

# -------------------------
# Helpers: file paths
# -------------------------
def repo_root_from_app():
    # this file lives in app/ ; repo root is parent
    return os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))

# -------------------------
# Load artifacts (cached)
# -------------------------
@st.cache_resource
def load_artifacts():
    try:
        root = repo_root_from_app()
        model_dir = os.path.join(root, "model")

        # expected filenames (flexible)
        lgb_path = os.path.join(model_dir, "model_lgb.joblib")
        xgb_path = os.path.join(model_dir, "model_xgb.joblib")
        tfidf_path = os.path.join(model_dir, "tfidf_vectorizer.joblib")
        label_enc_path = os.path.join(model_dir, "label_encoder.joblib")
        metadata_path = os.path.join(model_dir, "model_metadata.json")

        # find model
        model_path = None
        if os.path.exists(lgb_path):
            model_path = lgb_path
        elif os.path.exists(xgb_path):
            model_path = xgb_path
        else:
            # fallback: any .joblib in model_dir
            if os.path.exists(model_dir):
                for f in os.listdir(model_dir):
                    if f.endswith(".joblib"):
                        model_path = os.path.join(model_dir, f)
                        break

        missing = []
        if not model_path:
            missing.append("No .joblib model found in model/ folder")
        if not os.path.exists(tfidf_path):
            missing.append("Missing tfidf_vectorizer.joblib")
        if missing:
            return None, None, None, None, f"Missing artifacts: {missing}"

        # load
        model = joblib.load(model_path)
        vectorizer = joblib.load(tfidf_path)
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

        return model, vectorizer, label_encoder, metadata, model_path
    except Exception as e:
        tb = traceback.format_exc()
        return None, None, None, None, f"Exception while loading artifacts: {e}\n{tb}"

model, vectorizer, label_encoder, metadata, model_path_or_err = load_artifacts()
if model is None:
    st.error("Model load error:")
    st.code(model_path_or_err)
    st.stop()

st.success("✅ Model and vectorizer loaded successfully.")
st.caption(f"Loaded model: `{model_path_or_err}`")

if metadata:
    st.markdown("**Model metadata:**")
    st.json(metadata)

# -------------------------
# Prediction logic
# -------------------------
def predict_group_and_probs(model, vectorizer, label_encoder, text):
    """
    Returns: (pred_label (str), pred_prob (float), prob_dict {label:prob})
    Works for sklearn-like predict_proba, LightGBM/XGBoost boosters, and some fallback predict outputs.
    """
    X = vectorizer.transform([text])

    # sklearn-like
    if hasattr(model, "predict_proba"):
        probs = np.asarray(model.predict_proba(X))[0]
        idx = int(np.argmax(probs))
        pred_label = label_encoder.inverse_transform([idx])[0] if label_encoder is not None else str(idx)
        pred_prob = float(probs[idx])
        prob_dict = {label_encoder.inverse_transform([i])[0] if label_encoder else str(i): float(p) for i,p in enumerate(probs)}
        return pred_label, pred_prob, prob_dict

    # XGBoost Booster
    if xgb is not None and isinstance(model, xgb.Booster):
        dmat = xgb.DMatrix(X)
        pred = model.predict(dmat)
        pred = np.asarray(pred)
        if pred.ndim == 2:
            probs = pred[0]
            idx = int(np.argmax(probs))
            pred_label = label_encoder.inverse_transform([idx])[0] if label_encoder is not None else str(idx)
            return pred_label, float(probs[idx]), {label_encoder.inverse_transform([i])[0] if label_encoder else str(i): float(p) for i,p in enumerate(probs)}
        else:
            # scalar
            return (label_encoder.inverse_transform([int(pred.ravel()[0])])[0] if label_encoder else str(int(pred.ravel()[0]))), 1.0, {}

    # LightGBM booster
    if lgb is not None and isinstance(model, lgb.basic.Booster):
        pred = model.predict(X)
        pred = np.asarray(pred)
        if pred.ndim == 2:
            probs = pred[0]
            idx = int(np.argmax(probs))
            pred_label = label_encoder.inverse_transform([idx])[0] if label_encoder is not None else str(idx)
            return pred_label, float(probs[idx]), {label_encoder.inverse_transform([i])[0] if label_encoder else str(i): float(p) for i,p in enumerate(probs)}
        else:
            # if scalar or class
            try:
                idx = int(pred.ravel()[0])
                pred_label = label_encoder.inverse_transform([idx])[0] if label_encoder is not None else str(idx)
                return pred_label, 1.0, {pred_label: 1.0}
            except Exception:
                raise RuntimeError("Unexpected LightGBM output shape")

    # generic fallback predict
    try:
        pred = np.asarray(model.predict(X))
        if pred.ndim == 0:
            return str(pred.item()), 1.0, {str(pred.item()): 1.0}
        if pred.ndim == 1:
            if pred.shape[0] == 1:
                val = pred[0]
                if label_encoder is not None:
                    try:
                        return label_encoder.inverse_transform([int(val)])[0], 1.0, {label_encoder.inverse_transform([int(val)])[0]: 1.0}
                    except Exception:
                        return str(val), 1.0, {str(val): 1.0}
                return str(val), 1.0, {str(val): 1.0}
            else:
                # multi-output: return mapping (best effort)
                prob_dict = {}
                for i, v in enumerate(pred):
                    key = label_encoder.inverse_transform([i])[0] if label_encoder is not None else str(i)
                    prob_dict[key] = float(v)
                idx = int(np.argmax(pred))
                pred_label = label_encoder.inverse_transform([idx])[0] if label_encoder is not None else str(idx)
                return pred_label, float(prob_dict[pred_label]), prob_dict
        if pred.ndim == 2:
            probs = pred[0]
            idx = int(np.argmax(probs))
            pred_label = label_encoder.inverse_transform([idx])[0] if label_encoder is not None else str(idx)
            prob_dict = {label_encoder.inverse_transform([i])[0] if label_encoder else str(i): float(p) for i,p in enumerate(probs)}
            return pred_label, float(probs[idx]), prob_dict
    except Exception as e:
        raise RuntimeError(f"Generic predict failed: {e}")

    raise RuntimeError("Unable to extract prediction from model output.")

# -------------------------
# Rule-based quick overrides (optional, lightweight)
# -------------------------
RULES = {
    "not booting": "Hardware Support",
    "won't boot": "Hardware Support",
    "blue screen": "Hardware Support",
    "wifi": "Network Support",
    "network": "Network Support",
    "outlook": "Email Support",
    "email": "Email Support",
    "vpn": "Network Support",
    "password": "IT Help Desk"
}

def apply_rules(text):
    if not text:
        return None
    t = text.lower()
    for k, grp in RULES.items():
        if k in t:
            return grp
    return None

# -------------------------
# ServiceNow secrets (Streamlit Secrets)
# -------------------------
SERVICENOW_INSTANCE = st.secrets.get("SERVICENOW_INSTANCE", "")
SERVICENOW_USER = st.secrets.get("SERVICENOW_USER", "")
SERVICENOW_PWD  = st.secrets.get("SERVICENOW_PWD", "")

# -------------------------
# ServiceNow helper functions
# -------------------------
def get_group_sysid(instance, user, pwd, group_name):
    try:
        url = f"https://{instance}/api/now/table/sys_user_group"
        params = {"sysparm_query": f"name={group_name}", "sysparm_fields": "sys_id,name", "sysparm_limit": 1}
        r = requests.get(url, auth=HTTPBasicAuth(user, pwd), params=params, headers={"Accept":"application/json"}, timeout=15)
        r.raise_for_status()
        data = r.json()
        if data.get("result"):
            return data["result"][0]["sys_id"]
        return None
    except Exception as e:
        st.error(f"Error fetching group sys_id: {e}")
        return None

def update_incident_assignment(instance, user, pwd, ticket_number, group_sysid):
    try:
        # resolve incident sys_id by number
        lookup_url = f"https://{instance}/api/now/table/incident"
        lookup_params = {"sysparm_query": f"number={ticket_number}", "sysparm_fields": "sys_id", "sysparm_limit": 1}
        lookup = requests.get(lookup_url, auth=HTTPBasicAuth(user, pwd), params=lookup_params, headers={"Accept":"application/json"}, timeout=15)
        lookup.raise_for_status()
        res = lookup.json()
        if not res.get("result"):
            st.warning(f"Incident {ticket_number} not found.")
            return False
        sys_id = res["result"][0]["sys_id"]

        # patch assignment_group
        patch_url = f"https://{instance}/api/now/table/incident/{sys_id}"
        payload = {"assignment_group": group_sysid}
        patch = requests.patch(patch_url, auth=HTTPBasicAuth(user, pwd), json=payload, headers={"Content-Type":"application/json","Accept":"application/json"}, timeout=15)
        patch.raise_for_status()
        return True
    except Exception as e:
        st.error(f"Failed to update incident: {e}")
        return False

# -------------------------
# UI: Manual input
# -------------------------
st.header("Manual Ticket Prediction")
text_input = st.text_area("Enter ticket description:", height=150)
if st.button("Predict Assignment Group (manual)"):
    if not text_input.strip():
        st.warning("Please enter ticket text.")
    else:
        rule_group = apply_rules(text_input)
        if rule_group:
            st.success(f"🧭 Rule-based match: Assigned to **{rule_group}**")
        else:
            try:
                pred_label, pred_prob, prob_dict = predict_group_and_probs(model, vectorizer, label_encoder, text_input)
                st.write("**Predicted Assignment Group:**", pred_label)
                st.write("Confidence:", f"{pred_prob:.3f}")
                st.write("All group probabilities:")
                st.json(prob_dict)
            except Exception as e:
                st.error("Error during prediction:")
                st.code(str(e))

st.markdown("---")

# -------------------------
# REST handling (payload or text param)
# -------------------------
st.header("Incoming REST call (from ServiceNow)")

params = st.experimental_get_query_params() or {}
if params:
    st.write("Query params received:")
    st.json(params)

# Support two modes:
# 1) payload param: JSON string encoded as query param (recommended for Business Rule POST -> payload in URL)
# 2) legacy text param: use ?text=...&ticket_id=...&token=...
incoming_text = ""
ticket_id = ""
token = ""

# 1) payload param (URL encoded JSON)
if "payload" in params:
    raw = params.get("payload", [""])[0]
    try:
        # payload may be URL-decoded already by Streamlit
        payload_obj = json.loads(raw)
        st.write("Decoded JSON payload:")
        st.json(payload_obj)
        ticket_id = payload_obj.get("ticket_id", "") or ""
        incoming_text = (payload_obj.get("short_description", "") or "") + " " + (payload_obj.get("description", "") or "")
    except Exception as e:
        st.error(f"Failed to decode 'payload' JSON: {e}")
        incoming_text = ""
        ticket_id = ""

# 2) fallback: text param
if not incoming_text and "text" in params:
    incoming_text = params.get("text", [""])[0]
    ticket_id = params.get("ticket_id", [""])[0] if "ticket_id" in params else ""

# token
token = params.get("token", [""])[0] if "token" in params else ""

if incoming_text:
    st.subheader("Incoming REST Request")
    st.write("Ticket ID:", ticket_id or "(none provided)")
    st.write("Text:", incoming_text)

    # Token validation
    if API_TOKEN:
        if not token:
            st.error("❌ No token provided in query params.")
        elif token != API_TOKEN:
            st.error("❌ Unauthorized token. Check your Business Rule token.")
        else:
            # proceed
            rule_group = apply_rules(incoming_text)
            if rule_group:
                st.success(f"🧭 Rule-based match: Assigned to **{rule_group}**")
                pred_label = rule_group
                pred_prob = 1.0
                prob_dict = {pred_label: 1.0}
            else:
                try:
                    pred_label, pred_prob, prob_dict = predict_group_and_probs(model, vectorizer, label_encoder, incoming_text)
                    st.write("**Predicted Assignment Group:**", pred_label)
                    st.write("Confidence:", f"{pred_prob:.3f}")
                    st.write("All group probabilities:")
                    st.json(prob_dict)
                except Exception as e:
                    st.error("Error during REST prediction:")
                    st.code(str(e))
                    pred_label = None
                    pred_prob = None
                    prob_dict = {}

            # Attempt auto-update back to ServiceNow (wrapped safely)
            try:
                if SERVICENOW_INSTANCE and SERVICENOW_USER and SERVICENOW_PWD and pred_label and ticket_id:
                    st.info("🔁 Attempting to update ServiceNow incident assignment group...")
                    group_sysid = get_group_sysid(SERVICENOW_INSTANCE, SERVICENOW_USER, SERVICENOW_PWD, pred_label)
                    if group_sysid:
                        ok = update_incident_assignment(SERVICENOW_INSTANCE, SERVICENOW_USER, SERVICENOW_PWD, ticket_id, group_sysid)
                        if ok:
                            st.success(f"✅ Updated incident {ticket_id} with assignment group {pred_label}")
                        else:
                            st.warning("Auto-update attempted but failed (see messages).")
                    else:
                        st.warning(f"Group '{pred_label}' not found in ServiceNow — cannot auto-update.")
                else:
                    st.info("ℹ️ No ServiceNow credentials configured or missing ticket_id/prediction — skipping auto-update.")
            except Exception as ex:
                st.error(f"Unexpected error during auto-update: {ex}")
else:
    st.info("No incoming REST 'payload' or 'text' param detected. For testing call: `?payload=%7B...%7D&token=secret123` or `?text=printer%20jam&ticket_id=INC1&token=secret123`")

st.markdown("---")

# -------------------------
# Server-side log of recent incoming calls (in-memory)
# -------------------------
if "incoming_calls" not in st.session_state:
    st.session_state["incoming_calls"] = []

# Append a concise log entry if there was a REST call just now (non-empty ticket_id or text)
if incoming_text:
    log_entry = {
        "ticket": ticket_id or "",
        "text": incoming_text,
        "predicted": (pred_label if 'pred_label' in locals() else None),
        "confidence": (float(pred_prob) if 'pred_prob' in locals() and pred_prob is not None else None)
    }
    # keep last 20
    st.session_state.incoming_calls.insert(0, log_entry)
    st.session_state.incoming_calls = st.session_state.incoming_calls[:20]

st.subheader("Recent incoming REST calls (server log)")
for e in st.session_state.incoming_calls:
    ticket = e.get("ticket") or "(no id)"
    pred = e.get("predicted") or "(none)"
    conf = e.get("confidence")
    conf_str = f" — Confidence: {conf:.2f}" if conf is not None else ""
    st.write(f"{ticket} — Predicted: {pred}{conf_str}")
    st.write(f"Text: {e.get('text')}")
    st.markdown("---")

# -------------------------
# Footer
# -------------------------
st.caption("App running from: `" + repo_root_from_app() + "`")
st.caption("Tip: update code in GitHub and Streamlit will auto-redeploy.")
