# app/app.py
import streamlit as st
import joblib
import os
import traceback
import json
import numpy as np

SERVICENOW_INSTANCE = st.secrets.get("SERVICENOW_INSTANCE", "")
SERVICENOW_USER = st.secrets.get("SERVICENOW_USER", "")
SERVICENOW_PWD = st.secrets.get("SERVICENOW_PWD", "")

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
                # Attempt auto-update back to ServiceNow
if SERVICENOW_INSTANCE and SERVICENOW_USER and SERVICENOW_PWD:
    st.info("🔁 Attempting to update ServiceNow incident assignment group...")
    group_sysid = get_group_sysid(SERVICENOW_INSTANCE, SERVICENOW_USER, SERVICENOW_PWD, pred_label)
    if group_sysid:
        update_incident_assignment(SERVICENOW_INSTANCE, SERVICENOW_USER, SERVICENOW_PWD, ticket_id, group_sysid)
    else:
        st.warning(f"Group '{pred_label}' not found in ServiceNow.")
else:
    st.info("ℹ️ No ServiceNow credentials configured — skipping auto-update.")

            except Exception as e:
                st.error("Error during prediction:")
                st.code(str(e))

st.markdown("---")

# -------------------------
# 🌐 REST API CALL HANDLER (ServiceNow POST-with-payload query param)
# -------------------------
import urllib.parse
import requests
from requests.auth import HTTPBasicAuth
import time
from pathlib import Path

st.header("Incoming REST call (from ServiceNow)")

# optional local log file (keeps track of incoming calls)
INCOMING_LOG = Path("/tmp/streamlit_incoming_calls.jsonl")
def append_incoming_log(entry: dict):
    try:
        INCOMING_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(INCOMING_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass

def read_incoming_log(limit=25):
    if not INCOMING_LOG.exists():
        return []
    try:
        with open(INCOMING_LOG, "r", encoding="utf-8") as fh:
            lines = [l.strip() for l in fh.readlines() if l.strip()]
        return [json.loads(l) for l in lines[-limit:]]
    except Exception:
        return []

# Helper functions to update incident (optional)
def get_group_sysid(instance, user, pwd, group_name):
    try:
        q = "name=" + urllib.parse.quote(group_name)
        url = f"https://{instance}/api/now/table/sys_user_group?sysparm_query={q}&sysparm_fields=sys_id,name&sysparm_limit=1"
        r = requests.get(url, auth=HTTPBasicAuth(user, pwd), timeout=15)
        r.raise_for_status()
        res = r.json()
        if res.get("result"):
            return res["result"][0]["sys_id"]
    except Exception as ex:
        st.error(f"Group lookup failed: {ex}")
    return None

def update_incident_assignment(instance, user, pwd, ticket_id, group_sysid):
    try:
        # Resolve sys_id from incident number
        q = "number=" + urllib.parse.quote(ticket_id)
        url_lookup = f"https://{instance}/api/now/table/incident?sysparm_query={q}&sysparm_fields=sys_id&sysparm_limit=1"
        r = requests.get(url_lookup, auth=HTTPBasicAuth(user, pwd), timeout=15)
        r.raise_for_status()
        data = r.json()
        if not data.get("result"):
            return False, "incident_not_found"
        sysid = data["result"][0]["sys_id"]

        patch_url = f"https://{instance}/api/now/table/incident/{sysid}"
        payload = {"assignment_group": group_sysid}
        headers = {"Content-Type": "application/json"}
        r2 = requests.patch(patch_url, json=payload, auth=HTTPBasicAuth(user, pwd), headers=headers, timeout=15)
        r2.raise_for_status()
        return True, r2.json()
    except Exception as ex:
        return False, str(ex)

# ---- Parse params (supports POST with payload param) ----
params = st.experimental_get_query_params()
if params:
    st.write("Query params received:")
    st.json(params)

incoming_text = ""
ticket_id = ""
token = params.get("token", [""])[0] if "token" in params else ""

# Prefer 'payload' query param (contains JSON)
if "payload" in params:
    raw_payload = params.get("payload", [""])[0]
    try:
        decoded = urllib.parse.unquote(raw_payload)
        payload_obj = json.loads(decoded)
        ticket_id = payload_obj.get("ticket_id", "")
        incoming_text = (payload_obj.get("short_description", "") or "") + " " + (payload_obj.get("description", "") or "")
        st.subheader("Decoded JSON payload:")
        st.json(payload_obj)
    except Exception as e:
        st.error(f"Failed to decode payload JSON: {e}")

elif "text" in params:  # fallback for older GET calls
    incoming_text = params.get("text", [""])[0]
    ticket_id = params.get("ticket_id", [""])[0]

if incoming_text:
    st.subheader("Incoming REST Request")
    st.write("Ticket ID:", ticket_id)
    st.write("Text:", incoming_text)

    # token validation
    if API_TOKEN and token != API_TOKEN:
        st.error("❌ Unauthorized token. Check your Business Rule token.")
    else:
        # apply rule-based logic first
        rule_group = apply_rules(incoming_text)
        if rule_group:
            st.success(f"🧭 Rule-based match: Assigned to **{rule_group}**")
            pred_label = rule_group
            pred_prob = 1.0
            prob_dict = {rule_group: 1.0}
        else:
            try:
                pred_label, pred_prob, prob_dict = predict_group_and_probs(model, vectorizer, label_encoder, incoming_text)
                st.write("**Predicted Assignment Group:**", pred_label)
                st.write("Confidence:", f"{pred_prob:.3f}")
                st.write("All group probabilities:")
                st.json(prob_dict)
            except Exception as e:
                st.error(f"Error during prediction: {e}")
                pred_label = None
                pred_prob = None
                prob_dict = {}

        # Log to server file
        append_incoming_log({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "ticket_id": ticket_id,
            "pred_label": pred_label,
            "pred_prob": pred_prob,
            "text": incoming_text
        })

        # Optional: auto-update ServiceNow (if creds configured)
        SN_INSTANCE = st.secrets.get("SERVICENOW_INSTANCE")
        SN_USER = st.secrets.get("SERVICENOW_USER")
        SN_PWD  = st.secrets.get("SERVICENOW_PWD")
        if SN_INSTANCE and SN_USER and SN_PWD and pred_label:
            group_sysid = get_group_sysid(SN_INSTANCE, SN_USER, SN_PWD, pred_label)
            if group_sysid:
                ok, msg = update_incident_assignment(SN_INSTANCE, SN_USER, SN_PWD, ticket_id, group_sysid)
                if ok:
                    st.success(f"✅ Updated incident {ticket_id} with assignment group {pred_label}")
                else:
                    st.warning(f"⚠️ Could not update incident: {msg}")
            else:
                st.warning(f"⚠️ Group '{pred_label}' not found in ServiceNow instance.")
        else:
            st.info("ℹ️ No ServiceNow credentials configured — skipping auto-update.")
else:
    st.info("No incoming REST payload detected. Try calling: `?payload=%7B...%7D&token=secret123`")

# Show recent incoming calls
st.markdown("---")
st.subheader("📜 Recent incoming REST calls (server log)")
calls = read_incoming_log(limit=20)
if calls:
    for c in reversed(calls):
        st.write(f"**{c['timestamp']}** — Ticket: **{c['ticket_id']}** — Predicted: **{c['pred_label']}** — Confidence: {c['pred_prob']:.2f}")
        st.caption(c['text'])
else:
    st.info("No calls logged yet.")

# -------------------------
# 🔗 SERVICENOW API HELPERS
# -------------------------
import requests
from requests.auth import HTTPBasicAuth

def get_group_sysid(instance, user, pwd, group_name):
    """
    Fetch sys_id of a ServiceNow group by its name.
    """
    try:
        url = f"https://{instance}/api/now/table/sys_user_group"
        params = {
            "sysparm_query": f"name={group_name}",
            "sysparm_fields": "sys_id,name",
            "sysparm_limit": 1
        }
        r = requests.get(
            url,
            auth=HTTPBasicAuth(user, pwd),
            params=params,
            headers={"Accept": "application/json"},
            timeout=15
        )
        r.raise_for_status()
        data = r.json()
        if data.get("result"):
            return data["result"][0]["sys_id"]
        else:
            st.warning(f"Group '{group_name}' not found in ServiceNow.")
            return None
    except Exception as e:
        st.error(f"Error fetching group sys_id: {e}")
        return None


def update_incident_assignment(instance, user, pwd, ticket_number, group_sysid):
    """
    Update an incident's assignment group field using ServiceNow REST Table API.
    """
    try:
        # 1️⃣ Lookup incident sys_id by number
        lookup_url = f"https://{instance}/api/now/table/incident"
        lookup_params = {
            "sysparm_query": f"number={ticket_number}",
            "sysparm_fields": "sys_id",
            "sysparm_limit": 1
        }
        lookup = requests.get(
            lookup_url,
            auth=HTTPBasicAuth(user, pwd),
            params=lookup_params,
            headers={"Accept": "application/json"},
            timeout=15
        )
        lookup.raise_for_status()
        res = lookup.json()
        if not res.get("result"):
            st.warning(f"Incident {ticket_number} not found.")
            return False
        sys_id = res["result"][0]["sys_id"]

        # 2️⃣ PATCH the record to update assignment_group
        patch_url = f"https://{instance}/api/now/table/incident/{sys_id}"
        payload = {"assignment_group": group_sysid}
        patch = requests.patch(
            patch_url,
            auth=HTTPBasicAuth(user, pwd),
            json=payload,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            timeout=15
        )
        patch.raise_for_status()
        st.success(f"✅ Updated incident {ticket_number} assignment group successfully.")
        return True
    except Exception as e:
        st.error(f"❌ Failed to update incident: {e}")
        return False


# -------------------------
# 📋 FOOTER
# -------------------------
st.markdown("---")
st.caption("EVA Assignment Group Predictor - Streamlit Integration Demo")
st.caption("Developed for M.Tech Project (AI-based Ticket Routing)")
