# api/app_api.py
import os
import tempfile
import joblib
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Local expected paths inside repo when deployed
MODEL_LOCAL = "model/model_lgb.joblib"
VECT_LOCAL  = "model/tfidf_vectorizer.joblib"

def download_if_needed(local_path, env_var_name):
    # If file already exists in repo, use it. Otherwise download from env var URL.
    if os.path.exists(local_path):
        return local_path
    url = os.environ.get(env_var_name)
    if not url:
        raise RuntimeError(f"{local_path} not found and {env_var_name} not set")
    tmp = tempfile.NamedTemporaryFile(delete=False)
    r = requests.get(url, allow_redirects=True, timeout=60)
    r.raise_for_status()
    tmp.write(r.content)
    tmp.flush()
    return tmp.name

# Load model + vectorizer on startup
model = None
vectorizer = None
try:
    mpath = download_if_needed(MODEL_LOCAL, "MODEL_URL") if not os.path.exists(MODEL_LOCAL) else MODEL_LOCAL
    vpath = download_if_needed(VECT_LOCAL, "VECT_URL") if not os.path.exists(VECT_LOCAL) else VECT_LOCAL
    model = joblib.load(mpath)
    vectorizer = joblib.load(vpath)
    print("Loaded model and vectorizer")
except Exception as e:
    print("Model load error:", e)

@app.route("/health", methods=["GET"])
def health():
    ok = model is not None and vectorizer is not None
    return jsonify({"status": "ok" if ok else "error"}), (200 if ok else 500)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None or vectorizer is None:
        return jsonify({"error":"model not loaded"}), 500
    data = request.get_json(force=True)
    text = data.get("text", "")
    if not text:
        return jsonify({"error":"no text provided"}), 400

    X = vectorizer.transform([text])
    try:
        probs = model.predict_proba(X)[0].tolist()
    except Exception:
        probs = None
    pred = model.predict(X)[0]
    return jsonify({"prediction": str(pred), "probabilities": probs})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
