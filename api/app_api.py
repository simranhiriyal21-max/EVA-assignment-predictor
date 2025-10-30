from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load your model and vectorizer
model = joblib.load("../model/model_lgb.joblib")
vectorizer = joblib.load("../model/tfidf_vectorizer.joblib")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Transform text and predict
    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X).tolist()

    return jsonify({
        "prediction": str(prediction),
        "probabilities": probabilities
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
