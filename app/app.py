# top of app/app.py
import streamlit as st
import joblib
import os
import traceback

st.set_page_config(page_title="EVA Assignment Predictor")
st.title("🎯 EVA Assignment Predictor")
st.write("Starting app...")

def load_artifacts():
    try:
        base_dir = os.path.dirname(__file__)
        model_dir = os.path.normpath(os.path.join(base_dir, '..', 'model'))
        lgb_path = os.path.join(model_dir, 'model_lgb.joblib')
        tfidf_path = os.path.join(model_dir, 'tfidf_vectorizer.joblib')
        label_enc_path = os.path.join(model_dir, 'label_encoder.joblib')

        missing = []
        for p in [lgb_path, tfidf_path]:
            if not os.path.exists(p):
                missing.append(p)
        if missing:
            return None, None, f"Missing files: {missing}"

        # Try to load; catch exceptions and return message
        model = joblib.load(lgb_path)
        vectorizer = joblib.load(tfidf_path)
        label_encoder = None
        if os.path.exists(label_enc_path):
            label_encoder = joblib.load(label_enc_path)
        return model, vectorizer, None
    except Exception as e:
        tb = traceback.format_exc()
        return None, None, f"Exception while loading artifacts: {e}\n{tb}"

model, vectorizer, load_err = load_artifacts()

if load_err:
    st.error("Model load error:")
    st.code(load_err)
else:
    st.success("Model and vectorizer loaded.")
    # small UI to test
    text = st.text_area("Enter ticket text for prediction")
    if st.button("Predict"):
        if not text.strip():
            st.warning("Enter ticket text first.")
        else:
            try:
                X = vectorizer.transform([text])
                pred_prob = model.predict(X)[0]
                st.write("Pred prob:", float(pred_prob))
            except Exception as e:
                st.error("Error during prediction:")
                st.code(str(e))
