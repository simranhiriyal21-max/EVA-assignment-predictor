# put these imports at top of file if not already present
import numpy as np
import xgboost as xgb
import lightgbm as lgb

def get_prediction_prob(model, vectorizer, text):
    """
    Return a single probability (float) for the positive class if binary,
    or return a dict of class->prob for multi-class models.
    """
    X = vectorizer.transform([text])  # 1 x n_features (sparse or dense)

    # If model has predict_proba (sklearn API)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)  # shape: (1, n_classes)
        proba = np.asarray(proba)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            # binary: take class 1 prob; multi-class: return full vector
            if proba.shape[1] == 2:
                return float(proba[0, 1])
            else:
                return {int(i): float(p) for i, p in enumerate(proba[0])}

        # fallback: single probability
        return float(proba.ravel()[0])

    # If XGBoost Booster (raw xgb.Booster)
    if isinstance(model, xgb.Booster):
        # XGBoost expects DMatrix
        dmat = xgb.DMatrix(X)
        pred = model.predict(dmat)  # could be (1,) or (1, n_classes)
        pred = np.asarray(pred)
        if pred.ndim == 2:
            # multi-class -> return dict
            return {int(i): float(p) for i, p in enumerate(pred[0])}
        return float(pred.ravel()[0])

    # If LightGBM Booster (lgb.Booster)
    if isinstance(model, lgb.basic.Booster) or hasattr(model, 'predict') and not hasattr(model, 'predict_proba'):
        # Some LightGBM objects are Booster (from lgb.train)
        try:
            pred = model.predict(X)  # usually returns (1,) for binary
            pred = np.asarray(pred)
            if pred.ndim == 2:
                return {int(i): float(p) for i, p in enumerate(pred[0])}
            return float(pred.ravel()[0])
        except Exception:
            # fallthrough to generic predict
            pass

    # Generic fallback: call predict and try to extract sensible scalar
    try:
        pred = model.predict(X)
        pred = np.asarray(pred)
        if pred.ndim == 0:
            return float(pred.item())
        if pred.ndim == 1:
            # if shape (1,), return first element
            if pred.shape[0] == 1:
                return float(pred[0])
            # if longer, return array->list for debugging
            return {i: float(p) for i, p in enumerate(pred)}
        if pred.ndim == 2:
            # try probability for class 1
            if pred.shape[1] >= 2:
                return float(pred[0, 1])
            return {i: float(p) for i, p in enumerate(pred[0])}
    except Exception as e:
        raise RuntimeError(f"Generic predict failed: {e}")

    raise RuntimeError("Unable to extract scalar prediction from model output.")

# Usage inside app when user clicks Predict:
try:
    result = get_prediction_prob(model, vectorizer, text)
    if isinstance(result, dict):
        # multi-class returned as dict
        st.write("Predicted class probabilities:")
        st.json(result)
    else:
        # binary scalar probability returned
        st.write("Pred probability (class=1):", float(result))
        st.write("Predicted label (0/1 with 0.5 threshold):", int(float(result) >= 0.5))
except Exception as e:
    st.error("Error during prediction:")
    st.code(str(e))
