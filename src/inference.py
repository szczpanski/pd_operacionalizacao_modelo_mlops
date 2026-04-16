import pandas as pd
from src.features import build_features


def prepare_inference_input(data):
    if isinstance(data, dict):
        X = pd.DataFrame([data])
    elif isinstance(data, pd.DataFrame):
        X = data.copy()
    else:
        raise TypeError("A entrada deve ser dict ou pandas.DataFrame.")

    X = build_features(X)
    return X


def predict(model, X):
    X_prepared = prepare_inference_input(X)

    prediction = model.predict(X_prepared)
    probability = None

    if hasattr(model, 'predict_proba'):
        probability = model.predict_proba(X_prepared)[:, 1]

    return {
        'prediction': prediction.tolist(),
        'probability_inadimplencia': None if probability is None else probability.tolist()
    }
