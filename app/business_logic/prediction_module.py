import pickle
from functools import lru_cache

@lru_cache(maxsize=1)
def load_anomaly_model():
    with open("models/anomaly_detector.pkl", "rb") as f:
        return pickle.load(f)

def is_anomaly(energy):
    model = load_anomaly_model()
    return model.predict([[energy]])[0] == -1

def predict_category(prompt):
    return "General"  # Stub – update if classifier is implemented
