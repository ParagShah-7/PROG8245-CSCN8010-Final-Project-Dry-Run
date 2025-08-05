import pickle
from functools import lru_cache

@lru_cache(maxsize=1)
def load_energy_model():
    with open("models/energy_predictor.pkl", "rb") as f:
        return pickle.load(f)

@lru_cache(maxsize=1)
def load_anomaly_model():
    with open("models/anomaly_detector.pkl", "rb") as f:
        return pickle.load(f)

def predict_energy(layers, time, complexity):
    model = load_energy_model()
    return model.predict([[layers, time, complexity]])[0]

def is_anomaly(energy):
    model = load_anomaly_model()
    return model.predict([[energy]])[0] == -1

def predict_category(prompt):
    return "General"  # placeholder or use actual classifier if available
