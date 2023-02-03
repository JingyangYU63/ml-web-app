import pickle
from pathlib import Path

__version__ = "0.1.0"

BASE_DIR = Path(__file__).resolve(strict=True).parent


with open(f"{BASE_DIR}/trained linear model.pkl", "rb") as f:
    model = pickle.load(f)


def predict_pipeline(month):
    month = int(month)
    assert 1 <= month <= 12
    pred = model.iloc[month - 1].item()
    return str(pred)
