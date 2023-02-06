import pickle
from pathlib import Path

__version__ = "0.1.0"

BASE_DIR = Path(__file__).resolve(strict=True).parent


with open(f"{BASE_DIR}/trained linear model.pkl", "rb") as f:
    model = pickle.load(f)

monthToNumber = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12
}

abbrevToNum = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12
}

def predict_pipeline(month):
    if month in monthToNumber:
        month = monthToNumber[month]
    if month in abbrevToNum:
        month = abbrevToNum[month]
    month = int(month)
    assert 1 <= month <= 12, "Please enter a correct format of month input!"
    pred = model.iloc[month - 1].item()
    return str(pred)
