# model_predict.py

import joblib
import pandas as pd
from configs import MODEL_PATH

@staticmethod
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

model_features = [
    '연령대코드(5세단위)', '식전혈당(공복혈당)', '혈색소',
    '감마지티피', 'bmi', '트리글리세라이드', 'HDL콜레스테롤'
]

def predict_proba(features_dict):
    input_df = pd.DataFrame([{k: features_dict.get(k, 0) for k in model_features}])
    proba = model.predict_proba(input_df)[0][1]
    return proba
