import pandas as pd
import os
from datetime import datetime

HISTORY_FILE = "data/health_history.csv"

def save_history(user_id, age, sex, bmi, proba, risk_score, metabolic_score, health_index):
    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user_id": user_id,
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "risk_proba": proba,
        "risk_score": risk_score,
        "metabolic_score": metabolic_score,
        "health_index": health_index
    }
    
    df = pd.DataFrame([record])
    
    if os.path.exists(HISTORY_FILE):
        df.to_csv(HISTORY_FILE, mode='a', header=False, index=False)
    else:
        df.to_csv(HISTORY_FILE, mode='w', header=True, index=False)

def load_history(user_id=None):
    if not os.path.exists(HISTORY_FILE):
        return pd.DataFrame()
    
    df = pd.read_csv(HISTORY_FILE)
    
    if user_id:
        df = df[df["user_id"] == user_id]
    
    return df
