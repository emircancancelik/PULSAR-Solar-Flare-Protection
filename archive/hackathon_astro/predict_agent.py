import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import joblib
import numpy as np

class PredictionAgent:
    def __init__(self, model_path='solar_flare_model_v2.pkl'):
        try:
            self.model = joblib.load(model_path)
            print("🚀 ALAZ Fizik Motoru Yayında.")
        except Exception as e:
            print(f"Model yükleme hatası: {e}")
            self.model = None

    def predict_orbital_decay(self, kp, inc, mm):
        """
        30 dakika sonraki yörünge sürtünmesini (BSTAR) tahmin eder.
        Girdiler: Kp Skalası, Yörünge Eğimi, Günlük Tur Sayısı
        """
        if self.model:
            X_input = np.array([[kp, inc, mm]])
            prediction = self.model.predict(X_input)
            return prediction[0]
        return 0.0
        month_sin = np.sin(2 * np.pi * month / 12)
        hour_sin = np.sin(2 * np.pi * hour / 24)
        X_input = np.array([[flare_score, month_sin, hour_sin, year]])
        
        prediction = self.model.predict(X_input)
        return prediction[0]