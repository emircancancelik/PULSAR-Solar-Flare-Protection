import sqlite3
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

class SQLiteTrainerAgent:
    def __init__(self, db_path='space_weather_telemetry.db', model_path='telemetry_model.pkl'):
        self.db_path = db_path
        self.model_path = model_path
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        
    def train_from_live_data(self):
        print("1. Canlı SQLite veritabanına bağlanılıyor...")
        try:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query("SELECT * FROM telemetry", conn)
            conn.close()
        except Exception as e:
            print(f"Veritabanı hatası: {e}")
            return
        if len(df) < 50:
            print(f"⚠️ Yeterli canlı veri yok. Mevcut kayıt: {len(df)}. Eğitim için sistemin yaklaşık 50 dakika çalışması bekleniyor.")
            return

        print(f"2. Toplam {len(df)} adet canlı telemetri verisiyle model eğitiliyor...")
        X = df[['speed', 'bz', 'm_prob']]
        y = df['current_kp']
        self.model.fit(X, y)
        joblib.dump(self.model, self.model_path)
        print(f"3. BAŞARILI: Model kendi kendini güncelledi ve '{self.model_path}' olarak kaydedildi!")

if __name__ == "__main__":
    ai_trainer = SQLiteTrainerAgent()
    ai_trainer.train_from_live_data()