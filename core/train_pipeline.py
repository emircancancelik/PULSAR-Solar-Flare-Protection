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
            # Veritabanındaki tüm canlı akışı Pandas DataFrame'e çevir
            df = pd.read_sql_query("SELECT * FROM telemetry", conn)
            conn.close()
        except Exception as e:
            print(f"Veritabanı hatası: {e}")
            return
            
        # Güvenlik Kilidi: Modelin düzgün eğitilmesi için en az 50 veri noktası birikmeli
        if len(df) < 50:
            print(f"⚠️ Yeterli canlı veri yok. Mevcut kayıt: {len(df)}. Eğitim için sistemin yaklaşık 50 dakika çalışması bekleniyor.")
            return

        print(f"2. Toplam {len(df)} adet canlı telemetri verisiyle model eğitiliyor...")
        
        # Özellikler (Hız, Manyetik Alan, Patlama Olasılığı) ve Hedef (Kp Şiddeti)
        X = df[['speed', 'bz', 'm_prob']]
        y = df['current_kp']
        
        # Modeli Eğit
        self.model.fit(X, y)
        
        # Yeni Modeli Kaydet
        joblib.dump(self.model, self.model_path)
        print(f"3. BAŞARILI: Model kendi kendini güncelledi ve '{self.model_path}' olarak kaydedildi!")

if __name__ == "__main__":
    ai_trainer = SQLiteTrainerAgent()
    ai_trainer.train_from_live_data()