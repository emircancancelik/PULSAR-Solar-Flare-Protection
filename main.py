import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime

# DIŞARIDAN GELEN AI TAHMİN AJANI (predict_agent.py içindeki)
from predict_agent import PredictionAgent 

# VERİ ÇEKEN AJAN (İsmini SolarWindAgent yapıyoruz ki karışmasın)
class SolarWindAgent:
    def __init__(self):
        self.plasma_url = "https://services.swpc.noaa.gov/products/solar-wind/plasma-1-day.json"
        self.mag_url = "https://services.swpc.noaa.gov/products/solar-wind/mag-1-day.json"

    async def fetch_json(self, session, url):
        try:
            async with session.get(url, timeout=10) as response:
                if response.status == 200:
                    return await response.json()
                return None
        except Exception as e:
            print(f"Bağlantı Hatası: {e}")
            return None

    async def get_processed_data(self):
        async with aiohttp.ClientSession() as session:
            plasma_task = self.fetch_json(session, self.plasma_url)
            mag_task = self.fetch_json(session, self.mag_url)
            plasma_data, mag_data = await asyncio.gather(plasma_task, mag_task)

            if not plasma_data or not mag_data:
                print("Veri çekilemedi.")
                return None

            df_plasma = pd.DataFrame(plasma_data[1:], columns=plasma_data[0])
            df_mag = pd.DataFrame(mag_data[1:], columns=mag_data[0])

            df_plasma.columns = [c.lower() for c in df_plasma.columns]
            df_mag.columns = [c.lower() for c in df_mag.columns]

            if 'bz' not in df_mag.columns:
                bz_alt = [c for c in df_mag.columns if 'bz' in c]
                if bz_alt:
                    df_mag.rename(columns={bz_alt[0]: 'bz'}, inplace=True)

            for col in ['speed', 'bz']:
                if col in df_plasma.columns:
                    df_plasma[col] = pd.to_numeric(df_plasma[col], errors='coerce')
                if col in df_mag.columns:
                    df_mag[col] = pd.to_numeric(df_mag[col], errors='coerce')

            df_plasma['time_tag'] = pd.to_datetime(df_plasma['time_tag'])
            df_mag['time_tag'] = pd.to_datetime(df_mag['time_tag'])
            
            df_plasma['time_tag'] = df_plasma['time_tag'].dt.round('1min')
            df_mag['time_tag'] = df_mag['time_tag'].dt.round('1min')
            combined = pd.merge(df_plasma, df_mag, on='time_tag', how='inner').dropna()
            return combined

class DecisionAI:
    @staticmethod
    def analyze_risk(data):
        if data is None or data.empty:
            return None
        last_row = data.iloc[-1]
        speed = last_row.get('speed', 0)
        bz = last_row.get('bz', 0)
        risk_score = 0
        if speed > 450: risk_score += 30
        if speed > 600: risk_score += 40
        if bz < -5: risk_score += 30
        return {
            "time": last_row['time_tag'],
            "speed": speed,
            "bz": bz,
            "risk_score": min(risk_score, 100),
            "alert_level": "KRİTİK" if (speed > 600 or bz < -10) else "NORMAL"
        }

async def main_orchestrator():
    # Burada predict_agent.py'dan gelen sınıfı kullanıyoruz
    ai_predictor = PredictionAgent('solar_flare_model.pkl')
    solar_agent = SolarWindAgent()
    decision_ai = DecisionAI()
    
    print("\n--- Karaman Güneş Fırtınası Hibrit İzleme Sistemi Başlatıldı ---")
    
    # Simülasyon Patlama Tahmini
    test_flare = "X1.2"
    now = datetime.now()
    kp_prediction = ai_predictor.predict_kp(test_flare, now.year, now.month, now.hour)
    print(f"\n[AI ANALİZ] Olası {test_flare} patlaması için tahmini Kp: {kp_prediction:.2f}")

    # Canlı Veri Döngüsü
    while True:
        data = await solar_agent.get_processed_data()
        if data is not None and not data.empty:
            analysis = decision_ai.analyze_risk(data)
            if analysis:
                print(f"[{analysis['time']}] Canlı Veri -> Hız: {analysis['speed']:.1f} km/s | Bz: {analysis['bz']:.1f} nT | Risk: %{analysis['risk_score']}")
        await asyncio.sleep(60)

if __name__ == "__main__":
    try:
        asyncio.run(main_orchestrator())
    except KeyboardInterrupt:
        print("\nSistem kapatıldı.")