import asyncio
import logging
import numpy as np

from data_pipeline.ingestor import SpaceWeatherIngestor
from core.risk_orchestrator import RiskOrchestrator
from core.ml_engine import SpaceWeatherMLEngine
from agent.predict_agent import GNSSInferenceAgent 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PulsarUnifiedEngine:
    def __init__(self):
        logger.info("PULSAR Ana Orkestratörü Başlatılıyor (Asenkron Çift Lob Mimarisi)...")
        
        # 1. Modüller
        self.ingestor = SpaceWeatherIngestor(timeout_seconds=2.0)
        self.orchestrator = RiskOrchestrator(cost_false_alarm=10.0, cost_missed_detection=100.0)
        
        # 2. Yapay Zeka Ajanları
        self.sw_ai_engine = SpaceWeatherMLEngine(model_path='models/solar_flare_model.pkl')
        self.gnss_ai_engine = GNSSInferenceAgent(model_path='models/xgboost_gnss_v3.pkl') # GERÇEK GNSS MODELİN
        
        # 3. Paylaşımlı Hafıza (State)
        self.current_space_weather_context = {
            'kp_index': 0.0, 
            'xray_flux': 1e-8, 
            'predicted_kp': 0.0
        }

    async def space_weather_loop(self):
        """SAĞ LOB: Düşük frekanslı (örneğin 60 saniyede bir) NOAA API döngüsü"""
        while True:
            try:
                # NOAA'dan canlı veri çek
                raw_context = await self.ingestor.get_current_context()
                
                # ML Çıkarımı: Gelecek Fırtına
                predicted_kp = self.sw_ai_engine.predict_future_kp(raw_context['xray_flux'])
                
                # Global state'i güncelle (GNSS motoru burayı okuyacak)
                self.current_space_weather_context = {
                    'kp_index': raw_context.get('kp_index', 0.0),
                    'xray_flux': raw_context.get('xray_flux', 1e-8),
                    'predicted_kp': predicted_kp
                }
                logger.info(f"[SAĞ LOB GÜNCELLENDİ] Canlı Kp: {raw_context['kp_index']} | Tahmini Kp: {predicted_kp:.2f}")
                
                await asyncio.sleep(60.0) # Uzay havası yavaş değişir, sistemi yorma
            except Exception as e:
                logger.error(f"Uzay Havası Döngüsü Hatası: {e}")
                await asyncio.sleep(10.0)

    async def gnss_control_loop(self):
        """SOL LOB ve ORKESTRATÖR: Yüksek frekanslı (1 Hz / 10 Hz) İHA telemetri döngüsü"""
        # EKF'nin başlangıçta kalibre olması için biraz süre tanı
        await asyncio.sleep(2.0) 
        
        while True:
            try:
                # 1. İHA'dan gelen anlık sensör verisini simüle et (Gerçek sistemde MAVLink'ten gelir)
                current_doppler = 1500 + np.random.normal(0, 5)
                current_cn0 = 45 + np.random.normal(0, 1.5)
                
                # 2. SAĞ LOB'dan gelen en güncel 'predicted_kp' değerini al
                # Dikkat: İki yapay zeka burada birbirine veri aktarıyor!
                current_predicted_kp = self.current_space_weather_context['predicted_kp']
                
                # 3. SOL LOB (GNSS) Çıkarımı (random.uniform YERİNE EKF + XGBoost kullanıyoruz)
                gnss_state = self.gnss_ai_engine.update_and_predict(
                    doppler_hz=current_doppler, 
                    c_n0=current_cn0, 
                    realtime_kp=current_predicted_kp
                )
                
                if gnss_state["status"] == "CALIBRATING":
                    logger.debug("EKF Matrisleri kalibre ediliyor...")
                    await asyncio.sleep(1.0)
                    continue

                # 4. KARAR MOTORU: İki yapay zekanın çıktısını birleştir
                decision = self.orchestrator.evaluate_system_state(
                    gnss_anomaly_prob=gnss_state["anomaly_probability"], 
                    space_weather_context=self.current_space_weather_context
                )
                
               # 5. Eylem (Endüstriyel Loglama)
                logger.info(f"TELEMETRY | Doppler: {current_doppler:.1f} Hz | C/N0: {current_cn0:.1f} dB-Hz | EKF_Res: {gnss_state['ekf_residual']:.2f}")
                logger.info(f"AI_STATE  | GNSS_Prob: {gnss_state['anomaly_probability']:.4f} | Kp_Cur: {self.current_space_weather_context['kp_index']} | Kp_Pred: {current_predicted_kp:.2f}")
                
                if decision['threat_classification'] != "NOMINAL":
                    logger.warning(f"THREAT DETECTED | Class: {decision['threat_classification']} | Action: {decision['hardware_command']}")
                else:
                    logger.info(f"STATUS NOMINAL | Action: {decision['hardware_command']}")
                
                await asyncio.sleep(1.0) # İHA karar döngüsü saniyede 1 kez çalışır
                
            except Exception as e:
                logger.error(f"GNSS Kontrol Döngüsü Hatası: {e}")
                await asyncio.sleep(1.0)

    async def run(self):
        """Asenkron görevleri paralel olarak ateşler"""
        await asyncio.gather(
            self.space_weather_loop(),
            self.gnss_control_loop()
        )

if __name__ == "__main__":
    pulsar = PulsarUnifiedEngine()
    try:
        asyncio.run(pulsar.run())
    except KeyboardInterrupt:
        print("\n[!] PULSAR Motoru manuel olarak durduruldu.")