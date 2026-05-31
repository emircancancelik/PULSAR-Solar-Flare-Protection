import joblib
import numpy as np
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ALAZSpaceWeatherAgent:
    """
    PULSAR Sağ Lob: Uzay Havası Tahmin ve Bağlam Motoru.
    Güneş Rüzgarı (DSCOVR) verilerinden Kp, GOES verilerinden Flare etkisini hesaplar.
    """
    def __init__(self, kp_model_path: str = 'models/kp_predictor.pkl', flare_model_path: str = 'models/flare_impact.pkl'):
        try:
            # GELECEK VİZYONU: İki ayrı model olmalı
            # 1. Kp Tahmin Modeli (Girdi: Güneş Rüzgarı Hızı, Yoğunluk, Bz)
            # self.kp_model = joblib.load(kp_model_path) 
            
            # 2. Mevcut modelin (Geçici olarak kullanıyoruz ama acilen fiziksel verilerle yeniden eğitilmeli)
            self.legacy_model = joblib.load(kp_model_path)
            logger.info("ALAZ Uzay Havası Ajanı sistem belleğine yüklendi.")
        except Exception as e:
            logger.critical(f"FATAL ERROR: Uzay Havası modeli yüklenemedi. Sistem kör uçuşta! Detay: {e}")
            raise SystemExit(1) # Kritik bir hata varsa sistemi başlatma, durdur.

    def _parse_flare_class(self, flare_class: str) -> float:
        """Güneş patlaması sınıfını logaritmik enerji akısına çevirir (W/m^2)."""
        if not flare_class or not isinstance(flare_class, str):
            raise ValueError("Geçersiz flare sınıfı verisi.")
            
        mapping = {'X': 1e-4, 'M': 1e-5, 'C': 1e-6, 'B': 1e-7, 'A': 1e-8}
        scale = flare_class[0].upper()
        
        if scale not in mapping:
            return 1e-8 # Nominal değer
            
        try:
            value = float(flare_class[1:])
            # Gerçek X-Ray Flux hesaplaması (W/m^2)
            return mapping[scale] * value
        except ValueError:
            logger.warning(f"Flare değeri parse edilemedi: {flare_class}. Nominal değere dönülüyor.")
            return 1e-8

    def predict_context(self, flare_class: str, wind_speed: float, bz: float, density: float) -> Dict[str, Any]:
        """
        Orkestratörün (RiskOrchestrator) beklediği 30 dakikalık uzay havası bağlamını (context) üretir.
        """
        # 1. GNSS / İyonosferik Gürültü Tahmini (Flares -> D-RAP)
        xray_flux = self._parse_flare_class(flare_class)
        flare_prob = 1.0 if xray_flux >= 1e-5 else 0.0 # M sınıfı ve üzeri tehlikelidir
        
        # 2. Kp / Geomanyetik Fırtına Tahmini (Solar Wind -> Kp)
        # NOT: Mevcut legacy_model saat ve tarih istiyor. Bu geçici kodu buraya gömüyoruz
        # ancak gerçek model wind_speed, bz ve density almalı!
        
        # --- GEÇİCİ KOD (Senin modelini kırmamak için) ---
        # time_now = datetime.utcnow()
        # month_sin = np.sin(2 * np.pi * time_now.month / 12)
        # hour_sin = np.sin(2 * np.pi * time_now.hour / 24)
        # legacy_score = mapping_logic_here...
        # predicted_kp = self.legacy_model.predict([[legacy_score, month_sin, hour_sin, time_now.year]])[0]
        # ------------------------------------------------
        
        # GERÇEK OLMASI GEREKEN (Fiziksel Kp tahmini)
        # Kp, manyetik alanın (Bz) güneye dönmesi ve güneş rüzgarının hızlanmasıyla artar.
        # Bu basit ampirik bir yedek (fallback) algoritmasıdır:
        estimated_kp = 0.0
        if bz < 0: # Manyetosfer yırtılması
            estimated_kp = (wind_speed / 100) + abs(bz) * 0.5
        else:
            estimated_kp = (wind_speed / 200)
            
        predicted_kp = float(np.clip(estimated_kp, 0.0, 9.0)) # Kp skalası 0-9 arasıdır

        return {
            "xray_flux_w_m2": xray_flux,
            "flare_probability": flare_prob,
            "predicted_kp": predicted_kp,
            "predicted_bz": bz,
            "predicted_f107": 150.0 # Geçici statik değer
        }

# Modül Testi
if __name__ == "__main__":
    # Konsol çıktısı için log ayarı
    logging.basicConfig(level=logging.INFO)
    
    # Model dosyası henüz yoksa patlamaması için mock/fake path verelim
    agent = ALAZSpaceWeatherAgent(kp_model_path="dummy_path.pkl")
    
    # NOAA'dan gelen gerçek zamanlı bir X1.2 patlaması ve fırtına verisi simülasyonu
    context = agent.predict_context(
        flare_class="X1.2",
        wind_speed=650.0,  # km/s (Hızlı)
        bz=-12.5,          # nT (Güneye yönlü, tehlikeli)
        density=15.0       # particles/cm^3
    )
    
    print("\nÜRETİLEN UZAY HAVASI BAĞLAMI:")
    for k, v in context.items():
        print(f" - {k}: {v}")