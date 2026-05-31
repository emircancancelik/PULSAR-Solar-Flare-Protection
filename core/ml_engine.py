import joblib
import numpy as np
import logging
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
logger = logging.getLogger(__name__)

class SpaceWeatherMLEngine:
    """
    PULSAR Erken Uyarı Yapay Zeka Motoru.
    Güneş patlaması X-Ray akısını kullanarak gelecek olan Geomanyetik Fırtınanın (Kp)
    büyüklüğünü tahmin eder.
    """
    def __init__(self, model_path: str = 'models/solar_flare_model.pkl'):
        try:
            self.model = joblib.load(model_path)
            logger.info(f"AI Modeli başarıyla yüklendi: {model_path}")
        except FileNotFoundError:
            logger.error(f"Kritik Hata: {model_path} bulunamadı! Sistem simülasyon moduna geçiyor.")
            self.model = None

    def predict_future_kp(self, current_xray_flux: float) -> float:
        """
        Anlık X-Ray akısını alır, modelin beklediği özelliklere (features) çevirir
        ve tahmini Kp değerini döndürür.
        """
        if self.model is None:
            return 0.0

        # Fiziksel Dönüşüm: X-Ray Flux (W/m^2) -> Modelin anladığı Score
        # X1.0 sınıfı = 10^-4 W/m^2. Bunu 100 skoruna çevirmek için 1e6 ile çarpıyoruz.
        # Örn: 1e-5 (M1.0) * 1e6 = 10 (Modelin M sınıfı için beklediği skor)
        score = current_xray_flux * 1e6

        # Zamansal Döngüsel Özellikler (Cyclical Time Features)
        now = datetime.now()
        m_sin = np.sin(2 * np.pi * now.month / 12.0)
        h_sin = np.sin(2 * np.pi * now.hour / 24.0)
        year = now.year

        # Model Çıkarımı (Inference)
        features = np.array([[score, m_sin, h_sin, year]])
        try:
            predicted_kp = self.model.predict(features)[0]
            # Kp değeri fiziksel olarak 0 ile 9 arasında olmak zorundadır
            return float(np.clip(predicted_kp, 0.0, 9.0))
        except Exception as e:
            logger.error(f"Tahmin motoru çöktü: {e}")
            return 0.0