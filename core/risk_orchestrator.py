import logging
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any

logger = logging.getLogger(__name__)

@dataclass
class TurkeyGeoBounds:
    """Türkiye'nin coğrafi ve basitleştirilmiş geomanyetik sınırları"""
    LAT_MIN: float = 36.0
    LAT_MAX: float = 42.0
    LON_MIN: float = 26.0
    LON_MAX: float = 45.0
    GEOMAGNETIC_LAT_OFFSET: float = 29.5 

class RiskOrchestrator:
    """
    PULSAR Hibrit Karar Motoru.
    GNSS ML çıktıları, Uzay Havası tahminleri ve Yörünge mekaniğini birleştirerek
    Beklenen Maliyet (Expected Cost) üzerinden otonom donanım komutları üretir.
    """
    def __init__(self, cost_false_alarm: float = 10.0, cost_missed_detection: float = 100.0):
        # 1. UAV/GNSS Maliyet Matrisi
        self.c_fa = cost_false_alarm
        self.c_miss = cost_missed_detection
        self.decision_threshold = self.c_fa / (self.c_fa + self.c_miss)
        
        # 2. Uzay Havası Kritik Eşikleri
        self.KP_STORM_THRESHOLD = 5.0      # G1 seviyesi Geomanyetik Fırtına
        self.XRAY_FLARE_THRESHOLD = 1e-5   # M-Class Solar Flare (10^-5 W/m^2)
        
        # 3. Fiziksel Sabitler (Uydu ve GIC için)
        self.earth_radius_km = 6371.0
        self.mu_earth = 398600.4418        # Yerçekimi parametresi (km^3/s^2)

    def _calibrate_probability(self, raw_prob: float) -> float:
        """
        Platt/Isotonic Scaling kalibrasyonu.
        XGBoost'un ECE (Expected Calibration Error) optimizasyonu için rezerve edilmiştir.
        """
        return float(np.clip(raw_prob, 0.0, 1.0))

    def _calculate_local_db_dt(self, predicted_kp: float, predicted_bz: float) -> float:
        """Türkiye lokasyonu için tahmini manyetik alan değişim hızı (dB/dt) hesaplar."""
        base_fluctuation = np.exp(0.5 * predicted_kp) 
        bz_impact = abs(predicted_bz) if predicted_bz < 0 else 0 
        lat_factor = (TurkeyGeoBounds.LAT_MAX + TurkeyGeoBounds.LAT_MIN) / 2 / 90.0
        
        return (base_fluctuation + (bz_impact * 2.0)) * lat_factor

    def evaluate_uav_gnss_state(self, gnss_anomaly_prob: float, realtime_context: Dict[str, float]) -> Dict[str, Any]:
        """İHA ve Uçuş Sistemleri için Spoofing vs Uzay Havası ayrımı (Real-time)"""
        kp_index = realtime_context.get('kp_index', 0.0)
        xray_flux = realtime_context.get('xray_flux', 1e-8)
        
        calibrated_risk = self._calibrate_probability(gnss_anomaly_prob)
        is_space_weather_event = (kp_index >= self.KP_STORM_THRESHOLD) or (xray_flux >= self.XRAY_FLARE_THRESHOLD)
        is_risk_exceeded = calibrated_risk >= self.decision_threshold

        if is_risk_exceeded and is_space_weather_event:
            threat = "IONOSPHERIC_SCINTILLATION"
            action = "SWITCH_TO_INS_ONLY"
            diag = "Yüksek anomali. Kök neden: Uzay Havası."
        elif is_risk_exceeded and not is_space_weather_event:
            threat = "ELECTRONIC_WARFARE_SPOOFING"
            action = "SWITCH_TO_VISUAL_ODOMETRY"
            diag = "Yüksek anomali. Kök neden: Elektronik Harp (Spoofing)."
        elif 0.3 <= calibrated_risk < self.decision_threshold and not is_space_weather_event:
            threat = "RF_JAMMING_WARNING"
            action = "INCREASE_EKF_R_COV"
            diag = "Orta seviye anomali. Olası Jamming. EKF (R) matrisi esnetiliyor."
        else:
            threat = "NOMINAL"
            action = "CONTINUE_MISSION"
            diag = "Sistem stabil."

        return {
            "classification": threat,
            "hardware_command": action,
            "calibrated_risk": round(calibrated_risk, 4),
            "diagnostics": diag
        }

    def evaluate_pipeline_gic_risk(self, forecast_context: Dict[str, float]) -> Dict[str, Any]:
        """Türkiye doğalgaz boru hatları için 30 dk sonrasının Katodik Koruma kararı"""
        pred_kp = forecast_context.get('predicted_kp', 0.0)
        pred_bz = forecast_context.get('predicted_bz', 0.0)
        
        db_dt = self._calculate_local_db_dt(pred_kp, pred_bz)
        
        if db_dt > 100.0:
            action = "CRITICAL_GIC_ALERT: Katodik koruma voltajını artır."
            status = "CRITICAL"
        elif db_dt > 50.0:
            action = "WARNING: 30 dk içinde voltaj dalgalanması bekleniyor."
            status = "ELEVATED"
        else:
            action = "NOMINAL: Katodik koruma standart seviyede."
            status = "NORMAL"
            
        return {"status": status, "est_db_dt_nT_min": round(db_dt, 2), "action": action}

    def calculate_satellite_delta_v(self, current_alt_km: float, target_alt_km: float, 
                                    forecast_context: Dict[str, float], mass_kg: float = 250.0, area_m2: float = 2.5) -> Dict[str, Any]:
        """Güneş fırtınası kaynaklı yörünge kaybını önlemek için Delta-V hesaplaması"""
        f107 = forecast_context.get('predicted_f107', 90.0)
        
        if f107 < 150:
            return {"status": "NOMINAL", "delta_v_m_s": 0.0, "action": "İtki gereksinimi yok."}

        r_current = self.earth_radius_km + current_alt_km
        r_target = self.earth_radius_km + target_alt_km
        
        v_current = np.sqrt(self.mu_earth / r_current)
        v_target = np.sqrt(self.mu_earth / r_target)
        
        delta_v_m_s = abs(v_target - v_current) * 1000.0
        
        # Balistik katsayı bazlı risk ağırlığı (Drag denklemi basitleştirmesi)
        ballistic_coef = mass_kg / area_m2
        risk_multiplier = 1.0 + (f107 / 200.0) if ballistic_coef < 50 else 1.0
        required_dv = delta_v_m_s * risk_multiplier

        return {
            "status": "ORBIT_DECAY_WARNING", 
            "delta_v_m_s": round(required_dv, 3), 
            "action": f"30 dk tahmini: Yörünge kaybını engellemek için {round(required_dv, 3)} m/s itki planla."
        }

    def process_full_infrastructure(self, gnss_prob: float, realtime_ctx: Dict[str, float], forecast_ctx: Dict[str, float]) -> Dict[str, Any]:
        """Tüm otonom sistemleri koordine eden ana orkestrasyon fonksiyonu"""
        return {
            "timestamp_eval": "Real-time & T+30m Forecast",
            "uav_gnss_subsystem": self.evaluate_uav_gnss_state(gnss_prob, realtime_ctx),
            "pipeline_gic_subsystem": self.evaluate_pipeline_gic_risk(forecast_ctx),
            "leo_satellite_subsystem": self.calculate_satellite_delta_v(
                current_alt_km=400.0, target_alt_km=410.0, forecast_context=forecast_ctx
            )
        }

if __name__ == "__main__":
    orchestrator = RiskOrchestrator()
    
    # Anlık Telemetri (Sensör & Güncel NOAA API)
    realtime_data = {'kp_index': 1.33, 'xray_flux': 1.81e-06}
    gnss_anom_prob = 0.85 
    
    # 30 Dakika Sonrasının Tahmini (Güneş Fırtınası Modeli Çıktısı)
    forecast_data = {
        'predicted_kp': 6.5,     # Fırtına patlıyor
        'predicted_bz': -15.0,   # Manyetosfer yarılıyor
        'predicted_f107': 180.0  # Atmosfer genleşecek
    }
    
    decisions = orchestrator.process_full_infrastructure(gnss_anom_prob, realtime_data, forecast_data)
    
    print("\n--- PULSAR UNIFIED RESILIENCE ARCHITECTURE ---")
    for subsystem, output in decisions.items():
        if isinstance(output, dict):
            print(f"\n[{subsystem.upper()}]")
            for k, v in output.items():
                print(f"  - {k}: {v}")
        else:
            print(f"{subsystem}: {output}")