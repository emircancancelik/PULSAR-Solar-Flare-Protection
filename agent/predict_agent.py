import numpy as np
import joblib
from collections import deque
from typing import Dict, Any
from filterpy.kalman import ExtendedKalmanFilter
from filterpy.common import Q_discrete_white_noise

class GNSSInferenceAgent:
    """
    PULSAR Sol Lob: Gerçek Zamanlı GNSS Anomali Tespit Motoru.
    Statik Pandas DataFrame'leri yerine FIFO hafıza tamponları (deque) kullanır.
    """
    def __init__(self, model_path: str = "models/xgboost_gnss_v3.pkl"):
        # 1. Modeli Yükle (DİKKAT: Doğru modeli kaydettiğinden emin ol)
        self.model = joblib.load(model_path)
        
        # 2. Endüstriyel EKF Başlatma (Manevra toleranslı)
        self.ekf = ExtendedKalmanFilter(dim_x=2, dim_z=1)
        self.ekf.F = np.array([[1., 1.0], [0., 1.]])
        self.ekf.x = np.array([[0.], [0.]])
        self.ekf.P *= 500.
        self.ekf.Q = Q_discrete_white_noise(dim=2, dt=1.0, var=2.0)
        self.ekf.R = np.array([[25.0]])

        # 3. Akışkan Hafıza (Streaming Memory Buffers) - Max 15 eleman
        self.window_size = 15
        self.history = {
            'doppler_hz': deque(maxlen=self.window_size),
            'c_n0': deque(maxlen=self.window_size),
            'ekf_residual_abs': deque(maxlen=self.window_size)
        }
        
    def _hx_doppler(self, x): return np.array([x[0]])
    def _H_jacobian_doppler(self, x): return np.array([[1., 0.]])

    def update_and_predict(self, doppler_hz: float, c_n0: float, realtime_kp: float) -> Dict[str, Any]:
        """
        Gelen anlık tekil (1 Hz) telemetri noktasını işler.
        """
        # --- 1. EKF Güncellemesi ---
        self.ekf.predict()
        self.ekf.update(
            np.array([doppler_hz]), 
            HJacobian=self._H_jacobian_doppler, 
            Hx=self._hx_doppler
        )
        current_residual = abs(self.ekf.y[0].item())

        # --- 2. Hafıza Güncellemesi ---
        self.history['doppler_hz'].append(doppler_hz)
        self.history['c_n0'].append(c_n0)
        self.history['ekf_residual_abs'].append(current_residual)

        # Tampon dolana kadar (ilk 15 saniye) sistem nominal döner, kalibrasyon yapar
        if len(self.history['doppler_hz']) < self.window_size:
            return {"anomaly_probability": 0.0, "ekf_residual": current_residual, "status": "CALIBRATING"}

        # --- 3. Canlı Özellik Mühendisliği (Pandas OLMADAN) ---
        hist_doppler = list(self.history['doppler_hz'])
        hist_cn0 = list(self.history['c_n0'])
        hist_res = list(self.history['ekf_residual_abs'])

        # 'EKF_rolling_mean', 'EKF_rolling_std' (Window: 15)
        ekf_roll_mean = np.mean(hist_res)
        ekf_roll_std = np.std(hist_res)

        # 'EKF_residual_diff' (Anlık fark: current - previous)
        ekf_res_diff = hist_res[-1] - hist_res[-2]

        # 'doppler_rate' (current - previous)
        doppler_rate = hist_doppler[-1] - hist_doppler[-2]

        # 'residual_momentum' (current - value 5 steps ago)
        res_momentum = hist_res[-1] - hist_res[-6]

        # 'cn0_std' (Window: 10)
        cn0_roll_std = np.std(hist_cn0[-10:])

        # --- 4. Özellik Vektörü (Senin final_features sıranla BİREBİR AYNI olmalı) ---
        # ['doppler_hz', 'C_N0', 'Kp', 'EKF_residual_abs', 'EKF_rolling_mean', 
        #  'EKF_rolling_std', 'EKF_residual_diff', 'doppler_rate', 'residual_momentum', 'cn0_std']
        feature_vector = np.array([[
            doppler_hz, 
            c_n0, 
            realtime_kp, # Artık random değil, NOAA'dan gelen gerçek veri!
            current_residual, 
            ekf_roll_mean, 
            ekf_roll_std, 
            ekf_res_diff, 
            doppler_rate, 
            res_momentum, 
            cn0_roll_std
        ]])

        # --- 5. Tahmin ---
        # predict_proba çıktısı [[P(0), P(1)]] şeklindedir
        prob_anomaly = self.model.predict_proba(feature_vector)[0][1]

        return {
            "anomaly_probability": float(prob_anomaly),
            "ekf_residual": current_residual,
            "status": "RUNNING"
        }

# Sistem Testi (launcher.py veya main.py üzerinden çağrılacak şekli)
if __name__ == "__main__":
    # Not: Öncesinde defterindeki kodu düzeltip model_v3'ü doğru şekilde kaydetmelisin.
    # Örneğin: joblib.dump(model_v3, "models/xgboost_gnss_v3.pkl")
    
    agent = GNSSInferenceAgent(model_path="models/xgboost_gnss_v3.pkl")
    
    # 20 saniyelik sahte akış simülasyonu
    for sec in range(20):
        # Normal uçuş şartları
        mock_doppler = 1500 + np.random.normal(0, 5)
        mock_cn0 = 45 + np.random.normal(0, 1.5)
        real_kp = 2.0 # NOAA Ingestor'dan gelecek
        
        result = agent.update_and_predict(mock_doppler, mock_cn0, real_kp)
        print(f"T+{sec}sn -> Durum: {result['status']}, İhtimal: {result['anomaly_probability']:.4f}")