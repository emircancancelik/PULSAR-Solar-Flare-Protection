import scipy.io as sio
import pandas as pd
import numpy as np
import joblib
import os
import xgboost as xgb
import logging
from datetime import datetime, timedelta
from sklearn.frozen import FrozenEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss
from filterpy.kalman import ExtendedKalmanFilter
from filterpy.common import Q_discrete_white_noise

# --- LOGLAMA AYARLARI ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 1. VERİ KAYNAKLARI (MATLAB + SIMULATION) ---

def load_texbat_data(file_path, label):
    """Gerçek MATLAB dosyalarından veri çeker."""
    try:
        data = sio.loadmat(file_path)
        raw_matrix = data['channel']
        df = pd.DataFrame({
            'doppler_hz': raw_matrix[5, :],
            'C_N0': raw_matrix[8, :],
            'label': label
        })
        df['Kp'] = np.random.uniform(1, 3, len(df))
        return df
    except Exception as e:
        logger.warning(f"MATLAB dosyası okunamadı ({file_path}): {e}")
        return pd.DataFrame()

def generate_proxy_data(start_time, duration_seconds=5000, spoofing_start=2500):
    """MATLAB dosyaları yoksa simülasyon verisi üretir."""
    logger.info("Proxy (Simülasyon) verisi üretiliyor...")
    time_index = [start_time + timedelta(seconds=i) for i in range(duration_seconds)]
    cn0 = np.random.normal(45, 1.5, duration_seconds)
    doppler = 1500 * np.sin(np.linspace(0, 2*np.pi, duration_seconds)) + np.random.normal(0, 5, duration_seconds)
    label = np.zeros(duration_seconds)

    if spoofing_start < duration_seconds:
        drag_off = min(120, duration_seconds - spoofing_start)
        cn0[spoofing_start:spoofing_start+drag_off] -= np.linspace(0, 15, drag_off)
        cn0[spoofing_start+drag_off:] -= 15
        doppler[spoofing_start:] += np.linspace(0, 500, duration_seconds - spoofing_start)
        label[spoofing_start:] = 1

    df = pd.DataFrame({'time_tag': time_index, 'C_N0': cn0, 'doppler_hz': doppler, 'label': label})
    df['Kp'] = np.random.uniform(1, 3, duration_seconds)
    return df

# --- 2. SİNYAL İŞLEME ÇEKİRDEĞİ (EKF) ---

def process_features_industrial(df):
    """EKF Residual ve Zamansal Özellikleri çıkarır."""
    def hx(x): return np.array([x[0]])
    def hj(x): return np.array([[1., 0.]])

    ekf = ExtendedKalmanFilter(dim_x=2, dim_z=1)
    ekf.F = np.array([[1., 1.0], [0., 1.]])
    ekf.x = np.array([[0.], [0.]])
    ekf.P *= 500.
    ekf.Q = Q_discrete_white_noise(dim=2, dt=1.0, var=2.0)
    ekf.R = np.array([[25.0]])

    residuals = []
    for z in df['doppler_hz'].values:
        ekf.predict()
        ekf.update(np.array([z]), HJacobian=hj, Hx=hx)
        residuals.append(ekf.y[0].item())

    df = df.copy()
    df['EKF_residual_abs'] = np.abs(residuals)
    df['EKF_residual_diff'] = df['EKF_residual_abs'].diff().fillna(0)
    df['doppler_rate'] = df['doppler_hz'].diff().fillna(0)
    df['residual_momentum'] = df['EKF_residual_abs'].diff(5).fillna(0)
    df['cn0_std'] = df['C_N0'].rolling(window=10).std().fillna(0)

    window_size = 15
    df['EKF_rolling_mean'] = df['EKF_residual_abs'].rolling(window=window_size).mean()
    df['EKF_rolling_std'] = df['EKF_residual_abs'].rolling(window=window_size).std()

    return df.dropna()

# --- 3. EĞİTİM DÖNGÜSÜ ---

if __name__ == "__main__":
    # Veri Toplama
    df_clean = load_texbat_data('channel.mat', label=0)
    df_attack = load_texbat_data('channel-2.mat', label=1)

    if not df_clean.empty and not df_attack.empty:
        logger.info("Gerçek TEXBAT verileriyle eğitime başlanıyor...")
        df_all = pd.concat([df_clean, df_attack], ignore_index=True)
    else:
        df_all = generate_proxy_data(datetime.now())

    # Özellik Çıkarımı
    logger.info("EKF Süzgeci ve Özellik Mühendisliği uygulanıyor...")
    df_final = process_features_industrial(df_all)

    features = [
        'doppler_hz', 'C_N0', 'Kp', 'EKF_residual_abs', 
        'EKF_rolling_mean', 'EKF_rolling_std', 'EKF_residual_diff',
        'doppler_rate', 'residual_momentum', 'cn0_std'
    ]
    
    X = df_final[features]
    y = df_final['label']

    # Eğitim / Test Ayrımı
    split = int(len(X) * 0.75)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # XGBoost + Kalibrasyon
    logger.info("XGBoost eğitiliyor ve kalibre ediliyor...")
    base_model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    base_model.fit(X_train, y_train)

    # Scikit-Learn 1.6+ uyumlu FrozenEstimator kullanımı
    calibrated_model = CalibratedClassifierCV(FrozenEstimator(base_model), method='sigmoid')
    calibrated_model.fit(X_test, y_test)

    # Performans Analizi
    y_prob = calibrated_model.predict_proba(X_test)[:, 1]
    logger.info(f"Nihai ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
    logger.info(f"Brier Skoru: {brier_score_loss(y_test, y_prob):.4f}")

    # Modeli Mühürleme
    os.makedirs('models', exist_ok=True)
    joblib.dump(calibrated_model, 'models/xgboost_gnss.pkl')
    logger.info("PULSAR GNSS Motoru başarıyla mühürlendi: models/xgboost_gnss.pkl")