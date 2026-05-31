import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
from datetime import datetime
import time
import sqlite3

# --- VERİTABANI BAŞLATMA (Storage Layer) ---
def init_db():
    conn = sqlite3.connect('space_weather_telemetry.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS telemetry
                 (timestamp TEXT, speed REAL, bz REAL, current_kp REAL, m_prob REAL)''')
    conn.commit()
    return conn

conn = init_db()

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="TUA Kriz Yönetimi | Karaman", layout="wide", page_icon="🌍")

st.markdown("<style>.big-font { font-size:24px !important; font-weight: bold; color: #ffcc00; }</style>", unsafe_allow_html=True)
st.title("🛡️ Türkiye Uzay Havası Kriz Yönetim Merkezi")
st.markdown("**Geliştirici:** Karaman Takımı (Çoklu-Ajan Yapay Zeka Orkestrasyonu)")
st.divider()

@st.cache_resource
def load_model():
    try:
        return joblib.load('solar_flare_model.pkl')
    except:
        return None

ai_model = load_model()

# --- VERİ ÇEKME VE VERİTABANINA YAZMA (Data Ingestion) ---
@st.cache_data(ttl=60) # 60 saniyede bir cache patlar ve yeni veri çekilir
def fetch_and_log_data():
    try:
        kp_res = requests.get("https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json", timeout=5).json()
        prob_res = requests.get("https://services.swpc.noaa.gov/json/solar_probabilities.json", timeout=5).json()
        plasma = requests.get("https://services.swpc.noaa.gov/products/solar-wind/plasma-1-day.json", timeout=5).json()
        mag = requests.get("https://services.swpc.noaa.gov/products/solar-wind/mag-1-day.json", timeout=5).json()
        
        current_kp = float(kp_res[-1][1]) if len(kp_res) > 1 else 2.0
        m_prob = prob_res[0].get('m_class_1_day', 5) if prob_res else 5
        current_speed = float(plasma[-1][1]) if len(plasma)>1 and plasma[-1][1] is not None else 400.0
        current_bz = float(mag[-1][3]) if len(mag)>1 and mag[-1][3] is not None else -1.0
        
        # VERİYİ SQLITE'A KAYDET
        c = conn.cursor()
        c.execute("INSERT INTO telemetry VALUES (?, ?, ?, ?, ?)", 
                  (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), current_speed, current_bz, current_kp, m_prob))
        conn.commit()
        
        return current_kp, m_prob, current_speed, current_bz
    except Exception as e:
        return 2.0, 5, 400.0, -1.0

current_kp, m_flare_prob, wind_speed, wind_bz = fetch_and_log_data()

# --- ÜST PANEL: CANLI UZAY TELEMETRİSİ ---
st.subheader("🛰️ L1 Uyduları Canlı Telemetrisi (Otonom Loglama Aktif)")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Gerçekleşen Kp", f"{current_kp:.1f}")
col2.metric("M-Sınıfı Patlama İhtimali", f"%{m_flare_prob}")
col3.metric("Güneş Rüzgarı Hızı", f"{wind_speed:.1f} km/s")
col4.metric("Manyetik Alan (Bz)", f"{wind_bz:.1f} nT")

st.divider()

# --- ORTA PANEL: AI SİMÜLASYONU ---
st.subheader("🔮 Yapay Zeka Kriz Simülatörü")
col_sim, col_risk = st.columns([1, 2])

with col_sim:
    flare_input = st.selectbox("Patlama Senaryosu:", ["C1.0", "M1.0", "M5.0", "X1.0", "X5.0", "X15.0"])
    if ai_model:
        f_class = flare_input.split(" ")[0]
        mapping = {'X': 100, 'M': 10, 'C': 1, 'B': 0.1}
        score = mapping.get(f_class[0], 0) * float(f_class[1:])
        now = datetime.now()
        m_sin, h_sin = np.sin(2 * np.pi * now.month/12), np.sin(2 * np.pi * now.hour/24)
        predicted_kp = ai_model.predict([[score, m_sin, h_sin, now.year]])[0]
    else:
        predicted_kp = 0.0

    st.markdown(f"<p class='big-font'>Tahmini Kp Şiddeti: {predicted_kp:.2f}</p>", unsafe_allow_html=True)
    st.progress(min(predicted_kp / 9.0, 1.0))

with col_risk:
    st.markdown("#### 🏢 Ulusal Altyapı Otonom Emirleri")
    if predicted_kp >= 8.5:
        st.error("🚨 TÜRKSAT: SAFE-MODE'A ALIN! TEİAŞ: PLANLI KESİNTİ BAŞLATIN.")
    elif predicted_kp >= 7.0:
        st.warning("⚠️ TÜRKSAT: Yörünge sürüklenmesi arttı. BOTAŞ: Katodik korumayı artırın.")
    elif predicted_kp >= 5.0:
        st.info("ℹ️ DHMİ: HF Radyo iletişiminde parazitlenme rapor ediliyor.")
    else:
        st.success("✅ TÜM KURUMLAR: Uzay havası stabil.")

# --- OTONOM YENİLEME DÖNGÜSÜ ---
st.divider()
st.caption("Veritabanı: SQLite | Yenileme: 60 Saniyede Bir Otomatik")
time.sleep(60)
st.rerun()