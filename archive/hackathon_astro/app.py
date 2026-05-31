import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
from datetime import datetime
import sqlite3
import warnings

warnings.filterwarnings('ignore')

# --- MUST BE FIRST STREAMLIT CALL ---
st.set_page_config(page_title="ALAZ | Uzay Havası Kriz Yönetimi", layout="wide")

st.markdown("""
<style>
    .reportview-container { background: #0e1117; }
    .big-font { font-size: 22px !important; font-weight: 600; color: #00ffcc; border-left: 4px solid #00ffcc; padding-left: 10px;}
    .section-header { font-size: 18px; font-weight: bold; color: #a3a8b8; text-transform: uppercase; letter-spacing: 1.5px; border-bottom: 1px solid #333; padding-bottom: 5px;}
    .status-box { padding: 20px; border-radius: 10px; border: 1px solid #333; background-color: #161b22; text-align: center; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

# --- MODEL DEFINITIONS ---
MODELS = {
    "Fiziksel LEO Motoru (Orijinal)": "solar_flare_model.pkl",
}

@st.cache_resource
def load_alaz_models():
    loaded_models = {}
    for name, path in MODELS.items():
        try:
            loaded_models[name] = joblib.load(path)
        except Exception as e:
            loaded_models[name] = None
            st.sidebar.warning(f"⚠️ {name} yüklenemedi: {e}")
    return loaded_models

all_models = load_alaz_models()
# Ana tahmin modeli (T+30 için)
flare_model_name = "Fiziksel LEO Motoru (Orijinal)"
flare_model      = all_models.get(flare_model_name)
MODEL_FEATURES = {
    "Fiziksel LEO Motoru (Orijinal)":    ['kp_value', 'Yorunge_Egilimi_Inclo', 'Ortalama_Hareket_NoKozai'],
    "Gelişmiş Hasar Tahmini (Advanced)": ['kp_value', 'Yorunge_Egilimi_Inclo', 'Ortalama_Hareket_NoKozai'],
    "Uydu Outlier Veri Tespiti":         ['speed', 'bz', 'm_prob'],
    "Best Classification Model":         ['kp_value', 'Yorunge_Egilimi_Inclo', 'Ortalama_Hareket_NoKozai'],
}

# --- SIDEBAR: MODEL SELECTOR ---
st.sidebar.title("🛡️ ALAZ AI Kontrol Paneli")

available_models = [name for name, m in all_models.items() if m is not None]
st.sidebar.success(f"Yüklenen model sayısı: {len(available_models)}")

if not available_models:
    st.error("❌ Hiçbir model yüklenemedi. .pkl dosyalarının app.py ile aynı klasörde olduğundan emin ol.")
    st.stop()

selected_model_name = st.sidebar.selectbox("Aktif Yapay Zeka Beyni:", available_models)
active_model = all_models.get(selected_model_name)

# --- SIDEBAR: DEEP SPACE SENSORS ---
st.sidebar.subheader("📡 Derin Uzay Sensörleri (DSCOVR)")
s_speed = st.sidebar.number_input("Güneş Rüzgarı Hızı (km/s)", value=400.0)
s_bz    = st.sidebar.slider("Manyetik Alan Yönü (Bz)", -50.0, 50.0, 0.0)
s_prob  = st.sidebar.number_input("Manyetik Olasılık (m_prob)", value=0.1)

if st.sidebar.button("Erken Uyarı Analizi"):
    tele_model = all_models.get("Uydu Outlier Veri Tespiti")
    if tele_model:
        input_tele = pd.DataFrame([[s_speed, s_bz, s_prob]], columns=['speed', 'bz', 'm_prob'])
        try:
            impact_score = tele_model.predict(input_tele)[0]
            st.sidebar.warning(f"🚨 Beklenen Etki Skoru: {impact_score:.2f}")
            if s_bz < -10:
                st.sidebar.error("⚠️ Bz Negatif! Manyetik kalkan açıldı, enerji transferi yüksek!")
        except Exception as e:
            st.sidebar.error(f"Tahmin hatası: {e}")
    else:
        st.sidebar.error("Telemetri modeli yüklenemedi.")

# --- DATABASE ---
if "operation_logs" not in st.session_state:
    st.session_state.operation_logs = []

def init_db():
    conn = sqlite3.connect('space_weather_telemetry.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS telemetry
                 (timestamp TEXT, speed REAL, bz REAL, current_kp REAL, m_prob REAL)''')
    conn.commit()
    return conn

conn = init_db()

# --- DATA FETCHING ---
# --- DATA FETCHING (OTONOM & API-DIRENÇLİ HAT) ---
@st.cache_data(ttl=60)
def fetch_and_log_data():
    try:
        kp_url = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"
        kp_data = requests.get(kp_url, timeout=5).json()
        
        # JSON'ı DataFrame'e çevir ve Kp sütununu garantile
        df_kp = pd.DataFrame(kp_data[1:], columns=kp_data[0])
        current_kp = float(df_kp.iloc[-1]['Kp']) # Her zaman en güncel (son) satırı alır

        # 2. Diğer Uzay Havası Sensörleri
        prob_res = requests.get("https://services.swpc.noaa.gov/json/solar_probabilities.json", timeout=5).json()
        plasma   = requests.get("https://services.swpc.noaa.gov/products/solar-wind/plasma-1-day.json", timeout=5).json()
        mag      = requests.get("https://services.swpc.noaa.gov/products/solar-wind/mag-1-day.json", timeout=5).json()

        m_prob        = prob_res[0].get('m_class_1_day', 5) if prob_res else 5
        current_speed = float(plasma[-1][1])  if len(plasma) > 1 and plasma[-1][1] is not None else 400.0
        current_bz    = float(mag[-1][3])     if len(mag)    > 1 and mag[-1][3]    is not None else -1.0

        # 3. SQLite Veritabanına Otonom Loglama
        c = conn.cursor()
        c.execute("INSERT INTO telemetry VALUES (?, ?, ?, ?, ?)",
                  (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), current_speed, current_bz, current_kp, m_prob))
        conn.commit()

        return current_kp, m_prob, current_speed, current_bz
        
    except Exception as e:
        # API çökerse sistemi kilitlememek için Fallback (Yedek) değerler
        st.error(f"⚠️ Otonom Veri Çekme Hatası (Sensör Kaybı): {e}")
        return 2.0, 5, 400.0, -1.0

current_kp, m_flare_prob, wind_speed, wind_bz = fetch_and_log_data()

# --- LIVE SATELLITE DATA ---
ACTIVE_FLEET = {
    56211: "İMECE (Gözlem - LEO)",
    60232: "TÜRKSAT 6A (Haberleşme - GEO)",
    47344: "TÜRKSAT 5B (Haberleşme - GEO)",
    39152: "TÜRKSAT 4A (Haberleşme - GEO)",
    38991: "GÖKTÜRK-2 (Gözlem - LEO)"
}

@st.cache_data(ttl=3600)
def fetch_real_orbit_data():
    url = "https://www.space-track.org/basicspacedata/query/class/gp/COUNTRY_CODE/TURK/EPOCH/%3Enow-7/OBJECT_TYPE/PAYLOAD/DECAY_DATE/null-val/orderby/NORAD_CAT_ID%20asc/emptyresult/show"
    
    try:
        df = pd.read_csv(url + "/format/csv", timeout=5)
        df = df.rename(columns={
            'OBJECT_NAME': 'Uydu Adı',
            'NORAD_CAT_ID': 'NORAD ID',
            'MEAN_MOTION': 'Hız (Tur/Gün)',
            'BSTAR': 'Sürtünme (B* Drag)',
            'INCLINATION': 'Eğim (Derece)'
        })
        return df[['Uydu Adı', 'NORAD ID', 'Hız (Tur/Gün)', 'Sürtünme (B* Drag)', 'Eğim (Derece)']]
    except Exception as e:
        st.sidebar.error(f"Uydu verisi çekilemedi: {e}")
        backup_data = [
            {"Uydu Adı": "İMECE", "NORAD ID": 56211, "Hız (Tur/Gün)": 15.06, "Sürtünme (B* Drag)": 0.0001, "Eğim (Derece)": 97.4},
            {"Uydu Adı": "TÜRKSAT 6A", "NORAD ID": 60232, "Hız (Tur/Gün)": 1.00, "Sürtünme (B* Drag)": 0.0000, "Eğim (Derece)": 0.0}
        ]
        return pd.DataFrame(backup_data)

# --- HEADER & G-SCALE PANEL ---
st.markdown("<h1 style='text-align: center; color: #ffffff;'>ALAZ SİSTEMİ | MERKEZİ KOMUTA EKRANI</h1>", unsafe_allow_html=True)
st.caption("Geliştirici: Karaman Takımı | Durum: Otonom MLOps Aktif")

g_class, g_desc, color = "G0", "Nominal Durum", "#00ffcc"
if current_kp >= 9:   g_class, g_desc, color = "G5 (Extreme)", "Tam İzolasyon ve Güvenli Mod",         "#ff0000"
elif current_kp >= 8: g_class, g_desc, color = "G4 (Severe)",  "Kritik Yük Atma ve Korozyon Kontrolü", "#ff4b4b"
elif current_kp >= 7: g_class, g_desc, color = "G3 (Strong)",  "Trafo Koruma ve Frekans Takibi",        "#ffa500"
elif current_kp >= 6: g_class, g_desc, color = "G2 (Moderate)","Yüksek Enlem Voltaj Alarmı",            "#ffff00"
elif current_kp >= 5: g_class, g_desc, color = "G1 (Minor)",   "Havacılık ve HF İletişim Takibi",       "#00ffcc"

st.markdown(f"""
<div class="status-box" style="text-align: center;">
    <div style="color: #a3a8b8; font-size: 14px; text-transform: uppercase; letter-spacing: 2px;">NOAA Standartları Otonom Karar Mekanizması</div>
    <h1 style="color: {color}; margin: 15px 0; font-size: 48px; text-align: center;">{g_class}</h1>
    <h3 style="color: white; margin-bottom: 20px; text-align: center;">{g_desc}</h3>
    <table style="width:100%; color: white; border-collapse: collapse; font-size: 14px; background: rgba(255,255,255,0.05); border-radius: 10px;">
        <tr style="border-bottom: 1px solid #444;">
            <th style="padding: 12px; text-align: left;">Stratejik Kurum</th>
            <th style="padding: 12px; text-align: left;">Otonom Müdahale Protokolü</th>
            <th style="padding: 12px; text-align: right;">Sistem Durumu</th>
        </tr>
        <tr>
            <td style="padding: 12px;">⚡ TEİAŞ (Enerji)</td>
            <td style="padding: 12px;">{'Kritik Hat İzolasyonu' if current_kp >= 8 else 'Voltaj Stabilizasyonu'}</td>
            <td style="padding: 12px; text-align: right; color: {color}; font-weight: bold;">● AKTİF İZLEME</td>
        </tr>
        <tr>
            <td style="padding: 12px;">🔥 BOTAŞ (Boru Hattı)</td>
            <td style="padding: 12px;">{'Katodik Koruma Maksimum' if current_kp >= 7 else 'Nominal Korozyon Takibi'}</td>
            <td style="padding: 12px; text-align: right; color: {color}; font-weight: bold;">● AKTİF İZLEME</td>
        </tr>
        <tr>
            <td style="padding: 12px;">✈️ DHMİ (Havacılık)</td>
            <td style="padding: 12px;">{'Polar Rota Kısıtlaması' if current_kp >= 5 else 'Serbest Rota'}</td>
            <td style="padding: 12px; text-align: right; color: {color}; font-weight: bold;">● AKTİF İZLEME</td>
        </tr>
    </table>
</div>
""", unsafe_allow_html=True)

# --- METRICS ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Gerçekleşen Kp",   f"{current_kp:.1f}")
col2.metric("Patlama İhtimali", f"%{m_flare_prob}")
col3.metric("Rüzgar Hızı",      f"{wind_speed:.1f} km/s")
col4.metric("Manyetik Alan",    f"{wind_bz:.1f} nT")

# --- AI FUTURE PREDICTION (T+30) BÖLÜMÜ ---
st.markdown("<h4 style='color: #a3a8b8;'>🧠 ALAZ AI | GELECEK 30 DK ÖNGÖRÜSÜ</h4>", unsafe_allow_html=True)
ai_col1, ai_col2 = st.columns(2)

# Ajanı veya doğrudan modeli kullanarak 3 parametreli tahmin yapıyoruz
if flare_model:
    future_kp = current_kp + 0.5 
    
    try:
        macro_pred = flare_model.predict([[future_kp, 97.4, 15.06]])[0]
        scientific_pred = macro_pred / 100000 
        ai_col1.metric("Tahmini Hasar (BSTAR)", f"{scientific_pred:.6f}")
        
        status_msg = "STABİL" if macro_pred < 50 else "RİSKLİ" if macro_pred < 150 else "TEHLİKELİ"
        status_color = "#00ffcc" if status_msg == "STABİL" else "#ffa500" if status_msg == "RİSKLİ" else "#ff4b4b"
        
        ai_col2.markdown(f"<h3 style='color: {status_color};'>{status_msg}</h3>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Tahmin motoru hatası: {e}")
else:
    ai_col1.warning(f"⚠️ '{flare_model_name}' yüklenemedi.")
st.divider()

# --- ACTIVE MODEL: DAMAGE ANALYSIS ---
st.title(f"🛰️ {selected_model_name} Canlı Yayında")

col1, col2 = st.columns(2)
with col1:
    kp   = st.slider("G-Skalası (Kp Index)", 0, 90, 40) / 10.0
    f107 = st.number_input("Solar Radyo Akısı (F10.7)", value=150.0)
with col2:
    inc = st.number_input("Yörünge Eğimi (Inclin.)", value=97.0)
    mm  = st.number_input("Günlük Tur (Mean Motion)", value=15.0)
# --- ACTIVE MODEL: DAMAGE ANALYSIS BÖLÜMÜ ---
if st.button("Hasar Analizi Yap"):
    if active_model:
        features = MODEL_FEATURES.get(selected_model_name, ['kp_value', 'Yorunge_Egilimi_Inclo', 'Ortalama_Hareket_NoKozai'])
        
        if selected_model_name == "Uydu Outlier Veri Tespiti":
            input_data = pd.DataFrame([[wind_speed, wind_bz, m_flare_prob]], columns=features)
        else:
            input_data = pd.DataFrame([[kp, inc, mm]], columns=features)

        try:
            # Hesaplama
            macro_prediction = active_model.predict(input_data)[0]
            scientific_prediction = macro_prediction / 100000
            st.metric("Tahmin Edilen Hasar (Delta BSTAR)", f"{scientific_prediction:.8f}")
            if macro_prediction > 150.0:
                st.error("🚨 KRİTİK SEVİYE: Atmosferik sürtünme limitleri aşıldı!")
            else:
                st.success("✅ GÜVENLİ: Yörünge stabilitesi korunuyor.")
        except Exception as e:
            st.error(f"Buton Tahmin Hatası: {e}")

# --- REAL-TIME DATA STREAM ---
st.markdown("<div class='section-header'>📊 OTONOM VERİ AKIŞI (REAL-TIME)</div>", unsafe_allow_html=True)
df_logs = pd.read_sql("SELECT * FROM telemetry ORDER BY timestamp DESC LIMIT 20", conn)
c_chart, c_table = st.columns([2, 1])
with c_chart:
    if not df_logs.empty:
        st.area_chart(df_logs.set_index('timestamp')['speed'], color="#00ffcc")
    else:
        st.info("Henüz veri yok.")
with c_table:
    if not df_logs.empty:
        st.table(df_logs[['timestamp', 'speed', 'bz', 'current_kp']].head(5))

# --- TABS ---
tab_orbit, tab_earth, tab_ai = st.tabs(["[ UYDULAR ]", "[ ALTYAPI ]", "[ AI MOTORU ]"])

with tab_orbit:
    st.markdown("<div class='section-header'>GLOBAL UYDU RİSK İZLEME EKRANI</div>", unsafe_allow_html=True)
    satellite_db = fetch_real_orbit_data()

    if satellite_db is not None and not satellite_db.empty:
        st.selectbox("Görünüm:", ["Türkiye Aktif Filosu", "Tümü"])
        risk_status = "NOMİNAL"
        if current_kp >= 7:   risk_status = "YÜKSEK RİSK"
        elif current_kp >= 5: risk_status = "ORTA RİSK"
        display_df = satellite_db.copy()
        display_df["Risk Durumu"] = risk_status
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.warning("⚠️ Canlı uydu telemetri verisine şu an ulaşılamıyor (CelesTrak API Offline).")

    st.markdown("#### 📡 Otonom Sistem Günlükleri")
    now_str    = datetime.now().strftime("%H:%M:%S")
    log_status = "NOMİNAL" if current_kp < 5 else "RİSKLİ"
    new_log    = f"[{now_str}] Kp: {current_kp} | Durum: {log_status}"
    if not st.session_state.operation_logs or st.session_state.operation_logs[0] != new_log:
        st.session_state.operation_logs.insert(0, new_log)
    log_html = f"<div style='height: 100px; overflow-y: auto; background-color: black; color: #00ffcc; padding: 10px; font-family: monospace;'>{'<br>'.join(st.session_state.operation_logs[:10])}</div>"
    st.markdown(log_html, unsafe_allow_html=True)

with tab_earth:
    st.markdown("<div class='section-header'>STRATEJİK ALTYAPI ANALİZ MERKEZİ</div>", unsafe_allow_html=True)
    city_cols = st.columns(3)
    with city_cols[0]:
        st.markdown("#### ⚡ Ankara / İç Anadolu")
        st.caption("Yüksek Gerilim Hatları")
        if current_kp >= 7: st.error("RİSK: Yüksek GIC")
        else: st.success("DURUM: Nominal")
    with city_cols[1]:
        st.markdown("#### ✈️ İstanbul / Marmara")
        st.caption("GPS ve Havacılık")
        if current_kp >= 5: st.warning("UYARI: Sinyal Sapması")
        else: st.success("DURUM: Güvenli")
    with city_cols[2]:
        st.markdown("#### 🔥 İskenderun / Akdeniz")
        st.caption("Boru Hattı Korozyon")
        if current_kp >= 6: st.warning("RİSK: Voltaj Sapması")
        else: st.success("DURUM: Korumalı")

with tab_ai:
    st.markdown("<div class='section-header'>YAPAY ZEKA TAHMİN VE OPERASYON MERKEZİ</div>", unsafe_allow_html=True)
    
    sim_kp = st.slider("Simüle Edilecek Kp Şiddeti:", 0.0, 9.0, float(current_kp))

    if flare_model:
        try:
            input_data = pd.DataFrame(
                [[sim_kp, 97.4, 15.06]], 
                columns=['kp_value', 'Yorunge_Egilimi_Inclo', 'Ortalama_Hareket_NoKozai']
            )

            macro_prediction = flare_model.predict(input_data)[0]
            scientific_prediction = macro_prediction / 100000
            st.metric("Tahmin Edilen Hasar (Delta BSTAR)", f"{scientific_prediction:.8f}")

            if macro_prediction > 150.0:
                st.error("🚨 KRİTİK SEVİYE")
            else:
                st.success("✅ GÜVENLİ")

        except Exception as e:
            st.error(f"Tahmin Hatası: {e}")

# --- SIDEBAR METHODOLOGY ---
with st.sidebar:
    st.markdown("### 📑 Metodoloji")
    st.info("Sistem NOAA G-Skalası standartlarına göre çalışmaktadır.")

# --- FOOTER ---
st.divider()
f_col1, f_col2 = st.columns([1, 1])
with f_col1:
    st.markdown("### 📑 Bilimsel Metodoloji ve Operasyonel Standartlar")
    st.markdown("""
    ALAZ Sistemi'nin risk analizleri ve otonom karar mekanizmaları aşağıdaki akademik ve teknik altyapıyı kullanır:
    - **Tahmin Motoru:** Breiman, L. (2001) - Random Forests Regressor.
    - **Veri İşleme:** Harris et al. (2020) - NumPy & Pandas Vectorization.
    - **Yörünge Mekaniği:** SGP4 Propagator Model (Vallado et al.).
    - **Standart:** NOAA G-Scale (Geomagnetic Storm Classification).
    """)
with f_col2:
    st.markdown("### 🏛️ Stratejik Veri Sağlayıcılar ve Paydaşlar")
    partners = {
        "NOAA / SWPC": "Güneş rüzgarı, Kp endeksi ve F10.7 verilerinin ana sağlayıcısı.",
        "CelesTrak":   "Yörünge elemanları (TLE) veri tabanı.",
        "GFZ Potsdam": "Yüksek hassasiyetli manyetik alan ve Kp-Index doğrulama.",
        "NASA / CCMC": "Uzay havası modellerinin standardizasyonu ve akademik validasyon.",
    }
    for org, desc in partners.items():
        st.markdown(f"**{org}:** {desc}")

st.caption(f"© 2026 Karaman Takımı - ALAZ Solar Alert System | {datetime.now().strftime('%H:%M:%S')}")

# --- AUTO-REFRESH (non-blocking, 60s) ---
st.markdown("""
<script>
setTimeout(function() { window.location.reload(); }, 60000);
</script>
""", unsafe_allow_html=True)
