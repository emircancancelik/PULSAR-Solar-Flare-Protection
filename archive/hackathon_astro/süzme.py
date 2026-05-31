import pandas as pd
import requests
import io

def fetch_kp_data():
    all_rows = []
    years = [2023, 2024, 2025]
    
    for year in years:
        url = f"https://kpindex.gfz-potsdam.de/fileadmin/KpIndex/Kp_indices_since_1932/Kp_indices_{year}.txt"
        print(f"📡 {year} verisi çekiliyor: {url}")
        
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                lines = response.text.splitlines()
                for line in lines:
                    # Başlıkları ve boşlukları atla (Sayıyla başlayan satırlara odaklan)
                    if len(line) > 28 and line[0:2].strip().isdigit():
                        y = int(line[0:2])
                        m = int(line[2:4])
                        d = int(line[4:6])
                        
                        # 8 adet 3 saatlik Kp değeri (12. karakterden başlar, her biri 2 karakter)
                        for i in range(8):
                            start = 12 + (i * 2)
                            kp_raw = line[start:start+2].strip()
                            if kp_raw:
                                kp_val = int(kp_raw) / 10.0
                                hour = i * 3
                                ts = pd.Timestamp(year=2000+y, month=m, day=d, hour=hour)
                                all_rows.append({"timestamp": ts, "kp_value": kp_val})
                print(f"✅ {year} başarıyla işlendi.")
            else:
                print(f"❌ {year} sunucu hatası: {response.status_code}")
        except Exception as e:
            print(f"❌ {year} hatası: {str(e)}")

    return pd.DataFrame(all_rows)

# --- ANA AKIŞ ---
df_kp = fetch_kp_data()

if not df_kp.empty:
    # G-Skalası Ekleme
    def to_g(kp):
        if kp >= 9: return "G5"
        if kp >= 8: return "G4"
        if kp >= 7: return "G3"
        if kp >= 6: return "G2"
        if kp >= 5: return "G1"
        return "G0"
    
    df_kp['g_scale'] = df_kp['kp_value'].apply(to_g)
    
    print(f"\n🔥 Toplam {len(df_kp)} satır veri hazır!")
    print(df_kp.tail())
    
    # 5.6 milyonluk veriyle birleştirmek üzere kaydet
    df_kp.to_csv("G_Skalasi_Final.csv", index=False)
    print("\n📁 'G_Skalasi_Final.csv' başarıyla oluşturuldu.")
else:
    print("❌ Maalesef hiçbir veri çekilemedi. Bağlantını kontrol et.")