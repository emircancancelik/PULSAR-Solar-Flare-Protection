import json
import pandas as pd

def unified_alaz_parser(input_file, output_file):
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        # 1. SENARYO: GFZ POTSDAM / NESTED STRUCTURE (İç İçe Sözlük)
        # Eğer sözlükse ve içinde 'datetime' veya 'Kp' gibi dikey listeler varsa:
        if isinstance(raw_data, dict) and ('datetime' in raw_data or 'Kp' in raw_data):
            print("📦 Format: GFZ Potsdam (Dikey Liste Yapısı) Algılandı")
            payload = {
                'timestamp': raw_data.get('datetime', []),
                'kp_value': raw_data.get('Kp', [])
            }
            df = pd.DataFrame(payload)

        # 2. SENARYO: NOAA / LIST-IN-LIST STRUCTURE (Liste içinde Liste)
        elif isinstance(raw_data, list):
            if len(raw_data) > 0 and isinstance(raw_data[0], list):
                print("📡 Format: NOAA (Headers ilk satırda) Algılandı")
                df = pd.DataFrame(raw_data[1:], columns=raw_data[0])
            else:
                print("📋 Format: Standart Kayıt Listesi Algılandı")
                df = pd.DataFrame(raw_data)
        
        # 3. SENARYO: DİĞER SÖZLÜKLER (Standart API Key-Value)
        elif isinstance(raw_data, dict):
            print("📦 Format: Genel Sözlük (Dictionary) Algılandı")
            possible_keys = ['obs', 'data', 'entries', 'kp_index']
            found_key = next((k for k in possible_keys if k in raw_data), None)
            
            if found_key:
                df = pd.DataFrame(raw_data[found_key])
            else:
                # Tekil kayıt ise listeye alıp DataFrame yap
                df = pd.DataFrame([raw_data])
        
        else:
            raise ValueError("Tanımlanamayan JSON yapısı!")

        # --- ORTAK VERİ TEMİZLEME VE NORMALİZASYON ---
        # Sütun isimlerini küçült (Büyük-küçük harf hatasını önle)
        df.columns = [str(col).lower() for col in df.columns]

        # ALAZ Tahmin Motoru için standart isimlere zorla
        rename_map = {
            'time_tag': 'timestamp',
            'datetime': 'timestamp',
            'kp': 'kp_value',
            'kp_index': 'kp_value'
        }
        df = df.rename(columns=rename_map)

        # Veri Tiplerini Netleştir
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        if 'kp_value' in df.columns:
            df['kp_value'] = pd.to_numeric(df['kp_value'], errors='coerce')

        # Geçersiz verileri (NaN) temizle
        df = df.dropna(subset=['timestamp', 'kp_value'])

        # CSV Kaydet
        df.to_csv(output_file, index=False)
        print(f"✅ İşlem Başarılı: {output_file}")
        print(f"📊 Toplam Satır: {len(df)}")
        print("🔍 İlk 3 Satır:\n", df.head(3))

    except Exception as e:
        print(f"❌ Kritik Hata: {e}")
        print("💡 İpucu: JSON içeriğinin 'datetime' ve 'Kp' anahtarlarını içerdiğinden emin ol.")

# Çalıştır
unified_alaz_parser('json.json', 'kp_data_cleaned.csv')