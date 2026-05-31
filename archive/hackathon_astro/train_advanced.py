import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# 1. Mühendislik Harikası Veriyi Yükle
print("🚀 ALAZ AI: Gelişmiş Özellik (Feature Engineered) Seti Yükleniyor...")
df = pd.read_csv("ALAZ_Engineered_Training_Data.csv")

# NaN (Boş) değerleri temizle (Lag hesaplamalarından kaynaklı ilk satırlar boş olabilir)
df = df.dropna()

# 2. Girdiler (X) ve Çıktı (y)
# String (Metin) kolonları ve hedef kolonları ayıklayıp sadece sayısal sinyalleri modele veriyoruz
kullanilmayacak_kolonlar = [
    'SAT_NAME', 'EPOCH', 'DATE', 'NORAD_ID', 'NORAD_CAT_ID',
    'TARGET_NEXT_DELTA_BSTAR', 'TARGET_NEXT_DELTA_MEAN_MOTION', 
    'TARGET_BSTAR_UP', 'TARGET_MM_DROP', 'TARGET_ORBITAL_RESPONSE'
]

features = [col for col in df.columns if col not in kullanilmayacak_kolonlar]

# Hedefimiz: Sürtünmedeki "Gelecek" Değişim (Delta)
# Tahmini kolaylaştırmak için 100.000 ile makro ölçeğe çekiyoruz
X = df[features]
y = df['TARGET_NEXT_DELTA_BSTAR'] * 100000 

# 3. Eğitim Seti Ayrımı
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Model Eğitimi (Azure Container Cost Optimizasyonu için n_jobs=-1)
print(f"🧠 Model Eğitiliyor... Kullanılan Sinyal Sayısı: {len(features)}")
model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# 5. Performans Ölçümü
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
basari_yuzdesi = r2 * 100

print("-" * 50)
print(f"📉 Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"🎯 Gelecek Hasar (Delta BSTAR) Tahmin Başarısı: %{basari_yuzdesi:.2f}")
print("-" * 50)

# 6. Beyni Dışarı Aktar
joblib.dump(model, "alaz_advanced_model.pkl")

joblib.dump(features, "model_features.pkl")
print("✅ İşlem Tamam! 'alaz_advanced_model.pkl' ve 'model_features.pkl' hazır.")