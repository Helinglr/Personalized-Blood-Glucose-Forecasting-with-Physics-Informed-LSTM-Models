import os
import random
import numpy as np
import tensorflow as tf

# --- SABİTLEME AYARLARI (Reproducibility) ---
# Bu blok, her çalıştırmada AYNI sonucu almanı sağlar.
SEED_VALUE = 42

os.environ['PYTHONHASHSEED'] = str(SEED_VALUE)
random.seed(SEED_VALUE)
np.random.seed(SEED_VALUE)
tf.random.set_seed(SEED_VALUE)

# Deneysel özellikleri kapat (Deterministik çalışma için)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

# --- DİĞER IMPORTLAR ---
import pandas as pd
import glob
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

from src.drivers import InsulinDriver, carbsDriver
from src.time_context import TimeContext
from src.model import GlucoseModel
from src.solver import ParameterSolver
from src.loaders import OhioLoader
# --- AYARLAR ---
DATA_FOLDER = 'data/'
LOOK_BACK = 24
HORIZON = 6 
EPOCHS = 50 
BATCH_SIZE = 4 

def analyze_patient(file_path):
    patient_id = os.path.basename(file_path).split('.')[0]
    print(f"\n{'-'*60}")
    print(f"HASTA ANALİZİ: {patient_id}")
    
    # 1. VERİ YÜKLEME
    try:
        loader = OhioLoader()
        df = loader.load(file_path)
    except Exception: return None

    if len(df) < 100: return None

    # 2. TEMİZLİK
    df['bolus'] = df['bolus'].clip(upper=35) 
    df['carbs'] = df['carbs'].clip(upper=250)

    try:
        ins_driver = InsulinDriver()
        carb_driver = carbsDriver()
        df = ins_driver.calculate_inventory(df)
        df = carb_driver.calculate_inventory(df)
        time_ctx = TimeContext()
        df = time_ctx.add_context(df)
        df = df.dropna()
    except Exception: return None

    # 3. HAZIRLIK
    features = ['glucose', 'IOB', 'COB', 'sin_time', 'cos_time', 
                'is_morning', 'is_afternoon', 'is_evening', 'is_night']
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[features].values)

    X, y = [], []
    for i in range(len(scaled_data) - LOOK_BACK - HORIZON):
        X.append(scaled_data[i:(i + LOOK_BACK), :])
        y.append(scaled_data[i + LOOK_BACK + HORIZON, 0])
    
    X, y = np.array(X), np.array(y)
    
    # --- DOĞRULAMA İÇİN VERİYİ BÖL (Train / Test) ---
    # Son %20'lik kısmı modele HİÇ göstermeyeceğiz. Sınav yapacağız.
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # 4. EĞİTİM
    print(f"   > Model Eğitiliyor (Train: {len(X_train)}, Test: {len(X_test)})...")
    gm = GlucoseModel(n_timesteps=LOOK_BACK)
    
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    history = gm.model.fit(
        X_train, y_train,
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE, 
        validation_data=(X_test, y_test),
        verbose=1, # İlerlemeyi gör
        callbacks=[early_stop]
    )

    # --- 5. KALİTE KONTROL (SİHİRLİ ADIM) ---
    print(f"   > Doğrulama Testi Yapılıyor...")
    
    # Test verisi üzerinde tahmin yap
    preds_scaled = gm.model.predict(X_test, verbose=0)
    
    # Geriye Scale Et (Gerçek mg/dL değerlerine dön)
    # Scaler'ın sadece ilk sütunu (glucose) ile işlem yapıyoruz
    dummy_scaler = MinMaxScaler()
    dummy_scaler.min_, dummy_scaler.scale_ = scaler.min_[0], scaler.scale_[0]
    
    y_test_real = (y_test * (1/scaler.scale_[0])) + scaler.min_[0]
    preds_real = (preds_scaled.flatten() * (1/scaler.scale_[0])) + scaler.min_[0]
    
    # Hata Puanı (MAE): Ortalama kaç mg/dL sapıtmış?
    mae_score = mean_absolute_error(y_test_real, preds_real)
    print(f"   > HATA PUANI (MAE): {mae_score:.2f} mg/dL")
    
    # --- GRAFİK ÇİZ VE KAYDET ---
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_real[:200], label='Gerçek Şeker (CGM)', color='black', alpha=0.7)
    plt.plot(preds_real[:200], label='Yapay Zeka Tahmini', color='red', linestyle='--')
    plt.title(f"Hasta {patient_id} - Model Performansı (Hata: {mae_score:.1f} mg/dL)")
    plt.xlabel("Zaman (5dk adımlar)")
    plt.ylabel("Glikoz (mg/dL)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Kaydet
    grafik_adi = f"Grafik_{patient_id}.png"
    plt.savefig(grafik_adi)
    print(f"   > Grafik kaydedildi: {grafik_adi}")
    plt.close()

    # Eğer Hata çok yüksekse (>30 mg/dL), parametre hesaplama bile.
    if mae_score > 35:
        print("   ⚠️ UYARI: Model hatası çok yüksek. Sonuçlar güvenilir değil.")
    
    # 6. PARAMETRE ÇÖZME
    print(f"   > Parametreler Hesaplanıyor...")
    solver = ParameterSolver(gm.model, scaler)
    period_results = solver.extract_parameters_by_period(base_glucose=140)
    
    flat_result = {"Hasta_ID": patient_id, "MAE_Hata": round(mae_score, 2)}
    for period, vals in period_results.items():
        flat_result[f"IDF_{period}"] = max(0, vals['IDF'])
        flat_result[f"ICR_{period}"] = max(0, vals['ICR'])
        
    return flat_result

if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError: pass

    all_files = glob.glob(os.path.join(DATA_FOLDER, "*.csv"))
    full_report = []
    
    # Sadece şüpheli hastalara bakalım
    TARGET_IDS = ["544", "567"]
    
    for f in all_files:
        if "training" in f and any(tid in f for tid in TARGET_IDS):
            res = analyze_patient(f)
            if res:
                full_report.append(res)

    if full_report:
        final_df = pd.DataFrame(full_report)
        print("\n" + "="*50)
        print(final_df)
        final_df.to_csv("Analiz_Grafikli_Sonuc.csv", index=False)