import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# --- MODÜLLER ---
from src.loaders import OhioLoader
from src.drivers import InsulinDriver, CarbsDriver
from src.time_context import TimeContext
from src.model import PhysicsLSTM
from src.solver import ParameterSolver
from src.validator import PhysioValidator 

# --- AYARLAR ---
DATA_FOLDER = 'data/'
LOOK_BACK = 48          # 4 Saat Geçmiş
PREDICTION_STEPS = 6    # 30 DAKİKA (6 Adım)
EPOCHS = 60             
BATCH_SIZE = 32         

def preprocess_df(df, desc="Veri"):
    """
    Veri ön işleme, temizleme ve özellik üretimi
    """
    print(f"   > {desc} hazırlanıyor...", end=" ")
    
    rename_map = {'cbg': 'glucose', 'carbInput': 'carbs', 'carb': 'carbs', '5minute_intervals_timestamp': 'timestamp'}
    df = df.rename(columns=rename_map)
    
    for c in ['bolus', 'carbs', 'glucose']:
        if c not in df.columns and c != 'glucose': df[c] = 0.0
    
    df['bolus'] = df['bolus'].fillna(0)
    df['carbs'] = df['carbs'].fillna(0)
    
    if 'glucose' in df.columns:
        df['glucose'] = df['glucose'].interpolate(method='linear', limit=6)
        df = df.dropna(subset=['glucose'])
        df = df[(df['glucose'] > 30) & (df['glucose'] < 600)]
    else: 
        print("❌ Glikoz verisi yok!")
        return pd.DataFrame() 

    # Her dosya için standart bir zaman başlangıcı atıyoruz
    df.index = pd.date_range(start='1/1/2022', periods=len(df), freq='5min')
    
    df['ema'] = df['glucose'].ewm(span=12, adjust=False).mean()
    
    time_ctx = TimeContext()
    df = time_ctx.add_context(df)
    ins_driver = InsulinDriver(sampling_interval=5)
    carb_driver = CarbsDriver(sampling_interval=5)
    df = ins_driver.calculate_all(df)
    df = carb_driver.calculate_all(df)
    
    warmup = 72
    if len(df) > warmup: df = df.iloc[warmup:]
    
    print(f"Tamam. ({len(df)} kayıt)")
    return df

def train_and_test_patient(patient_id, train_path, test_path):
    print(f"\n{'='*80}")
    print(f"🏥 HASTA {patient_id} ANALİZİ BAŞLIYOR")
    print(f"{'='*80}")
    
    # 1. VERİ YÜKLEME
    try:
        raw_train = pd.read_csv(train_path)
        raw_test = pd.read_csv(test_path)
        df_train = preprocess_df(raw_train, "Eğitim Seti")
        df_test = preprocess_df(raw_test, "Test Seti")
    except Exception as e:
        print(f"❌ Kritik Hata: {e}")
        return

    # --- BÖLÜM 1: FİZYOLOJİK ANALİZ (SOLVER) ---
    print(f"\n🧠 [BÖLÜM 1] Fizyolojik Parametre Çıkarımı (Solver)...")
    solver = ParameterSolver()
    
    # Eğitim ve Test verisini birleştiriyoruz (Cascade Solver için)
    combined_raw = pd.concat([df_train, df_test])
    combined_raw = combined_raw.reset_index(drop=True)
    combined_raw.index = pd.date_range(start='1/1/2022', periods=len(combined_raw), freq='5min')

    physio_params = solver.analyze_historical_events(combined_raw)
    
    print("\n   >>> HASTANIN METABOLİK KARNESİ <<<")
    print(f"   {'Zaman':<15} | {'ISF':<10} | {'ICR':<10} | {'CS':<10}")
    print("   " + "-"*60)
    for slot, vals in physio_params.items():
        print(f"   {slot:<15} | {vals['ISF']:<10} | {vals['ICR']:<10} | {vals['CarbSens']:<10}")
    print("   " + "-"*60 + "\n")

    # --- YENİ: KANIT/SAĞLAMA AŞAMASI ---
    validator = PhysioValidator(solver.time_slots)
    test_portion_len = len(df_test)
    validation_data = combined_raw.iloc[-test_portion_len:]
    validator.validate_params(validation_data, physio_params, patient_id)

    # --- BÖLÜM 2: TAHMİN MODELİ (PREDICTION) ---
    print(f"🔮 [BÖLÜM 2] Yapay Zeka Tahmin Modeli Eğitimi (Hybrid CNN-LSTM)...")
    
    combined = pd.concat([df_train, df_test], axis=0)
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    features = ['glucose', 'insulin_rate', 'glucose_rate', 'IOB', 'COB', 'bolus', 'carbs', 'sin_time', 'cos_time', 'ema']
    for col in features:
        if col not in combined.columns: combined[col] = 0
    
    scaler.fit(combined[features])
    
    scaled_train = pd.DataFrame(scaler.transform(df_train[features]), columns=features)
    scaled_test = pd.DataFrame(scaler.transform(df_test[features]), columns=features)
    
    lstm_engine = PhysicsLSTM(look_back=LOOK_BACK, prediction_horizon=PREDICTION_STEPS)
    
    # y_train_packed ve y_test_packed boyut kontrolü
    X_train, y_train_packed, _ = lstm_engine.prepare_data(scaled_train)
    X_test, y_test_packed, _ = lstm_engine.prepare_data(scaled_test)
    
    if len(X_train) == 0: 
        print("⚠️ Yetersiz veri, eğitim atlanıyor.")
        return

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
    ]
    
    history = lstm_engine.model.fit(
        X_train, y_train_packed,
        validation_data=(X_test, y_test_packed),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True, 
        callbacks=callbacks,
        verbose=1
    )
    
    # --- BÖLÜM 3: GÖRSELLEŞTİRME VE METRİKLER ---
    print(f"\n📊 [BÖLÜM 3] Sonuçlar Görselleştiriliyor...")
    pred_scaled = lstm_engine.model.predict(X_test).flatten()
    
    # Hata düzeltme (1D/2D kontrolü)
    if y_test_packed.ndim > 1:
        actual_scaled = y_test_packed[:, 0]
    else:
        actual_scaled = y_test_packed
    
    def unscale_col(val_array, col_idx=0):
        dummy = np.zeros((len(val_array), 10))
        dummy[:, col_idx] = val_array
        return scaler.inverse_transform(dummy)[:, col_idx]

    inv_pred = unscale_col(pred_scaled)
    inv_actual = unscale_col(actual_scaled)
    inv_input = unscale_col(X_test[:, -1, 0])

    # --- METRİK HESAPLAMA ---
    mse = mean_squared_error(inv_actual, inv_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(inv_actual, inv_pred)
    
    # MARD (Mean Absolute Relative Difference) - Diyabet için Altın Standart
    # Sıfıra bölme hatasını önlemek için +1 ekliyoruz
    mard = np.mean(np.abs((inv_actual - inv_pred) / (inv_actual + 1e-5))) * 100

    print(f"   📈 PERFORMANS METRİKLERİ:")
    print(f"      RMSE : {rmse:.2f} mg/dL (Ortalama Hata Varyansı)")
    print(f"      MAE  : {mae:.2f} mg/dL (Ortalama Sapma)")
    print(f"      MARD : %{mard:.2f} (Diyabet Doğruluk Standardı)")

    # --- GRAFİK ---
    plt.figure(figsize=(16, 8))
    
    steps_show = 288 # 24 Saatlik kesit gösterelim (Daha detaylı olsun)
    if len(inv_pred) > steps_show:
        start_idx = len(inv_pred) - steps_show
    else:
        start_idx = 0
        
    slice_input = inv_input[start_idx:]
    slice_actual = inv_actual[start_idx:]
    slice_pred = inv_pred[start_idx:]
    
    t_input = np.arange(len(slice_input))
    t_target = t_input + PREDICTION_STEPS 
    
    plt.plot(t_input, slice_input, label='Geçmiş Veri', color='#3498db', alpha=0.4, linewidth=2)
    plt.plot(t_target, slice_actual, label='Gerçek (30dk Sonra)', color='black', alpha=0.8, linewidth=2.5)
    
    # Tahmin Çizgisi (Gecikmeyi görmek için)
    plt.plot(t_target, slice_pred, label='Yapay Zeka Tahmini', color='#e74c3c', linestyle='--', linewidth=2.5)
    
    # Saatlik işaretler
    hour_ticks = np.arange(0, len(t_input) + PREDICTION_STEPS, 12)
    hour_labels = [f"{i}s" for i in range(len(hour_ticks))]
    plt.xticks(hour_ticks, hour_labels, fontsize=10)
    
    # Bölgeler
    plt.axhspan(0, 70, color='red', alpha=0.1, label='Hipoglisemi (<70)')
    plt.axhspan(180, 400, color='yellow', alpha=0.1, label='Hiperglisemi (>180)')

    title_text = (
        f"HASTA {patient_id} ANALİZ RAPORU\n"
        f"Doğruluk: MARD %{mard:.1f} | Ortalama Hata: {mae:.1f} mg/dL | RMSE: {rmse:.1f}\n"
        f"(ISF: {physio_params['Öğle (11-16)']['ISF']} | Tahmin Ufku: 30dk)"
    )
    plt.title(title_text, fontsize=14, fontweight='bold')
    plt.xlabel("Zaman (Saat)", fontsize=12)
    plt.ylabel("Glikoz (mg/dL)", fontsize=12)
    plt.legend(loc='upper left', frameon=True)
    plt.grid(True, alpha=0.3)
    
    filename = f"Patient_{patient_id}_FullReport.png"
    plt.savefig(filename)
    print(f"✅ Rapor Kaydedildi: {filename}")

def main():
    if not os.path.exists(DATA_FOLDER):
        print("❌ 'data/' klasörü bulunamadı!")
        return

    all_files = glob.glob(os.path.join(DATA_FOLDER, "*.csv"))
    patient_ids = set()
    for f in all_files:
        filename = os.path.basename(f)
        pid = filename.split('-')[0].split('_')[0]
        patient_ids.add(pid)
    
    print(f"🔎 Bulunan Hastalar: {patient_ids}")
    
    for pid in patient_ids:
        train_file = None
        test_file = None
        for f in all_files:
            if pid in f and "training" in f.lower(): train_file = f
            if pid in f and "test" in f.lower(): test_file = f
            
        if train_file and test_file:
            train_and_test_patient(pid, train_file, test_file)
        else:
            print(f"⚠️ Hasta {pid} için eksik dosya.")

if __name__ == "__main__":
    main()