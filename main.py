import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Modüller
from src.loaders import OhioLoader
from src.drivers import InsulinDriver, CarbsDriver
from src.time_context import TimeContext
from src.model import PhysicsLSTM

# --- AYARLAR ---
DATA_FOLDER = 'data/'
LOOK_BACK = 48          # 4 Saat Geçmiş
PREDICTION_STEPS = 6    # 30 DAKİKA (6 Adım)
EPOCHS = 60             
BATCH_SIZE = 32         

def preprocess_df(df, desc="Veri"):
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
    else: return pd.DataFrame() 

    df.index = pd.date_range(start='1/1/2022', periods=len(df), freq='5min')
    
    # --- YENİ: EMA (Exponential Moving Average) ---
    # Gürültüyü azaltılmış, trendi netleştirilmiş glikoz verisi.
    # Span=12 (1 saatlik ortalama ağırlıklı)
    df['ema'] = df['glucose'].ewm(span=12, adjust=False).mean()
    
    time_ctx = TimeContext()
    df = time_ctx.add_context(df)
    ins_driver = InsulinDriver(sampling_interval=5)
    carb_driver = CarbsDriver(sampling_interval=5)
    df = ins_driver.calculate_all(df)
    df = carb_driver.calculate_all(df)
    
    warmup = 72
    if len(df) > warmup: df = df.iloc[warmup:]
    
    return df

def train_and_test_patient(patient_id, train_path, test_path):
    print(f"\n{'='*60}")
    print(f"🏥 HASTA {patient_id}: HYBRID CNN-LSTM (Gürültü Önleyici)")
    print(f"{'='*60}")
    
    try:
        raw_train = pd.read_csv(train_path)
        raw_test = pd.read_csv(test_path)
        df_train = preprocess_df(raw_train, "Train")
        df_test = preprocess_df(raw_test, "Test")
    except Exception as e:
        print(f"❌ Dosya Hatası: {e}")
        return

    # Scaling (10 Feature)
    combined = pd.concat([df_train, df_test], axis=0)
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # 'ema' eklendi
    features = ['glucose', 'insulin_rate', 'glucose_rate', 'IOB', 'COB', 'bolus', 'carbs', 'sin_time', 'cos_time', 'ema']
    
    for col in features:
        if col not in combined.columns: combined[col] = 0
    
    scaler.fit(combined[features])
    
    scaled_train = pd.DataFrame(scaler.transform(df_train[features]), columns=features)
    scaled_test = pd.DataFrame(scaler.transform(df_test[features]), columns=features)
    
    # Model
    lstm_engine = PhysicsLSTM(look_back=LOOK_BACK, prediction_horizon=PREDICTION_STEPS)
    
    X_train, y_train, _ = lstm_engine.prepare_data(scaled_train)
    X_test, y_test, _ = lstm_engine.prepare_data(scaled_test)
    
    if len(X_train) == 0: return

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
    ]
    
    lstm_engine.model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True, 
        callbacks=callbacks,
        verbose=1
    )
    
    print(f"\n📊 Grafik Çiziliyor...")
    pred_scaled = lstm_engine.model.predict(X_test).flatten()
    
    # Unscale (10 feature olduğu için fonksiyonu güncelledik)
    def unscale_glucose(val_array):
        dummy = np.zeros((len(val_array), 10)) # 10 sütun
        dummy[:, 0] = val_array # 0. sütun glucose
        return scaler.inverse_transform(dummy)[:, 0]

    inv_pred = unscale_glucose(pred_scaled)
    inv_actual = unscale_glucose(y_test)
    inv_input = unscale_glucose(X_test[:, -1, 0])

    plt.figure(figsize=(16, 8))
    
    steps_show = 144 
    if len(inv_pred) > steps_show:
        start_idx = len(inv_pred) - steps_show
    else:
        start_idx = 0
        
    slice_input = inv_input[start_idx:]
    slice_actual = inv_actual[start_idx:]
    slice_pred = inv_pred[start_idx:]
    
    t_input = np.arange(len(slice_input))
    t_target = t_input + PREDICTION_STEPS 
    
    plt.plot(t_input, slice_input, label='Mavi: Geçmiş Şeker', color='#3498db', alpha=0.4, linewidth=2)
    plt.plot(t_target, slice_actual, label='Siyah: Gerçek (30dk Sonra)', color='black', alpha=0.8, linewidth=2.5)
    plt.plot(t_target, slice_pred, label='Kırmızı: CNN-LSTM Tahmini', color='#e74c3c', linestyle='--', linewidth=2.5)
    
    hour_ticks = np.arange(0, steps_show + 1, 12)
    hour_labels = [f"{i}s" for i in range(len(hour_ticks))]
    plt.xticks(hour_ticks, hour_labels, fontsize=11)
    
    plt.axhspan(0, 70, color='red', alpha=0.1, label='Hipo')
    plt.axhspan(180, 400, color='yellow', alpha=0.1, label='Hiper')

    plt.title(f"HASTA {patient_id}: Hybrid CNN-LSTM Sonucu (Düşük Gürültü)", fontsize=14)
    plt.xlabel("Zaman (Saat)", fontsize=12)
    plt.ylabel("Glikoz (mg/dL)", fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.savefig(f"Patient_{patient_id}_Hybrid.png")
    print(f"✅ Kaydedildi: Patient_{patient_id}_Hybrid.png")

def main():
    if not os.path.exists(DATA_FOLDER):
        print("❌ Klasör yok.")
        return

    all_files = glob.glob(os.path.join(DATA_FOLDER, "*.csv"))
    patient_ids = set()
    for f in all_files:
        filename = os.path.basename(f)
        pid = filename.split('-')[0].split('_')[0]
        patient_ids.add(pid)
    
    print(f"🔎 Hastalar: {patient_ids}")
    
    for pid in patient_ids:
        train_file = None
        test_file = None
        for f in all_files:
            if pid in f and "training" in f.lower(): train_file = f
            if pid in f and "test" in f.lower(): test_file = f
            
        if train_file and test_file:
            train_and_test_patient(pid, train_file, test_file)

if __name__ == "__main__":
    main()