
import os
import pandas as pd
import numpy as np
import glob
import warnings
import matplotlib.pyplot as plt 
from sklearn.preprocessing import MinMaxScaler
from src.drivers import InsulinDriver, CarbsDriver
from src.time_context import TimeContext
from src.solver import ParameterSolver
from src.model import PhysicsLSTM

warnings.filterwarnings('ignore')
DATA_FOLDER = 'data/' 
pd.options.display.float_format = '{:.2f}'.format

def get_period_name(h):
    if 6 <= h < 11: return "Morning (06-11)"
    elif 11 <= h < 16: return "Afternoon (11-16)"
    elif 16 <= h < 23: return "Evening (16-23)"
    else: return "Night (23-06)"

def preprocess_for_solver(df):
    df = df.rename(columns={'cbg': 'glucose', 'carbInput': 'carbs', 'carb': 'carbs'})
    df['glucose'] = pd.to_numeric(df['glucose'], errors='coerce')
    df = df.dropna(subset=['glucose'])
    for c in ['bolus', 'carbs']: df[c] = pd.to_numeric(df.get(c, 0), errors='coerce').fillna(0)
    df.index = pd.date_range(start='1/1/2022', periods=len(df), freq='5min')
    df['slot'] = [get_period_name(h) for h in df.index.hour]
    return df

def preprocess_for_ai(df, physio_params, scaler=None, fit=False):
    """Injects each patient’s individual solver parameters (ISF/ICR) into the AI training."""
    df = df.rename(columns={'cbg': 'glucose', 'carbInput': 'carbs', 'carb': 'carbs'})
    df['glucose'] = pd.to_numeric(df['glucose'], errors='coerce').interpolate(limit=6)
    df = df.dropna(subset=['glucose'])
    
    df.index = pd.date_range(start='1/1/2022', periods=len(df), freq='5min')
    df['slot'] = [get_period_name(h) for h in df.index.hour]
    
    # ✅ RESEARCH TIER: Feature Injection 
    df['ISF'] = df['slot'].map(lambda s: physio_params.get(s, {}).get('ISF', 45.0))
    df['ICR'] = df['ISF'] / (df['slot'].map(lambda s: physio_params.get(s, {}).get('CS', 3.0)) + 1e-5)
    df['Conf'] = df['slot'].map(lambda s: physio_params.get(s, {}).get('Conf', 0.5))

    # Drivers 
    df = CarbsDriver().calculate_all(df)
    df = InsulinDriver().calculate_all(df)
    df = TimeContext().add_context(df)
    
    # Feature Engineering & Dynamic factors
    df['glucose_rate'] = df['glucose'].diff().fillna(0)
    df['dyn_ICR_factor'] = 1.0
    df.loc[df['glucose'] > 180, 'dyn_ICR_factor'] = 0.9 
    df.loc[df['glucose'] < 90, 'dyn_ICR_factor'] = 1.1 
    
    df['adapt_ISF_factor'] = 1.0
    df.loc[df['glucose_rate'] > 1.5, 'adapt_ISF_factor'] = 0.9 
    
    df['ema_fast'] = df['glucose'].ewm(span=3).mean()
    df['ema_slow'] = df['glucose'].ewm(span=12).mean()
    df['ema_diff'] = df['ema_fast'] - df['ema_slow']
    
    # Scaling
    scale_cols = [
        'insulin_rate', 'glucose_rate', 'IOB', 'COB', 'bolus', 'carbs', 
        'sin_time', 'cos_time', 'ema_fast', 'ema_slow', 'ema_diff', 
        'carb_absorption', 'ISF', 'ICR', 'dyn_ICR_factor', 'adapt_ISF_factor'
    ]
    
    if fit: df[scale_cols] = scaler.fit_transform(df[scale_cols])
    else: df[scale_cols] = scaler.transform(df[scale_cols])
    return df

def plot_patient_results(pid, res):
    """Patient-level graph output."""
    plt.figure(figsize=(12, 6))
    plt.plot(res.index, res['Actual'], label='Actual glucose', color='black', alpha=0.5)
    plt.plot(res.index, res['Pred'], '--', label='AI Prediction (P50)', color='blue')
    if 'P10' in res.columns and 'P90' in res.columns:
        plt.fill_between(res.index, res['P10'], res['P90'], color='blue', alpha=0.1, label='Confidence Interval')
    plt.axhline(y=70, color='red', linestyle=':', label='Hypo Threshold')
    plt.axhline(y=180, color='orange', linestyle=':', label='Hyper Threshold')
    plt.title(f"Patient {pid} - Clinical prediction graph")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.show()

def analyze_patient(pid, train_f, test_f):
    print(f"\n>>> [PATİENT {pid}] INDIVIDUAL ANALYSIS STARTING...")
    try:
        raw_train, raw_test = pd.read_csv(train_f), pd.read_csv(test_f)
        if raw_train.empty or raw_test.empty: raise ValueError("Empty file.")

        # 1. SOLVER (Patient-specific parameters)
        df_solver = preprocess_for_solver(raw_train)
        solver = ParameterSolver()
        physio = solver.analyze_historical_events(df_solver)
        
        # 2. AI (Patient-specific training)
        scaler = MinMaxScaler()
        df_ai_tr = preprocess_for_ai(raw_train, physio, scaler=scaler, fit=True)
        df_ai_ts = preprocess_for_ai(raw_test, physio, scaler=scaler, fit=False)
        
        # Conf normalize
        max_c = max([p.get('Conf', 0.5) for p in physio.values()]) + 1e-5
        df_ai_tr['Conf'] = df_ai_tr['Conf'] / max_c
        df_ai_ts['Conf'] = df_ai_ts['Conf'] / max_c

        ai = PhysicsLSTM()
        X_tr, y_tr, w_tr = ai.prepare_data(df_ai_tr)
        X_ts, y_ts, w_ts = ai.prepare_data(df_ai_ts)
        
        print(f"    🤖 Model training...", end="", flush=True)
        ai.model.fit(X_tr, y_tr, sample_weight=w_tr, epochs=50, batch_size=32, verbose=0)
        print(" Done.")
        
        # 3. EVALUATION
        preds = ai.model.predict(X_ts, verbose=0)
        p10_30, p50_30, p90_30 = preds[:, 5*3], preds[:, 5*3+1], preds[:, 5*3+2]
        
        last_g = df_ai_ts['glucose'].values[ai.look_back - 1 : ai.look_back - 1 + len(p50_30)]
        res = df_ai_ts.iloc[ai.look_back : ai.look_back + len(p50_30)].copy()
        res['Actual'] = last_g + y_ts[:, 5]
        res['Pred'], res['P10'], res['P90'] = last_g + p50_30, last_g + p10_30, last_g + p90_30
        
        perf = res.groupby('slot').agg({'Pred': lambda x: np.mean(np.abs(res.loc[x.index,'Actual']-x)/(res.loc[x.index,'Actual']+1e-3))*100}).rename(columns={'Pred':'AI MARD (%)'})
        report = pd.DataFrame.from_dict(physio, orient='index').join(perf)
        report['ICR'] = report['ISF'] / (report['CS'] + 1e-5)

        print(f"\n📊 PATİENT {pid} CLINICAL REPORT")
        print("-" * 140)
        print(report[['ISF', 'Conf', 'n', 'ICR', 'CS', 'Drift', 'AI MARD (%)']].to_string())
        print("-" * 140)
        
        plot_patient_results(pid, res)

    except Exception as e: print(f"    ! ERROR: {str(e)}")

def main():
    all_files = glob.glob(os.path.join(DATA_FOLDER, "*.csv"))
    pids = sorted(list({os.path.basename(f).split('-')[0].split('_')[0] for f in all_files}))
    for pid in pids:
        tr = next((f for f in all_files if pid in f and "train" in f.lower()), None)
        ts = next((f for f in files if pid in f and "test" in f.lower()), None) 
        
def main_fixed():
    all_files = glob.glob(os.path.join(DATA_FOLDER, "*.csv"))
    pids = sorted(list({os.path.basename(f).split('-')[0].split('_')[0] for f in all_files}))
    for pid in pids:
        train_f = next((f for f in all_files if pid in f and "train" in f.lower()), None)
        test_f = next((f for f in all_files if pid in f and "test" in f.lower()), None)
        if train_f and test_f: analyze_patient(pid, train_f, test_f)

if __name__ == "__main__": main_fixed()
=======
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
>>>>>>> 2a81eba280340d2cb40cb8ad7f61c4161f82e581
