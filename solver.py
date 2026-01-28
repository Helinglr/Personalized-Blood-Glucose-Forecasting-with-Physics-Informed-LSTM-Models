import numpy as np
import pandas as pd
from scipy.optimize import least_squares

class ParameterSolver:
    """
    DEBUG MODE SOLVER (4. Saat Hedefli - HATA DÜZELTİLDİ)
    - Pandas Slice Hatası Giderildi.
    - Sadece 70-180 mg/dL (Öğlijemik) aralığındaki temiz verileri analiz eder.
    """
    def __init__(self, model=None, scaler_X=None, scaler_y=None):
        pass 
        
    def analyze_historical_events(self, df):
        print(f"   > Debug Batch Analizi...", end=" ", flush=True)
        
        # VERİ HAVUZU
        batches = {
            'Sabah': {'C': [], 'I': [], 'dG': []},
            'Öğle': {'C': [], 'I': [], 'dG': []},
            'Akşam':{'C': [], 'I': [], 'dG': []},
            'Gece':  {'C': [], 'I': [], 'dG': []}
        }
        
        # Olayları bul
        event_indices = df[(df['bolus'] > 0.5) | (df['carbs'] > 10)].index
        processed_intervals = set()
        valid_count = 0
        
        for start_time in event_indices:
            # Çakışma kontrolü
            if any((start_time >= p_start) and (start_time < p_end) for p_start, p_end in processed_intervals):
                continue
                
            end_time = start_time + pd.Timedelta(hours=4)
            if end_time > df.index[-1]: continue
            
            segment = df[start_time : end_time]
            
            # --- HATA DÜZELTİLDİ BURASI ---
            # Eskisi: look_ahead = segment[pd.Timedelta(minutes=30) :] (HATALI)
            # Yenisi: Başlangıç zamanına 30 dk ekleyip oradan sonrasına bakıyoruz.
            check_time = start_time + pd.Timedelta(minutes=30)
            look_ahead = segment[check_time:]
            
            # --- FİLTRE 1: EK DOZ YOK ---
            if (look_ahead['bolus'].sum() > 0.5) or (look_ahead['carbs'].sum() > 10):
                continue 
            
            # --- FİLTRE 2: 70-180 HEDEFİ ---
            start_glucose = segment['glucose'].iloc[0]
            end_glucose = segment['glucose'].iloc[-3:].mean() # Son 15 dk ortalaması
            
            if pd.isna(start_glucose) or pd.isna(end_glucose): continue
            
            # Sadece "Başarılı" yönetimleri öğren
            if not (70 <= start_glucose <= 180): continue
            if not (70 <= end_glucose <= 180): continue
            
            # --- VERİ ÇIKAR ---
            total_carbs = segment['carbs'].sum()
            total_bolus = segment['bolus'].sum()
            delta_glucose = end_glucose - start_glucose
            
            hour = start_time.hour
            period = 'Sabah' if 6<=hour<12 else 'Öğle' if 12<=hour<18 else 'Akşam' if 18<=hour<24 else 'Gece'
            
            batches[period]['C'].append(total_carbs)
            batches[period]['I'].append(total_bolus)
            batches[period]['dG'].append(delta_glucose)
            
            processed_intervals.add((start_time, end_time))
            valid_count += 1

        # --- OPTİMİZASYON (SINIRSIZ) ---
        results = {}
        
        all_C = np.concatenate([b['C'] for b in batches.values()])
        all_I = np.concatenate([b['I'] for b in batches.values()])
        all_dG = np.concatenate([b['dG'] for b in batches.values()])
        
        if len(all_C) == 0:
            print("(HİÇ 'SAF' VERİ YOK!)")
            # Hata vermesin diye 0 dönüyoruz
            return {k: {'IDF': 0.0, 'ICR': 0.0} for k in batches.keys()}
        
        global_params = self._optimize_raw(all_C, all_I, all_dG)
        
        for p_name, data in batches.items():
            if len(data['C']) >= 2:
                params = self._optimize_raw(data['C'], data['I'], data['dG'])
                results[p_name] = params
            else:
                results[p_name] = global_params
        
        print(f"Tamamlandı. ({valid_count} temiz olay)")
        return results

    def _optimize_raw(self, C_list, I_list, dG_list):
        C = np.array(C_list)
        I = np.array(I_list)
        Y = np.array(dG_list)
        
        def loss_function(params):
            cs, idf, bias = params
            prediction = (C * cs) - (I * idf) + bias
            return Y - prediction

        initial_guess = [1.0, 1.0, 0.0] 
        
        try:
            res = least_squares(loss_function, initial_guess, 
                              bounds=(-np.inf, np.inf), 
                              loss='soft_l1', f_scale=15.0)
            
            cs_opt = res.x[0]
            idf_opt = res.x[1]
            
            if cs_opt == 0: icr_opt = 0
            else: icr_opt = idf_opt / cs_opt
            
            return {'IDF': idf_opt, 'ICR': icr_opt}
            
        except Exception as e:
            print(f" [Matematik Çökmesi: {e}]")
            return {'IDF': 0.0, 'ICR': 0.0}