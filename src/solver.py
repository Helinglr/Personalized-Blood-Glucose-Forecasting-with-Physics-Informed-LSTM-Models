import numpy as np
import pandas as pd
from scipy.special import gammainc 

class ParameterSolver:
    """
    IOB/COB AWARE SOLVER (Geçmiş Yükü Hesaplayan Çözücü)
    
    Kullanıcı Tespiti: "14:30'da yemek yendiğinde, 12:00'deki iğnenin etkisi hala sürüyor.
    Bunu hesaba katmazsak model 'Yemek şekeri yükseltmedi' sanıp CS=0 buluyor."
    
    Çözüm:
    Sadece analiz penceresindeki olayları değil, pencerenin 6 SAAT ÖNCESİNDEN (Lookback)
    gelen 'Sarkan Etkileri' (Residuals) de hesaba katıyoruz.
    """
    def __init__(self):
        self.time_slots = {
            'Sabah (06-11)': (6, 11),
            'Öğle (11-16)': (11, 16),
            'Akşam (16-22)': (16, 22),
            'Gece (22-06)': (22, 6)
        }
        self.INSULIN_DELAY = 20
        self.CARB_DELAY = 20
        self.HISTORY_LOOKBACK = 360 # 6 Saat geriye bak

    def _gamma_cdf(self, t_minutes, peak_time_min, delay_min):
        """ Kümülatif Gamma: t anına kadar toplam ne kadar emildi? """
        if t_minutes <= delay_min: return 0.0
        adjusted_time = t_minutes - delay_min
        adjusted_peak = peak_time_min - delay_min
        if adjusted_peak <= 0: adjusted_peak = 1
        shape_param = 2.5 
        scale_param = adjusted_peak / (shape_param - 1)
        return gammainc(shape_param, adjusted_time / scale_param)

    def _calculate_residual_effect(self, df, t_start, t_end, col_name, peak_time, delay):
        """
        GEÇMİŞTEN SARKAN BAKİYE HESABI (Critical Fix)
        
        Mantık:
        Bir iğne 12:00'de yapıldı. Biz 14:00-17:00 arasını inceliyoruz.
        Bu iğnenin bu aralıktaki etkisi: CDF(17:00) - CDF(14:00) farkıdır.
        """
        total_residual = 0.0
        
        # Pencere başlangıcından 6 saat öncesine git
        lookback_start = t_start - pd.Timedelta(minutes=self.HISTORY_LOOKBACK)
        
        # O aralıktaki ve pencere içindeki tüm olayları al
        # (t_end'e kadar olan her şey potansiyel etkidir)
        relevant_window = df[lookback_start : t_end]
        events = relevant_window[relevant_window[col_name] > 0]
        
        for t_event, row in events.iterrows():
            val = row[col_name]
            
            # 1. Pencere Başlangıcında ne kadarı bitmişti?
            mins_to_start = (t_start - t_event).total_seconds() / 60
            cdf_start = self._gamma_cdf(mins_to_start, peak_time, delay)
            
            # 2. Pencere Bitişinde ne kadarı bitmişti?
            mins_to_end = (t_end - t_event).total_seconds() / 60
            cdf_end = self._gamma_cdf(mins_to_end, peak_time, delay)
            
            # 3. BU PENCEREDE AKTİFLEŞEN MİKTAR (Fark)
            # Eğer olay pencere içindeyse cdf_start 0'a yakın olur, fark tüm etkiyi verir.
            # Eğer olay çok eskiyse cdf_start ve cdf_end ikisi de 1.0 olur, fark 0 olur (etki bitmiş).
            active_fraction_in_window = cdf_end - cdf_start
            
            if active_fraction_in_window > 0:
                total_residual += val * active_fraction_in_window
                
        return total_residual

    def _solve_raw(self, A_list, b_list):
        A = np.array(A_list)
        b = np.array(b_list)
        if len(A) == 0: return 0.0, 0.0, 0.0, "NoData"
        try:
            x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            cs, isf = x[0], x[1]
            if abs(cs) > 1e-5: icr = isf / cs
            else: icr = 0.0
            return float(isf), float(icr), float(cs), f"Raw(n={len(A)})"
        except Exception as e:
            return 0.0, 0.0, 0.0, "Error"

    def analyze_historical_events(self, df):
        print(f"   > Metabolik Analiz (IOB/COB Geçmiş Taramalı)...", end=" ", flush=True)
        
        results = {}
        slot_data = {k: {'A': [], 'b': []} for k in self.time_slots.keys()}
        
        candidates = df[df['bolus'] > 0.5].index
        
        for t_start in candidates:
            start_glucose = df.loc[t_start, 'glucose']
            t_end_time = t_start + pd.Timedelta(minutes=180)
            if t_end_time not in df.index: continue
            
            window = df[t_start : t_end_time] # Sadece glikoz değişimi için
            
            # --- KRİTİK HESAPLAMA ---
            # Artık sadece pencere içine bakmıyoruz.
            # "Bu 3 saatlik sürede, hem yeni iğnelerden hem ESKİ iğnelerden toplam kaç ünite kana karıştı?"
            
            total_active_insulin = self._calculate_residual_effect(
                df, t_start, t_end_time, 'bolus', peak_time=60, delay=self.INSULIN_DELAY
            )
            
            total_digested_carbs = self._calculate_residual_effect(
                df, t_start, t_end_time, 'carbs', peak_time=45, delay=self.CARB_DELAY
            )
            
            end_glucose = window.iloc[-3:]['glucose'].mean()
            delta_g = end_glucose - start_glucose 

            # Slot Bulma
            hour = t_start.hour
            slot_name = 'Gece (22-06)'
            for name, (start, end) in self.time_slots.items():
                if start < end:
                    if start <= hour < end: slot_name = name
                else:
                    if hour >= start or hour < end: slot_name = name

            # Veri Ekleme (Matrise Girecek Değerler Artık Geçmiş Yükünü de İçeriyor)
            if total_digested_carbs > 0.1 or total_active_insulin > 0.1:
                slot_data[slot_name]['A'].append([total_digested_carbs, -total_active_insulin])
                slot_data[slot_name]['b'].append(delta_g)

        # ÇÖZÜM
        for slot_name, data in slot_data.items():
            isf, icr, cs, info = self._solve_raw(data['A'], data['b'])
            results[slot_name] = {
                'ISF': round(isf, 2),
                'ICR': round(icr, 2),
                'CarbSens': round(cs, 2),
                'Info': info
            }
            
        print("Tamam.")
        return results