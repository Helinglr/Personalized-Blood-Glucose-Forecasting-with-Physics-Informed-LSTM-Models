import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import gammainc

class PhysioValidator:
    def __init__(self, time_slots):
        self.time_slots = time_slots
        self.INSULIN_DELAY = 20
        self.CARB_DELAY = 20
        self.HISTORY_LOOKBACK = 360 # 6 Saat geriye bak

    def _gamma_cdf(self, t_minutes, peak_time_min, delay_min):
        if t_minutes <= delay_min: return 0.0
        adjusted_time = t_minutes - delay_min
        adjusted_peak = peak_time_min - delay_min
        if adjusted_peak <= 0: adjusted_peak = 1
        shape_param = 2.5
        scale_param = adjusted_peak / (shape_param - 1)
        return gammainc(shape_param, adjusted_time / scale_param)

    def _calculate_residual_effect(self, df, t_start, t_end, col_name, peak_time, delay):
        """ Geçmişten sarkan + Yeni eklenen toplam aktif yükü hesaplar """
        total_residual = 0.0
        lookback_start = t_start - pd.Timedelta(minutes=self.HISTORY_LOOKBACK)
        relevant_window = df[lookback_start : t_end]
        events = relevant_window[relevant_window[col_name] > 0]
        
        for t_event, row in events.iterrows():
            val = row[col_name]
            mins_to_start = (t_start - t_event).total_seconds() / 60
            cdf_start = self._gamma_cdf(mins_to_start, peak_time, delay)
            
            mins_to_end = (t_end - t_event).total_seconds() / 60
            cdf_end = self._gamma_cdf(mins_to_end, peak_time, delay)
            
            active_fraction = cdf_end - cdf_start
            if active_fraction > 0:
                total_residual += val * active_fraction
        return total_residual

    def validate_params(self, df, calculated_params, patient_id):
        print(f"\n🔍 [KANIT] Fizyolojik Sağlama (IOB ve COB Dahil)...")
        
        for slot_name, (start_h, end_h) in self.time_slots.items():
            if start_h < end_h:
                mask = (df.index.hour >= start_h) & (df.index.hour < end_h)
            else:
                mask = (df.index.hour >= start_h) | (df.index.hour < end_h)
            
            slot_df = df[mask]
            # Olay Filtresi
            sig_events = slot_df[(slot_df['bolus'] > 1.0) | (slot_df['carbs'] > 20)]
            if sig_events.empty: continue
            
            best_t = sig_events['bolus'].idxmax()
            if pd.isna(best_t) or sig_events.loc[best_t, 'bolus'] == 0:
                 best_t = sig_events['carbs'].idxmax()

            self._plot_validation(df, best_t, calculated_params, patient_id, slot_name)

    def _plot_validation(self, df, t_bolus, params, pid, slot_name):
        t_bolus = pd.to_datetime(t_bolus)
        # Grafikte 3 saat geriyi de göster ki "Geçmiş Yemekleri" görelim
        t_start_plot = t_bolus - pd.Timedelta(minutes=180) 
        t_end_plot = t_bolus + pd.Timedelta(hours=4)
        
        if t_end_plot not in df.index: return
        window = df[t_start_plot : t_end_plot]
        if window.empty: return

        p = params.get(slot_name, {'ISF': 0.0, 'ICR': 0.0, 'CarbSens': 0.0})
        isf = p['ISF']
        icr = p['ICR']
        cs = p['CarbSens']
        
        # Matematik (IOB & COB)
        start_glucose = df.loc[t_bolus, 'glucose']
        t_target = t_bolus + pd.Timedelta(hours=3)
        
        # Toplam Aktif Yük (Geçmiş + Yeni)
        tot_active_ins = self._calculate_residual_effect(
            df, t_bolus, t_target, 'bolus', peak_time=60, delay=self.INSULIN_DELAY
        )
        tot_active_carb = self._calculate_residual_effect(
            df, t_bolus, t_target, 'carbs', peak_time=45, delay=self.CARB_DELAY
        )

        # Sadece "Bu Pencerede" yenenler (Meraklısına bilgi)
        current_window_carbs = df[t_bolus : t_target]['carbs'].sum()
        past_ghost_carbs = max(0, tot_active_carb - (current_window_carbs * 0.9)) # Yaklaşık bir gösterge

        rise = tot_active_carb * cs
        drop = tot_active_ins * isf
        pred_val = start_glucose + rise - drop
        
        idx_target = df.index.get_indexer([t_target], method='nearest')[0]
        actual_val = df.iloc[idx_target]['glucose']

        # --- GÖRSELLEŞTİRME ---
        plt.figure(figsize=(12, 7))
        times_min = (window.index - t_bolus).total_seconds() / 60
        plt.plot(times_min, window['glucose'], color='black', linewidth=2, label='Gerçek')
        plt.axvline(0, color='blue', linestyle='--', alpha=0.5, label='Olay (t=0)')
        
        # GEÇMİŞ OLAYLAR (Solda kalanlar)
        # 1. Geçmiş İğneler (Mor Üçgen)
        past_boluses = window[(window['bolus'] > 0) & (window.index < t_bolus)]
        for pt, row in past_boluses.iterrows():
             m = (pt - t_bolus).total_seconds() / 60
             plt.scatter([m], [row['glucose']], color='purple', s=60, marker='v', 
                         label='Geçmiş İğne (IOB)' if 'Geçmiş İğne (IOB)' not in plt.gca().get_legend_handles_labels()[1] else "")

        # 2. Geçmiş Yemekler (Turuncu Kare) -> YENİ EKLENDİ
        past_meals = window[(window['carbs'] > 0) & (window.index < t_bolus)]
        for pt, row in past_meals.iterrows():
             m = (pt - t_bolus).total_seconds() / 60
             plt.scatter([m], [row['glucose']], color='orange', s=60, marker='s', 
                         label='Geçmiş Yemek (COB)' if 'Geçmiş Yemek (COB)' not in plt.gca().get_legend_handles_labels()[1] else "")

        target_m = (t_target - t_bolus).total_seconds() / 60
        plt.scatter([0], [start_glucose], color='blue', s=100, label='Başlangıç')
        plt.scatter([target_m], [pred_val], color='red', s=150, marker='X', label='Tahmin')
        plt.scatter([target_m], [actual_val], color='green', s=100, label='Gerçek')

        # Başlıkta COB Detayı
        title = (
            f"RAPOR: {slot_name} (IOB & COB DAHİL)\n"
            f"Parametreler -> CS: {cs:.2f} | ISF: {isf:.2f}\n"
            f"Aktif YEMEK: {tot_active_carb:.1f}g (Bunun ~{past_ghost_carbs:.1f}g kadarı geçmişten sarkıyor)\n"
            f"Aktif İNSÜLİN: {tot_active_ins:.1f}u\n"
            f"Denklem: {start_glucose:.0f} + ({tot_active_carb:.1f}*{cs:.1f}) - ({tot_active_ins:.1f}*{isf:.1f}) = {pred_val:.0f}"
        )
        
        plt.title(title, fontsize=10, fontfamily='monospace', loc='left')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(f"Report_{pid}_{slot_name.split()[0]}.png")
        plt.close()