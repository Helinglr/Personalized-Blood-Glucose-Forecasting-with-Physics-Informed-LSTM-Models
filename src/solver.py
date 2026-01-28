import numpy as np
import pandas as pd
from scipy.optimize import least_squares

class ParameterSolver:
    def __init__(self, tau_days=7):
        self.time_slots = ["Morning (06-11)", "Afternoon (11-16)", "Evening (16-23)", "Night (23-06)"]
        self.isf_horizon = 24   # 120 mn (Insulin Effect)
        self.carb_horizon = 12  # 60 mn (Carbohydrate Effect)
        self.tau = tau_days * 288 
        self.k_min = 15         # Bayesian shrinkage intensity

    def analyze_historical_events(self, df):
        # 1. OUTLIER FİLTER
        clean_df = self._filter_outliers(df)
        
        # 2. GLOBAL PRIOR (Hierarchical anchor)
        global_isf, global_std, n_global = self._solve_robust_isf(clean_df)
        isf_prior = global_isf if (not pd.isna(global_isf) and n_global > 5) else 45.0
        
        raw_results = {}
        for slot in self.time_slots:
            slot_data = clean_df[clean_df['slot'] == slot]
            isf_raw, isf_std, n = self._solve_robust_isf(slot_data)
            
            # 3. PARTIAL POOLING (Global shrinkage with increasing uncertainty)
            uncertainty = (isf_std / (isf_prior + 1e-5)) if not pd.isna(isf_std) else 2.0
            k_dynamic = self.k_min * (1 + uncertainty)
            w = n / (n + k_dynamic) if n > 0 else 0
            
            # Log-Space Shrinkage
            if not pd.isna(isf_raw) and isf_raw > 0:
                isf_final = np.exp(w * np.log(isf_raw) + (1 - w) * np.log(isf_prior))
            else:
                isf_final = isf_prior

            # 4. CS & ICR  (Clinical lower bound: 0.2)
            cs_raw, avg_carbs = self._solve_carb_sensitivity(slot_data, isf_final)
            cs_final = np.clip(cs_raw, 0.2, 5.0) if not pd.isna(cs_raw) else 0.2
            
            # 5. DRIFT GUARDRAIL (Physical safety limit)
            drift_raw = self._calculate_model_based_drift(slot_data, isf_final, cs_final)
            # Drift equation: Divides the hourly residual by 5 minutes and applies clipping.
            drift_final = np.clip(drift_raw, -3.0, 3.0) 
            
            # Confidence Score
            conf = (n / (1 + (isf_std / (isf_final + 1e-5)))) if not pd.isna(isf_std) else 0

            raw_results[slot] = {
                'ISF': isf_final, 'ISF_std': isf_std, 'n': n, 
                'CS': cs_final, 'Drift': drift_final, 'Conf': conf,
                'avg_carbs': avg_carbs
            }

            

        return self._apply_circadian_smoothing(raw_results)

    def _solve_robust_isf(self, df):
        """Huber Loss ve MAD tabanlı standart sapma ile ISF çözümü"""
        events = self._extract_pure_insulin_events(df)
        if len(events) < 3: return np.nan, np.nan, len(events)
        
        b, dg, weights = events[:, 0], events[:, 1], events[:, 2]
        
        # Model: delta_g ≈ -ISF * bolus
        res = least_squares(lambda p: weights * ((-p[0] * b) - dg), x0=[45.0], 
                            bounds=(5, 300), loss='huber', f_scale=10)
        
        isf_val = float(res.x[0])
        residuals = weights * ((-isf_val * b) - dg)
        isf_std = np.median(np.abs(residuals - np.median(residuals))) * 1.4826 / (np.mean(b) + 1e-5)
        
        return isf_val, isf_std, len(events)

    def _extract_pure_insulin_events(self, df):
        events = []
        for i in range(len(df) - self.isf_horizon):
            row = df.iloc[i]
            if row['bolus'] > 0 and row['carbs'] < 1:
                delta_g = df.iloc[i + self.isf_horizon]['glucose'] - row['glucose']
                if delta_g < 0:
                    dt = len(df) - i
                    w = np.exp(-dt / self.tau)
                    events.append((row['bolus'], delta_g, w))
        return np.array(events) if events else np.array([])

    def _solve_carb_sensitivity(self, df, isf_final):
        events = []
        for i in range(len(df) - self.carb_horizon):
            row = df.iloc[i]
            if row['carbs'] >= 5:
                delta_g = df.iloc[i + self.carb_horizon]['glucose'] - row['glucose']
                events.append((row['carbs'], row['bolus'], delta_g))
        
        ev_arr = np.array(events)
        if len(ev_arr) < 3: return np.nan, 0
        
        c, b, dg = ev_arr[:, 0], ev_arr[:, 1], ev_arr[:, 2]
        y = dg + (isf_final * b)
        res = least_squares(lambda p: (p[0] * c) - y, x0=[3.0], bounds=(0.1, 30.0), loss='soft_l1')
        return float(res.x[0]), np.mean(c)

    def _calculate_model_based_drift(self, df, isf, cs):
        if pd.isna(isf): return 0.0
        resids = []
        for i in range(0, len(df) - 12, 6):
            row = df.iloc[i]
            actual = df.iloc[i + 12]['glucose'] - row['glucose']
            pred = (row['carbs'] * (cs if not pd.isna(cs) else 0.2)) - (row['bolus'] * isf)
            resids.append(actual - pred)
        return (float(np.nanmedian(resids)) / 12.0) if resids else 0.0

    def _filter_outliers(self, df):
        df = df.copy()
        for col in ['bolus', 'carbs']:
            if col in df.columns:
                q95 = df[df[col] > 0][col].quantile(0.95)
                df.loc[df[col] > q95, col] = 0
        return df

    def _apply_circadian_smoothing(self, results):
        slots = self.time_slots
        n = len(slots)
        new_res = results.copy()
        for i in range(n):
            curr, prev, nxt = slots[i], slots[(i-1)%n], slots[(i+1)%n]
            new_res[curr]['ISF'] = (results[curr]['ISF']*0.7 + 
                                    results[prev]['ISF']*0.15 + 
                                    results[nxt]['ISF']*0.15)
        return new_res