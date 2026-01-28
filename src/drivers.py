import numpy as np
import pandas as pd
from scipy.stats import gamma

class InsulinDriver:
    def __init__(self, duration_min=300, peak_min=75, sampling_interval=5):
        self.dt = sampling_interval
        self.curve = self._generate_curve(duration_min, peak_min)

    def _generate_curve(self, dur, peak):
        t = np.linspace(0, dur, int(dur/self.dt))
        curve = gamma.pdf(t, 3.0, scale=peak/2.0)
        return curve / curve.sum() if curve.sum() > 0 else curve

    def calculate_all(self, df):
        vals = df['bolus'].fillna(0).values if 'bolus' in df.columns else np.zeros(len(df))
        rate = np.convolve(vals, self.curve, mode='full')[:len(df)]
        df['insulin_rate'] = rate
        df['IOB'] = np.maximum(np.cumsum(vals) - np.cumsum(rate), 0)
        return df

class CarbsDriver:
    def __init__(self, sampling_interval=5):
        self.dt = sampling_interval

    def _generate_curve(self, dur, peak):
        t = np.linspace(0, dur, int(dur/self.dt))
        curve = gamma.pdf(t, 2.5, scale=peak/1.5)
        return curve / curve.sum() if curve.sum() > 0 else curve

    def calculate_all(self, df):
        target = 'carbs' if 'carbs' in df.columns else 'carbInput'
        vals = df[target].fillna(0).values if target in df.columns else np.zeros(len(df))
        
        # RESEARCH TIER: Extended COB (Protein/Fat Impact)
        fast_curve = self._generate_curve(dur=240, peak=60)
        slow_curve = self._generate_curve(dur=480, peak=120) # Pizza etkisi: 8 saat
        
        fast_rate = np.convolve(vals, fast_curve, mode='full')[:len(df)]
        slow_rate = np.convolve(vals, slow_curve, mode='full')[:len(df)]
        
        # 40% slow-wave modulation blending during the evening period
        is_evening = df['slot'].apply(lambda x: 1.0 if "Akşam" in str(x) else 0.0)
        rate = (1.0 - 0.4 * is_evening) * fast_rate + (0.4 * is_evening) * slow_rate
        
        df['carb_absorption'] = rate
        df['COB'] = np.maximum(np.cumsum(vals) - np.cumsum(rate), 0)
        return df