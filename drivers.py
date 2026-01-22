import numpy as np
import pandas as pd
from scipy.stats import gamma

class InsulinDriver:
    def __init__(self, duration_min=300, peak_min=75, sampling_interval=5):
        self.duration = duration_min
        self.peak = peak_min
        self.dt = sampling_interval
        self.curve = self._generate_curve()

    def _generate_curve(self):
        steps = int(self.duration / self.dt)
        t = np.linspace(0, self.duration, steps)
        shape = 3.0 
        scale = self.peak / (shape - 1)
        curve = gamma.pdf(t, shape, scale=scale)
        if curve.sum() > 0: curve = curve / curve.sum()
        return curve

    def calculate_all(self, df):
        if not isinstance(df, pd.DataFrame): return df
        vals = df['bolus'].fillna(0).values if 'bolus' in df.columns else np.zeros(len(df))
        rate = np.convolve(vals, self.curve, mode='full')[:len(df)]
        df['insulin_rate'] = rate
        cum_input = np.cumsum(vals)
        cum_spent = np.cumsum(rate)
        df['IOB'] = np.maximum(cum_input - cum_spent, 0)
        return df

class CarbsDriver:
    def __init__(self, duration_min=240, peak_min=60, sampling_interval=5):
        self.duration = duration_min
        self.peak = peak_min
        self.dt = sampling_interval
        self.curve = self._generate_curve()

    def _generate_curve(self):
        steps = int(self.duration / self.dt)
        t = np.linspace(0, self.duration, steps)
        shape = 2.5
        scale = self.peak / (shape - 1)
        curve = gamma.pdf(t, shape, scale=scale)
        if curve.sum() > 0: curve = curve / curve.sum()
        return curve

    def calculate_all(self, df):
        if not isinstance(df, pd.DataFrame): return df
        target = 'carbs' if 'carbs' in df.columns else 'carbInput'
        vals = df[target].fillna(0).values if target in df.columns else np.zeros(len(df))
        rate = np.convolve(vals, self.curve, mode='full')[:len(df)]
        df['glucose_rate'] = rate
        cum_input = np.cumsum(vals)
        cum_absorbed = np.cumsum(rate)
        df['COB'] = np.maximum(cum_input - cum_absorbed, 0)
        return df