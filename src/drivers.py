import numpy as np
import pandas as pd
from scipy.stats import gamma
from abc import ABC, abstractmethod

class BioDriver(ABC):
    @abstractmethod
    def calculate_inventory(self, df= pd.DataFrame()) -> pd.DataFrame:
        pass
    
    
    def gamma_kernel(self, duration_min, peak_min):
        t = np.arange(0, duration_min , 5)
        a = 2.0
        scale = (peak_min / (a-1))
        pdf = gamma.pdf(t, a, scale=scale)
        return pdf / np.sum(pdf)
    
class InsulinDriver(BioDriver):
    def calculate_inventory(self, df=pd.DataFrame()) -> pd.DataFrame:
        df = df.copy()
        curve = self.gamma_kernel(duration_min=360, peak_min=60)
        remaining_curve = 1- np.cumsum(curve)

        bolus = df['bolus'].fillna(0).values
        df['IOB'] = np.convolve(bolus, remaining_curve, mode='full')[:len(df)]

        return df
    
class carbsDriver(BioDriver):
    def calculate_inventory(self, df=pd.DataFrame()) -> pd.DataFrame:
        df = df.copy()
        curve = self.gamma_kernel(duration_min=240, peak_min=45)
        remaining_curve = 1- np.cumsum(curve)

        carbs = df['carbs'].fillna(0).values

        df['COB'] = np.convolve(carbs, remaining_curve, mode='full')[:len(df)]

        return df




