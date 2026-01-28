import numpy as np
import pandas as pd

class TimeContext:
    def add_context(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if isinstance(df.index, pd.DatetimeIndex):
            dates = df.index
        elif 'timestamp' in df.columns:
            dates = pd.to_datetime(df['timestamp'])
        else:
            dates = pd.date_range(start='1/1/2022', periods=len(df), freq='5min')

        hours = dates.hour
        minutes = dates.minute + hours * 60
        
        # Physiological slot definitions
        def get_slot(h):
            if 6 <= h < 11: return "Morning (06-11)"
            elif 11 <= h < 16: return "Afternoon (11-16)"
            elif 16 <= h < 23: return "Evening (16-23)"
            else: return "Night (23-06)"

        df['slot'] = [get_slot(h) for h in hours]
        
        #Even if the AI is inactive, cyclical time features should remain in the dataset.
        df['sin_time'] = np.sin(2 * np.pi * minutes / 1440)
        df['cos_time'] = np.cos(2 * np.pi * minutes / 1440)
        return df
