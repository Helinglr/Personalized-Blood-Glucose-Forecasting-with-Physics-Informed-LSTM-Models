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

        minutes = dates.minute + dates.hour * 60
        df['sin_time'] = np.sin(2 * np.pi * minutes / 1440)
        df['cos_time'] = np.cos(2 * np.pi * minutes / 1440)
        return df