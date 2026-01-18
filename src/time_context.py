import numpy as np
import pandas as pd

class TimeContext:
    def add_context(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        minutes = df.index.minute+ df.index.hour * 60

        df['sin_time'] = np.sin(2 * np.pi * minutes / 1440)
        df['cos_time'] = np.cos(2 * np.pi * minutes / 1440)

        df['is_morning'] = ((df.index.hour >= 6) & (df.index.hour < 12)).astype(int)
        df['is_afternoon'] = ((df.index.hour >= 12) & (df.index.hour < 18)).astype(int)
        df['is_evening'] = ((df.index.hour >= 18) & (df.index.hour < 24)).astype(int)
        df['is_night'] = ((df.index.hour >= 0) & (df.index.hour < 6)).astype(int)

        return df
    
