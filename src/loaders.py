import pandas as pd
import os

class OhioLoader:
    def load(self, file_path: str) -> pd.DataFrame:
        """
        Ohio Veri setini yükler ve standart formata çevirir.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dosya bulunamadı: {file_path}")
            
        df = pd.read_csv(file_path)
        
        # Timestamp Düzeltme
        if '5minute_intervals_timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['5minute_intervals_timestamp'] * 60, unit='s')
        elif 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
        df = df.set_index('timestamp').sort_index()
        
        # NaN temizliği ve İsimlendirme
        # get methodu ile sütun yoksa hata vermesini engelleriz
        df['bolus'] = df.get('bolus', pd.Series(0)).fillna(0)
        
        # Farklı isimlendirmeleri yönet (carbInput veya carbs)
        if 'carbInput' in df.columns:
            df['carbs'] = df['carbInput'].fillna(0)
        else:
            df['carbs'] = df.get('carbs', pd.Series(0)).fillna(0)
            
        df = df.rename(columns={'cbg': 'glucose'})
        
        # Sadece gerekli sütunları döndür
        return df[['glucose', 'bolus', 'carbs']].dropna()