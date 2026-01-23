import pandas as pd
import os

class OhioLoader:
    def load(self, file_path: str) -> pd.DataFrame:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dosya yok: {file_path}")
        return pd.read_csv(file_path)