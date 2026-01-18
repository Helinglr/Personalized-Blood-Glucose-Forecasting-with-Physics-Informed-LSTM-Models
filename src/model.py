from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.regularizers import l2

class GlucoseModel:
    def __init__(self, n_timesteps, n_features=9):
        self.n_timesteps = n_timesteps
        self.n_features = n_features
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        
        # Girdi Katmanı
        model.add(Input(shape=(self.n_timesteps, self.n_features)))
        
        # 1. LSTM Katmanı (Dropout ve L2 Regularization Eklendi)
        # return_sequences=True: Bir sonraki LSTM katmanına veri aktarır
        model.add(LSTM(
            64, 
            return_sequences=True,
            kernel_regularizer=l2(0.001), # Aşırı büyük değerleri (119 gibi) cezalandırır
            recurrent_regularizer=l2(0.001)
        ))
        model.add(Dropout(0.2)) # Nöronların %20'sini rastgele kapatır (Ezberi bozar)
        
        # 2. LSTM Katmanı
        model.add(LSTM(
            32, 
            return_sequences=False,
            kernel_regularizer=l2(0.001)
        ))
        model.add(Dropout(0.2)) # Yine %20 körleme
        
        # Çıktı Katmanı (Dense)
        model.add(Dense(16, activation='relu', kernel_regularizer=l2(0.001)))
        model.add(Dense(1)) # Tek bir glikoz değeri tahmin et
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model