import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization, Concatenate, Flatten, Conv1D, MaxPooling1D
import tensorflow.keras.backend as K

class PhysicsLSTM:
    def __init__(self, look_back=48, prediction_horizon=6): # 30 dk
        self.look_back = look_back
        self.prediction_horizon = prediction_horizon
        self.model = self._build_model()

    def _hybrid_loss(self, y_true, y_pred):
        """
        HUBER + ANTI-LAG KAYIP FONKSİYONU
        Huber Loss: Gürültüye (outliers) karşı dirençlidir.
        """
        # 1. Huber Loss (Standart Hata yerine bunu kullanıyoruz)
        # Hata küçükse karesini, büyükse mutlak değerini alır.
        # Bu, modelin anlık zıplamalara (gürültüye) aşırı tepki vermesini engeller.
        huber = tf.keras.losses.Huber(delta=0.05)(y_true, y_pred)
        
        # 2. Tembellik Cezası (Anti-Lag)
        # Modelin "kopyala-yapıştır" yapmasını engellemek için.
        # (Yöntem olarak basit bir fark cezası kullanıyoruz)
        diff = y_true - y_pred
        
        # Gerçek artarken model düşüyorsa (veya tam tersi) ekstra ceza
        # Bunu basitçe "Büyük Hata Cezası" olarak ekleyebiliriz
        large_error_penalty = K.maximum(K.abs(diff) - 0.05, 0.0) * 10.0
        
        return huber + large_error_penalty

    def _build_model(self):
        # Girdi: [Look_Back, 10] (Feature sayısı 1 artarak 10 oldu: EMA eklendi)
        input_layer = Input(shape=(self.look_back, 10))
        
        # --- 1. CNN BLOKU (Gürültü Temizleyici) ---
        # 1D Konvolüsyon: Verideki anlık titreşimleri süzer, trendi çıkarır.
        x_cnn = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(input_layer)
        x_cnn = BatchNormalization()(x_cnn)
        # MaxPooling kullanmıyoruz çünkü zaman serisinin uzunluğunu korumak istiyoruz
        
        # --- 2. LSTM BLOKU (Zaman Öğrenici) ---
        x_lstm = LSTM(128, return_sequences=True)(x_cnn) # CNN çıktısını alır
        x_lstm = Dropout(0.3)(x_lstm)
        x_lstm = BatchNormalization()(x_lstm)
        
        x_lstm = LSTM(64, return_sequences=False)(x_lstm)
        x_lstm = Dropout(0.3)(x_lstm)
        x_lstm = BatchNormalization()(x_lstm)
        
        # --- 3. WIDE BLOKU (Anlık Tepki) ---
        # Son anlık durumu direkt çıkışa bağla (Lag önlemek için)
        x_wide = Flatten()(input_layer)
        x_wide = Dense(32, activation='relu')(x_wide)
        
        # BİRLEŞTİRME
        combined = Concatenate()([x_lstm, x_wide])
        x = Dense(64, activation='relu')(combined)
        
        # ÇIKIŞ
        output_layer = Dense(1, activation='linear')(x)
        
        model = Model(inputs=input_layer, outputs=output_layer)
        
        # Adam Optimizer (Hafifçe yavaşlatılmış öğrenme hızı)
        opt = tf.keras.optimizers.Adam(learning_rate=0.0005)
        model.compile(optimizer=opt, loss=self._hybrid_loss)
        return model

    def prepare_data(self, scaled_df):
        # ARTIK 10 FEATURE VAR (EMA Eklendi)
        cols = ['glucose', 'insulin_rate', 'glucose_rate', 'IOB', 'COB', 'bolus', 'carbs', 'sin_time', 'cos_time', 'ema']
        
        data_input = scaled_df[cols].values
        data_target = scaled_df['glucose'].values 
        
        X, y = [], []
        limit = len(data_input) - self.look_back - self.prediction_horizon
        
        for i in range(limit):
            X.append(data_input[i : (i + self.look_back), :])
            future_idx = i + self.look_back - 1 + self.prediction_horizon
            y.append(data_target[future_idx])
            
        return np.array(X), np.array(y), None