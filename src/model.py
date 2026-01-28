import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization, Concatenate, Flatten, Conv1D, Embedding
import tensorflow.keras.backend as K

class PhysicsLSTM:
    def __init__(self, look_back=48, prediction_horizon=18):
        self.look_back = look_back
        self.prediction_horizon = prediction_horizon
        self.quantiles = [0.1, 0.5, 0.9]
        self.feature_cols = [
            'insulin_rate', 'glucose_rate', 'IOB', 'COB', 'bolus', 'carbs', 
            'sin_time', 'cos_time', 'ema_fast', 'ema_slow', 'ema_diff', 
            'carb_absorption', 'ISF', 'ICR', 'dyn_ICR_factor', 'adapt_ISF_factor'
        ]
        self.model = self._build_model()

    def _quantile_loss(self, y_true, y_pred):
        losses = []
        alpha = 0.05
        for i, q in enumerate(self.quantiles):
            pred_q = y_pred[:, i::len(self.quantiles)]
            error = y_true - pred_q
            q_loss = K.maximum(q * error, (q - 1) * error)
            for h in range(self.prediction_horizon):
                weight = 1.0 + (alpha * h)
                losses.append(K.mean(q_loss[:, h]) * weight)
        return K.sum(losses)

    def _build_model(self):
        phys_input = Input(shape=(self.look_back, len(self.feature_cols)), name="phys_input")
        slot_input = Input(shape=(1,), name="slot_input")
        
        slot_emb = Embedding(4, 8)(slot_input)
        slot_emb = Flatten()(slot_emb)
        
        x = Conv1D(32, 5, activation='relu', padding='same')(phys_input)
        x = BatchNormalization()(x)
        x = LSTM(64, return_sequences=False)(x)
        x = Dropout(0.2)(x)
        
        combined = Concatenate()([x, slot_emb])
        x = Dense(64, activation='relu')(combined)
        output_layer = Dense(self.prediction_horizon * 3, activation='linear')(x)
        
        model = Model(inputs=[phys_input, slot_input], outputs=output_layer)
        model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=self._quantile_loss)
        return model

    def prepare_data(self, df):
        data_input = df[self.feature_cols].values
        glucose_values = df['glucose'].values
        slot_map = {"Morning (06-11)": 0, "Afternoon (11-16)": 1, "Evening (16-23)": 2, "Night (23-06)": 3}
        
        X_phys, X_slot, y_traj, weights = [], [], [], []
        limit = len(df) - self.look_back - self.prediction_horizon
        for i in range(limit):
            X_phys.append(data_input[i : (i + self.look_back)])
            X_slot.append(slot_map.get(df['slot'].iloc[i + self.look_back - 1], 0))
            g_now = glucose_values[i + self.look_back - 1]
            y_traj.append([glucose_values[i + self.look_back + h] - g_now for h in range(self.prediction_horizon)])
            # Confidence-Weighted Learning Weight
            weights.append(df['Conf'].iloc[i + self.look_back - 1])
            
        return [np.array(X_phys), np.array(X_slot)], np.array(y_traj), np.array(weights)