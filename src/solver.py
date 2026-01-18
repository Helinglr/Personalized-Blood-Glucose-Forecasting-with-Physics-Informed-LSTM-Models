import numpy as np
from src.drivers import InsulinDriver, carbsDriver

class ParameterSolver:
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler
        self.ins_driver = InsulinDriver()
        self.carb_driver = carbsDriver()
        
        # --- DİNAMİK HAFIZA AYARI (HATA ÇÖZÜMÜ BURADA) ---
        # Modelin eğitim sırasında kaç adım kullandığını otomatik algılar.
        # model.input_shape genelde (None, LOOK_BACK, Features) döner.
        self.look_back = model.input_shape[1] 
        
        # Kernel Eğrileri
        self.ins_decay_curve = 1.0 - np.cumsum(self.ins_driver.gamma_kernel(360, 60))
        self.carb_decay_curve = 1.0 - np.cumsum(self.carb_driver.gamma_kernel(240, 45))

    def _get_time_features(self, hour):
        """Belirli bir saat için zaman vektörü üretir."""
        minutes = hour * 60
        sin_t = np.sin(2 * np.pi * minutes / 1440)
        cos_t = np.cos(2 * np.pi * minutes / 1440)
        
        is_m = 1 if 6 <= hour < 12 else 0
        is_a = 1 if 12 <= hour < 18 else 0
        is_e = 1 if 18 <= hour < 24 else 0
        is_n = 1 if 0 <= hour < 6 else 0
        
        return [sin_t, cos_t, is_m, is_a, is_e, is_n]

    def _run_single_simulation(self, base_glucose, time_vector, bolus_scenario=0, carb_scenario=0, steps=48):
        trajectory = []
        
        # Başlangıç Glikozunu Scale Et
        dummy_row = [[base_glucose] + [0]*8]
        base_glucose_scaled = self.scaler.transform(dummy_row)[0][0]
        
        # --- DİNAMİK HAFIZA KULLANIMI ---
        # Eskiden 12 sabitti, şimdi self.look_back (24) kullanıyor.
        current_state = np.zeros((1, self.look_back, 9))
        
        for i in range(self.look_back):
            current_state[:, i, 0] = base_glucose_scaled
            current_state[:, i, 3:] = time_vector

        # Simülasyon Döngüsü
        for t in range(steps):
            # 1. Driver
            curr_iob = 0
            curr_cob = 0
            if t < len(self.ins_decay_curve):
                curr_iob = bolus_scenario * self.ins_decay_curve[t]
            if t < len(self.carb_decay_curve):
                curr_cob = carb_scenario * self.carb_decay_curve[t]
            
            # 2. State Hazırlama
            dummy_input = [[base_glucose, curr_iob, curr_cob] + [0]*6]
            scaled_vals = self.scaler.transform(dummy_input)[0]
            
            next_step = current_state[:, -1, :].copy()
            next_step[0, 1] = scaled_vals[1] # IOB
            next_step[0, 2] = scaled_vals[2] # COB
            
            # 3. Model Tahmini
            temp_state = current_state.copy()
            temp_state[:, -1, :] = next_step
            
            pred_scaled = self.model.predict(temp_state, verbose=0)[0][0]
            
            # Geri Çevirme
            pred_real = (pred_scaled * self.scaler.data_range_[0]) + self.scaler.data_min_[0]
            trajectory.append(pred_real)
            
            # 4. Kaydır
            current_state = np.roll(current_state, -1, axis=1)
            final_step = next_step.copy()
            final_step[0, 0] = pred_scaled 
            current_state[:, -1, :] = final_step
            
        return np.array(trajectory)

    def extract_parameters_by_period(self, base_glucose=140):
        period_ranges = {
            'Sabah': range(6, 12),
            'Öğle': range(12, 18),
            'Akşam': range(18, 24),
            'Gece': range(0, 6)
        }
        
        final_results = {}
        # Progress Bar için kütüphane kullanmıyoruz, manuel nokta koyuyoruz
        print(f"   > Parametre Taraması (Hafıza: {self.look_back} adım)...", end=" ")
        
        for p_name, hours in period_ranges.items():
            idf_samples = []
            icr_samples = []
            
            for h in hours:
                print(".", end="", flush=True) # İlerlemeyi göster
                t_vec = self._get_time_features(h)
                
                traj_base = self._run_single_simulation(base_glucose, t_vec, bolus_scenario=0, carb_scenario=0)
                traj_ins = self._run_single_simulation(base_glucose, t_vec, bolus_scenario=1.0, carb_scenario=0)
                traj_carb = self._run_single_simulation(base_glucose, t_vec, bolus_scenario=0, carb_scenario=10.0)
                
                max_drop = np.max(traj_base - traj_ins)
                max_rise = np.max(traj_carb - traj_base)
                
                carb_sens = max_rise / 10.0
                icr = max_drop / carb_sens if carb_sens > 0 else 0
                
                idf_samples.append(max_drop)
                icr_samples.append(icr)
            
            final_results[p_name] = {
                "IDF": round(np.median(idf_samples), 2),
                "ICR": round(np.median(icr_samples), 2)
            }
            
        print(" Bitti!")
        return final_results