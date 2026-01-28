Personalized Blood Glucose Forecasting with Physics-Informed LSTMThis repository contains a high-performance, hybrid decision-support system designed to forecast blood glucose levels for Type 1 Diabetes patients across 30, 60, and 90-minute horizons. Unlike standard "black-box" models, this system integrates physiological priors (Insulin Sensitivity, Carb Ratios, and Circadian Drift) into a deep learning architecture.

🚀Key Features
Physiological Solver: A custom-built solver using Bayesian Shrinkage to calculate patient-specific metabolic coefficients (ISF, ICR) from historical "pure events."
Physics-Informed LSTM: A neural network architecture that processes not just raw sensor data, but also simulated Insulin on Board (IOB) and Carbs on Board (COB) kinetics.
Delta-Trajectory Learning: Instead of predicting absolute values, the model learns the glucose delta to minimize the "lag" effect common in CGM-based forecasting.
Quantile Forecasting (P10, P50, P90): Provides probabilistic outcomes to estimate clinical risk, allowing for hypoglycemia/hyperglycemia early warning systems.
Extended COB Modeling: Simulates the delayed impact of proteins and fats (the "Pizza Effect") through a dual-gamma absorption engine.
Circadian Awareness: Utilizes Slot Embeddings to account for time-dependent insulin resistance (Dawn Phenomenon, etc.).

🧠 System Architecture
The system operates through a two-stage pipeline:
The Solver Layer: Analyzes glucose response to isolated insulin and carb inputs to stabilize patient metabolic parameters.
The Deep Learning Layer: A multi-head architecture combining 1D-CNN (for local trend extraction) and LSTM (for long-term temporal dependencies), conditioned on the solver's output.

🛠 Tech StackDeep Learning: TensorFlow / Keras
Optimization: Scipy (Least Squares)
Data Processing: Pandas, NumPy, Scikit-learn
Visualization: Matplotlib (Clinical Dashboard)

📊 Evaluation Metric
The model is evaluated using the Mean Absolute Relative Difference (MARD), the gold standard in clinical diabetes research

📂├── data/               # Raw patient CSV files (ID-train.csv, ID-test.csv)
├── src/
│   ├── loader.py       # Data loading, cleaning, and synchronization
│   ├── solver.py       # Metabolic coefficient calculator (ISF/ICR)
│   ├── model.py        # Physics-Informed LSTM architecture
│   ├── drivers.py      # Kinetic simulators for Insulin and Carbohydrates
│   └── time_context.py # Circadian rhythm and slot encoding
├── main.py             # Orchestrator for training and clinical reporting
└── validator.py        # Empirical validation tool for clinical parameters


⚠️ DisclaimerThis software is developed for research and educational purposes only. It is NOT a medical device. Clinical decisions should never be based solely on these predictions.
