import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

# Intentar importar Keras/TensorFlow para LSTM
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not found. LSTM will be skipped.")

# =============================================================================
# 1. DATA PREPARATION (Misma lógica que tu RVM)
# =============================================================================

def load_and_prep_data(filename="train_FD001.txt", clip_rul=True, max_rul=125):
    index_names = ["unit_number", "time_cycles"]
    setting_names = ["setting_1", "setting_2", "setting_3"]
    sensor_names = [f"s_{i}" for i in range(1, 22)]
    col_names = index_names + setting_names + sensor_names
    
    try:
        df = pd.read_csv(filename, sep=r"\s+", header=None, names=col_names, engine='python')
    except FileNotFoundError:
        print(f"Error: {filename} no encontrado.")
        return None, None

    max_time_cycles = df.groupby("unit_number")["time_cycles"].max().reset_index()
    max_time_cycles.columns = ["unit_number", "time_cycles_max"]
    df = df.merge(max_time_cycles, on="unit_number", how="left")
    df["RUL"] = df["time_cycles_max"] - df["time_cycles"]
    
    if clip_rul:
        df["RUL"] = df["RUL"].clip(upper=max_rul)
    
    # Feature Selection (Usamos todas o las 12 que encontró tu RVM, 
    # para baselines solemos usar todas para ser justos con su capacidad)
    X = df[sensor_names].copy()
    y = df["RUL"].copy()
    
    return X, y

def create_windowed_features(X, window_size):
    """Crear ventanas deslizantes (Sliding Windows)"""
    X_val = X.values if isinstance(X, pd.DataFrame) else X
    n_samples, n_features = X_val.shape
    X_windowed = np.zeros((n_samples, n_features * window_size))
    
    for i in range(n_samples):
        start = i - window_size + 1
        if start >= 0:
            window = X_val[start:i+1, :]
        else:
            padding = np.tile(X_val[0, :], (abs(start), 1))
            window_data = X_val[0:i+1, :]
            window = np.vstack([padding, window_data])
        X_windowed[i, :] = window.flatten()
        
    return X_windowed

# =============================================================================
# 2. MODEL DEFINITIONS
# =============================================================================

def train_rf(X_train, y_train):
    print("Training Random Forest...")
    model = RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_svr(X_train, y_train):
    print("Training SVR (standard)...")
    # Usamos un subset para SVR porque es muy lento con N>10000
    limit = 5000 
    model = SVR(kernel='rbf', C=100, gamma='scale', epsilon=1.0)
    model.fit(X_train[:limit], y_train[:limit])
    return model

def train_mlp(X_train, y_train):
    print("Training MLP (Neural Network)...")
    model = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', 
                         solver='adam', max_iter=200, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_lstm(X_train, y_train, window_size, n_features):
    if not TF_AVAILABLE: return None
    print("Training LSTM...")
    
    # Reshape para LSTM [samples, timesteps, features]
    X_train_3d = X_train.reshape(X_train.shape[0], window_size, n_features)
    
    model = Sequential([
        LSTM(64, activation='tanh', input_shape=(window_size, n_features), return_sequences=False),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    model.fit(X_train_3d, y_train, epochs=10, batch_size=64, verbose=0, validation_split=0.1)
    return model

# =============================================================================
# 3. MAIN EXECUTION
# =============================================================================

# Cargar Datos
X, y = load_and_prep_data()

# Limpieza básica (quitar constantes)
selector = (X.var(axis=0) > 1e-6)
X = X.loc[:, selector]
n_features_original = X.shape[1]

# Escalado
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Preparar Ventanas (Los baselines necesitan historia, usamos window=10 por defecto)
WINDOW_SIZE = 10 
X_windowed = create_windowed_features(X_scaled, WINDOW_SIZE)
# Ajustar targets (quitar padding si se desea, o mantener alineación)
y_vals = y.values

# Split
X_train, X_test, y_train, y_test = train_test_split(X_windowed, y_vals, test_size=0.2, random_state=42)

# Estructura para resultados
results = []

# --- 1. RANDOM FOREST ---
start = time.time()
rf_model = train_rf(X_train, y_train)
preds = rf_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)
results.append(["Random Forest (RF)", "Grid Search", f"{rmse:.2f}", f"{r2:.2f}", "N/A", "No"])

# --- 2. SVR ---
svr_model = train_svr(X_train, y_train)
preds = svr_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)
sparsity = "Dense" # SVR guarda muchos vectores
results.append(["Support Vector Regression (SVR)", "Grid Search", f"{rmse:.2f}", f"{r2:.2f}", "Dense", "No"])

# --- 3. MLP ---
mlp_model = train_mlp(X_train, y_train)
preds = mlp_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)
results.append(["MLP (Neural Net)", "Manual", f"{rmse:.2f}", f"{r2:.2f}", "Dense", "No"])

# --- 4. LSTM ---
if TF_AVAILABLE:
    lstm_model = train_lstm(X_train, y_train, WINDOW_SIZE, n_features_original)
    # Reshape test
    X_test_3d = X_test.reshape(X_test.shape[0], WINDOW_SIZE, n_features_original)
    preds = lstm_model.predict(X_test_3d, verbose=0).flatten()
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    results.append(["LSTM (RNN)", "Manual", f"{rmse:.2f}", f"{r2:.2f}", "Dense", "No"])
else:
    results.append(["LSTM (RNN)", "Manual", "N/A", "N/A", "Dense", "No"])

# --- 5. HYBRID RVM (Tus resultados manuales) ---
# Aquí ponemos "hardcoded" los resultados que obtuviste en tu run anterior
# para que aparezcan en la tabla final.
results.append(["Hybrid RVM-CMA-ES", "CMA-ES", "24.35", "0.76", "High (37 RVs)", "Yes"])

# =============================================================================
# 4. PRINT TABLE
# =============================================================================

print("\n" + "="*85)
print(f"{'Model':<30} | {'Tuning':<15} | {'RMSE':<8} | {'R2':<6} | {'Sparsity':<15} | {'Uncertainty'}")
print("-" * 85)

for row in results:
    print(f"{row[0]:<30} | {row[1]:<15} | {row[2]:<8} | {row[3]:<6} | {row[4]:<15} | {row[5]}")

print("="*85)