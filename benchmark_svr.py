import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import time

from main import create_windowed_features, load_and_prep_data

# --- COPIA AQUÍ TUS FUNCIONES load_and_prep_data Y create_windowed_features ---
# (O impórtalas si las tienes en un módulo)

# 1. Cargar Datos
X, y = load_and_prep_data(clip_rul=True) # ¡Importante usar el mismo clipping!

# 2. Preprocesado Básico (Sin features optimizadas, usamos todas para ser justos con SVR)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Usamos Window Size fijo de 10 (estándar razonable) o 1 (para comparar con tu resultado actual)
WINDOW_SIZE = 1 # Pon 1 para comparar manzanas con manzanas con tu RVM actual
X_windowed = create_windowed_features(X_scaled, WINDOW_SIZE)
# Ajustar y (quitar primeros N-1)
y_windowed = y[WINDOW_SIZE-1:].values
X_windowed = X_windowed[WINDOW_SIZE-1:]

# 3. Split (Mismo split 60/20/20 aprox)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_windowed, y_windowed, test_size=0.2, random_state=42)

# 4. Entrenar SVR Estándar
print("Entrenando SVR (esto puede tardar un poco)...")
start_time = time.time()

# Parámetros 'por defecto' razonables
model = SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1)
model.fit(X_train, y_train)

end_time = time.time()

# 5. Evaluar
preds = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
n_support = model.support_vectors_.shape[0]

print("\n" + "="*30)
print(" SVR BENCHMARK RESULTS")
print("="*30)
print(f"RMSE: {rmse:.4f}")
print(f"Training Time: {end_time - start_time:.2f} sec")
print(f"Support Vectors: {n_support} (vs 37 de tu RVM!)")
print(f"Sparsity Ratio: {n_support/len(X_train)*100:.2f}% de los datos retenidos")