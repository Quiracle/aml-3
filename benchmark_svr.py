import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import time

from main import create_windowed_features, load_and_prep_data

X, y = load_and_prep_data(clip_rul=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

WINDOW_SIZE = 1 
X_windowed = create_windowed_features(X_scaled, WINDOW_SIZE)
y_windowed = y[WINDOW_SIZE-1:].values
X_windowed = X_windowed[WINDOW_SIZE-1:]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_windowed, y_windowed, test_size=0.2, random_state=42)

print("Training SVR (this may take a while)...")
start_time = time.time()

# Reasonable 'default' parameters for SVR on this dataset
model = SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1)
model.fit(X_train, y_train)

end_time = time.time()

# 5. Evaluate the model
preds = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
n_support = model.support_vectors_.shape[0]

print("\n" + "="*30)
print(" SVR BENCHMARK RESULTS")
print("="*30)
print(f"RMSE: {rmse:.4f}")
print(f"Training Time: {end_time - start_time:.2f} sec")
print(f"Support Vectors: {n_support} (vs 37 of RVM!)")
print(f"Sparsity Ratio: {n_support/len(X_train)*100:.2f}% of the training data retained as support vectors")