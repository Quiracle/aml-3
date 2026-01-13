import numpy as np
import pandas as pd
import cma
from skrvm import RVR  # The Fast-RVM Regression class
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# =============================================================================
# 1. Data Preparation
# =============================================================================

def load_and_prep_data():
    """
    Load and prepare NASA CMAPSS FD001 dataset.
    
    Returns:
        X (pd.DataFrame): Feature matrix (settings + sensors)
        y (pd.Series): RUL (Remaining Useful Life) target variable
    """
    # --- Define Column Names ---
    index_names = ["unit_number", "time_cycles"]
    setting_names = ["setting_1", "setting_2", "setting_3"]
    sensor_names = [f"s_{i}" for i in range(1, 22)]
    col_names = index_names + setting_names + sensor_names
    
    # --- Load Training Data ---
    fd1 = pd.read_csv(
        "train_FD001.txt",
        sep=r"\s+",
        header=None,
        names=col_names,
        index_col=False,
        engine='python'
    )
    
    # --- Calculate RUL ---
    # For each unit, find the maximum time cycles
    max_time_cycles = fd1.groupby("unit_number")["time_cycles"].max().reset_index()
    max_time_cycles.columns = ["unit_number", "time_cycles_max"]
    
    # Merge back and calculate RUL
    fd1 = fd1.merge(max_time_cycles, on="unit_number", how="left")
    fd1["RUL"] = fd1["time_cycles_max"] - fd1["time_cycles"]
    
    # --- Extract Features and Target ---
    # Use only sensor features (s_1 to s_21), exclude settings
    feature_cols = sensor_names
    X = fd1[feature_cols].copy()
    y = fd1["RUL"].copy()
    
    return X, y

def clean_data(X):
    """
    Remove constant columns to prevent Singular Matrix errors in RVM.
    """
    # Remove columns with 0 variance
    selector = (X.var(axis=0) > 1e-6)
    X_clean = X[:, selector]
    return X_clean, selector

def create_windowed_features(X, window_size):
    """
    Create windowed features by stacking recent observations.
    
    Args:
        X (np.ndarray): Input data of shape (n_samples, n_features)
        window_size (int): Number of timesteps to include in each window
    
    Returns:
        np.ndarray: Windowed features of shape (n_samples, n_features * window_size)
    """
    n_samples, n_features = X.shape
    X_windowed = np.zeros((n_samples, n_features * window_size))
    
    for i in range(n_samples):
        start_idx = max(0, i - window_size + 1)
        window = X[start_idx:i+1]  # Get window up to current timestep
        
        # Pad with zeros if window is smaller than window_size
        if len(window) < window_size:
            padding = np.zeros((window_size - len(window), n_features))
            window = np.vstack([padding, window])
        
        # Flatten and store
        X_windowed[i] = window.flatten()
    
    return X_windowed

# Load Data
X, y = load_and_prep_data()

# Convert to NumPy arrays for processing
X_raw = X.values
y_array = y.values

# Clean & Scale (Crucial for RBF Kernels)
X_clean, valid_feats_idx = clean_data(X_raw)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clean)

# Split: Train (for RVM fit), Validation (for CMA-ES fitness), Test (Final eval)
X_train_full, X_temp, y_train_full, y_temp = train_test_split(X_scaled, y_array, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# --- SUBSAMPLING STRATEGY ---
# To speed up the CMA-ES search, we train on a small subset of data.
# RVM training is O(N^3) or O(NM^2), so keeping N small during search is vital.
subset_size = 300
subset_idx = np.random.choice(len(X_train_full), subset_size, replace=False)
X_sub = X_train_full[subset_idx]
y_sub = y_train_full[subset_idx]

print(f"Data Loaded. Full Train: {X_train_full.shape}, Validation: {X_val.shape}")
print(f"Optimization will run on subset size: {X_sub.shape[0]}")

# Check for NaN/Inf values
print(f"NaN values in X_sub: {np.isnan(X_sub).sum()}, y_sub: {np.isnan(y_sub).sum()}")
print(f"Inf values in X_sub: {np.isinf(X_sub).sum()}, y_sub: {np.isinf(y_sub).sum()}")
print(f"Data ranges - X_sub: [{X_sub.min():.3f}, {X_sub.max():.3f}], y_sub: [{y_sub.min():.3f}, {y_sub.max():.3f}]")

# =============================================================================
# 2. Fitness Function (The Bi-Level Optimization)
# =============================================================================

def rvm_fitness(theta):
    """
    Objective function for CMA-ES to minimize.
    
    Args:
        theta (np.array): 
            theta[0]: Log of Window Size (minimum 1, maximum 50)
            theta[1]: Log of Kernel Width (sigma)
            theta[2:]: Feature selection weights (continuous)
    
    Returns:
        float: RMSE on Validation Set (or high penalty if invalid)
    """
    PENALTY = 1e6
    
    # --- A. Decode Hyperparameters ---
    
    # 1. Window Size
    # Map from log-space to positive integer [1, 50]
    try:
        window_size = int(np.clip(np.exp(theta[0]), 1, 50))
    except (OverflowError, ValueError):
        return PENALTY
    
    # 2. Kernel Width (sigma)
    # Map from log-space to positive real space
    try:
        sigma = np.exp(theta[1])
    except OverflowError:
        return PENALTY

    # Bounds check for stability
    if sigma < 1e-3 or sigma > 1e3 or np.isnan(sigma):
        return PENALTY
    
    # Convert sigma to sklearn's 'gamma' parameter
    # Formula: gamma = 1 / (2 * sigma^2)
    gamma_val = 1.0 / (2.0 * sigma**2)

    # 3. Feature Masking
    # Map continuous CMA-ES output to binary mask
    feature_weights = theta[2:]
    mask = feature_weights > 0 # Threshold at 0
    
    # Constraint: Must select at least 1 feature
    if np.sum(mask) == 0:
        return PENALTY

    # --- B. Create Windowed Features ---
    # First apply feature mask, then create windows
    try:
        # Select features first
        X_sub_masked = X_sub[:, mask]
        X_val_masked = X_val[:, mask]
        
        # Apply rolling window to create temporal features
        X_tr_windowed = create_windowed_features(X_sub_masked, window_size)
        X_val_windowed = create_windowed_features(X_val_masked, window_size)
        
        # Skip first window_size-1 samples (not enough history)
        X_tr_windowed = X_tr_windowed[window_size-1:]
        y_sub_windowed = y_sub[window_size-1:]
        
        X_val_windowed = X_val_windowed[window_size-1:]
        y_val_windowed = y_val[window_size-1:]
        
        if len(X_tr_windowed) == 0 or len(X_val_windowed) == 0:
            return PENALTY
    except Exception as e:
        print(f"Windowing error: {e}")
        return PENALTY
    
    try:
        # Initialize Fast RVM (RVR)
        # RVR uses coef1 parameter for gamma in RBF kernel
        model = RVR(kernel='rbf', coef1=gamma_val)
        
        # Fit on SUBSET
        model.fit(X_tr_windowed, y_sub_windowed)
        
        # Check if any relevance vectors were found
        if model.relevance_.shape[0] == 0:
            return PENALTY
            
        # --- C. Evaluate (Outer Loop Objective) ---
        preds = model.predict(X_val_windowed)
        rmse = np.sqrt(mean_squared_error(y_val_windowed, preds))
        
        # Optional: Soft Regularization for Sparsity (Occam's Razor)
        # Adds a tiny penalty for every extra feature used
        sparsity_penalty = 0.01 * np.sum(mask)
        
        return rmse + sparsity_penalty

    except Exception as e:
        # Print error for debugging
        print(f"RVM fitness error: {e}")
        return PENALTY

# =============================================================================
# 3. Run CMA-ES (Outer Loop)
# =============================================================================

n_opt_features = X_sub.shape[1]
# Initial Guess: 
# theta[0] = ln(5) -> window_size = exp(ln(5)) = 5
# theta[1] = 0.0 -> sigma = exp(0) = 1.0
# theta[2:] = 1.0 -> All features selected initially
start_theta = np.concatenate([[np.log(10)], [0.0], np.ones(n_opt_features)])
sigma_iter = 0.5 # Step size for exploration

print("\nStarting CMA-ES Optimization...")
print("Optimizing Window Size, Kernel Width (Sigma) and Feature Mask...")

# Run CMA-ES
# options={'maxiter': 50} limits the run time. Increase for better results.
es = cma.CMAEvolutionStrategy(start_theta, sigma_iter, {'maxiter': 150, 'verbose': -1})
es.optimize(rvm_fitness)

# Get best results
best_theta = es.result.xbest
best_fitness = es.result.fbest

# =============================================================================
# 4. Final Evaluation & Report
# =============================================================================

# Decode optimal parameters
final_window_size = int(np.clip(np.exp(best_theta[0]), 1, 50))
final_sigma = np.exp(best_theta[1])
final_gamma = 1.0 / (2.0 * final_sigma**2)
final_mask = best_theta[2:] > 0
selected_indices = np.where(final_mask)[0]

print("\n" + "="*50)
print(" OPTIMIZATION RESULTS")
print("="*50)
print(f"Best Validation RMSE (approx): {best_fitness:.4f}")
print(f"Optimal Window Size: {final_window_size}")
print(f"Optimal Sigma: {final_sigma:.4f} (Gamma: {final_gamma:.4f})")
print(f"Selected Features: {len(selected_indices)} out of {n_opt_features}")
print(f"Feature Indices: {selected_indices}")

# --- Retrain on FULL Training Set ---
print("\nRetraining Final Model on FULL dataset...")

# Apply feature mask first
X_train_masked = X_train_full[:, final_mask]
X_test_masked = X_test[:, final_mask]

# Create windowed features with optimal window size
X_train_windowed = create_windowed_features(X_train_masked, final_window_size)
X_test_windowed = create_windowed_features(X_test_masked, final_window_size)

# Remove first window_size-1 samples (not enough history)
X_train_final = X_train_windowed[final_window_size-1:]
y_train_final = y_train_full[final_window_size-1:]

X_test_final = X_test_windowed[final_window_size-1:]
y_test_final = y_test[final_window_size-1:]

final_model = RVR(kernel='rbf', coef1=final_gamma)
final_model.fit(X_train_final, y_train_final)

# Final Test Prediction
test_preds = final_model.predict(X_test_final)
test_rmse = np.sqrt(mean_squared_error(y_test_final, test_preds))
r_squared = final_model.score(X_test_final, y_test_final)
n_rv = final_model.relevance_.shape[0]

print("-" * 30)
print(f"FINAL TEST RMSE: {test_rmse:.4f}")
print(f"FINAL TEST R^2:  {r_squared:.4f}")
print(f"Sparsity: {n_rv} Relevance Vectors (from {len(X_train_final)} samples)")
print("-" * 30)

# Example of obtaining predictive variance (Uncertainty)
# RVR provides MSE; std = sqrt(MSE)
_, test_mse = final_model.predict(X_test_final, eval_MSE=True)
test_std = np.sqrt(test_mse)
print(f"Avg Predictive Uncertainty (StdDev): {np.mean(test_std):.4f}")