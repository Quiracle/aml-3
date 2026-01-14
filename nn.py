import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def load_and_prep_data(train_path="train_FD001.txt"):
    """Load FD001 and compute RUL for every timestep.

    Returns features (np.ndarray) and targets (np.ndarray).
    """
    col_idx = ["unit_number", "time_cycles"]
    settings = ["setting_1", "setting_2", "setting_3"]
    sensors = [f"s_{i}" for i in range(1, 22)]
    col_names = col_idx + settings + sensors

    df = pd.read_csv(train_path, sep=r"\s+", header=None, names=col_names, engine="python")

    # compute RUL per engine
    max_cycles = df.groupby("unit_number")["time_cycles"].max().reset_index()
    max_cycles.columns = ["unit_number", "time_cycles_max"]
    df = df.merge(max_cycles, on="unit_number", how="left")
    df["RUL"] = df["time_cycles_max"] - df["time_cycles"]

    feature_cols = sensors  # use sensor readings
    X = df[feature_cols].copy()
    y = df["RUL"].copy()

    return X.values.astype(np.float32), y.values.astype(np.float32)


def remove_constant_columns(X):
    var = X.var(axis=0)
    mask = var > 1e-6
    return X[:, mask], mask


class RegressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class MLPRegressor(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def get_device():
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_model(X_train, y_train, X_val, y_val, device, epochs=50, batch_size=256, lr=1e-3):
    in_dim = X_train.shape[1]
    model = MLPRegressor(in_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    train_ds = RegressionDataset(X_train, y_train)
    val_ds = RegressionDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            preds = model(xb)
            loss = loss_fn(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)

        train_loss = running / len(train_ds)

        # validation
        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb)
                val_running += loss_fn(preds, yb).item() * xb.size(0)

        val_loss = val_running / len(val_ds)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch {epoch}/{epochs}  Train MSE: {train_loss:.4f}  Val MSE: {val_loss:.4f}")

    # load best
    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def evaluate(model, X_test, y_test, device):
    model.eval()
    ds = RegressionDataset(X_test, y_test)
    loader = DataLoader(ds, batch_size=512, shuffle=False)
    preds = []
    trues = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            out = model(xb).cpu().numpy()
            preds.append(out)
            trues.append(yb.numpy())

    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    mse = mean_squared_error(trues, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(trues, preds)
    return rmse, r2, preds, trues


def main(args):
    # load train data and prepare train/val splits
    X_train_all, y_train_all = load_and_prep_data(args.train_path)

    # remove constant columns based on train
    X_train_all, mask = remove_constant_columns(X_train_all)

    scaler = StandardScaler()
    X_train_all = scaler.fit_transform(X_train_all)

    X_train, X_val, y_train, y_val = train_test_split(X_train_all, y_train_all, test_size=0.2, random_state=42)

    # load test data (uses separate RUL file) and apply same preprocessing
    def load_test_data(test_path, rul_path):
        col_idx = ["unit_number", "time_cycles"]
        settings = ["setting_1", "setting_2", "setting_3"]
        sensors = [f"s_{i}" for i in range(1, 22)]
        col_names = col_idx + settings + sensors

        df_test = pd.read_csv(test_path, sep=r"\s+", header=None, names=col_names, engine="python")

        # read RULs - one per unit
        rul = pd.read_csv(rul_path, sep=r"\s+", header=None, engine="python").iloc[:, 0].values

        # For each unit in test, compute RUL for each timestep as RUL_file + (T_test_max - time_cycle)
        max_cycles = df_test.groupby("unit_number")["time_cycles"].max().reset_index()
        max_cycles.columns = ["unit_number", "time_cycles_max"]
        df_test = df_test.merge(max_cycles, on="unit_number", how="left")

        # map unit_number -> RUL from file (units are 1-indexed)
        rul_map = {i + 1: int(rul[i]) for i in range(len(rul))}
        df_test["RUL"] = df_test.apply(lambda row: rul_map[int(row["unit_number"])] + (row["time_cycles_max"] - row["time_cycles"]), axis=1)

        X_t = df_test[sensors].values.astype(np.float32)
        y_t = df_test["RUL"].values.astype(np.float32)
        return X_t, y_t

    X_test_raw, y_test_raw = load_test_data(args.test_path, args.rul_path)

    # apply mask and scaler
    X_test = X_test_raw[:, mask]
    X_test = scaler.transform(X_test)

    device = get_device()
    print(f"Using device: {device}")

    model = train_model(X_train, y_train, X_val, y_val, device, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)

    rmse, r2, preds, trues = evaluate(model, X_test, y_test_raw, device)
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test R^2: {r2:.4f}")

    # save model and scaler
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pth"))
    np.save(os.path.join(out_dir, "mask.npy"), mask)
    np.save(os.path.join(out_dir, "scaler_mean.npy"), scaler.mean_)
    np.save(os.path.join(out_dir, "scaler_scale.npy"), scaler.scale_)
    print(f"Model and preprocessing saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50, help="number of training epochs")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out-dir", dest="out_dir", default="artifacts")
    parser.add_argument("--train-path", dest="train_path", default="train_FD001.txt")
    parser.add_argument("--test-path", dest="test_path", default="test_FD001.txt")
    parser.add_argument("--rul-path", dest="rul_path", default="RUL_FD001.txt")
    args = parser.parse_args()
    main(args)


