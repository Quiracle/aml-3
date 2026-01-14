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


def load_train_df(train_path="train_FD001.txt"):
    col_idx = ["unit_number", "time_cycles"]
    settings = ["setting_1", "setting_2", "setting_3"]
    sensors = [f"s_{i}" for i in range(1, 22)]
    col_names = col_idx + settings + sensors
    df = pd.read_csv(train_path, sep=r"\s+", header=None, names=col_names, engine="python")
    max_cycles = df.groupby("unit_number")["time_cycles"].max().reset_index()
    max_cycles.columns = ["unit_number", "time_cycles_max"]
    df = df.merge(max_cycles, on="unit_number", how="left")
    df["RUL"] = df["time_cycles_max"] - df["time_cycles"]
    return df


def load_test_df(test_path, rul_path):
    col_idx = ["unit_number", "time_cycles"]
    settings = ["setting_1", "setting_2", "setting_3"]
    sensors = [f"s_{i}" for i in range(1, 22)]
    col_names = col_idx + settings + sensors
    df_test = pd.read_csv(test_path, sep=r"\s+", header=None, names=col_names, engine="python")
    rul = pd.read_csv(rul_path, sep=r"\s+", header=None, engine="python").iloc[:, 0].values
    max_cycles = df_test.groupby("unit_number")["time_cycles"].max().reset_index()
    max_cycles.columns = ["unit_number", "time_cycles_max"]
    df_test = df_test.merge(max_cycles, on="unit_number", how="left")
    rul_map = {i + 1: int(rul[i]) for i in range(len(rul))}
    df_test["RUL"] = df_test.apply(lambda row: rul_map[int(row["unit_number"])] + (row["time_cycles_max"] - row["time_cycles"]), axis=1)
    return df_test


def remove_constant_columns(X):
    var = X.var(axis=0)
    mask = var > 1e-6
    return X[:, mask], mask


def create_sequences(df, sensors, window_size=30):
    # sliding windows per unit
    seqs = []
    targets = []
    for unit, grp in df.groupby("unit_number"):
        arr = grp[sensors].values.astype(np.float32)
        rul = grp["RUL"].values.astype(np.float32)
        n = len(arr)
        for t in range(n):
            start = max(0, t - window_size + 1)
            window = arr[start:t+1]
            if window.shape[0] < window_size:
                pad = np.zeros((window_size - window.shape[0], arr.shape[1]), dtype=np.float32)
                window = np.vstack([pad, window])
            seqs.append(window)
            targets.append(rul[t])
    return np.stack(seqs), np.array(targets, dtype=np.float32)


class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)


class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2, bidirectional=False):
        super().__init__()
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        head_in = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Linear(head_in, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


def get_device():
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_model(model, train_loader, val_loader, device, epochs=50, lr=1e-3, weight_decay=0.0, clip_norm=0.0, scheduler_name="none", patience=10):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    best_val = float("inf")
    best_state = None
    history = {"train_loss": [], "val_loss": []}

    # scheduler setup
    scheduler = None
    if scheduler_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    elif scheduler_name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    no_improve = 0
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
            if clip_norm and clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
            running += loss.item() * xb.size(0)
        train_loss = running / len(train_loader.dataset)

        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb)
                val_running += loss_fn(preds, yb).item() * xb.size(0)
        val_loss = val_running / len(val_loader.dataset)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # scheduler step
        if scheduler_name == "plateau" and scheduler is not None:
            scheduler.step(val_loss)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if scheduler_name == "step" and scheduler is not None:
            scheduler.step()

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch {epoch}/{epochs} Train MSE: {train_loss:.4f} Val MSE: {val_loss:.4f}")

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch} (no improvement in {patience} epochs)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def evaluate(model, loader, device):
    model.eval()
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
    sensors = [f"s_{i}" for i in range(1, 22)]
    df_train = load_train_df(args.train_path)

    # determine non-constant sensors and fit scaler on those
    X_train_raw = df_train[sensors].values.astype(np.float32)
    X_train_raw, mask = remove_constant_columns(X_train_raw)
    sensors_masked = [s for s, m in zip(sensors, mask) if m]
    scaler = StandardScaler()
    df_train_scaled = df_train.copy()
    df_train_scaled[sensors_masked] = scaler.fit_transform(df_train[sensors_masked].values)

    # create sequences using only masked sensors
    X_seq, y_seq = create_sequences(df_train_scaled, sensors_masked, window_size=args.window_size)

    # split train/val
    X_tr, X_val, y_tr, y_val = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

    # datasets/loaders
    train_ds = SequenceDataset(X_tr, y_tr)
    val_ds = SequenceDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # prepare test sequences
    df_test = load_test_df(args.test_path, args.rul_path)
    df_test_scaled = df_test.copy()
    # apply mask and scaler to masked sensors
    df_test_scaled[sensors_masked] = scaler.transform(df_test[sensors_masked].values)
    X_test_seq, y_test_seq = create_sequences(df_test_scaled, sensors_masked, window_size=args.window_size)
    test_ds = SequenceDataset(X_test_seq, y_test_seq)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    device = get_device()
    print(f"Using device: {device}")

    model = LSTMRegressor(input_size=X_seq.shape[2], hidden_size=args.hidden_size, num_layers=args.num_layers, dropout=args.dropout, bidirectional=args.bidirectional)
    model, history = train_model(model, train_loader, val_loader, device, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay, clip_norm=args.clip_norm, scheduler_name=args.scheduler, patience=args.patience)

    rmse, r2, preds, trues = evaluate(model, test_loader, device)
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test R^2: {r2:.4f}")

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, "lstm_model.pth"))
    np.save(os.path.join(out_dir, "mask.npy"), mask)
    np.save(os.path.join(out_dir, "scaler_mean.npy"), scaler.mean_)
    np.save(os.path.join(out_dir, "scaler_scale.npy"), scaler.scale_)
    # save history and predictions
    np.save(os.path.join(out_dir, "history.npy"), history)
    np.save(os.path.join(out_dir, "test_preds.npy"), preds)
    np.save(os.path.join(out_dir, "test_trues.npy"), trues)
    print(f"Saved model, preprocessing, history and predictions to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", default="train_FD001.txt")
    parser.add_argument("--test-path", default="test_FD001.txt")
    parser.add_argument("--rul-path", default="RUL_FD001.txt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out-dir", default="artifacts")
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--bidirectional", action="store_true", help="use bidirectional LSTM")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="optimizer weight decay")
    parser.add_argument("--clip-norm", type=float, default=0.0, help="max grad norm (0 disables)")
    parser.add_argument("--scheduler", choices=["none", "plateau", "step"], default="plateau")
    parser.add_argument("--patience", type=int, default=10, help="early stopping patience")
    args = parser.parse_args()
    main(args)
