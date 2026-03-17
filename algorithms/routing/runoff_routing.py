# algorithms/routing/runoff_routing.py
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

from .config import RUNOFF_FILE, RUNOFF_OUTPUT_DIR
from .utils import calculate_metrics
from .data_loader import create_window_dataset


class HydrologyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def run_runoff_routing(
    file_path=None,
    time_steps=60,
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
    epochs=60,
    batch_size=32,
    lr=0.001,
    patience=12,
    hidden_size=64,
    num_layers=2,
    dropout=0.2,
    device=None,
    save_model_path=None
):
    file_path = file_path or RUNOFF_FILE
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"未找到 Runoff Routing 数据文件: {file_path}")

    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio 必须等于 1.0")

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(file_path, parse_dates=["date"], index_col="date")

    feature_cols = [
        "precipitation", "temp_mean_c", "temp_max_c", "temp_min_c",
        "sm_vol_l1", "et_mm_day", "swe_mm", "snowmelt_mm_day"
    ]
    target_col = "runoff"

    missing_cols = [c for c in feature_cols + [target_col] if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Runoff Routing 数据缺少列: {missing_cols}")

    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.interpolate(method="time").ffill().bfill()

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    scaled_x = scaler_x.fit_transform(df[feature_cols])
    scaled_y = scaler_y.fit_transform(df[[target_col]])

    X_windowed, y_windowed = create_window_dataset(scaled_x, scaled_y, time_steps)

    n_samples = X_windowed.shape[0]
    if n_samples == 0:
        raise ValueError("Runoff Routing 窗口样本为空，请检查数据长度或 time_steps 参数。")

    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))

    if train_end <= 0 or val_end <= train_end or val_end >= n_samples:
        raise ValueError("训练/验证/测试集划分失败，请检查比例或数据量。")

    train_loader = DataLoader(
        TensorDataset(X_windowed[:train_end], y_windowed[:train_end]),
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_windowed[train_end:val_end], y_windowed[train_end:val_end]),
        batch_size=batch_size,
        shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(X_windowed[val_end:], y_windowed[val_end:]),
        batch_size=batch_size,
        shuffle=False
    )

    model = HydrologyLSTM(
        input_size=len(feature_cols),
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=1,
        dropout=dropout
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    if save_model_path is None:
        save_model_path = os.path.join(RUNOFF_OUTPUT_DIR, "best_model.pth")

    best_v_loss = float("inf")
    early_stop_count = 0
    train_losses = []
    val_losses = []

    for _ in range(epochs):
        model.train()
        batch_losses = []

        for bx, by in train_loader:
            bx = bx.to(device)
            by = by.to(device)

            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())

        train_loss = float(np.mean(batch_losses)) if batch_losses else np.nan
        train_losses.append(train_loss)

        model.eval()
        v_losses = []
        with torch.no_grad():
            for bx, by in val_loader:
                bx = bx.to(device)
                by = by.to(device)
                v_losses.append(criterion(model(bx), by).item())

        curr_v_loss = float(np.mean(v_losses)) if v_losses else np.nan
        val_losses.append(curr_v_loss)

        if curr_v_loss < best_v_loss:
            best_v_loss = curr_v_loss
            torch.save(model.state_dict(), save_model_path)
            early_stop_count = 0
        else:
            early_stop_count += 1
            if early_stop_count >= patience:
                break

    if os.path.exists(save_model_path):
        model.load_state_dict(torch.load(save_model_path, map_location=device))

    model.eval()
    all_p, all_t = [], []

    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(device)
            all_p.append(model(bx).cpu().numpy())
            all_t.append(by.numpy())

    y_pred = scaler_y.inverse_transform(np.concatenate(all_p))
    y_true = scaler_y.inverse_transform(np.concatenate(all_t))

    rmse, nse, kge = calculate_metrics(y_true, y_pred)

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "y_true": y_true,
        "y_pred": y_pred,
        "rmse": float(rmse),
        "nse": float(nse),
        "kge": float(kge),
        "feature_names": feature_cols,
        "target_name": target_col,
        "models": model,
        "model_path": save_model_path,
    }