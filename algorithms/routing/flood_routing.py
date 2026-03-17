# algorithms/routing/flood_routing.py
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from .config import FLOOD_FILE
from .utils import get_nse
from .data_loader import create_sequence_dataset


class FloodLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3
        )
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def run_flood_routing(
    file_path=None,
    look_back=10,
    epochs=50,
    batch_size=32,
    lr=0.001,
    train_ratio=0.8,
    device=None
):
    file_path = file_path or FLOOD_FILE
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"未找到 Flood Routing 数据文件: {file_path}")

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(file_path, parse_dates=["date"])

    features = [
        "precipitation", "temp_mean_c", "temp_max_c", "temp_min_c",
        "sm_vol_l1", "et_mm_day", "swe_mm", "snowmelt_mm_day", "runoff"
    ]
    targets = ["Downstream_WaterLevel_m", "Upstream-Water_Level_m"]

    missing_cols = [c for c in features + targets if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Flood Routing 数据缺少列: {missing_cols}")

    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=features + targets).reset_index(drop=True)

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_x.fit_transform(df[features])
    y_scaled = scaler_y.fit_transform(df[targets])

    X_seq, y_seq = create_sequence_dataset(X_scaled, y_scaled, look_back)

    if len(X_seq) == 0:
        raise ValueError("Flood Routing 序列样本为空，请检查数据长度或 look_back 参数。")

    X_tensor = torch.from_numpy(X_seq).float()
    y_tensor = torch.from_numpy(y_seq).float()

    train_size = int(len(X_tensor) * train_ratio)
    if train_size <= 0 or train_size >= len(X_tensor):
        raise ValueError("训练集划分失败，请检查 train_ratio 或数据量。")

    train_dataset = TensorDataset(X_tensor[:train_size], y_tensor[:train_size])
    test_dataset = TensorDataset(X_tensor[train_size:], y_tensor[train_size:])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = FloodLSTM(input_size=len(features)).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []

    for _ in range(epochs):
        model.train()
        epoch_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(len(train_loader), 1)
        train_losses.append(float(avg_loss))

    model.eval()
    with torch.no_grad():
        test_preds = model(X_tensor[train_size:].to(device)).cpu().numpy()
        test_actuals = y_tensor[train_size:].numpy()

    preds_real = scaler_y.inverse_transform(test_preds)
    actuals_real = scaler_y.inverse_transform(test_actuals)

    nse_downstream = get_nse(actuals_real[:, 0], preds_real[:, 0])
    nse_upstream = get_nse(actuals_real[:, 1], preds_real[:, 1])

    return {
        "train_losses": train_losses,
        "preds_real": preds_real,
        "actuals_real": actuals_real,
        "nse_downstream": float(nse_downstream),
        "nse_upstream": float(nse_upstream),
        "feature_names": features,
        "target_names": targets,
        "models": model,
        "test_loader_size": len(test_loader),
    }