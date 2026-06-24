"""LSTM 模型训练脚本 — 日度气象 → 月度径流预测"""
from pathlib import Path
from matplotlib import ticker
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import DataLoader, TensorDataset
import joblib
import os
import xarray as xr

from .lstm_model import Seq2SeqLSTM, FEATURE_COLS_FINAL

# ======================= 资源目录 =======================
_RESOURCE_DIR = Path(__file__).resolve().parent / "resources"
_DATA_DIR = _RESOURCE_DIR / "data"
_MODELS_DIR = _RESOURCE_DIR / "models"
_SCALERS_DIR = _RESOURCE_DIR / "scalers"

# ========================== 超参数 ==========================
SEQ_LEN_DAYS = 365
OUTPUT_STEPS_MONTHS = 12
BATCH_SIZE = 32
HIDDEN_SIZE = 64
NUM_LAYERS = 2
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 100
PATIENCE = 25
BEST_MODEL_PATH = str(_MODELS_DIR / 'best.pth')

FEATURE_COLS = ['discharge', 'smlt', 'ssrd', 'e', 'u10', 'v10', 'sp', 'skt']
TARGET_COL = 'discharge'


def create_cross_scale_sequences(df, feature_cols, target_col, seq_len_days, output_steps_months):
    """利用日历进行滚动切片，生成跨尺度样本对"""
    X, y = [], []
    dates = df['date'].values
    features_data = df[feature_cols].values
    target_data = df[target_col].values

    total_days = len(df)
    max_start_idx = total_days - seq_len_days - (output_steps_months * 31)

    for i in range(0, max_start_idx, 2):
        end_idx = i + seq_len_days
        x_seq = features_data[i:end_idx]

        y_seq = []
        current_date = pd.to_datetime(dates[end_idx])

        valid_sample = True
        for m in range(output_steps_months):
            target_month_start = current_date + pd.DateOffset(months=m)
            target_month_start = target_month_start.replace(day=1)
            target_month_end = target_month_start + pd.offsets.MonthEnd(1)

            mask = (df['date'] >= target_month_start) & (df['date'] <= target_month_end)
            month_data = target_data[mask]

            if len(month_data) == 0:
                valid_sample = False
                break
            y_seq.append(np.mean(month_data))

        if valid_sample and len(y_seq) == output_steps_months:
            X.append(x_seq)
            y.append(y_seq)

    return np.array(X), np.array(y)


class PeakWeightedMSELoss(nn.Module):
    """峰值加权损失函数，加大对洪峰预测误差的惩罚"""
    def __init__(self, peak_threshold=0.7, peak_weight=5.0):
        super(PeakWeightedMSELoss, self).__init__()
        self.peak_threshold = peak_threshold
        self.peak_weight = peak_weight

    def forward(self, y_pred, y_true):
        base_error = (y_pred - y_true) ** 2
        peak_mask = (y_true > self.peak_threshold).float()
        weights = 1.0 + peak_mask * (self.peak_weight - 1.0)
        weighted_loss = base_error * weights
        return torch.mean(weighted_loss)


def calculate_metrics(y_true, y_pred):
    """计算回归评估指标"""
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, mape, r2


# ============================================================
# 独立训练入口
# ============================================================
if __name__ == "__main__":
    print("------------------")
    df = pd.read_csv(str(_DATA_DIR / 'ERA5_daily_with_discharge.csv'), parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)

    df['day_of_year_sin'] = np.sin(2 * np.pi * df['date'].dt.dayofyear / 365.25)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['date'].dt.dayofyear / 365.25)

    # 归一化
    scaler_feat = MinMaxScaler()
    scaler_target = MinMaxScaler()

    scaled_features_data = scaler_feat.fit_transform(df[FEATURE_COLS_FINAL].values)
    df_scaled_features = pd.DataFrame(scaled_features_data, columns=FEATURE_COLS_FINAL)
    df_combined = pd.concat([df[['date', TARGET_COL]], df_scaled_features], axis=1)

    print("正在生成跨尺度样本对...")
    X_raw, y_raw = create_cross_scale_sequences(
        df_combined, FEATURE_COLS_FINAL, TARGET_COL, SEQ_LEN_DAYS, OUTPUT_STEPS_MONTHS)

    y_raw_reshaped = y_raw.reshape(-1, 1)
    y_scaled_reshaped = scaler_target.fit_transform(y_raw_reshaped)
    y = y_scaled_reshaped.reshape(y_raw.shape)
    X = X_raw

    print(f"生成的样本数量: {len(X)}")
    print(f"X 形状: {X.shape} (样本数, 天数, 特征数)")
    print(f"y 形状: {y.shape} (样本数, 预测月数)")

    # 分数据集
    total_size = len(X)
    train_end = int(total_size * 0.8)
    val_end = int(total_size * 0.9)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    input_size = X.shape[2]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Seq2SeqLSTM(input_size, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_STEPS_MONTHS).to(device)

    criterion = PeakWeightedMSELoss(peak_threshold=0.6, peak_weight=5.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_val_loss = float('inf')
    patience_counter = 0
    history_train_loss = []
    history_val_loss = []

    target_range = scaler_target.data_max_[0] - scaler_target.data_min_[0]
    mse_scale_factor = target_range ** 2

    print(f"\n开始训练跨尺度 Seq2Seq 模型，设备: {device}")
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)

        avg_train_loss = train_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0
        all_val_pred = []
        all_val_true = []
        with torch.no_grad():
            for batch_X_val, batch_y_val in val_loader:
                batch_X_val, batch_y_val = batch_X_val.to(device), batch_y_val.to(device)
                val_out = model(batch_X_val)
                loss = criterion(val_out, batch_y_val)
                val_loss += loss.item() * batch_X_val.size(0)
                all_val_pred.append(val_out.cpu().numpy())
                all_val_true.append(batch_y_val.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader.dataset)
        avg_train_loss_orig = avg_train_loss * mse_scale_factor
        avg_val_loss_orig = avg_val_loss * mse_scale_factor

        history_train_loss.append(avg_train_loss_orig)
        history_val_loss.append(avg_val_loss_orig)

        val_pred = np.concatenate(all_val_pred, axis=0)
        val_true = np.concatenate(all_val_true, axis=0)
        val_pred_orig = scaler_target.inverse_transform(val_pred)
        val_true_orig = scaler_target.inverse_transform(val_true)
        rmse, mae, mape, r2 = calculate_metrics(val_true_orig, val_pred_orig)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{NUM_EPOCHS} | Train Loss(Orig): {avg_train_loss_orig:.2f} | "
                  f"Val Loss(Orig): {avg_val_loss_orig:.2f}")
            print(f"  Val Metrics: RMSE={rmse:.2f}, MAE={mae:.2f}, MAPE={mape:.2f}%, R2={r2:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            patience_counter = 0
            if (epoch + 1) % 5 == 0:
                print(f"  >>> 保存最佳模型，当前最佳验证损失(Orig) {avg_val_loss_orig:.2f}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"早停于 epoch {epoch + 1}")
                break

    # 绘制 Loss 曲线
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(10, 6))
    actual_epochs = len(history_train_loss)
    plt.plot(range(1, actual_epochs + 1), history_train_loss, label='训练集 Loss', color='#1f77b4', linewidth=2)
    plt.plot(range(1, actual_epochs + 1), history_val_loss, label='验证集 Loss', color='#d62728', linewidth=2, linestyle='--')
    plt.title('训练与验证损失 (反归一化后)', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE Loss [(m³/s)²]', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # 测试集评估
    print("\n加载最佳模型进行测试...")
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    model.eval()

    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)
    all_test_pred = []
    with torch.no_grad():
        for batch_X_test, _ in test_loader:
            batch_X_test = batch_X_test.to(device)
            test_out = model(batch_X_test)
            all_test_pred.append(test_out.cpu().numpy())

    test_pred = np.concatenate(all_test_pred, axis=0)
    test_pred_orig = scaler_target.inverse_transform(test_pred)
    test_true_orig = scaler_target.inverse_transform(y_test.numpy())

    rmse_t, mae_t, mape_t, r2_t = calculate_metrics(test_true_orig, test_pred_orig)
    print("\n========== 测试集评估 ==========")
    print(f"RMSE: {rmse_t:.2f}, MAE: {mae_t:.2f}, MAPE: {mape_t:.2f}%, R2: {r2_t:.4f}")

    # 保存归一化器
    joblib.dump(scaler_feat, str(_SCALERS_DIR / 'seq2seq_scaler_feat.pkl'))
    joblib.dump(scaler_target, str(_SCALERS_DIR / 'seq2seq_scaler_target.pkl'))
    print(f"模型和归一化器已保存到 {_SCALERS_DIR}")

    # 测试样本可视化
    num_test_samples = test_true_orig.shape[0]
    if num_test_samples > 0:
        sample_indices = [0, num_test_samples // 2, num_test_samples - 1]
        unique_indices = sorted(list(set(idx for idx in sample_indices if idx < num_test_samples)))

        plt.figure(figsize=(15, 6 * len(unique_indices)))
        for i, idx in enumerate(unique_indices):
            plt.subplot(len(unique_indices), 1, i + 1)
            true_vals = test_true_orig[idx]
            pred_vals = test_pred_orig[idx]
            x = range(1, len(true_vals) + 1)
            plt.plot(x, true_vals, label='真实月均径流', color='blue', marker='o', linestyle='-', linewidth=2)
            plt.plot(x, pred_vals, label='预测值', color='red', marker='x', linestyle='--', linewidth=2)
            plt.title(f'测试集样本 {idx + 1}：基于过去365天预测未来 {len(true_vals)} 个月径流量趋势')
            plt.ylabel('平均径流量 (m³/s)')
            plt.xticks(x)
            plt.legend(loc='upper right')
            plt.grid(True, linestyle='--', alpha=0.7)
            ax = plt.gca()
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
            ax.yaxis.get_major_formatter().set_scientific(False)
            y_min, y_max = ax.get_ylim()
            y_ticks = np.arange(np.floor(y_min / 200) * 200,
                               np.ceil(y_max / 200) * 200 + 1, 200)
            ax.set_yticks(y_ticks)
            ax.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.show()


# ============================================================
# 外部训练接口
# ============================================================

def train_from_external_data(data_path, hyperparams=None, progress_callback=None):
    """
    使用外部数据 (CSV/Excel/NC) 重新训练 LSTM 模型

    参数:
        data_path:  外部数据文件路径 (.csv / .xlsx / .xls / .nc) 或目录
        hyperparams: 超参数字典
        progress_callback: 回调函数 (epoch, epochs, train_loss, val_loss, rmse, mae)

    返回:
        dict: 训练结果指标
    """
    import glob as glob_mod

    hp = hyperparams or {}
    seq_len = hp.get('seq_len_days', SEQ_LEN_DAYS)
    output_steps = hp.get('output_steps_months', OUTPUT_STEPS_MONTHS)
    hidden = hp.get('hidden_size', HIDDEN_SIZE)
    num_layers_param = hp.get('num_layers', NUM_LAYERS)
    batch = hp.get('batch_size', BATCH_SIZE)
    lr = hp.get('learning_rate', LEARNING_RATE)
    wd = hp.get('weight_decay', WEIGHT_DECAY)
    epochs = hp.get('num_epochs', NUM_EPOCHS)
    patience = hp.get('patience', PATIENCE)
    model_path = hp.get('model_path', str(_MODELS_DIR / 'best.pth'))

    # --- 加载外部数据 ---
    df = None
    if data_path.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(data_path, parse_dates=['date'])
    elif data_path.endswith('.csv'):
        df = pd.read_csv(data_path, parse_dates=['date'])
    elif data_path.endswith('.nc'):
        ds = xr.open_dataset(data_path)
        df = ds.to_dataframe().reset_index()
        if 'time' in df.columns:
            df = df.rename(columns={'time': 'date'})
        ds.close()
    elif os.path.isdir(data_path):
        nc_files = sorted(glob_mod.glob(os.path.join(data_path, '*.nc')))
        parts = []
        for f in nc_files:
            ds = xr.open_dataset(f)
            part = ds.to_dataframe().reset_index()
            if 'time' in part.columns:
                part = part.rename(columns={'time': 'date'})
            parts.append(part)
            ds.close()
        if parts:
            df = pd.concat(parts, ignore_index=True)
    else:
        raise ValueError(f"不支持的数据格式: {data_path}")

    if df is None or len(df) == 0:
        raise ValueError("无法读取数据或数据为空")

    df = df.sort_values('date').reset_index(drop=True)
    if 'day_of_year_sin' not in df.columns:
        df['day_of_year_sin'] = np.sin(2 * np.pi * df['date'].dt.dayofyear / 365.25)
        df['day_of_year_cos'] = np.cos(2 * np.pi * df['date'].dt.dayofyear / 365.25)

    feature_cols = FEATURE_COLS + ['day_of_year_sin', 'day_of_year_cos']
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"数据缺少必要列: {missing}")

    # --- 归一化 ---
    scaler_f = MinMaxScaler()
    scaler_t = MinMaxScaler()
    scaled_features = scaler_f.fit_transform(df[feature_cols].values)
    df_scaled_f = pd.DataFrame(scaled_features, columns=feature_cols)
    df_combined = pd.concat([df[['date', TARGET_COL]], df_scaled_f], axis=1)

    # --- 生成样本 ---
    X_raw, y_raw = create_cross_scale_sequences(
        df_combined, feature_cols, TARGET_COL, seq_len, output_steps)

    y_scaled = scaler_t.fit_transform(y_raw.reshape(-1, 1)).reshape(y_raw.shape)
    X, y = X_raw, y_scaled

    # --- 划分数据集 ---
    n = len(X)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)
    X_train_t = torch.tensor(X[:train_end], dtype=torch.float32)
    y_train_t = torch.tensor(y[:train_end], dtype=torch.float32)
    X_val_t = torch.tensor(X[train_end:val_end], dtype=torch.float32)
    y_val_t = torch.tensor(y[train_end:val_end], dtype=torch.float32)
    X_test_t = torch.tensor(X[val_end:], dtype=torch.float32)
    y_test_t = torch.tensor(y[val_end:], dtype=torch.float32)

    train_ld = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch, shuffle=True)
    val_ld = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch, shuffle=False)

    # --- 训练 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Seq2SeqLSTM(X.shape[2], hidden, num_layers_param, output_steps).to(device)
    criterion = PeakWeightedMSELoss(peak_threshold=0.6, peak_weight=5.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    target_range = scaler_t.data_max_[0] - scaler_t.data_min_[0]
    mse_scale = target_range ** 2

    best_val = float('inf')
    patience_ctr = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for bx, by in train_ld:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * bx.size(0)
        avg_train = train_loss / len(train_ld.dataset)

        model.eval()
        val_loss = 0
        vp, vt = [], []
        with torch.no_grad():
            for bx, by in val_ld:
                bx, by = bx.to(device), by.to(device)
                out = model(bx)
                val_loss += criterion(out, by).item() * bx.size(0)
                vp.append(out.cpu().numpy())
                vt.append(by.cpu().numpy())
        avg_val = val_loss / len(val_ld.dataset)

        vp_np = np.concatenate(vp, axis=0)
        vt_np = np.concatenate(vt, axis=0)
        vp_orig = scaler_t.inverse_transform(vp_np)
        vt_orig = scaler_t.inverse_transform(vt_np)
        rmse_v = np.sqrt(mean_squared_error(vt_orig.flatten(), vp_orig.flatten()))
        mae_v = mean_absolute_error(vt_orig.flatten(), vp_orig.flatten())

        if progress_callback:
            progress_callback(epoch + 1, epochs, avg_train * mse_scale,
                            avg_val * mse_scale, rmse_v, mae_v)

        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), model_path)
            joblib.dump(scaler_f, str(_SCALERS_DIR / 'seq2seq_scaler_feat.pkl'))
            joblib.dump(scaler_t, str(_SCALERS_DIR / 'seq2seq_scaler_target.pkl'))
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                break

    # --- 测试评估 ---
    test_ld = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=batch, shuffle=False)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    tp = []
    with torch.no_grad():
        for bx, _ in test_ld:
            tp.append(model(bx.to(device)).cpu().numpy())
    tp_np = np.concatenate(tp, axis=0)
    tp_orig = scaler_t.inverse_transform(tp_np)
    tt_orig = scaler_t.inverse_transform(y_test_t.numpy())
    rmse_t = np.sqrt(mean_squared_error(tt_orig.flatten(), tp_orig.flatten()))
    mae_t = mean_absolute_error(tt_orig.flatten(), tp_orig.flatten())
    r2_t = r2_score(tt_orig.flatten(), tp_orig.flatten())

    return {
        'best_val_loss': best_val * mse_scale,
        'test_rmse': rmse_t,
        'test_mae': mae_t,
        'test_r2': r2_t,
        'model_path': model_path,
    }
