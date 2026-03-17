# algorithms/routing/data_loader.py
import numpy as np
import torch


def create_sequence_dataset(X, y, look_back):
    """
    用于 Flood Routing
    X, y 都强制转为 numpy，避免 pandas 索引歧义
    """
    X = np.asarray(X)
    y = np.asarray(y)

    xs, ys = [], []
    for i in range(len(X) - look_back):
        xs.append(X[i:i + look_back])
        ys.append(y[i + look_back])

    return np.array(xs), np.array(ys)


def create_window_dataset(X, y, time_steps):
    """
    用于 Runoff Routing
    """
    X = np.asarray(X)
    y = np.asarray(y)

    xs, ys = [], []
    for i in range(len(X) - time_steps + 1):
        xs.append(X[i:i + time_steps])
        ys.append(y[i + time_steps - 1])

    return (
        torch.tensor(np.array(xs), dtype=torch.float32),
        torch.tensor(np.array(ys), dtype=torch.float32),
    )