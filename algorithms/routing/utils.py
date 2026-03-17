# algorithms/routing/utils.py
import numpy as np
from sklearn.metrics import mean_squared_error


def get_nse(actual, predicted):
    actual = np.asarray(actual).flatten()
    predicted = np.asarray(predicted).flatten()

    denominator = np.sum((actual - np.mean(actual)) ** 2)
    if denominator == 0:
        return np.nan

    numerator = np.sum((actual - predicted) ** 2)
    return 1 - (numerator / denominator)


def calculate_metrics(actual, pred):
    actual = np.asarray(actual).flatten()
    pred = np.asarray(pred).flatten()

    rmse = np.sqrt(mean_squared_error(actual, pred))

    denominator = np.sum((actual - np.mean(actual)) ** 2)
    if denominator == 0:
        nse = np.nan
    else:
        nse = 1 - (np.sum((actual - pred) ** 2) / denominator)

    if len(actual) > 1 and np.std(actual) != 0 and np.std(pred) != 0:
        r = np.corrcoef(actual, pred)[0, 1]
    else:
        r = np.nan

    alpha = np.std(pred) / np.std(actual) if np.std(actual) != 0 else np.nan
    beta = np.mean(pred) / np.mean(actual) if np.mean(actual) != 0 else np.nan

    if np.isnan(r) or np.isnan(alpha) or np.isnan(beta):
        kge = np.nan
    else:
        kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

    return rmse, nse, kge