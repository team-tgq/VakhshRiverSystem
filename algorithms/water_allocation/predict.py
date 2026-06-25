"""GM(1,1) 下游三国需水预测模型"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 资源目录
_RESOURCE_DIR = Path(__file__).resolve().parent / "resources"
_DATA_DIR = _RESOURCE_DIR / "data"

# 瓦赫什河流量占阿姆河总流量的比例系数
DOWNSTREAM_SHARE_FACTOR = 0.2


def gm11_predict_exact(data, forecast_steps):
    """GM(1,1) 灰色预测模型"""
    x0 = np.array(data)
    min_val = np.min(x0)
    shift = 0
    if min_val <= 0:
        shift = abs(min_val) + 1
        x0 = x0 + shift

    x1 = np.cumsum(x0)
    z1 = 0.5 * (x1[:-1] + x1[1:])

    B = np.vstack([-z1, np.ones(len(z1))]).T
    Y = x0[1:].reshape(-1, 1)

    try:
        u = np.linalg.inv(B.T @ B) @ B.T @ Y
        a, b = u[0, 0], u[1, 0]
    except np.linalg.LinAlgError:
        return np.array([np.nan] * forecast_steps)

    n = len(x0)
    predict_x1 = np.zeros(n + forecast_steps)
    predict_x1[0] = x0[0]
    for k in range(1, n + forecast_steps):
        predict_x1[k] = (x0[0] - b / a) * np.exp(-a * k) + b / a

    predict_x0 = np.zeros(n + forecast_steps)
    predict_x0[0] = predict_x1[0]
    for i in range(1, len(predict_x1)):
        predict_x0[i] = predict_x1[i] - predict_x1[i - 1]

    forecast = predict_x0[n:] - shift
    return forecast


def distribute_to_months(annual_total, month_ratios=None):
    """将年总用水量分配到各月份"""
    if month_ratios is None:
        month_ratios = [0.05, 0.05, 0.07, 0.08, 0.09, 0.12, 0.13, 0.12, 0.10, 0.09, 0.08, 0.06]
    ratios = np.array(month_ratios) / sum(month_ratios)
    monthly_values = annual_total * ratios
    return monthly_values


def _load_afghanistan_data():
    """加载阿富汗用水数据 (AQUASTAT 多变量 CSV 格式)"""
    csv_path = str(_DATA_DIR / 'Afghanistan_water_discharger.csv')
    df = pd.read_csv(csv_path)
    mask = df['Variable'] == 'Total water withdrawal'
    afg = df[mask][['Year', 'Value']].drop_duplicates().sort_values('Year')
    return afg['Year'].values.astype(int), afg['Value'].values.astype(float)


def _predict_one_country(historical_values, hist_years, target_year):
    """用 GM(1,1) 预测单个国家的年总用水量 (Billion m³)"""
    last_year = int(hist_years[-1])
    if target_year <= last_year:
        idx = np.where(hist_years == target_year)[0]
        if len(idx) > 0:
            return historical_values[idx[0]]
        return historical_values[-1]

    forecast_steps = target_year - last_year
    pred = gm11_predict_exact(historical_values, forecast_steps)
    return pred[-1]


def predict_monthly_water_demand(target_year):
    """
    预测指定年份土库曼斯坦 + 乌兹别克斯坦的月度用水量 (Billion m³)
    保留向后兼容
    """
    df_tkm = pd.read_csv(str(_DATA_DIR / 'Turkmenistan_water_data.csv'))
    df_uzb = pd.read_csv(str(_DATA_DIR / 'Uzbekistan_water_data.csv'))

    target = 'Total water withdrawal'

    last_year_tkm = int(df_tkm['Year'].max())
    last_year_uzb = int(df_uzb['Year'].max())

    if target_year <= last_year_tkm and target_year <= last_year_uzb:
        tkm_value = df_tkm[df_tkm['Year'] == target_year][target].values[0]
        uzb_value = df_uzb[df_uzb['Year'] == target_year][target].values[0]
        tkm_monthly = distribute_to_months(tkm_value)
        uzb_monthly = distribute_to_months(uzb_value)
        return {
            'Turkmenistan': tkm_monthly,
            'Uzbekistan': uzb_monthly,
            'Turkmenistan_annual': tkm_value,
            'Uzbekistan_annual': uzb_value,
        }

    forecast_steps_tkm = target_year - last_year_tkm
    forecast_steps_uzb = target_year - last_year_uzb
    tkm_hist = df_tkm[target].values
    uzb_hist = df_uzb[target].values

    tkm_annual = gm11_predict_exact(tkm_hist, forecast_steps_tkm)[-1] if forecast_steps_tkm > 0 else tkm_hist[-1]
    uzb_annual = gm11_predict_exact(uzb_hist, forecast_steps_uzb)[-1] if forecast_steps_uzb > 0 else uzb_hist[-1]

    tkm_annual *= DOWNSTREAM_SHARE_FACTOR
    uzb_annual *= DOWNSTREAM_SHARE_FACTOR

    tkm_monthly = distribute_to_months(tkm_annual) * DOWNSTREAM_SHARE_FACTOR
    uzb_monthly = distribute_to_months(uzb_annual) * DOWNSTREAM_SHARE_FACTOR

    return {
        'Turkmenistan': tkm_monthly,
        'Uzbekistan': uzb_monthly,
        'Turkmenistan_annual': tkm_annual,
        'Uzbekistan_annual': uzb_annual,
    }


def predict_downstream_total(target_year):
    """
    预测指定年份下游三国 (土库曼斯坦 + 乌兹别克斯坦 + 阿富汗)
    合并后的月度用水量，已应用 0.2 系数

    返回:
        dict: {
            'downstream_monthly':    (12,) numpy array, 月度合并用水量 (Billion m³)
            'downstream_annual':     float, 年总用水量 (Billion m³)
            'Turkmenistan_annual':   float,
            'Uzbekistan_annual':     float,
            'Afghanistan_annual':    float,
        }
    """
    # --- 土库曼斯坦 ---
    df_tkm = pd.read_csv(str(_DATA_DIR / 'Turkmenistan_water_data.csv'))
    tkm_hist = df_tkm['Total water withdrawal'].values
    tkm_years = df_tkm['Year'].values.astype(int)
    tkm_annual = _predict_one_country(tkm_hist, tkm_years, target_year) * DOWNSTREAM_SHARE_FACTOR

    # --- 乌兹别克斯坦 ---
    df_uzb = pd.read_csv(str(_DATA_DIR / 'Uzbekistan_water_data.csv'))
    uzb_hist = df_uzb['Total water withdrawal'].values
    uzb_years = df_uzb['Year'].values.astype(int)
    uzb_annual = _predict_one_country(uzb_hist, uzb_years, target_year) * DOWNSTREAM_SHARE_FACTOR

    # --- 阿富汗 ---
    afg_years, afg_vals = _load_afghanistan_data()
    afg_annual = _predict_one_country(afg_vals, afg_years, target_year) * DOWNSTREAM_SHARE_FACTOR

    # --- 合并 ---
    total_annual = tkm_annual + uzb_annual + afg_annual
    total_monthly = distribute_to_months(total_annual) * DOWNSTREAM_SHARE_FACTOR

    return {
        'downstream_monthly': total_monthly,
        'downstream_annual': total_annual,
        'Turkmenistan_annual': tkm_annual,
        'Uzbekistan_annual': uzb_annual,
        'Afghanistan_annual': afg_annual,
    }


# 测试代码
if __name__ == "__main__":
    df_tkm = pd.read_csv(str(_DATA_DIR / 'Turkmenistan_water_data.csv'))
    df_uzb = pd.read_csv(str(_DATA_DIR / 'Uzbekistan_water_data.csv'))

    target = 'Total water withdrawal'
    last_year_tkm = int(df_tkm['Year'].max())
    last_year_uzb = int(df_uzb['Year'].max())

    forecast_horizon = 15
    future_years_tkm = np.arange(last_year_tkm + 1, last_year_tkm + 1 + forecast_horizon)
    future_years_uzb = np.arange(last_year_uzb + 1, last_year_uzb + 1 + forecast_horizon)

    tkm_hist = df_tkm[target].values
    uzb_hist = df_uzb[target].values
    tkm_pred = gm11_predict_exact(tkm_hist, forecast_horizon)
    uzb_pred = gm11_predict_exact(uzb_hist, forecast_horizon)

    print("--- Turkmenistan: Total water withdrawal Forecast ---")
    for y, p in zip(future_years_tkm, tkm_pred):
        print(f"Year {y}: {p:.3f}")

    print("\n--- Uzbekistan: Total water withdrawal Forecast ---")
    for y, p in zip(future_years_uzb, uzb_pred):
        print(f"Year {y}: {p:.3f}")

    # 测试三国合并
    print("\n--- Downstream Three Countries (x0.2) ---")
    for y in [2025, 2026, 2030, 2035]:
        r = predict_downstream_total(y)
        print(f"Year {y}: TKM={r['Turkmenistan_annual']:.3f}, "
              f"UZB={r['Uzbekistan_annual']:.3f}, "
              f"AFG={r['Afghanistan_annual']:.3f}, "
              f"Total={r['downstream_annual']:.3f} Billion m³")

    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(df_tkm['Year'], tkm_hist, label='Historical', color='blue', marker='o')
    axes[0].plot(future_years_tkm, tkm_pred, label='GM(1,1) Forecast', color='red', linestyle='--', marker='x')
    axes[0].set_title(f'Turkmenistan: {target}')
    axes[0].set_ylabel('Billion m³ / Year')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df_uzb['Year'], uzb_hist, label='Historical', color='blue', marker='o')
    axes[1].plot(future_years_uzb, uzb_pred, label='GM(1,1) Forecast', color='red', linestyle='--', marker='x')
    axes[1].set_title(f'Uzbekistan: {target}')
    axes[1].set_ylabel('Billion m³ / Year')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('total_water_forecast.png', dpi=150)
    print("\n预测图表已保存到 total_water_forecast.png")
