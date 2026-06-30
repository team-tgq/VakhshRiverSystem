"""
水资源分配核心算法模块
包含: NSGA-II 多时间尺度部门配水优化 + 努列克坝物理参数与LSTM径流预测
"""
from __future__ import annotations

import calendar
import os
from pathlib import Path
from typing import Dict, List, Optional, Callable

import numpy as np
import pandas as pd
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from .lstm_model import FEATURE_COLS_FINAL, Seq2SeqLSTM
from .predict import predict_downstream_total

# ======================= 资源目录 =======================
RESOURCE_DIR = Path(__file__).resolve().parent / "resources"
DATA_DIR = RESOURCE_DIR / "data"
MODELS_DIR = RESOURCE_DIR / "models"
SCALERS_DIR = RESOURCE_DIR / "scalers"


def _resource_path(*parts: str) -> Path:
    """获取资源文件的绝对路径"""
    return RESOURCE_DIR.joinpath(*parts)


# ======================= 部门常量 =======================
SECTOR_LIVE = "生活"
SECTOR_ECO = "生态"
SECTOR_AGR = "农业"
SECTOR_IND = "工业"
SECTOR_DOWN = "下游国家"
SECTOR_ORDER = [SECTOR_LIVE, SECTOR_ECO, SECTOR_AGR, SECTOR_IND]
SECTOR_ORDER_V2 = [SECTOR_LIVE, SECTOR_ECO, SECTOR_AGR, SECTOR_IND, SECTOR_DOWN]


# ============================================================
# 第一部分  NSGA-II 多时间尺度水资源分配问题
# ============================================================

class NurekWaterAllocation(ElementwiseProblem):
    """
    多时间尺度水资源分配问题 (日/月/年)

    参数:
        n_sources:   水源数 (2: 地表水 + 地下水)
        n_regions:   区域数 (1: 哈特隆州)
        m_sectors:   部门数 (5: 生活/生态/农业/工业/下游三国)
        n_periods:   时间期数 (1=单时段, 12=月度, 365=日度)
        time_scale:  "daily" / "monthly" / "yearly"
        a, b:        经济效益系数矩阵
        T:           部门收益权重
        D:           需水量矩阵 (n_periods, m_sectors)
        W:           供水量向量 (n_sources,) 或 (n_periods, n_sources)
        F_min, F_max: 分配上下限
        loss_rates:  管网传输损耗率 (n_regions,)
    """

    def __init__(self, n_sources, n_regions, m_sectors, a, b, T, D, W, F_min, F_max,
                 loss_rates, n_periods=1, time_scale="monthly"):
        self.n, self.r, self.m = n_sources, n_regions, m_sectors
        self.n_periods = n_periods
        self.time_scale = time_scale
        self.a, self.b, self.T = a, b, T
        self.loss_rates = loss_rates

        # W: 每期供水 -> shape (n_periods, n_sources)
        self.W = np.atleast_2d(W)
        if self.W.shape[0] != self.n_periods:
            self.W = np.tile(self.W, (self.n_periods, 1))

        # D: 每期需水 -> shape (n_periods, m_sectors)
        self.D = np.atleast_2d(D)
        if self.D.shape[0] != self.n_periods:
            self.D = np.tile(self.D, (self.n_periods, 1))

        # F_min/F_max: 统一为 (n_periods, n_regions, m_sectors) 方便索引
        self.F_min = np.atleast_2d(F_min)
        self.F_max = np.atleast_2d(F_max)
        if self.F_min.ndim == 2:
            if self.F_min.shape[0] == self.n_periods:
                self.F_min = self.F_min[:, np.newaxis, :]   # (P, M) -> (P, 1, M)
                self.F_max = self.F_max[:, np.newaxis, :]
            else:
                self.F_min = self.F_min[np.newaxis, :, :]   # (R, M) -> (1, R, M)
                self.F_max = self.F_max[np.newaxis, :, :]

        n_var = self.n * self.r * self.m * self.n_periods

        xl = np.zeros(n_var)
        xu = np.zeros(n_var)
        idx = 0
        for t in range(self.n_periods):
            for i in range(self.n):
                for k in range(self.r):
                    for j in range(self.m):
                        if i == 0:  # 地表水
                            denom = (1 - self.loss_rates[k]) if self.loss_rates[k] < 1 else 1.0
                            xu[idx] = min(self.W[t, i], self.F_max[t, k, j] / denom)
                        else:       # 地下水
                            xu[idx] = min(self.W[t, i], self.F_max[t, k, j])
                        idx += 1

        n_ieq = (self.n + self.r) * self.n_periods + 2 * self.r * self.m * self.n_periods
        super().__init__(n_var=n_var, n_obj=3, n_ieq_constr=n_ieq, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        X = x.reshape((self.n_periods, self.n, self.r, self.m))
        R = np.zeros((self.n_periods, self.r, self.m))

        for t in range(self.n_periods):
            for k in range(self.r):
                for j in range(self.m):
                    R[t, k, j] = X[t, 0, k, j] * (1 - self.loss_rates[k]) + X[t, 1, k, j]

        # --- 目标 1: 经济效益 ---
        profit = 0
        for t in range(self.n_periods):
            for k in range(self.r):
                for j in range(self.m):
                    M_kj = X[t, 0, k, j]
                    OW_kj = X[t, 1, k, j]
                    margin_surface = self.a[0, j] - self.b[0, j]
                    margin_ground = self.a[1, j] - self.b[1, j]
                    EP_kj = (margin_surface * M_kj * (1 - self.loss_rates[k])
                             + margin_ground * OW_kj) * self.T[j]
                    profit += EP_kj
        f1 = -profit

        # --- 目标 2: 总缺水量 ---
        shortage = np.sum(np.maximum(0, self.D[:, np.newaxis, :] - R))
        f2 = shortage

        # --- 目标 3: Gini 公平性 ---
        coverage = np.zeros(self.m)
        y_util = np.zeros(self.m)
        for j in range(self.m):
            D_sum = 0.0
            R_sum = 0.0
            EP_sum = 0.0
            for t in range(self.n_periods):
                D_sum += self.D[t, j]
                R_total = R[t, 0, j]
                R_sum += R_total
                M_j = X[t, 0, 0, j]
                OW_j = X[t, 1, 0, j]
                margin_surface = self.a[0, j] - self.b[0, j]
                margin_ground = self.a[1, j] - self.b[1, j]
                EP_sum += (margin_surface * M_j * (1 - self.loss_rates[0])
                           + margin_ground * OW_j) * self.T[j]
            coverage[j] = R_sum / D_sum if D_sum > 0 else 1.0
            y_util[j] = EP_sum / R_sum if R_sum > 0 else 0.0

        if np.all(coverage >= 1.0):
            gini = 0.0
        else:
            capped = np.minimum(coverage, 1.0)
            sum_cap = np.sum(capped)
            gini_cov = 0.0
            if sum_cap > 0:
                diff_cov = sum(abs(capped[a] - capped[b]) for a in range(self.m) for b in range(self.m))
                gini_cov = diff_cov / (2 * self.m * sum_cap)

            sum_y = np.sum(y_util)
            gini_econ = 0.0
            if sum_y > 0:
                diff_y = sum(abs(y_util[a] - y_util[b]) for a in range(self.m) for b in range(self.m))
                gini_econ = diff_y / (2 * self.m * sum_y)

            gini = 0.6 * gini_cov + 0.4 * gini_econ

        f3 = gini
        out["F"] = [f1, f2, f3]

        # --- 约束 ---
        g = []
        for t in range(self.n_periods):
            g.append(X[t, 0, :, :].sum() - self.W[t, 0])  # 地表水不超供
            g.append(X[t, 1, :, :].sum() - self.W[t, 1])  # 地下水不超供
        for t in range(self.n_periods):
            for k in range(self.r):
                g.append(R[t, k, :].sum() - 1.5 * self.D[t, :].sum())
        for t in range(self.n_periods):
            for k in range(self.r):
                for j in range(self.m):
                    g.append(self.F_min[t, k, j] - R[t, k, j])
                    g.append(R[t, k, j] - self.F_max[t, k, j])

        out["G"] = g


def run_nsga2_opt(problem_params, pop_size=150, n_gen=400):
    """运行 NSGA-II 优化"""
    problem = NurekWaterAllocation(**problem_params)
    algorithm = NSGA2(pop_size=pop_size)
    res = minimize(problem, algorithm, get_termination("n_gen", n_gen), seed=1, verbose=False)
    return res


# ============================================================
# 第二部分  努列克坝物理参数与 LSTM 径流预测 (数据服务层)
# ============================================================

class NurekDamParameters:
    """
    大坝物理参数 + LSTM 入库径流预测
    """

    def __init__(self, elec_price=0.8, unit_water_margin=1.6, data_path="",
                 v_initial=84.0, **kwargs):
        update_callback = kwargs.get('update_callback', None)
        self.target_year = kwargs.get('target_year', 2026)

        # 物理常数
        self.g = 9.81
        self.eta = 0.90
        self.hours_per_month = 24 * 30.4
        self.P_cap = 3015

        # 库容与水位
        self.V_initial = v_initial
        self.V_max = 105.0
        self.V_min = 60.0
        self.H_max = 265.0
        self.V_flood = 95.0

        # 流量与约束
        self.Q_safe = 2000.0
        self.Q_eco_base = 60.0
        self.Q_max_release = 3500.0
        self.Q_min_release = 10.0
        self.months = np.arange(1, 13)

        self.elec_price = elec_price
        self.unit_water_margin = unit_water_margin
        self.update_callback = update_callback

        # 获取下游三国合并月径流数据
        self.Q_downstream_monthly = self._get_downstream_monthly_flow()

        # LSTM 预测未来 12 个月入库流量
        self.Q_in = self._predict_monthly_inflow(data_path)

    def _get_downstream_monthly_flow(self):
        """获取下游三国合并月径流, 转换为 m³/s"""
        try:
            result = predict_downstream_total(self.target_year)
            seconds_per_month = 30.4 * 24 * 3600
            monthly_flow = result['downstream_monthly'] * 1e9 / seconds_per_month
            return monthly_flow
        except Exception as e:
            print(f"获取下游三国径流数据失败: {e}")
            return np.zeros(12)

    def _predict_monthly_inflow(self, data_path=""):
        """LSTM 预测未来 12 个月入库径流 (m³/s)"""
        import torch
        import joblib

        HIDDEN_SIZE = 64
        NUM_LAYERS = 2
        OUTPUT_STEPS_MONTHS = 12
        SEQ_LEN_DAYS = 365

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 从 checkpoint 自动检测 INPUT_SIZE
        best_model_path = str(MODELS_DIR / 'best.pth')
        ckpt = torch.load(best_model_path, map_location=device, weights_only=False)
        ckpt_input_size = ckpt['encoder.weight_ih_l0'].shape[1]
        print(f"[LSTM] checkpoint 输入维度: {ckpt_input_size}")

        model = Seq2SeqLSTM(ckpt_input_size, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_STEPS_MONTHS)
        model.load_state_dict(ckpt)
        model.to(device)
        model.eval()

        # 加载归一化器
        scaler_feat = None
        scaler_target = None
        try:
            scaler_feat = joblib.load(str(SCALERS_DIR / 'seq2seq_scaler_feat.pkl'))
            scaler_target = joblib.load(str(SCALERS_DIR / 'seq2seq_scaler_target.pkl'))
        except Exception as e:
            print(f"[LSTM] 归一化器加载失败: {e}")

        # 获取历史特征数据
        historical_data = self._get_recent_days(SEQ_LEN_DAYS, data_path)

        available_cols = [c for c in FEATURE_COLS_FINAL if c in historical_data.columns]
        if len(available_cols) < ckpt_input_size:
            print(f"[LSTM] 可用特征 {len(available_cols)} < 模型需要 {ckpt_input_size}")
        base_cols = [c for c in FEATURE_COLS_FINAL if c in historical_data.columns]
        if len(base_cols) > ckpt_input_size:
            base_cols = base_cols[:ckpt_input_size]
        last_features = historical_data[base_cols].values.astype(np.float32)
        if last_features.shape[1] < ckpt_input_size:
            pad = np.zeros((last_features.shape[0], ckpt_input_size - last_features.shape[1]),
                          dtype=np.float32)
            last_features = np.concatenate([last_features, pad], axis=1)

        if scaler_feat is not None:
            try:
                last_features_norm = scaler_feat.transform(last_features)
            except ValueError as ve:
                print(f"[LSTM] 归一化失败: {ve}, 使用原始值")
                last_features_norm = last_features
        else:
            last_features_norm = last_features

        input_tensor = torch.tensor(last_features_norm[np.newaxis, :, :],
                                    dtype=torch.float32).to(device)

        with torch.no_grad():
            pred_norm = model(input_tensor).cpu().numpy()

        if scaler_target is not None:
            pred_orig = scaler_target.inverse_transform(pred_norm).flatten()
        else:
            pred_orig = pred_norm.flatten()

        if self.update_callback is not None:
            self.update_callback(pred_orig)

        return pred_orig

    def _get_recent_days(self, n_days, data_path=""):
        """
        获取最近 n_days 天的气象特征数据
        支持: .nc (小时/日级), .csv, .xlsx/.xls, 文件夹
        """
        import glob as glob_mod
        import xarray as xr

        df = None

        agg_funcs = {
            'smlt': 'sum', 'ssrd': 'sum', 'e': 'sum',
            'u10': 'mean', 'v10': 'mean', 'sp': 'mean', 'skt': 'mean',
        }
        variables = list(agg_funcs.keys())

        if os.path.isdir(data_path):
            nc_files = glob_mod.glob(os.path.join(data_path, '*.nc'))
            if not nc_files:
                raise FileNotFoundError(f"文件夹 {data_path} 中没有找到 .nc 文件")
            print(f"找到 {len(nc_files)} 个.nc文件，正在处理...")
            all_daily_data = []
            for file in nc_files:
                try:
                    ds = xr.open_dataset(file)
                    missing_vars = [v for v in variables if v not in ds.data_vars]
                    if missing_vars:
                        ds.close()
                        continue
                    time_coord = 'valid_time' if 'valid_time' in ds.coords else 'time'
                    ds_spatial = ds.mean(dim=['latitude', 'longitude'], skipna=True)
                    data_dict = {'datetime': ds_spatial[time_coord].values}
                    for var in variables:
                        if var in ds_spatial.data_vars:
                            data_dict[var] = ds_spatial[var].values
                    df_hourly = pd.DataFrame(data_dict)
                    df_hourly['date'] = pd.to_datetime(df_hourly['datetime']).dt.date
                    df_daily = df_hourly.groupby('date').agg(agg_funcs).reset_index()
                    all_daily_data.append(df_daily)
                    ds.close()
                except Exception as e:
                    print(f"  处理文件 {os.path.basename(file)} 时出错: {e}")
                    continue
            if all_daily_data:
                df = pd.concat(all_daily_data, ignore_index=True)
                df['date'] = pd.to_datetime(df['date'])

        elif data_path.endswith('.nc'):
            ds = xr.open_dataset(data_path)
            time_coord = 'valid_time' if 'valid_time' in ds.coords else 'time'
            times = ds[time_coord].values
            is_hourly = len(times) > 365
            if is_hourly:
                ds_spatial = ds.mean(dim=['latitude', 'longitude'], skipna=True)
                data_dict = {'datetime': ds_spatial[time_coord].values}
                for var in variables:
                    if var in ds_spatial.data_vars:
                        data_dict[var] = ds_spatial[var].values
                df_hourly = pd.DataFrame(data_dict)
                df_hourly['date'] = pd.to_datetime(df_hourly['datetime']).dt.date
                df = df_hourly.groupby('date').agg(agg_funcs).reset_index()
                df['date'] = pd.to_datetime(df['date'])
            else:
                df = ds.to_dataframe().reset_index()
                if 'time' in df.columns:
                    df = df.rename(columns={'time': 'date'})
                spatial_cols = [c for c in ['latitude', 'longitude', 'lat', 'lon'] if c in df.columns]
                if spatial_cols:
                    df = df.groupby('date').mean(numeric_only=True).reset_index()
            ds.close()

        elif data_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(data_path, parse_dates=['date'])
            print(f"Excel文件读取完成，共 {len(df)} 天")

        elif data_path.endswith('.csv'):
            df = pd.read_csv(data_path, parse_dates=['date'])
            print(f"CSV文件读取完成，共 {len(df)} 天")

        # 回退到默认数据
        if df is None:
            default_csv = str(DATA_DIR / 'ERA5_daily_with_discharge.csv')
            print(f"使用默认数据文件: {default_csv}")
            df = pd.read_csv(default_csv, parse_dates=['date'])

        df = df.sort_values('date').reset_index(drop=True)

        # 日历循环编码
        df['day_of_year_sin'] = np.sin(2 * np.pi * df['date'].dt.dayofyear / 365.25)
        df['day_of_year_cos'] = np.cos(2 * np.pi * df['date'].dt.dayofyear / 365.25)

        recent = df.iloc[-n_days:].copy()
        if len(recent) < n_days:
            raise ValueError(f"数据量不足！模型需要过去 {n_days} 天的数据，但仅找到 {len(recent)} 天。")

        available_cols = [c for c in FEATURE_COLS_FINAL if c in recent.columns]
        missing_cols = [c for c in FEATURE_COLS_FINAL if c not in recent.columns]
        if missing_cols:
            print(f"[数据警告] 缺少以下特征列: {missing_cols}, 将用 0 填充")
            for c in missing_cols:
                recent[c] = 0.0
        return recent[FEATURE_COLS_FINAL]

    def get_head(self, V_storage):
        if V_storage <= 0:
            return 0
        return self.H_max * (V_storage / self.V_max) ** (1 / 3)


# ============================================================
# 第三部分  辅助计算函数 (ET0, 需水, 经济参数)
# ============================================================

def calculate_et0(params: Dict[str, float]) -> float:
    """FAO Penman-Monteith 蒸散发计算"""
    numerator = (
        0.408 * params["delta"] * (params["Rn"] - params["G"])
        + params["gamma"] * (900 / (params["T"] + 278)) * params["u2"] * (params["es"] - params["ea"])
    )
    denominator = params["delta"] + params["gamma"] * (1 + 0.34 * params["u2"])
    return numerator / denominator if denominator != 0 else 0.0


def calculate_monthly_demands(
    month: int,
    pop_wan: float,
    urban_rate_percent: float,
    gdp_yi: float,
    reuse_percent: float,
    irrigation_eff: float,
    eco_base: float,
    et0_daily: float,
    crop_rows: List[dict],
    fao_kc: Dict[str, Dict[str, float]],
    target_year: int = 2026,
    urban_quota: float = 145.0,
    rural_quota: float = 80.0,
) -> Dict[str, float]:
    """计算月度各部门需水量 

    urban_quota / rural_quota: 城市/农村人均用水定额 (L/天), 默认 145 / 80
    target_year: 下游国家径流预测目标年份
    """
    days_in_month = calendar.monthrange(2026, int(month))[1]
    pop = float(pop_wan) * 10000
    urban_rate = float(urban_rate_percent) / 100.0
    reuse_rate = float(reuse_percent) / 100.0

    # 生活: 可配置人均用水定额 (L/天)
    pop_urban = pop * urban_rate
    pop_rural = pop * (1 - urban_rate)
    live_m3 = (pop_urban * urban_quota / 1000 + pop_rural * rural_quota / 1000) * days_in_month
    live = live_m3 / 1_000_000

    # 生态: 关联生活用水 + 生态保障基数
    eco = 0.1 * live + float(eco_base)

    # 农业: 面积单位 km² → m²
    agr = 0.0
    for crop in crop_rows:
        area_str = str(crop.get("area", "")).strip()
        if not area_str:
            continue
        try:
            crop_type = str(crop["type"])
            crop_stage = str(crop["stage"])
            c_area = float(area_str) * 1_000_000  # km² → m²
            kc = float(fao_kc[crop_type][crop_stage])
        except (KeyError, ValueError, TypeError):
            continue

        etc_monthly = kc * et0_daily * days_in_month
        water_m3 = etc_monthly * 0.001 * c_area
        agr += (water_m3 / 1_000_000) / irrigation_eff if irrigation_eff > 0 else 0.0

    # 工业: 万元GDP定额 140 m³ + 季节系数 
    INDUSTRIAL_WATER_QUOTA = 140
    annual_industrial_water = float(gdp_yi) * 10000 * INDUSTRIAL_WATER_QUOTA * (1 - reuse_rate)
    season_factors = [0.85, 0.80, 0.90, 0.95, 1.05, 1.10, 1.15, 1.15, 1.05, 0.95, 0.85, 0.80]
    ind = annual_industrial_water * season_factors[int(month) - 1] / 12 / 1_000_000

    # 下游国家: LSTM 合并月径流预测 (Billion m³ → 百万m³)
    try:
        result = predict_downstream_total(int(target_year))
        downstream = float(result['downstream_monthly'][int(month) - 1]) * 1000
    except Exception as e:
        print(f"下游国家预测数据获取失败: {e}")
        downstream = 0.0

    return {
        SECTOR_LIVE: round(live, 2),
        SECTOR_ECO: round(eco, 2),
        SECTOR_AGR: round(agr, 2),
        SECTOR_IND: round(ind, 2),
        SECTOR_DOWN: round(downstream, 2),
    }


def estimate_economic_params(
    crop_rows: List[dict],
    agr_water_demand_million_m3: float,
    hydro_pmax: float,
    hydro_qmax: float,
    hydro_price: float,
):
    """估算经济效益系数"""
    p_max = float(hydro_pmax)
    q_max = float(hydro_qmax)
    elec_price = float(hydro_price)
    a_hydro = ((p_max * 1000) / (q_max * 3600)) * elec_price

    total_revenue_yuan = 0.0
    for crop in crop_rows:
        area_str = str(crop.get("area", "")).strip()
        if not area_str:
            continue
        try:
            area_mu = float(area_str) * 10000
            crop_yield = float(crop.get("yield", 0))
            crop_price = float(crop.get("price", 0))
        except (ValueError, TypeError):
            continue
        total_revenue_yuan += area_mu * crop_yield * crop_price

    agr_water_demand_m3 = float(agr_water_demand_million_m3) * 1_000_000
    # 抬高农业经济权重: alpha 0.5→1.0, 并设下限 6.0 (接近工业 9.0),
    # 使优化器有动力稳定农业分配, 避免其在 [0.2D, 2.5D] 区间内剧烈抖动。
    alpha = 1.0
    AGR_MARGIN_FLOOR = 6.0
    a_agr = (total_revenue_yuan / agr_water_demand_m3) * alpha if agr_water_demand_m3 > 0 else 0.8
    a_agr = max(a_agr, AGR_MARGIN_FLOOR)

    a_dom, a_eco, a_ind = 1.1, 1.0, 9.0
    a_surface = [a_dom + a_hydro, a_eco + a_hydro, a_agr + a_hydro, a_ind + a_hydro]
    b_surface = [0.005, 0.105, 0.005, 1.505]
    a_ground = [a_dom, a_eco, a_agr, a_ind]
    b_ground = [a_dom + 0.4, 0.1, a_agr + 0.3, a_ind + 0.5]
    return np.array([a_surface, a_ground]), np.array([b_surface, b_ground]), a_hydro, a_agr
