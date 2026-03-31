from __future__ import annotations

import calendar
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.util.ref_dirs import get_reference_directions

from .lstm_model import FEATURE_COLS_FINAL, Seq2SeqLSTM

SECTOR_LIVE = "生活"
SECTOR_ECO = "生态"
SECTOR_AGR = "农业"
SECTOR_IND = "工业"
SECTOR_ORDER = [SECTOR_LIVE, SECTOR_ECO, SECTOR_AGR, SECTOR_IND]

RESOURCE_DIR = Path(__file__).resolve().parent / "resources"


class NurekWaterAllocation(ElementwiseProblem):
    def __init__(self, n_sources, n_regions, m_sectors, a, b, T, D, W, F_min, F_max, loss_rates):
        self.n, self.r, self.m = n_sources, n_regions, m_sectors
        self.a, self.b, self.T = a, b, T
        self.D, self.W = D, W
        self.F_min, self.F_max = F_min, F_max
        self.loss_rates = loss_rates

        n_var = self.n * self.r * self.m
        xl = np.zeros(n_var)
        xu = np.zeros(n_var)

        idx = 0
        for i in range(self.n):
            for k in range(self.r):
                for j in range(self.m):
                    if i == 0:
                        if self.loss_rates[k] < 1:
                            xu[idx] = min(self.W[i], self.F_max[k, j] / (1 - self.loss_rates[k]))
                        else:
                            xu[idx] = self.W[i]
                    else:
                        xu[idx] = min(self.W[i], self.F_max[k, j])
                    idx += 1

        super().__init__(
            n_var=n_var,
            n_obj=3,
            n_ieq_constr=self.n + self.r + 2 * self.r * self.m,
            xl=xl,
            xu=xu,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        X = x.reshape((self.n, self.r, self.m))
        R = np.zeros((self.r, self.m))

        for k in range(self.r):
            for j in range(self.m):
                R[k, j] = X[0, k, j] * (1 - self.loss_rates[k]) + X[1, k, j]

        profit = 0.0
        for k in range(self.r):
            for j in range(self.m):
                M_kj = X[0, k, j]
                OW_kj = X[1, k, j]
                margin_surface = self.a[0, j] - self.b[0, j]
                margin_ground = self.a[1, j] - self.b[1, j]
                EP_kj = (margin_surface * M_kj * (1 - self.loss_rates[k]) + margin_ground * OW_kj) * self.T[j]
                profit += EP_kj

        f1 = -profit
        f2 = np.sum(np.maximum(0, self.D - R))

        y_util = np.zeros(self.m)
        for j in range(self.m):
            M_j = X[0, 0, j]
            OW_j = X[1, 0, j]
            R_total_j = R[0, j]
            if R_total_j > 0:
                margin_surface = self.a[0, j] - self.b[0, j]
                margin_ground = self.a[1, j] - self.b[1, j]
                EP_j = (margin_surface * M_j * (1 - self.loss_rates[0]) + margin_ground * OW_j) * self.T[j]
                y_util[j] = EP_j / R_total_j
            else:
                y_util[j] = 0.0

        sum_y = np.sum(y_util)
        if sum_y == 0:
            gini = 0.0
        else:
            diff_sum = sum(abs(y_util[a] - y_util[b]) for a in range(self.m) for b in range(self.m))
            gini = diff_sum / (2 * self.m * sum_y)

        out["F"] = [f1, f2, gini]

        g = []
        g.append(X[0, :, :].sum() - self.W[0])
        g.append(X[1, :, :].sum() - self.W[1])
        for k in range(self.r):
            g.append(R[k, :].sum() - 1.5 * self.D[k, :].sum())
        for k in range(self.r):
            for j in range(self.m):
                g.append(self.F_min[k, j] - R[k, j])
                g.append(R[k, j] - self.F_max[k, j])
        out["G"] = g


def calculate_et0(params: Dict[str, float]) -> float:
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
) -> Dict[str, float]:
    _ = eco_base

    days_in_month = calendar.monthrange(2026, int(month))[1]
    pop = float(pop_wan) * 10000
    urban_rate = float(urban_rate_percent) / 100.0
    reuse_rate = float(reuse_percent) / 100.0

    pop_urban = pop * urban_rate
    pop_rural = pop * (1 - urban_rate)
    live_m3 = (pop_urban * 0.145 + pop_rural * 0.08) * days_in_month
    live = live_m3 / 1_000_000

    # 与 v2.0 一致：生态需水按生活需水的 10% 动态估计。
    eco = 0.1 * live

    agr = 0.0
    for crop in crop_rows:
        area_str = str(crop.get("area", "")).strip()
        if not area_str:
            continue
        try:
            crop_type = str(crop["type"])
            crop_stage = str(crop["stage"])
            area_mu = float(area_str) * 10000
            kc = float(fao_kc[crop_type][crop_stage])
        except (KeyError, ValueError, TypeError):
            continue

        etc_monthly = kc * et0_daily * days_in_month
        water_m3 = etc_monthly * 0.001 * area_mu * 666.67 * 0.6
        agr += (water_m3 / 1_000_000) / irrigation_eff if irrigation_eff > 0 else 0.0

    ind = (float(gdp_yi) * 130 / 100) * (1 - reuse_rate) / 12

    return {
        SECTOR_LIVE: round(live, 2),
        SECTOR_ECO: round(eco, 2),
        SECTOR_AGR: round(agr, 2),
        SECTOR_IND: round(ind, 2),
    }


def estimate_economic_params(
    crop_rows: List[dict],
    agr_water_demand_million_m3: float,
    hydro_pmax: float,
    hydro_qmax: float,
    hydro_price: float,
):
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
    alpha = 0.5
    a_agr = (total_revenue_yuan / agr_water_demand_m3) * alpha if agr_water_demand_m3 > 0 else 0.8

    a_dom, a_eco, a_ind = 1.1, 1.0, 9.0
    a_surface = [a_dom + a_hydro, a_eco + a_hydro, a_agr + a_hydro, a_ind + a_hydro]
    b_surface = [0.005, 0.105, 0.005, 1.505]
    a_ground = [a_dom, a_eco, a_agr, a_ind]
    b_ground = [a_dom + 0.4, 0.1, a_agr + 0.3, a_ind + 0.5]
    return np.array([a_surface, a_ground]), np.array([b_surface, b_ground]), a_hydro, a_agr


def run_nsga2_opt(problem_params, pop_size=150, n_gen=400):
    problem = NurekWaterAllocation(**problem_params)
    algorithm = NSGA2(pop_size=int(pop_size))
    return minimize(problem, algorithm, get_termination("n_gen", int(n_gen)), seed=1, verbose=False)


def run_water_allocation_optimization(input_data: Dict) -> Dict:
    month = int(input_data["month"])
    W_supply = np.array([float(input_data["w_surface"]), float(input_data["w_ground"])], dtype=float)

    D_demand = np.zeros((1, 4), dtype=float)
    for idx, sec in enumerate(SECTOR_ORDER):
        D_demand[0, idx] = float(input_data["demands"][sec])

    loss_rates = np.zeros(1, dtype=float)
    loss_rates[0] = float(input_data["loss_percent"]) / 100.0

    a_matrix, b_matrix, a_hydro, a_agr = estimate_economic_params(
        crop_rows=input_data["crop_rows"],
        agr_water_demand_million_m3=float(input_data["demands"][SECTOR_AGR]),
        hydro_pmax=float(input_data["hydro_pmax"]),
        hydro_qmax=float(input_data["hydro_qmax"]),
        hydro_price=float(input_data["hydro_price"]),
    )

    raw_f_min = D_demand * 0.6
    total_min_required = float(raw_f_min.sum())
    total_receivable = float(W_supply[0] * max(0.0, 1.0 - loss_rates[0]) + W_supply[1])
    if total_min_required > 0 and total_min_required > total_receivable:
        # 水资源不足时，按比例放宽最小供水约束，避免无可行解。
        relax_ratio = max(0.0, total_receivable / total_min_required) * 0.999
        F_min = raw_f_min * relax_ratio
    else:
        F_min = raw_f_min

    F_max = np.maximum(D_demand * 1.2, F_min + 1e-6)

    problem_params = {
        "n_sources": 2,
        "n_regions": 1,
        "m_sectors": 4,
        "a": a_matrix,
        "b": b_matrix,
        "T": np.array([1, 1, 1, 1], dtype=float),
        "D": D_demand,
        "W": W_supply,
        "F_min": F_min,
        "F_max": F_max,
        "loss_rates": loss_rates,
    }

    res = run_nsga2_opt(problem_params)
    if res is None or res.F is None or res.X is None:
        # 二次兜底：完全放开最小供水约束，确保缺水场景仍能给出分配结果。
        problem_params["F_min"] = np.zeros_like(D_demand)
        problem_params["F_max"] = np.maximum(D_demand * 1.2, 1e-6)
        res = run_nsga2_opt(problem_params)
    if res is None or res.F is None or res.X is None:
        raise RuntimeError("NSGA-II 未生成有效解，请检查供水、需水与损耗参数。")

    pref_weights = np.array(
        [
            float(input_data["w_econ"]),
            float(input_data["w_short"]),
            float(input_data["w_gini"]),
        ],
        dtype=float,
    )
    F = np.atleast_2d(np.asarray(res.F, dtype=float))
    X = np.atleast_2d(np.asarray(res.X, dtype=float))
    if F.shape[0] != X.shape[0]:
        raise RuntimeError("NSGA-II 返回结果维度异常。")
    if F.shape[1] < 3:
        raise RuntimeError("NSGA-II 返回目标维度不足。")
    if X.shape[1] != 8:
        raise RuntimeError("NSGA-II 返回决策变量维度异常。")

    F_min_norm = F.min(axis=0)
    F_max_norm = F.max(axis=0)
    F_range = np.where(F_max_norm - F_min_norm == 0, 1e-9, F_max_norm - F_min_norm)
    best_idx = int(np.argmin(np.linalg.norm(((F - F_min_norm) / F_range) * pref_weights, axis=1)))

    return {
        "month": month,
        "profit": float(-F[best_idx, 0]),
        "shortage": float(F[best_idx, 1]),
        "gini": float(F[best_idx, 2]),
        "X_opt": X[best_idx].reshape((2, 1, 4)),
        "D_demand": D_demand,
        "loss_rates": loss_rates,
        "W_supply": W_supply,
        "a_hydro": float(a_hydro),
        "a_agr": float(a_agr),
        "res": res,
    }


def format_result_text(result: Dict, sectors: List[str]) -> str:
    month = int(result["month"])
    profit = float(result["profit"])
    shortage = float(result["shortage"])
    gini = float(result["gini"])
    X_opt = np.asarray(result["X_opt"], dtype=float)
    D = np.asarray(result["D_demand"], dtype=float)
    loss = np.asarray(result["loss_rates"], dtype=float)
    W = np.asarray(result["W_supply"], dtype=float)
    a_hydro = float(result["a_hydro"])

    if gini < 0.1:
        gini_diag = "(完全公平)"
    elif gini < 0.2:
        gini_diag = "(满意度较均衡)"
    elif gini < 0.3:
        gini_diag = "(分配偏向高产值部门)"
    else:
        gini_diag = "(偏科严重，存在明显受损部门)"

    lines = [
        f"第 {month} 月哈特隆州配水优化方案",
        "=" * 85,
        f"系统总综合经济效益: {profit:,.2f} 万元",
        f"系统总缺水量      : {shortage:,.2f} 百万m³",
        f"部门公平性(Gini)  : {gini:.4f} {gini_diag}",
        "=" * 85,
        f"地区: 哈特隆州 (管网传输损耗率: {loss[0] * 100:.1f}%)",
        f"{'部门':<10} | {'需水量':<10} | {'水库放水量':<10} | {'最终实收水量':<10} | {'满足率'}",
        "-" * 75,
    ]

    for j, sec in enumerate(sectors):
        demand = D[0, j]
        surf_out = X_opt[0, 0, j]
        received = surf_out * (1 - loss[0]) + X_opt[1, 0, j]
        ratio = (received / demand * 100) if demand > 0 else 100.0
        lines.append(f"{sec:<10} | {demand:<13.2f} | {surf_out:<15.2f} | {received:<16.2f} | {ratio:.1f}%")

    total_surf = float(X_opt[0, 0, :].sum())
    total_hydro_profit = total_surf * a_hydro
    lines.extend(
        [
            "",
            "=" * 85,
            f"水库大坝放水总量: {total_surf:.2f} / {W[0]:.2f} 百万m³",
            f"区域地下水抽水量: {X_opt[1, 0, :].sum():.2f} / {W[1]:.2f} 百万m³",
            f"大坝水力发电独立贡献: 约 {total_hydro_profit:,.2f} 万元",
        ]
    )
    return "\n".join(lines)


def _resource_path(file_name: str) -> Path:
    path = RESOURCE_DIR / file_name
    if not path.exists():
        raise FileNotFoundError(f"缺少资源文件: {path}")
    return path


class NurekDamParameters:
    def __init__(self, elec_price=0.8, unit_water_margin=1.6, data_path="", v_initial=84.0):
        self.g = 9.81
        self.eta = 0.90
        self.hours_per_month = 24 * 30.4
        self.P_cap = 3015

        self.V_initial = float(v_initial)
        self.V_max = 105.0
        self.V_min = 60.0
        self.H_max = 265.0
        self.V_flood = 95.0

        self.Q_safe = 2000.0
        self.Q_eco = 100.0
        self.Q_max_release = 3500.0
        self.Q_min_release = 10.0
        self.months = np.arange(1, 13)

        self.elec_price = float(elec_price)
        self.unit_water_margin = float(unit_water_margin)
        self.Q_in = self._predict_monthly_inflow(data_path)

    def _predict_monthly_inflow(self, data_path=""):
        import joblib
        import torch

        input_size = 9
        hidden_size = 64
        num_layers = 2
        output_steps_months = 12
        seq_len_days = 365

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = Seq2SeqLSTM(input_size, hidden_size, num_layers, output_steps_months)
        model.load_state_dict(torch.load(str(_resource_path("best_seq2seq_lstm.pth")), map_location=device))
        model.to(device)
        model.eval()

        scaler_feat = joblib.load(str(_resource_path("seq2seq_scaler_feat.pkl")))
        scaler_target = joblib.load(str(_resource_path("seq2seq_scaler_target.pkl")))

        historical_data = self._get_recent_days(seq_len_days, data_path)
        last_features = np.asarray(historical_data[FEATURE_COLS_FINAL].values, dtype=np.float32)

        # 兼容 sklearn 全局 transform_output='pandas' 场景，强制转为 ndarray。
        last_features_norm = np.asarray(scaler_feat.transform(last_features), dtype=np.float32)
        if last_features_norm.ndim != 2:
            raise ValueError(f"特征归一化结果维度异常: {last_features_norm.shape}")

        input_tensor = torch.tensor(last_features_norm[np.newaxis, :, :], dtype=torch.float32).to(device)
        with torch.no_grad():
            pred_norm = model(input_tensor).cpu().numpy()

        pred_orig = np.asarray(scaler_target.inverse_transform(pred_norm), dtype=np.float32)
        return pred_orig.reshape(-1)

    def _get_recent_days(self, n_days, data_path=""):
        df = None
        if data_path and os.path.exists(data_path):
            lower = data_path.lower()
            if os.path.isdir(data_path):
                try:
                    import xarray as xr
                except ImportError as e:
                    raise ImportError("读取 nc 目录需要安装 xarray") from e
                nc_files = [str(x) for x in Path(data_path).glob("*.nc")]
                if not nc_files:
                    raise FileNotFoundError(f"目录中没有找到 .nc 文件: {data_path}")
                ds = xr.open_mfdataset(nc_files, combine="by_coords")
                df = ds.to_dataframe().reset_index()
            elif lower.endswith(".nc"):
                try:
                    import xarray as xr
                except ImportError as e:
                    raise ImportError("读取 nc 文件需要安装 xarray") from e
                ds = xr.open_dataset(data_path)
                df = ds.to_dataframe().reset_index()
            elif lower.endswith(".csv"):
                df = pd.read_csv(data_path, parse_dates=["date"])
            else:
                raise ValueError("仅支持 .csv、.nc 文件或包含 .nc 的目录")

            if df is not None and "date" not in df.columns:
                if "time" in df.columns:
                    df = df.rename(columns={"time": "date"})
                elif "valid_time" in df.columns:
                    df = df.rename(columns={"valid_time": "date"})

            if df is not None:
                spatial_cols = [c for c in ("latitude", "longitude", "lat", "lon") if c in df.columns]
                if spatial_cols:
                    df = df.groupby("date").mean(numeric_only=True).reset_index()

        if df is None:
            df = pd.read_csv(str(_resource_path("ERA5_daily_with_discharge.csv")), parse_dates=["date"])

        if "date" not in df.columns:
            raise ValueError("输入数据缺少 date 列")

        df = df.sort_values("date").reset_index(drop=True)
        df["day_of_year_sin"] = np.sin(2 * np.pi * df["date"].dt.dayofyear / 365.25)
        df["day_of_year_cos"] = np.cos(2 * np.pi * df["date"].dt.dayofyear / 365.25)

        missing_cols = [c for c in FEATURE_COLS_FINAL if c not in df.columns]
        if missing_cols:
            raise ValueError(f"输入数据缺少必要特征列: {missing_cols}")

        recent = df.iloc[-int(n_days) :].copy()
        if len(recent) < int(n_days):
            raise ValueError(f"数据量不足: 需要 {n_days} 天，实际 {len(recent)} 天")
        return recent[FEATURE_COLS_FINAL]

    def get_head(self, V_storage):
        if V_storage <= 0:
            return 0.0
        return self.H_max * (V_storage / self.V_max) ** (1 / 3)


class NurekDamSchedulingProblem(ElementwiseProblem):
    def __init__(self, params):
        self.p = params
        super().__init__(
            n_var=12,
            n_obj=3,
            n_ieq_constr=12,
            xl=np.array([params.Q_min_release] * 12),
            xu=np.array([params.Q_max_release] * 12),
        )

    def _evaluate(self, x, out, *args, **kwargs):
        releases = x
        V_t = self.p.V_initial
        V_storage = np.zeros(12)
        H_t = np.zeros(12)
        constraints = []

        for t in range(12):
            factor = (24 * 3600 * 30.4) / 1e8
            inflow_vol = self.p.Q_in[t] * factor
            release_vol = releases[t] * factor
            V_next = V_t + inflow_vol - release_vol

            V_storage[t] = V_next
            H_t[t] = self.p.get_head((V_t + V_next) / 2)

            c_upper = max(0.0, V_next - self.p.V_max)
            c_lower = max(0.0, self.p.V_min - V_next)
            constraints.append(c_upper + c_lower)
            V_t = V_next

        total_economic_profit = 0.0
        for t in range(12):
            p_mw = 9.81 * self.p.eta * releases[t] * H_t[t] / 1000
            p_mw = min(p_mw, self.p.P_cap)
            energy_kwh = p_mw * self.p.hours_per_month * 1000
            hydro_revenue = energy_kwh * self.p.elec_price

            release_vol_m3 = releases[t] * (24 * 3600 * 30.4)
            water_revenue = release_vol_m3 * self.p.unit_water_margin
            total_economic_profit += hydro_revenue + water_revenue

        f1 = -(total_economic_profit / 10000)

        flood_risk = 0.0
        summer_idx = [4, 5, 6, 7, 8]
        for t in range(12):
            if releases[t] > self.p.Q_safe:
                flood_risk += ((releases[t] - self.p.Q_safe) / self.p.Q_safe) ** 2
            if t in summer_idx and V_storage[t] > self.p.V_flood:
                flood_risk += ((V_storage[t] - self.p.V_flood) / self.p.V_flood) ** 2

        eco_deficit = 0.0
        for t in range(12):
            if releases[t] < self.p.Q_eco:
                eco_deficit += ((self.p.Q_eco - releases[t]) / self.p.Q_eco) ** 2

        out["F"] = [f1, flood_risk, eco_deficit]
        out["G"] = constraints


def run_nsga3_opt(params_obj, pop_size=100, n_gen=200):
    problem = NurekDamSchedulingProblem(params_obj)
    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)
    pop_size = max(int(pop_size), len(ref_dirs))
    algorithm = NSGA3(pop_size=pop_size, ref_dirs=ref_dirs, n_offsprings=50, eliminate_duplicates=True)
    return minimize(problem, algorithm, get_termination("n_gen", int(n_gen)), seed=1, verbose=False)


def run_dam_scheduling_optimization(input_data: Dict) -> Dict:
    params = NurekDamParameters(
        elec_price=float(input_data.get("elec_price", 0.8)),
        unit_water_margin=float(input_data.get("unit_water_margin", 1.6)),
        data_path=str(input_data.get("data_path", "") or ""),
        v_initial=float(input_data.get("v_initial", 84.0)),
    )

    pop_size = int(input_data.get("nsga3_pop", 100))
    n_gen = int(input_data.get("nsga3_gen", 200))
    res = run_nsga3_opt(params, pop_size=pop_size, n_gen=n_gen)

    if res is None or res.F is None or res.X is None:
        raise RuntimeError("NSGA-III 未生成有效解，请提高种群规模或迭代代数。")

    F = np.atleast_2d(np.asarray(res.F, dtype=float))
    X = np.atleast_2d(np.asarray(res.X, dtype=float))
    if F.shape[0] != X.shape[0]:
        raise RuntimeError("NSGA-III 返回结果维度异常。")

    econ = -F[:, 0]
    risk = F[:, 1]
    eco = F[:, 2]
    best_idx = int(np.argmax(econ / (np.max(econ) + 1e-9) - risk / (np.max(risk) + 0.001)))

    return {
        "best_release": X[best_idx],
        "econ": float(econ[best_idx]),
        "risk": float(risk[best_idx]),
        "eco": float(eco[best_idx]),
        "inflow": np.asarray(params.Q_in, dtype=float),
        "params": params,
        "res": res,
    }


def format_dam_result_text(result: Dict) -> str:
    best_sol = np.asarray(result["best_release"], dtype=float)
    econ_val = float(result["econ"])
    risk_val = float(result["risk"])
    eco_val = float(result["eco"])

    lines = [
        "NSGA-III 推荐折中调度方案",
        "=" * 50,
        f"全年综合经济总收益: {econ_val:,.2f} 万元",
        f"洪水风险指数: {risk_val:.4f}",
        f"生态缺水指数: {eco_val:.4f} (越低越好)",
        "=" * 50,
        "每月下泄流量 (m³/s):",
    ]

    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    for m, val in zip(months, best_sol):
        lines.append(f"{m:<5}: {val:.1f}")

    lines.append("")
    lines.append("每月预测入库流量 (m³/s):")
    for m, val in zip(months, np.asarray(result["inflow"], dtype=float)):
        lines.append(f"{m:<5}: {val:.1f}")

    return "\n".join(lines)
