import calendar
import numpy as np

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination


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
                        xu[idx] = min(
                            self.W[i],
                            self.F_max[k, j] / (1 - self.loss_rates[k]) if self.loss_rates[k] < 1 else self.W[i]
                        )
                    else:
                        xu[idx] = min(self.W[i], self.F_max[k, j])
                    idx += 1

        super().__init__(
            n_var=n_var,
            n_obj=3,
            n_ieq_constr=self.n + self.r + 2 * self.r * self.m,
            xl=xl,
            xu=xu
        )

    def _evaluate(self, x, out, *args, **kwargs):
        X = x.reshape((self.n, self.r, self.m))
        R = np.zeros((self.r, self.m))

        for k in range(self.r):
            for j in range(self.m):
                R[k, j] = X[0, k, j] * (1 - self.loss_rates[k]) + X[1, k, j]

        # 目标1：最大化经济效益
        profit = 0.0
        for k in range(self.r):
            for j in range(self.m):
                M_kj = X[0, k, j]
                OW_kj = X[1, k, j]

                margin_surface = self.a[0, j] - self.b[0, j]
                margin_ground = self.a[1, j] - self.b[1, j]

                EP_kj = (
                    margin_surface * M_kj * (1 - self.loss_rates[k])
                    + margin_ground * OW_kj
                ) * self.T[j]
                profit += EP_kj

        f1 = -profit

        # 目标2：最小化总缺水量
        shortage = np.sum(np.maximum(0, self.D - R))
        f2 = shortage

        # 目标3：最小化部门公平性Gini
        y_util = np.zeros(self.m)
        for j in range(self.m):
            M_j = X[0, 0, j]
            OW_j = X[1, 0, j]
            R_total_j = R[0, j]

            if R_total_j > 0:
                margin_surface = self.a[0, j] - self.b[0, j]
                margin_ground = self.a[1, j] - self.b[1, j]
                EP_j = (
                    margin_surface * M_j * (1 - self.loss_rates[0])
                    + margin_ground * OW_j
                ) * self.T[j]
                y_util[j] = EP_j / R_total_j
            else:
                y_util[j] = 0.0

        sum_y = np.sum(y_util)
        if sum_y == 0:
            gini = 0.0
        else:
            diff_sum = sum(
                abs(y_util[a] - y_util[b])
                for a in range(self.m)
                for b in range(self.m)
            )
            gini = diff_sum / (2 * self.m * sum_y)

        f3 = gini

        out["F"] = [f1, f2, f3]

        # 约束
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


def calculate_et0(params: dict) -> float:
    numerator = (
        0.408 * params["delta"] * (params["Rn"] - params["G"])
        + params["gamma"] * (900 / (params["T"] + 278)) * params["u2"] * (params["es"] - params["ea"])
    )
    denominator = params["delta"] + params["gamma"] * (1 + 0.34 * params["u2"])
    if denominator == 0:
        return 0.0
    return numerator / denominator


def calculate_monthly_demands(
    month: int,
    pop_wan: float,
    urban_rate_percent: float,
    gdp_yi: float,
    reuse_percent: float,
    irrigation_eff: float,
    eco_base: float,
    et0_daily: float,
    crop_rows: list,
    fao_kc: dict,
) -> dict:
    days_in_month = calendar.monthrange(2026, month)[1]

    pop = pop_wan * 10000
    urban_rate = urban_rate_percent / 100.0
    reuse_rate = reuse_percent / 100.0

    pop_urban = pop * urban_rate
    pop_rural = pop * (1 - urban_rate)

    live_m3 = (pop_urban * 0.145 + pop_rural * 0.08) * days_in_month
    live = live_m3 / 1_000_000

    eco = max(eco_base, 0.1 * live)

    agr = 0.0
    for crop in crop_rows:
        area_wanmu = crop["area"]
        if area_wanmu <= 0:
            continue

        crop_type = crop["type"]
        crop_stage = crop["stage"]
        kc = fao_kc[crop_type][crop_stage]

        area_mu = area_wanmu * 10000
        etc_monthly = kc * et0_daily * days_in_month
        water_m3 = etc_monthly * 0.001 * area_mu * 666.67 * 0.6

        if irrigation_eff > 0:
            agr += (water_m3 / 1_000_000) / irrigation_eff

    ind = (gdp_yi * 130 / 100) * (1 - reuse_rate) / 12

    return {
        "生活": round(live, 2),
        "生态": round(eco, 2),
        "农业": round(agr, 2),
        "工业": round(ind, 2),
    }


def estimate_economic_params(crop_rows: list, agr_water_demand_million_m3: float,
                             hydro_pmax: float, hydro_qmax: float, hydro_price: float):
    # 发电经济估算
    kwh_per_m3 = (hydro_pmax * 1000) / (hydro_qmax * 3600)
    a_hydro = kwh_per_m3 * hydro_price
    b_hydro = 0.005

    # 农业NPV分摊净产值估算
    total_revenue_yuan = 0.0
    for crop in crop_rows:
        area_wanmu = crop["area"]
        c_yield = crop["yield"]
        c_price = crop["price"]

        if area_wanmu <= 0:
            continue

        area_mu = area_wanmu * 10000
        total_revenue_yuan += area_mu * c_yield * c_price

    agr_water_demand_m3 = agr_water_demand_million_m3 * 1_000_000
    alpha = 0.5

    if agr_water_demand_m3 > 0:
        a_agr = (total_revenue_yuan / agr_water_demand_m3) * alpha
    else:
        a_agr = 0.8

    b_agr = 0.0

    a_dom, b_dom = 1.1, 0.0
    a_eco, b_eco = 1.0, 0.1
    a_ind, b_ind = 9.0, 1.5

    a_surface = [a_dom + a_hydro, a_eco + a_hydro, a_agr + a_hydro, a_ind + a_hydro]
    b_surface = [b_dom + b_hydro, b_eco + b_hydro, b_agr + b_hydro, b_ind + b_hydro]

    a_ground = [a_dom, a_eco, a_agr, a_ind]
    b_ground = [b_dom + 0.4, b_eco, b_agr + 0.3, b_ind + 0.5]

    return np.array([a_surface, a_ground]), np.array([b_surface, b_ground]), a_hydro, a_agr


def run_water_allocation_optimization(input_data: dict) -> dict:
    month = int(input_data["month"])

    W_supply = np.array([
        float(input_data["w_surface"]),
        float(input_data["w_ground"])
    ])

    D_demand = np.zeros((1, 4))
    demand_order = ["生活", "生态", "农业", "工业"]
    for idx, sec in enumerate(demand_order):
        D_demand[0, idx] = float(input_data["demands"][sec])

    loss_rates = np.zeros(1)
    loss_rates[0] = float(input_data["loss_percent"]) / 100.0

    a_matrix, b_matrix, a_hydro, a_agr = estimate_economic_params(
        crop_rows=input_data["crop_rows"],
        agr_water_demand_million_m3=float(input_data["demands"]["农业"]),
        hydro_pmax=float(input_data["hydro_pmax"]),
        hydro_qmax=float(input_data["hydro_qmax"]),
        hydro_price=float(input_data["hydro_price"]),
    )

    T_weights = np.array([1, 1, 1, 1])
    F_min = D_demand * 0.6
    F_max = D_demand * 1.2

    pref_weights = np.array([
        float(input_data["w_econ"]),
        float(input_data["w_short"]),
        float(input_data["w_gini"]),
    ])

    problem = NurekWaterAllocation(
        2, 1, 4,
        a_matrix, b_matrix, T_weights,
        D_demand, W_supply, F_min, F_max, loss_rates
    )

    algorithm = NSGA2(pop_size=150)
    res = minimize(
        problem,
        algorithm,
        get_termination("n_gen", 400),
        seed=1,
        verbose=False
    )

    F = res.F
    F_min_norm = F.min(axis=0)
    F_max_norm = F.max(axis=0)
    F_range = np.where(F_max_norm - F_min_norm == 0, 1e-9, F_max_norm - F_min_norm)

    F_normalized = ((F - F_min_norm) / F_range) * pref_weights
    distances = np.linalg.norm(F_normalized, axis=1)
    best_idx = np.argmin(distances)

    best_profit = -res.F[best_idx, 0]
    best_shortage = res.F[best_idx, 1]
    best_gini = res.F[best_idx, 2]
    X_opt = res.X[best_idx].reshape((2, 1, 4))

    return {
        "month": month,
        "profit": float(best_profit),
        "shortage": float(best_shortage),
        "gini": float(best_gini),
        "X_opt": X_opt,
        "D_demand": D_demand,
        "loss_rates": loss_rates,
        "W_supply": W_supply,
        "a_hydro": float(a_hydro),
        "a_agr": float(a_agr),
    }


def format_result_text(result: dict, sectors: list) -> str:
    month = result["month"]
    profit = result["profit"]
    shortage = result["shortage"]
    gini = result["gini"]
    X_opt = result["X_opt"]
    D = result["D_demand"]
    loss = result["loss_rates"]
    W = result["W_supply"]
    a_hydro = result["a_hydro"]

    if gini < 0.1:
        gini_diag = "(完全公平)"
    elif 0.1 <= gini < 0.2:
        gini_diag = "(满意度较平均)"
    elif 0.2 <= gini < 0.3:
        gini_diag = "(分配偏向高产值部门)"
    else:
        gini_diag = "(偏科严重，存在明显受损部门)"

    lines = [
        f"🎯 第 {month} 月度 哈特隆州配置方案",
        "=" * 85,
        f"💰 系统总综合经济效益 : {profit:,.2f} 万元",
        f"📉 系统总缺水量       : {shortage:,.2f} 百万m³",
        f"⚖️ 部门公平性 Gini   : {gini:.4f} {gini_diag}",
        "=" * 85,
        f"",
        f"📍 地区：哈特隆州 (管网传输损耗率: {loss[0] * 100:.1f}%)",
        f"{'部门':<10} | {'需水量':<10} | {'水库放水量':<12} | {'最终实收水量':<12} | {'满足率'}",
        "-" * 75
    ]

    for j, sec in enumerate(sectors):
        demand = D[0, j]
        surf_out = X_opt[0, 0, j]
        received = surf_out * (1 - loss[0]) + X_opt[1, 0, j]
        ratio = (received / demand * 100) if demand > 0 else 100

        lines.append(
            f"{sec:<10} | {demand:<13.2f} | {surf_out:<15.2f} | {received:<16.2f} | {ratio:.1f}%"
        )

    total_surf = X_opt[0, 0, :].sum()
    total_hydro_profit = total_surf * a_hydro

    lines.extend([
        "",
        "=" * 85,
        f"🌊 水库大坝放水总量: {total_surf:.2f} / {W[0]:.2f} 百万m³",
        f"💧 区域地下水抽水量: {X_opt[1, 0, :].sum():.2f} / {W[1]:.2f} 百万m³",
        f"🔌 大坝水力发电独立贡献: 约 {total_hydro_profit:,.2f} 万元",
    ])

    return "\n".join(lines)