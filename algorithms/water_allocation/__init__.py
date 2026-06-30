"""
Water Allocation Module v2.0 (09模块)
瓦赫什河流域水资源动态优化配置系统

主要功能:
- 多目标优化算法进行多时间尺度部门用水分配 (日/月/年)
- LSTM Seq2Seq 模型预测入库径流量
- GM(1,1) 下游三国(土库曼斯坦/乌兹别克斯坦/阿富汗)用水量预测
- ET0 蒸散发计算与需水量估算
- FTW 遥感耕地面积提取
- GEE 在线数据获取

子模块:
- core: 核心优化算法 (NSGA-II + 径流预测)
- lstm_model: Seq2SeqLSTM 模型定义
- train: LSTM 模型训练脚本
- predict: GM(1,1) 下游需水预测
- remote_sensing: 遥感处理子模块 (FTW + GEE)
- main_ui: 独立 Tkinter 界面
- resources: 数据/模型/权重资源
"""
from .core import (
    # 常量
    SECTOR_LIVE,
    SECTOR_ECO,
    SECTOR_AGR,
    SECTOR_IND,
    SECTOR_DOWN,
    SECTOR_ORDER,
    SECTOR_ORDER_V2,
    # 核心类
    NurekWaterAllocation,
    NurekDamParameters,
    # 核心函数
    run_nsga2_opt,
    # 辅助函数
    calculate_et0,
    calculate_monthly_demands,
    estimate_economic_params,
)
from .lstm_model import Seq2SeqLSTM, FEATURE_COLS_FINAL, FEATURE_COLS_RAW
from .train import train_from_external_data
from .predict import (
    predict_downstream_total,
    predict_monthly_water_demand,
    gm11_predict_exact,
    distribute_to_months,
    DOWNSTREAM_SHARE_FACTOR,
)

__all__ = [
    # 常量
    "SECTOR_LIVE",
    "SECTOR_ECO",
    "SECTOR_AGR",
    "SECTOR_IND",
    "SECTOR_DOWN",
    "SECTOR_ORDER",
    "SECTOR_ORDER_V2",
    "FEATURE_COLS_FINAL",
    "FEATURE_COLS_RAW",
    "DOWNSTREAM_SHARE_FACTOR",
    # 核心类
    "NurekWaterAllocation",
    "NurekDamParameters",
    "Seq2SeqLSTM",
    # 核心函数
    "run_nsga2_opt",
    "train_from_external_data",
    # 预测
    "predict_downstream_total",
    "predict_monthly_water_demand",
    "gm11_predict_exact",
    "distribute_to_months",
    # 辅助
    "calculate_et0",
    "calculate_monthly_demands",
    "estimate_economic_params",
]
