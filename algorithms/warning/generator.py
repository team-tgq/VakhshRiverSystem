# ==================== 数据生成器 ====================
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings

from algorithms.warning.config import WARNING_THRESHOLDS
from algorithms.warning.core import FloodEarlyWarningSystem

warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier

import matplotlib
class FloodDataGenerator:
    """生成模拟数据，并调用预警系统处理"""
    def __init__(self, warning_system: FloodEarlyWarningSystem):
        self.warning_system = warning_system
        self.warning_levels = ['蓝', '黄', '橙', '红']
        self.warning_colors = {'蓝':'#1f77b4', '黄':'yellow', '橙':'#ff9900', '红':'#d62728'}
        self.stations = [f'站点{i:02d}' for i in range(1,13)]

    def generate_weather_data(self, hours=24):
        now = datetime.now()
        timestamps = [now - timedelta(hours=i) for i in range(hours)][::-1]
        weather_data = pd.DataFrame({
            '时间': timestamps,
            '降雨量(mm/h)': np.random.exponential(10, hours) * (1 + 0.5 * np.sin(np.arange(hours)*0.5)),
            '温度(℃)': 25 + 5 * np.sin(np.arange(hours)*0.3),
            '湿度(%)': 70 + 20 * np.sin(np.arange(hours)*0.2),
            '风速(m/s)': 3 + 2 * np.sin(np.arange(hours)*0.4),
            '气压(hPa)': 1013 + 5 * np.sin(np.arange(hours)*0.25)
        })
        return weather_data

    def generate_hydrology_data(self, hours=24):
        now = datetime.now()
        timestamps = [now - timedelta(hours=i) for i in range(hours)][::-1]
        hydro_data = pd.DataFrame({
            '时间': timestamps,
            '水位(m)': 5 + 0.5 * np.sin(np.arange(hours)*0.3) + np.random.randn(hours)*0.1,
            '流量(m³/s)': 50 + 20 * np.sin(np.arange(hours)*0.25) + np.random.randn(hours)*5,
            '流速(m/s)': 1.5 + 0.5 * np.sin(np.arange(hours)*0.2) + np.random.randn(hours)*0.1,
            '水温(℃)': 15 + 3 * np.sin(np.arange(hours)*0.2)
        })
        return hydro_data

    def generate_reservoir_data(self, hours=24):
        now = datetime.now()
        timestamps = [now - timedelta(hours=i) for i in range(hours)][::-1]
        res_data = pd.DataFrame({
            '时间': timestamps,
            '水位(m)': 50 + 5 * np.sin(np.arange(hours)*0.2) + np.random.randn(hours)*2,
            '出库流量(m³/s)': 20 + 5 * np.sin(np.arange(hours)*0.3) + np.random.randn(hours)*2,
            '入库流量(m³/s)': 25 + 8 * np.sin(np.arange(hours)*0.25) + np.random.randn(hours)*3
        })
        return res_data

    def generate_all_data(self):
        """生成所有数据并调用预警系统处理"""
        weather = self.generate_weather_data(24)
        hydro = self.generate_hydrology_data(24)
        reservoir = self.generate_reservoir_data(24)

        # 转换为预警系统所需的字典格式（注意列名匹配）
        real_time_data = {
            'weather_data': weather.rename(columns={'时间':'timestamp', '降雨量(mm/h)':'rainfall',
                                                     '风速(m/s)':'wind_speed'}),
            'hydrology_data': hydro.rename(columns={'时间':'timestamp', '水位(m)':'level',
                                                     '流量(m³/s)':'flow', '流速(m/s)':'velocity'}),
            'reservoir_data': reservoir.rename(columns={'时间':'timestamp', '水位(m)':'level',
                                                         '出库流量(m³/s)':'outflow', '入库流量(m³/s)':'inflow'})
        }

        # 处理数据，得到预警结果（整体流域情景）
        result = self.warning_system.process_real_time_data(real_time_data)

        # 根据实际情况构造站点表格数据，每个站点独立
        station_data = []
        # 区域平均水位（取水文数据最后一小时的平均值）
        base_water_level = hydro['水位(m)'].iloc[-1]  # 最后时刻水位
        base_rainfall = weather['降雨量(mm/h)'].iloc[-1]
        base_flow = hydro['流量(m³/s)'].iloc[-1]

        # 为每个站点生成不同的水位和等级
        for i, station in enumerate(self.stations):
            # 根据站点索引模拟空间变异性（上游到下游）
            # 假设经度越小越上游，水位越高
            lon = 116 + (i - 6) * 0.5  # 使经度从约113到119分布
            lat = 40 + (i % 3 - 1) * 1.0  # 纬度略有变化
            # 水位：上游（经度小）水位高，下游水位低
            water_level = base_water_level + (116 - lon) * 0.5 + np.random.randn() * 0.3
            water_level = max(0, water_level)

            # 根据水位确定预警等级
            if water_level < WARNING_THRESHOLDS['blue']:
                level = '蓝'
                level_idx = 1
            elif water_level < WARNING_THRESHOLDS['yellow']:
                level = '黄'
                level_idx = 2
            elif water_level < WARNING_THRESHOLDS['orange']:
                level = '橙'
                level_idx = 3
            else:
                level = '红'
                level_idx = 4

            # 置信度根据水位与阈值的距离设定
            if level == '蓝':
                dist = (WARNING_THRESHOLDS['blue'] - water_level) / WARNING_THRESHOLDS['blue']
                confidence = min(0.95, 0.7 + dist * 0.25)
            elif level == '黄':
                dist_high = WARNING_THRESHOLDS['yellow'] - water_level
                confidence = 0.7 + 0.2 * (1 - dist_high / (WARNING_THRESHOLDS['yellow'] - WARNING_THRESHOLDS['blue']))
            elif level == '橙':
                dist_high = WARNING_THRESHOLDS['orange'] - water_level
                confidence = 0.7 + 0.2 * (1 - dist_high / (WARNING_THRESHOLDS['orange'] - WARNING_THRESHOLDS['yellow']))
            else:  # 红
                dist = (water_level - WARNING_THRESHOLDS['red']) / WARNING_THRESHOLDS['red']
                confidence = min(0.95, 0.7 + dist * 0.25)
            confidence = np.clip(confidence, 0.5, 0.98)

            # 流量与水位相关
            flow = base_flow * (water_level / max(base_water_level, 0.1)) + np.random.randn() * 10

            station_data.append({
                '站点名称': station,
                '经度': lon,
                '纬度': lat,
                '预警等级': level,
                '等级值': level_idx,
                '置信度': confidence,
                '水位(m)': water_level,
                '流量(m³/s)': flow,
                '降雨量(mm/h)': base_rainfall * (1 + 0.3 * np.random.randn()),
                '风速(m/s)': weather['风速(m/s)'].iloc[-1] + np.random.randn() * 1
            })

        return {
            'weather': weather,
            'hydro': hydro,
            'stations': pd.DataFrame(station_data),
            'timestamp': datetime.now(),
            'scenario': result['scenario'],
            'confidence': result['confidence'],
            'warning_levels': result['warning_levels']  # 整体流域预警等级
        }