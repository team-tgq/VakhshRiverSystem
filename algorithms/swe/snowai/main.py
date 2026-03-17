import datetime

from snowai.density import (
    JonasDensity,
    PistochiDensity,
    SturmDensity,
    MachineLearningDensity
)
from snowai.swe import (
    MachineLearningSWE,
    HillSWE,
    StatisticalModels
)

from snowai.utils import ConvertData, clean_cache

from metloom.pointdata import SnotelPointData
from pathlib import Path
import sys
import rioxarray
sys.excepthook = sys.__excepthook__

# 当前文件路径
current_file = Path(__file__)

# 项目根目录
root_dir = current_file.parents[1]
# 从 SNOTEL（美国雪遥测站）系统获取 Banner Summit 站点在 2024 年 2 月的每日雪和气温数据，并存成一个 pandas DataFrame。
# Let's select banner summit
pt = SnotelPointData("312:ID:SNTL", "Banner Summit")

# start data and end date
start_date = datetime.datetime(2024, 2, 1)
end_date = datetime.datetime(2024, 2, 29)


# Notice this is a list
# 还包含经纬度以及高程信息
variables = [
    pt.ALLOWED_VARIABLES.SWE,
    pt.ALLOWED_VARIABLES.TEMPMIN,
    pt.ALLOWED_VARIABLES.TEMPMAX,
    pt.ALLOWED_VARIABLES.TEMPAVG,
    pt.ALLOWED_VARIABLES.SNOWDEPTH,
]

# request the data
df = pt.get_daily_data(start_date, end_date, variables)
print(df.head(3))
'''
Data Cleaning（数据清理）简要整理
本步骤主要包括以下内容：
# 变量选择（Variable selection）
    snow_class：雪类（alpine/maritime/...）
    elevation：海拔（m）
    snow_depth：雪深（m）
    tavg：平均温度（°C）
    tmin：最低温（°C）
    tmax：最高温（°C）
    DOY：日期/日序（该库里会按“10月1日为起点”转换）

# 特征工程（Feature engineering）
    -将英寸（inches）转换为米（meters），因为所有模型要求雪深和海拔单位为米。
    -将温度转换为摄氏度（°C）。
    -获取 雪况类别（snow class），因为 Sturm 模型和机器学习模型需要该信息。
'''
# raster路径
raster_path = root_dir / "SnowClass_NA_300m_10.0arcsec_2021_v01.0.nc"
print(raster_path)
raster = rioxarray.open_rasterio(raster_path)

clean_data=(
    df
    .reset_index()
    .filter(items=["datetime", "geometry", "MIN AIR TEMP", "MAX AIR TEMP", "AVG AIR TEMP", "SNOWDEPTH", "SWE"])
    .assign(
        Elevation_m=lambda x: ConvertData.inches_to_metric(x.geometry.map(lambda x: x.coords[0][2]), unit="meters"),
        Latitude=lambda x: x.geometry.map(lambda x: x.coords[0][1]),
        Longitude=lambda x: x.geometry.map(lambda x: x.coords[0][0]),
        TAVG_degC=lambda x: ConvertData.fah_to_cel(x["AVG AIR TEMP"]),
        TMIN_degC=lambda x: ConvertData.fah_to_cel(x["MIN AIR TEMP"]),
        TMAX_degC=lambda x: ConvertData.fah_to_cel(x["MAX AIR TEMP"]),
        Month=lambda x: x.datetime.dt.month,
        Snow_Depth_cm=lambda x: ConvertData.inches_to_metric(x["SNOWDEPTH"], unit="cm"),
        Snow_Class=lambda x: ConvertData.get_snow_class(lons=x.Longitude, lats=x.Latitude, raster=raster),
        Snow_Depth_m=lambda x: ConvertData.inches_to_metric(x["SNOWDEPTH"], unit="meters"),
        SWE_cm=lambda x: ConvertData.inches_to_metric(x["SWE"], unit="cm"),
        Snow_Density_gcm=lambda x: x.SWE_cm / x.Snow_Depth_cm

    )
    .drop(columns=[
        "geometry", "AVG AIR TEMP",
        "MIN AIR TEMP", "MAX AIR TEMP",
        "SNOWDEPTH", "Snow_Depth_cm",
        "Latitude", "Longitude"
    ])
)
print(clean_data.head(3))
clean_data.info()


from ..snowai.swe.statistical_models import StatisticalModels
from ..snowai.swe.machine_learning_model import MachineLearningSWE  # 按你的实际模块路径调整
# 如果 MachineLearningSWE 不在这个路径，请改成你项目真实 import 路径

preds_swe = (
    clean_data
    .assign(
        # 1) Default: 已有雪深 + 已有雪密度
        SWE_Default_cm=lambda x: StatisticalModels(
            algorithm="default",
            return_type="pandas"
        ).predict(
            data=x,
            depth_col="Snow_Depth_m",          # 雪深(m)
            density_col="Snow_Density_gcm"     # 雪密度(g/cm^3)
        ),

        # 2) Sturm: Sturm 密度模型 -> SWE
        SWE_Sturm_cm=lambda x: StatisticalModels(
            algorithm="sturm",
            return_type="pandas"
        ).predict(
            data=x,
            snow_depth="Snow_Depth_m",
            DOY="datetime",
            snow_class="Snow_Class"
        ),

        # 3) Jonas: Jonas 密度模型 -> SWE
        SWE_Jonas_cm=lambda x: StatisticalModels(
            algorithm="jonas",
            return_type="pandas"
        ).predict(
            data=x,
            snow_depth="Snow_Depth_m",
            month="Month",
            elevation="Elevation_m"
        ),

        # 4) Pistochi: Pistochi 密度模型 -> SWE
        SWE_Pistochi_cm=lambda x: StatisticalModels(
            algorithm="pistochi",
            return_type="pandas"
        ).predict(
            data=x,
            snow_depth="Snow_Depth_m",
            DOY="datetime"
        ),

        # 5) Hill: 需要你在 clean_data 里准备 pptwt / TD 两列（否则注释掉）
        # SWE_Hill_cm=lambda x: StatisticalModels(
        #     algorithm="hill",
        #     return_type="pandas"
        # ).predict(
        #     data=x,
        #     pptwt="pptwt_mm",        # 你自己的列名：冬季降水(mm)
        #     TD="TD_degC",            # 你自己的列名：温差(°C)
        #     DOY="datetime",
        #     snow_depth="Snow_Depth_m",
        #     DOY_=180
        # ),

        # 6) ML SWE: 先 ML 预测密度 -> SWE = density * depth * 100
        SWE_ML_cm=lambda x: MachineLearningSWE(
            return_type="pandas"
        ).predict(
            data=x,
            snow_depth="Snow_Depth_m",
            DOY="datetime",
            snow_class="Snow_Class",
            elevation="Elevation_m",
            tavg="TAVG_degC",
            tmin="TMIN_degC",
            tmax="TMAX_degC"
        ),
    )
)

# 查看结果
print(preds_swe.head(3))

# 表1
table_swe = preds_swe[[
    "datetime",
    "SWE_Default_cm",
    "SWE_Sturm_cm",
    "SWE_Jonas_cm",
    "SWE_Pistochi_cm",
    "SWE_ML_cm"
]]

# 表2
table_ml = clean_data.assign(
    SWE_ML_cm=lambda x: MachineLearningSWE(return_type="pandas").predict(
        data=x,
        snow_depth="Snow_Depth_m",
        DOY="datetime",
        snow_class="Snow_Class",
        elevation="Elevation_m",
        tavg="TAVG_degC",
        tmin="TMIN_degC",
        tmax="TMAX_degC"
    )
)

# 保存
save_root = root_dir / "output"
table_swe.to_csv(f"{save_root}/All_SWE_Methods.csv", index=False)
table_ml.to_csv(f"{save_root}/CleanData_with_ML_SWE.csv", index=False)
# 在 get_snow_class 里打开了一个大栅格，如果没有显式关闭，有些环境会在退出时清理触发
try:
    raster.close()
except Exception:
    pass
