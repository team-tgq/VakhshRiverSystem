# SWE 雪水当量评估与洪涝灾害风险等级评估模块技术说明

本文档面向项目汇报、模块交付与后续维护，系统整理当前分支中两个业务模块的功能定位、输入数据格式、算法实现流程、输出成果与界面使用方式。

适用模块：

- SWE 雪水当量评估模块
- 洪涝灾害风险等级评估模块

适用系统入口：

```powershell
E:\anaconda\envs\VakhshRiverSystem\python.exe main.py
```

相关代码入口：

| 模块 | GUI 插件 | 算法入口 | 核心说明 |
|---|---|---|---|
| SWE 雪水当量评估 | `plugins/swe_plugin/swe_widget.py` | `algorithms/swe/swe_assessment.py`、`algorithms/swe/daily_ml_pipeline.py` | 日尺度 GFS 驱动、VIIRS 雪盖约束、机器学习估算 SWE，并支持 DEM 温度订正 |
| 洪涝灾害风险等级评估 | `plugins/flood_plugin/flood_widget.py` | `algorithms/flood/risk_assessment_6factors_entropy.py`、`algorithms/flood/risk_assessment_6factors.py` | 六因子综合风险评估、日尺度动态输入、土地利用解释性增强、结果分位数五分类 |

## 1. 总体数据组织

系统采用“静态地理数据 + 日尺度动态数据 + 模型或综合评价算法”的组织方式。

静态地理数据主要包括研究区边界、DEM、土地利用、河网等。这类数据在短期内不随日期变化，因此模块会尽量自动读取，不要求用户每次重复输入。

动态数据主要包括逐日气象或遥感数据，例如降水、土壤湿度、气温、雪盖等。这类数据随日期变化，是两个模块进行日尺度评估的关键。

## 2. SWE 雪水当量评估模块

### 2.1 模块目标

SWE 模块用于估算研究区每日雪水当量分布，输出空间栅格、流域平均 SWE、融雪量、质量标识和日序列表。

该模块重点解决的问题是：

- 将近实时气象驱动转换为研究区日尺度 SWE 空间分布。
- 使用 VIIRS 雪盖产品约束是否存在积雪，减少无雪区误判。
- 使用 DEM 对 GFS 气温进行高程订正，提高山区温度驱动的合理性。
- 首次运行时显示训练进度，已有模型优先复用，避免每次重复训练。

### 2.2 运行入口与界面功能

GUI 入口为 `plugins/swe_plugin/swe_widget.py`。

界面主要功能：

| 功能 | 说明 |
|---|---|
| 更新最新 SWE | 优先更新当前业务日；如果当天驱动或雪盖数据不完整，自动回退到最近可用业务日 |
| 回算最近 N 天 | 对最近指定天数进行批量回算 |
| 加载已有结果 | 不重新计算，直接读取 `manifest.json` 中已有成果 |
| 重新训练模型 | 勾选后跳过已有模型，重新构建 SWE 机器学习模型 |
| 运行日志 | 展示模型复用、首次训练、数据获取、推理输出等进度 |
| SWE 分布图 | 显示当前选中业务日的 SWE 栅格，可视化长边按高精度设置展示 |

SWE 插件不是直接在 GUI 线程里运行重计算，而是通过 `algorithms.swe.swe_assessment` 启动独立计算子进程。这样可以避免长时间下载、训练或推理阻塞桌面界面，并把进度事件实时返回到日志框。

### 2.3 输入数据

#### 2.3.1 静态输入数据

| 数据 | 默认路径或来源 | 格式 | 内容说明 | 用途 |
|---|---|---|---|---|
| 研究区边界 | `algorithms/swe/study_area.shp` | Shapefile | Vakhsh 流域或目标研究区边界 | 裁剪输出、计算流域统计、限定有效区域 |
| DEM | 环境变量 `SWE_DEM_PATH` 或 `SWE_TEMPERATURE_DEM_PATH`；也会自动查找候选路径 | GeoTIFF | 真实地表高程 | 用于气温 DEM 订正 |
| GFS 模式地形 | NOAA GFS 下载的 `HGT/orography` | GRIB2/转换后数组 | GFS 模式自身使用的地形高度 | 作为 GFS 2 米气温的参考高程 |
| 派生地形因子 | 运行时生成或缓存 | 数组/缓存文件 | 高程、坡度、坡向、分区等 | 作为机器学习模型特征 |

DEM 自动查找候选路径包括：

- `algorithms/swe/dem_clip.tif`
- `algorithms/swe/dem.tif`
- `algorithms/flood/data/processed/dem_clip.tif`
- `algorithms/flood/data/processed/dem.tif`
- `algorithms/flood/dem_clip.tif`
- `algorithms/flood/dem.tif`

如果没有检测到 DEM，SWE 模块仍可运行，但 DEM 温度订正不会启用，界面摘要会显示 DEM 状态为未启用或缺失。

#### 2.3.2 近实时动态输入数据

| 数据 | 数据源 | 时间尺度 | 格式 | 内容说明 | 用途 |
|---|---|---|---|---|---|
| 2 米气温 | NOAA GFS 0.25° NOMADS | 日尺度，由多个预报时效聚合 | GRIB2 | 研究区每日气温，包含日均、日最低、日最高 | 计算温度特征、正积温、雨雪分相 |
| 总降水 | NOAA GFS 0.25° NOMADS | 日尺度 | GRIB2 | 累积降水，模块内部转换为逐日降水量 | 计算降水和固态降水输入 |
| WEASD / SDWE | NOAA GFS 0.25° NOMADS | 日尺度 | GRIB2 | GFS 雪水当量或雪状态字段 | 冷启动时辅助生成初始 SWE 状态 |
| VIIRS 近实时雪盖 | NASA LANCE / MODAPS `VNP10_NRT` | 日尺度 | HDF/遥感产品 | 日雪盖分数 | 约束当天是否有雪 |
| VIIRS 历史标准雪盖 | NASA Earthdata / NSIDC `VNP10C1` | 日尺度 | HDF5 | 历史标准 VIIRS 雪盖产品 | 训练和回算时作为补充雪盖约束 |

VIIRS 近实时产品可能需要配置令牌。模块会读取下列环境变量：

- `VIIRS_NRT_TOKEN`
- `EARTHDATA_TOKEN`

历史标准 VIIRS `VNP10C1` 读取依赖 `h5py`。如果运行环境无法导入 `h5py`，历史标准 VIIRS 数据无法读取，但近实时路径和已有结果加载不一定受影响。

#### 2.3.3 训练标签数据

SWE 模型使用 ERA5-Land 历史数据构建伪标签训练集。

| 数据 | 数据源 | 时间窗口 | 变量 | 用途 |
|---|---|---|---|---|
| ERA5-Land | Copernicus CDS | `2024-02-01` 至 `2025-01-31` | `snow_depth`、`snow_depth_water_equivalent`、`2m_temperature` | 构建 SWE 训练标签和历史气象特征 |

训练数据不是每次运行都重新下载和训练。模块会优先复用当前分支已有模型；如果当前分支没有可用模型，还会扫描本机其它工作树中的可用模型并同步复用。只有在模型缺失、签名不匹配或用户勾选“重新训练模型”时才进入首次训练。

### 2.4 输入数据格式要求

#### 2.4.1 栅格数据格式

SWE 模块内部输出和主要空间输入均按 GeoTIFF 组织。

推荐栅格要求：

| 项目 | 要求 |
|---|---|
| 坐标系 | 优先使用经纬度或可被 `rasterio` 正确识别的 CRS |
| 数据类型 | 连续变量使用 `float32`，质量标识使用 `int16` |
| NoData | 连续变量默认使用 `-9999.0`，质量标识默认使用 `-1` |
| 空间范围 | 覆盖研究区边界 |
| 空间对齐 | 模块会对 GFS 与 DEM 等数据进行必要插值或重采样 |

#### 2.4.2 业务日期

SWE 模块采用业务日期概念，时区为 `Asia/Dushanbe`。更新最新 SWE 时，系统会优先尝试当天业务日；如果当天 GFS 或 VIIRS 约束尚未完整到位，会自动回退到前一天继续更新。

### 2.5 算法实现

SWE 模块采用“近实时气象驱动 + DEM 温度订正 + VIIRS 雪盖约束 + 机器学习回归 + 递归状态更新”的流程。

整体流程如下：

```text
研究区边界 / DEM
        ↓
下载或读取 GFS 气温、降水、模式地形
        ↓
DEM 与 GFS 模式地形对比，进行气温高程订正
        ↓
根据订正后温度进行雨雪分相，得到固态降水
        ↓
读取前一日 SWE 状态与近几日滚动统计
        ↓
读取 VIIRS 日雪盖分数作为积雪约束
        ↓
构建特征表，输入 HistGradientBoostingRegressor
        ↓
输出当日 SWE、融雪量、QA、流域平均值和诊断文件
```

#### 2.5.1 DEM 温度订正

GFS 的 2 米气温是贴着 GFS 模式地形表面的 2 米气温，不是海平面气温。因此模块不直接用 DEM 绝对高程重新修正，而是比较真实 DEM 与 GFS 模式地形之间的高程差。

当前默认高程直减率为：

```text
6.5 °C / km
```

公式为：

```text
DEM高程差 = 真实DEM高程 - GFS参考高程
温度订正量 = -6.5 × DEM高程差 / 1000
订正后温度 = GFS原始温度 + 温度订正量
```

解释：

- 如果真实 DEM 高于 GFS 模式地形，则温度订正量为负，订正后温度降低。
- 如果真实 DEM 低于 GFS 模式地形，则温度订正量为正，订正后温度升高。
- 订正只针对“真实地形与模式地形之间的差异”，避免对 GFS 已经包含的模式地形影响重复修正。

直减率可通过环境变量覆盖：

```text
SWE_DEM_LAPSE_RATE_C_PER_KM
```

#### 2.5.2 雨雪分相与固态降水

模块使用订正后的日最低温、日最高温判断降水中固态降水比例，形成：

- 原始固态降水 `solid_precip_raw_mm`
- DEM 温度订正后的固态降水 `solid_precip_mm`
- 固态降水订正差值 `solid_precip_correction_mm`

这一步很关键，因为 SWE 的新增量主要来自固态降水，而固态降水比例对气温非常敏感。

#### 2.5.3 机器学习模型

当前模型版本：

```text
daily_swe_gbr_viirs_v4
```

模型类型：

```text
HistGradientBoostingRegressor
```

主要训练参数：

| 参数 | 值 |
|---|---|
| `loss` | `squared_error` |
| `learning_rate` | `0.05` |
| `max_depth` | `8` |
| `max_iter` | `350` |
| `min_samples_leaf` | `80` |
| `random_state` | `42` |

主要特征包括：

| 特征类型 | 字段示例 | 含义 |
|---|---|---|
| 空间位置 | `latitude`、`longitude` | 像元空间位置 |
| 地形 | `elevation_m`、`slope_deg`、`aspect_deg`、`zone_id` | 高程、坡度、坡向和分区 |
| 时间 | `doy` | 年内日序 |
| 气温 | `temp_mean_c`、`temp_min_c`、`temp_max_c`、`temp_range_c` | DEM 订正后的日温度特征 |
| 正积温 | `positive_degree_c`、`positive_degree_3d`、`positive_degree_7d` | 反映融雪热量条件 |
| 降水 | `precipitation_mm`、`solid_precip_mm` | 总降水与固态降水 |
| 固态降水滚动统计 | `solid_precip_3d`、`solid_precip_7d`、`solid_precip_14d`、`solid_precip_30d` | 反映近期积雪补给 |
| 温度滚动统计 | `temp_mean_3d`、`temp_mean_7d`、`temp_mean_14d`、`temp_mean_30d` | 反映近期消融背景 |
| 前期 SWE 状态 | `prev_swe_mm`、`prev_swe_3d_mean`、`prev_swe_7d_mean` | 递归状态输入 |
| 雪盖约束 | `snow_cover_fraction`、`viirs_available`、`snow_cover_persist_3d`、`snow_cover_persist_7d` | VIIRS 雪盖及其持续性 |

#### 2.5.4 递归状态更新

模型预测得到当日像元 SWE 后，模块会结合前一日状态与固态降水进行物理约束：

```text
基础SWE = 前一日SWE + 当日固态降水
最终SWE = min(模型预测SWE, 基础SWE)
最终SWE = max(最终SWE, 0)
融雪量 = 基础SWE - 最终SWE
```

如果 VIIRS 显示某像元雪盖比例很低，模块会进一步压低该像元 SWE，减少无雪区出现高 SWE 的可能。

#### 2.5.5 QA 标识

SWE 输出包含质量标识栅格，用于说明结果是否存在约束或冷启动情况。

| QA 标识 | 数值 | 含义 |
|---|---:|---|
| `QA_VIIRS_MISSING` | `1` | VIIRS 雪盖缺失 |
| `QA_VIIRS_CONSTRAINED` | `2` | VIIRS 雪盖对 SWE 做了约束 |
| `QA_COLD_START_EXTERNAL` | `4` | 使用外部或冷启动 SWE 状态 |

### 2.6 输出数据

SWE 模块主要输出目录：

```text
algorithms/swe/output/daily_ml/
```

#### 2.6.1 空间栅格输出

| 文件 | 格式 | 内容 |
|---|---|---|
| `rasters/SWE_mm_YYYYMMDD.tif` | GeoTIFF / float32 | 当日 SWE 空间分布，单位 mm |
| `rasters/Snowmelt_mm_day_YYYYMMDD.tif` | GeoTIFF / float32 | 当日融雪量，单位 mm/day |
| `rasters/SWE_QA_YYYYMMDD.tif` | GeoTIFF / int16 | 当日 SWE 质量标识 |

#### 2.6.2 温度 DEM 订正诊断输出

当 DEM 温度订正启用时，模块会输出诊断栅格：

| 文件 | 内容 |
|---|---|
| `rasters/diagnostics/TempMeanRaw_C_YYYYMMDD.tif` | 插值到研究区网格后的原始 GFS 日均温 |
| `rasters/diagnostics/TempMeanCorrected_C_YYYYMMDD.tif` | DEM 订正后的日均温 |
| `rasters/diagnostics/TempCorrection_C_YYYYMMDD.tif` | 温度订正量 |
| `rasters/diagnostics/DEM_minus_GFS_Elevation_m_YYYYMMDD.tif` | 真实 DEM 与 GFS 参考高程差 |
| `rasters/diagnostics/SolidPrecipCorrection_mm_YYYYMMDD.tif` | 因温度订正导致的固态降水差异 |
| `rasters/diagnostics/TempMeanComparison_C_YYYYMMDD.png` | 温度订正前后对比图 |

这些文件可用于解释“为什么山区温度订正后 SWE 结果更合理”。

#### 2.6.3 序列、模型与缓存输出

| 文件 | 内容 |
|---|---|
| `manifest.json` | 每个业务日的结果清单、路径、数据源状态、诊断信息 |
| `series/daily_basin_series.csv` | 流域平均 SWE、融雪量、驱动周期、VIIRS 状态等日序列 |
| `models/daily_swe_gbr.joblib` | 已训练或复用的 SWE 模型 |
| `models/training_metrics.json` | 训练指标、样本量、特征列表、训练窗口等 |
| `cache/` | GFS、VIIRS、forcing、state 等运行缓存 |
| `../routing/data/SWE_daily_series.csv` | 供下游汇流或水文模块使用的 SWE 日序列 |

### 2.7 汇报解释口径

SWE 模块可概括为：

> 本模块以 GFS 近实时气象数据为日尺度驱动，结合 VIIRS 日雪盖约束和历史 ERA5-Land 伪标签训练的机器学习模型，实现研究区 SWE 空间分布估算。针对山区高程起伏明显的问题，模块进一步引入真实 DEM 与 GFS 模式地形之间的高程差，对 2 米气温进行直减率订正，从而改善雨雪分相和固态降水估计，使 SWE 结果更符合山区地形控制特征。

## 3. 洪涝灾害风险等级评估模块

### 3.1 模块目标

洪涝模块用于输出研究区洪涝灾害风险指数和五级风险等级图，并提供土地利用类型与风险结果之间的解释性统计。

该模块重点解决的问题是：

- 将原来的月尺度动态输入升级为逐日气象输入。
- 静态地理数据自动读取，减少重复输入。
- 如果目标日期实时数据暂不可用，日志中说明原因，并自动使用最近可用日期继续运行。
- 强化地形地貌与土地利用类型对洪涝风险的解释性。
- 在连续风险指数计算完成后，再基于当前结果进行五分类，避免固定阈值导致类别过少。

### 3.2 运行入口与界面功能

GUI 入口为 `plugins/flood_plugin/flood_widget.py`。

默认算法入口为熵权组合版本：

```text
algorithms/flood/risk_assessment_6factors_entropy.py
```

备用或非熵权版本：

```text
algorithms/flood/risk_assessment_6factors.py
```

界面主要功能：

| 功能 | 说明 |
|---|---|
| 目标日期 | 选择需要评估的日期，系统按日期匹配逐日降雨与土壤湿度 |
| 运行风险评估 | 自动读取静态数据，匹配或获取动态数据，并生成风险结果 |
| 加载已有结果 | 读取上一次生成的风险图、土地利用底图、统计表和交互地图 |
| 风险栅格页签 | 显示风险结果栅格 |
| 土地利用页签 | 显示土地利用类型底图 |
| 类型统计页签 | 显示土地利用类型与风险结果的统计表 |
| 交互地图页签 | 显示风险图层、土地利用图层和研究区边界，可进行图层切换 |
| 运行日志 | 显示实际使用日期、数据路径、输出路径和解释性摘要 |

### 3.3 输入数据

#### 3.3.1 静态地理输入

| 数据 | 推荐路径 | 格式 | 内容说明 | 用途 |
|---|---|---|---|---|
| 研究区边界 | `algorithms/flood/study_area.shp` | Shapefile | 研究区边界 | 裁剪、地图边界、静态数据准备 |
| DEM 裁剪栅格 | `algorithms/flood/data/processed/dem_clip.tif` | GeoTIFF | 研究区 DEM | 计算低海拔敏感性和坡度 |
| 原始 DEM | `algorithms/flood/dem.tif` 或 `algorithms/flood/data/raw/dem.tif` | GeoTIFF | 原始高程数据 | 当 `dem_clip.tif` 缺失时自动裁剪 |
| 土地利用栅格 | `algorithms/flood/data/processed/landcover_demgrid.tif` | GeoTIFF | ESA WorldCover 分类，已对齐 DEM 网格 | 计算土地利用易涝敏感性与解释统计 |
| 河网矢量 | `algorithms/flood/data/raw/hydrorivers.gpkg` | GeoPackage | HydroRIVERS 河网 | 计算距河道远近敏感性 |

静态数据解析由 `algorithms/flood/input_resolver.py` 负责。

如果缺少 `landcover_demgrid.tif` 或 `hydrorivers.gpkg`，但已有 DEM 和研究区边界，系统会尝试自动准备：

- 从原始 DEM 生成 `dem_clip.tif`
- 下载并重采样 ESA WorldCover，生成 `landcover_demgrid.tif`
- 下载并裁剪 HydroRIVERS，生成 `hydrorivers.gpkg`

#### 3.3.2 动态逐日输入

洪涝模块当前动态数据包括两类：

| 数据 | 文件命名 | 格式 | 单位或含义 | 用途 |
|---|---|---|---|---|
| 逐日降雨 | `rain_mm_demgrid_YYYY-MM-DD.tif` | GeoTIFF | mm/day，已对齐 DEM 网格 | 反映当日致灾降水强度 |
| 表层土壤湿度 | `soil_moist_demgrid_YYYY-MM-DD.tif` | GeoTIFF | 0-0.1 m 表层土壤湿度 | 反映下垫面前期湿润程度和产流敏感性 |

动态文件放置目录：

```text
algorithms/flood/data/processed/
```

也可以放在其子目录中，例如：

```text
algorithms/flood/data/processed/daily/
```

输入解析器会递归扫描 `processed` 目录及子目录，并按文件名中的日期自动匹配。文件名可以使用 `YYYY-MM-DD`、`YYYYMMDD` 或 `YYYY_MM_DD` 日期形式，但推荐统一使用：

```text
rain_mm_demgrid_2026-04-27.tif
soil_moist_demgrid_2026-04-27.tif
```

#### 3.3.3 近实时动态数据自动获取

如果所选日期没有完整的本地逐日降雨和土壤湿度，模块会尝试从 NOAA GFS 近实时数据源自动获取。

当前近实时数据源：

| 数据 | NOAA GFS 变量 | 层次 | 说明 |
|---|---|---|---|
| 逐日降雨 | `APCP` | `surface` | 24 小时累积降水 |
| 表层土壤湿度 | `SOILW` | `0-0.1 m below ground` | 表层土壤湿度 |

自动处理流程：

```text
下载 GFS GRIB2
        ↓
提取 APCP / SOILW 波段
        ↓
转换为临时 GeoTIFF
        ↓
重采样并对齐到 DEM 网格
        ↓
输出 rain_mm_demgrid_YYYY-MM-DD.tif 和 soil_moist_demgrid_YYYY-MM-DD.tif
```

如果目标日期数据尚未同步或超出在线保留范围，模块不会强行生成未来或无数据日期结果。当前逻辑会在日志中说明目标日期不可用，并自动选择本地已有的最近可用日期继续运行；如果本地也没有任何可用逐日数据，则给出友好错误说明。

#### 3.3.4 ERA5-Land 历史日尺度预处理

模块仍保留 ERA5-Land 历史数据预处理能力，可将历史 NetCDF 数据转换为日尺度 GeoTIFF 并对齐 DEM 网格。

ERA5-Land 变量：

| 变量 | 含义 | 处理方式 |
|---|---|---|
| `total_precipitation` / `tp` | 总降水 | 按日累积并转换为 mm |
| `volumetric_soil_water_layer_1` / `swvl1` | 第一层土壤体积含水量 | 按日平均 |

处理后输出：

```text
rain_mm_demgrid_YYYY-MM-DD.tif
soil_moist_demgrid_YYYY-MM-DD.tif
```

### 3.4 洪涝算法实现

洪涝风险评估采用六因子综合指数法，并在默认版本中加入熵权法对主观权重进行轻度修正。

#### 3.4.1 六个风险因子

| 因子键名 | 中文含义 | 数据来源 | 计算方式 | 风险含义 |
|---|---|---|---|---|
| `rain` | 逐日降水 | 日降雨栅格 | Min-Max 归一化 | 降雨越大，洪涝触发风险越高 |
| `soil_moist` | 表层土壤湿度 | 日土壤湿度栅格 | Min-Max 归一化 | 土壤越湿，入渗余量越小，产流风险越高 |
| `elev_low` | 低海拔敏感性 | DEM | 对高程裁剪后归一化，再取 `1 - normalized elevation` | 低洼区更容易汇水和积水 |
| `slope_low` | 低坡度敏感性 | DEM 派生坡度 | 坡度裁剪到 0-10° 后归一化，再取 `1 - normalized slope` | 坡度越缓，排水越慢，积水风险越高 |
| `landuse_suscept` | 土地利用易涝敏感性 | ESA WorldCover | 按地类编码映射敏感性值 | 不同土地利用类型对洪涝敏感性不同 |
| `river_near` | 近河道敏感性 | HydroRIVERS | 河网栅格化后计算距离，使用指数衰减 `exp(-distance / 1500m)` | 越靠近河道，受河流漫溢影响越明显 |

#### 3.4.2 土地利用敏感性映射

土地利用因子从原先偏弱的“不透水因子”升级为 `landuse_suscept`，即“土地利用易涝敏感性”。该因子既参与最终风险计算，也作为解释性输出的主线。

默认 ESA WorldCover 映射如下：

| 编码 | 类型 | 敏感性 | 是否参与解释性排名 | 解释 |
|---:|---|---:|---|---|
| 10 | Tree cover / 林地 | 0.35 | 是 | 植被覆盖较好，滞蓄和入渗能力较强 |
| 20 | Shrubland / 灌丛 | 0.45 | 是 | 较林地稍敏感 |
| 30 | Grassland / 草地 | 0.55 | 是 | 地表覆盖中等，风险中等 |
| 40 | Cropland / 农田 | 0.80 | 是 | 农田多位于较平坦区域，且排水受耕作格局影响 |
| 50 | Built-up / 建成区 | 1.00 | 是 | 不透水面高，径流形成快，风险敏感性最高 |
| 60 | Bare sparse / 裸地或稀疏植被 | 0.50 | 是 | 覆盖弱但不一定低洼，取中等敏感 |
| 70 | Snow ice / 冰雪 | 0.10 | 是 | 在洪涝解释中直接易涝性较低 |
| 80 | Water bodies / 水体 | 0.05 | 否 | 保留统计但不参与解释性排名，避免开阔水体干扰结论 |
| 90 | Wetland / 草本湿地 | 0.85 | 是 | 天然蓄滞洪或长期湿润区域，易出现高风险等级 |
| 95 | Mangroves / 红树林 | 0.75 | 是 | 湿地型生态系统，敏感性较高 |
| 100 | Moss lichen / 苔藓地衣 | 0.40 | 是 | 覆盖类型敏感性偏低 |
| 未识别有限值 | Unknown / 未识别 | 0.50 | 是 | 取中性敏感值，避免直接丢弃有效像元 |

草本湿地高风险占比偏高并不表示“经济损失最高”，而是说明该类型地表具有天然低洼、湿润、蓄滞洪或易淹没特征，因此在灾害发生可能性维度上更容易被判为较高风险。

#### 3.4.3 权重设置

默认主观权重：

| 因子 | 权重 |
|---|---:|
| `rain` | 0.20 |
| `soil_moist` | 0.16 |
| `elev_low` | 0.15 |
| `slope_low` | 0.12 |
| `landuse_suscept` | 0.25 |
| `river_near` | 0.12 |

默认熵权组合参数：

```text
alpha_subjective = 0.85
beta_entropy = 0.15
entropy_sample_size = 5000
```

组合权重计算：

```text
final_weight_i = alpha_subjective × subjective_weight_i
               + (1 - alpha_subjective) × entropy_weight_i
```

随后对最终权重进行归一化，确保权重和为 1。

这样设计的原因是：主观权重明确体现洪涝机理和土地利用解释性，熵权只作为数据分布层面的微调，避免土地利用和地形地貌的重要性被纯数据离散度过度冲淡。

#### 3.4.4 连续风险指数计算

六个因子统一转换到 0-1 风险敏感性空间后，计算综合风险指数：

```text
Risk = Σ(final_weight_i × factor_i)
```

其中：

- `Risk` 越接近 1，表示洪涝风险越高。
- 任一关键因子缺失的像元会被置为 NoData。
- DEM 缺失区域同样不参与计算。

#### 3.4.5 五级风险等级划分

当前分支采用“先计算连续风险指数，再基于当前结果分位数五分类”的方法。

方法标识：

```text
result_quantile_5classes
```

五个等级：

| 等级编码 | 等级名称 | 含义 |
|---:|---|---|
| 1 | 低 | 当前结果中风险相对最低区域 |
| 2 | 较低 | 风险偏低区域 |
| 3 | 中 | 中等风险区域 |
| 4 | 较高 | 风险偏高区域 |
| 5 | 高 | 当前结果中风险最高区域 |

注意：五级分类不是固定按 `0.2/0.4/0.6/0.8` 阈值切分，而是在连续风险指数已经计算完成后，按当前结果的分位数进行五分类。这样可以保证每次结果图都能完整展示五个风险等级，更适合空间解释和汇报展示。

### 3.5 输出数据

洪涝模块主要输出目录：

```text
algorithms/flood/outputs/
```

#### 3.5.1 栅格与地图输出

| 文件 | 格式 | 内容 |
|---|---|---|
| `risk_6factors.tif` | GeoTIFF / float32 | 连续洪涝风险指数，范围约 0-1 |
| `risk_6factors_level.tif` | GeoTIFF / uint8 | 五级风险等级栅格，1-5 分别对应低至高 |
| `flood_risk_map.html` | HTML / Folium | 交互地图，包含风险等级图层、土地利用图层和研究区边界 |

#### 3.5.2 权重与解释性输出

| 文件 | 格式 | 内容 |
|---|---|---|
| `final_weights.txt` | TXT | 主观权重、熵权、最终组合权重、因子含义 |
| `landuse_risk_stats.csv` | CSV / UTF-8 BOM | 各土地利用类型的面积、平均风险、高风险占比和主导风险等级 |
| `landuse_risk_summary.txt` | TXT | 面向汇报的土地利用解释摘要 |

`landuse_risk_stats.csv` 固定字段：

| 字段 | 含义 |
|---|---|
| `landcover_code` | 土地利用编码 |
| `landcover_name` | 土地利用类型名称 |
| `included_in_ranking` | 是否参与解释性排名 |
| `pixel_count` | 有效像元数量 |
| `area_km2` | 面积，单位 km² |
| `mean_risk` | 该土地利用类型内平均连续风险指数 |
| `p90_risk` | 该土地利用类型内风险指数 90 分位值 |
| `high_risk_ratio` | 该类型中“较高 + 高”两档像元占比 |
| `dominant_risk_level` | 该类型内占比最高的风险等级 |

#### 3.5.3 函数返回结果

`run_risk_assessment()` 会返回关键路径和统计信息，供 GUI 日志与展示页签使用。

主要返回字段包括：

| 字段 | 含义 |
|---|---|
| `risk_tif` | 连续风险指数栅格路径 |
| `risk_level_tif` | 五级风险等级栅格路径 |
| `map_html` | 交互地图路径 |
| `weights_txt` | 权重报告路径 |
| `landuse_stats_csv` | 土地利用风险统计表路径 |
| `landuse_summary_txt` | 土地利用解释摘要路径 |
| `landuse_factor_name` | 固定为 `landuse_suscept` |
| `rain_path` | 实际使用的逐日降雨栅格 |
| `soil_path` | 实际使用的逐日土壤湿度栅格 |
| `requested_target_date` | 用户请求日期 |
| `resolved_target_date` | 系统实际使用日期 |
| `dynamic_scale` | 动态数据尺度，例如 `daily` |
| `risk_level_breaks` | 当前结果分位数五分类区间 |
| `risk_level_distribution` | 五级风险等级像元数与占比 |
| `top_high_risk_landuse` | 高风险占比最高的前三类土地利用 |
| `top_mean_risk_landuse` | 平均风险最高的前三类土地利用 |

### 3.6 汇报解释口径

洪涝模块可概括为：

> 本模块在六因子综合风险评估框架下，引入逐日降水和表层土壤湿度作为动态触发因子，以 DEM 派生的低海拔、低坡度和距河道距离刻画地形地貌控制作用，并将土地利用升级为强解释性敏感因子。最终先计算连续风险指数，再基于当前结果分位数划分五级风险等级，同时输出土地利用类型统计表和解释摘要，使土地利用底图与最终风险图之间能够形成可解释对应关系。

## 4. 两个模块的对比

| 对比项 | SWE 雪水当量评估 | 洪涝灾害风险等级评估 |
|---|---|---|
| 主要目标 | 估算每日 SWE 空间分布和流域平均值 | 评估每日洪涝风险等级 |
| 时间尺度 | 日尺度 | 日尺度 |
| 主要动态数据 | GFS 气温、降水、WEASD；VIIRS 雪盖 | 日降雨、表层土壤湿度 |
| 主要静态数据 | 研究区边界、DEM、地形因子 | 研究区边界、DEM、土地利用、河网 |
| 算法类型 | 机器学习回归 + 物理约束 | 六因子综合指数 + 熵权修正 |
| 关键改进 | DEM 温度订正、已有模型优先复用、训练进度可见 | 地形地貌强化、土地利用解释性增强、结果分位数五分类 |
| 主要输出 | SWE 栅格、融雪栅格、QA、日序列 | 连续风险栅格、五级风险栅格、交互地图、土地利用统计 |
| 汇报亮点 | 高程订正改善山区气温和雨雪分相 | 土地利用底图可解释最终风险空间格局 |

## 5. 运行与验收建议

### 5.1 系统启动

在仓库根目录运行：

```powershell
E:\anaconda\envs\VakhshRiverSystem\python.exe main.py
```

### 5.2 SWE 模块验收点

| 验收项 | 预期结果 |
|---|---|
| 点击“加载已有结果” | 能显示已有 SWE 业务日列表和分布图 |
| 点击“更新最新 SWE” | 日志显示模型复用或训练进度，最终生成最新可用业务日结果 |
| 勾选“重新训练模型” | 日志持续显示训练样本准备、模型训练和模型保存过程 |
| DEM 可用 | 摘要中显示 DEM 温度订正已启用，并输出温度诊断栅格 |
| VIIRS 缺失 | 结果仍可输出，但 QA 标识记录 VIIRS 缺失 |

### 5.3 洪涝模块验收点

| 验收项 | 预期结果 |
|---|---|
| 选择目标日期并运行 | 日志显示请求日期、实际使用日期、降雨和土壤湿度路径 |
| 目标日期无实时数据但有历史可用日 | 日志提示目标日期不可用，并自动使用最近可用日期 |
| 静态数据完整 | 自动读取 DEM、土地利用、河网，无需用户重复输入 |
| 静态数据缺失但有原始 DEM | 自动尝试生成 `dem_clip.tif`、`landcover_demgrid.tif` 和 `hydrorivers.gpkg` |
| 运行完成 | 生成连续风险栅格、五级风险栅格、交互地图、权重报告、土地利用统计表和解释摘要 |
| 查看结果页签 | 风险栅格、土地利用、类型统计、交互地图四个页签均可打开 |

