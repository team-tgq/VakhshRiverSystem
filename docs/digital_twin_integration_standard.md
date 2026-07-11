# 瓦赫什流域统一数据与独立模块集成标准

本文档定义当前阶段唯一有效的数据目录、来源标记、插件路径接口、模块输入输出和验收规则。旧的“方案 -> 时段 -> raster/table”结构不再使用。

## 1. 当前范围

当前只要求以下模块使用各自已有的真实、公开或原生验证数据独立运行：

- M01 SegFormer 积雪/水体识别
- M02 雪水当量估算
- M04 淹没区识别
- M05 RAFT 光流测速
- M06 积雪状态识别
- M07 洪涝灾害风险等级评估
- M08 水资源分配
- M09 库区水量估算

本阶段不建立模块间输入输出关系，不统一业务年份，不定义调度工况。M03 洪水演进源码和 Unity build 保留，但不运行、不生成输入输出，也不得写入 `processed`。

## 2. 强制原则

1. `baseline` 只保存固定公共基准数据。
2. `raw` 按数据类型组织，不能按模块编号建立一级目录。
3. `processed` 按模块组织，只保存算法实际运行结果。
4. 正式模块不得从 `algorithms/*/data` 读取；该位置只能作为迁移来源或只读备份。
5. 同一份 raw 数据只保留一个正式副本，通过元数据记录消费模块。
6. 禁止联网下载、mock、随机、占位、空文件和结果改名冒充运行。
7. 缺少输入或算法未实现时，保留缺项并报告，不生成“看起来完整”的成果。
8. `processed` 不保存 raw 副本、权重、缓存、日志、临时帧或说明文档。
9. 所有正式文件必须具有可追溯的 `.meta.json`。
10. 所有插件必须使用统一路径 API，不得硬编码项目绝对路径。

## 3. 目录规范

```text
data/瓦赫什流域孪生数据/
├─ baseline/
│  ├─ DEM.tif
│  ├─ 流域边界.*
│  ├─ 河网.*
│  └─ 水库边界.*
├─ raw/
│  ├─ remote_sensing/
│  │  ├─ optical_rgb/
│  │  │  ├─ segformer_snow/{val,test}/{images,masks}/
│  │  │  └─ segformer_water/{val,test}/{images,masks}/
│  │  ├─ sentinel1_sar/inundation_weak_labeled/
│  │  ├─ sentinel2_multispectral/inundation_hand_labeled/
│  │  └─ gee/snow_state/
│  ├─ meteorology/
│  │  ├─ temperature/
│  │  ├─ precipitation/
│  │  ├─ solid_precipitation/
│  │  └─ daily_forcing/
│  ├─ land_surface/
│  │  ├─ soil_moisture/
│  │  └─ land_cover/
│  ├─ snow_hydrology/
│  │  ├─ previous_swe_state/
│  │  ├─ snow_cover/
│  │  └─ snow_density/
│  ├─ reservoir/
│  │  ├─ parameters/
│  │  ├─ observations/
│  │  └─ hypsometry/
│  ├─ socioeconomic/
│  │  ├─ population/
│  │  ├─ industry/
│  │  ├─ agriculture/
│  │  └─ water_demand/
│  ├─ configuration/water_allocation/
│  │  ├─ supply/
│  │  ├─ demand/
│  │  ├─ crops/
│  │  └─ decision_weights/
│  └─ video/river_velocity/
└─ processed/
   ├─ M01_segformer/
   ├─ M02_swe/
   ├─ M04_inundation/
   ├─ M05_raft/
   ├─ M06_snow_state/
   ├─ M07_flood_risk/
   ├─ M08_water_allocation/
   └─ M09_reservoir_estimation/
```

不存在的 test 数据不得补造，也不要创建空 test 目录。允许某种数据类型当前没有文件，此时不创建空目录冒充已接入。

## 4. 数据真实性和用途

| 类型 | 当前数据 | 允许表述 |
| --- | --- | --- |
| 模块测试数据 | 2026-05 清水河道流速测试视频 | 非现场、非现地实测视频 |
| 公开或官方数据 | 气象、降水、土壤湿度、土地覆盖、GEE 产品 | 按 sidecar 中实际机构或产品来源描述 |
| 模块原生验证集 | SegFormer snow/water、Bolivia S1、India S2 | 模块验证数据，不是瓦赫什业务数据 |
| 用户配置 | M08/M09 参数 CSV | 用户配置或算法参数，不是观测 |
| 来源未完全核实 | M09 原模块 2022 面积记录等 | unverified/module-native validation，不得写成实测 |

除视频外，任何文件出现 `is_field_observation=true` 都是错误。GEE 输出属于遥感产品，不是现场采样。

## 5. 模块契约

### 5.1 M01 SegFormer

- 输入：`raw/remote_sensing/optical_rgb/segformer_{snow,water}/`。
- 环境：`E:/anaconda/envs/segformer/python.exe`。
- 输出：每张输入对应一个 mask 和 overlay；有有效真值时才计算指标。
- 约束：保持现有模型、权重、预处理、推理和界面效果。不得默认使用临时下载的 Sentinel 数据。
- 已知输入问题：water val 中 8 个原标签是零字节文件，不能补造，因此只推理、不计指标。

### 5.2 M06 积雪状态

- 输入：`raw/remote_sensing/gee/snow_state/*.tif`。
- 输出：`snow_type`、确定性 `snow_density` 和统计表。
- 必须记录波段、编码、dtype、CRS、NoData 和日期。
- 当前雪密度来自状态类别映射，不是实测雪密度。

### 5.3 M02 雪水当量

- 输入：统一 daily forcing 和 `baseline/DEM.tif`，字段以代码真实契约为准。
- 输出：SWE、Snowmelt 和统计表。
- 当前算法没有径流计算逻辑，禁止生成 runoff。
- forcing 与 DEM 网格不一致时跳过并报告，禁止静默拉伸。

### 5.4 M09 库区水量估算

- 独立运行，不读取 M03。
- `reservoir_parameters.csv`：`reservoir_id,reservoir_name,parameter,value,unit,source`。
- `reservoir_observations.csv`：`date,reservoir_id,sensor,water_level_m,surface_area_km2,source,quality_status`。
- `reservoir_hypsometry.csv`：`reservoir_id,elevation_m,area_km2,storage_mcm`。
- 输出：`reservoir_storage.csv` 和 `estimation_summary.json`。
- 当前算法不计算独立出库流量，因此不生成 outflow。

### 5.5 M08 水资源分配

- 独立运行，不读取 M09 或任何其他 processed 成果。
- 配置文件必须完整恢复界面，界面修改后可回写 CSV。
- `global_supply_config.csv`：全局时段和初始供水。
- `monthly_inflow.csv`：1-12 月显式入流。
- `demand_parameters.csv`：人口、城镇化、GDP、产业和需水参数。
- `crops.csv`：作物、生育期、面积、产量、价格和 Kc。
- `decision_weights.csv`：偏好、部门和水电权重。
- 缺少月入流时直接失败，禁止使用固定 `500 m3/s` 回退。
- 输出：`allocation_plan.csv` 和 `allocation_summary.csv`。

### 5.6 M07 洪涝风险

- 输入：降水、土壤湿度、土地覆盖、DEM、河网和流域边界。
- 当前独立运行，不读取 M03 或 M04。
- 输出：风险指数、五级风险栅格、权重、土地利用统计和 HTML 地图。

### 5.7 M04 淹没区识别

- 输入：原模块的 Sentinel-1 SAR 弱标注样例和 Sentinel-2 多光谱手工标注样例。
- 必须用 rasterio 检查波段、dtype、CRS 和范围，不能只依赖文件名。
- 两类数据分别保存 mask 和 overlay。
- Bolivia/India 样例不是瓦赫什数据，禁止命名为“瓦赫什实测淹没范围”。

### 5.8 M05 RAFT

- 输入：`raw/video/river_velocity/` 中完整清水河道测试视频。
- 权重：`algorithms/raft/raft-sintel.pth`。
- 必须处理完整视频，不能只处理前两帧或界面指定的少量帧。
- CSV 字段至少包含 `frame_index,timestamp_s,velocity_px_frame,velocity_m_s,valid_pixel_count,confidence,source_video`。
- 没有可信空间标定时 `velocity_m_s` 必须留空。

## 6. 统一路径 API

```python
from app.digital_twin_standard import (
    ensure_raw_source_path,
    module_processed_dir,
    raw_data_dir,
)

input_dir = raw_data_dir("remote_sensing", "sentinel1_sar", "inundation_weak_labeled")
input_file = ensure_raw_source_path(input_dir / "example.tif")
mask_dir = module_processed_dir("M04", "sentinel1_sar_weak", "masks", create=True)
```

`ensure_raw_source_path()` 会拒绝统一 raw 之外的正式输入。`module_processed_dir()` 会把模块成果限制到自己的 processed 目录。

## 7. 元数据

raw sidecar 至少包含：

```text
data_type, dataset_name, source_files, source_origin, consumer_modules,
split, sensor, bands, dtype, crs, date, is_module_native, checksum, created_at
```

processed sidecar 至少包含：

```text
module_code, output_type, source_files, model_weight, threshold_or_config,
shape, dtype, crs, checksum, created_at
```

processed 的 `source_files` 必须指向本次实际使用的 baseline/raw 文件，不能指向另一个模块的 processed。模型权重单独记录到 `model_weight`。

## 8. 数据准备和运行

```bash
# 只生成迁移和删除审计清单
python tools/prepare_real_twin_data.py --audit-only

# 迁移本地原生数据并清理已确认旧结构
python tools/prepare_real_twin_data.py --migrate

# 独立运行八个模块
python tools/prepare_real_twin_data.py --run-modules --modules "M01,M06,M02,M09,M08,M07,M04,M05"
```

脚本禁止联网；迁移时先复制并核对 SHA-256，再清理旧时段/方案副本。临时下载的 Sentinel 数据及其低质量派生成果应按审计清单删除，模型权重、源码、原生验证集、GEE 产品、清水河道测试视频和 baseline 必须保留。

## 9. 三轮验收

```bash
python tools/validate_twin_data.py --stage baseline-raw
python tools/validate_twin_data.py --stage full
```

第一轮检查根目录、baseline、raw 分类、文件非空、sidecar、SHA-256 和来源标记。

第二轮检查八个模块目录、预期成果、真实来源链和 `finish.tag`，并确认没有 M03、方案目录或跨模块 processed 输入。

第三轮实际打开 GeoTIFF、图片、CSV 和视频，检查 CRS、NoData、有效像元、尺寸对应、字段、行数和完整视频帧对关系。不能用“文件存在”代替运行验证。

报告位于：

- `reports/data_architecture_migration_audit.json`
- `reports/independent_module_run_report.json`
- `reports/twin_data_validation_report.json`

## 10. 当前质量限制

1. M02 `2017-07-01` 的 SWE 和 Snowmelt 是全零模型结果，验证器保留常值警告。
2. M02 `forcing_20260525.npz` 为 `9×21`，与 `521×1051` DEM 不一致，当前跳过。
3. M04 两幅 S1 样例淹没比例约 `99.98%`，当前 7 通道模型对 2 波段 SAR 的适配质量较差。
4. M09 的 2022 面积记录缺少底层影像，只能作为原模块验证记录。
5. M05 没有空间标定，只能报告像素位移速度。

这些限制必须在演示、报告和后续数据交付中保留，不能通过改名、填默认值或删除警告掩盖。
