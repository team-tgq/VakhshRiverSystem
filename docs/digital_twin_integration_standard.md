# 瓦赫什流域数字孪生系统集成整改实施说明

本文档依据《瓦赫什流域河流域数字孪生系统集成整改规范（初步）》整理，用于约束本仓库后续模块对接、数据目录、字段单位、页面演示顺序和样例数据组织。

## 1. 整改目标

当前系统整改要解决五类问题：

- 各模块地图坐标、流域范围和时间范围不统一，三维图层叠加错位。
- 文件格式、参数单位和命名不统一，模块输出无法被下游模块直接读取。
- 外地参考数据和瓦赫什流域正式数据混算，导致结果不合理。
- 前后算法因果关系不清晰，无法说明先算哪个、后算哪个。
- 页面顺序不符合流域业务逻辑，演示过程不连贯。

本次已在项目中落地：

- `app/digital_twin_standard.py`：统一 CRS、时间、字段单位、模块链路和成果目录工具。
- `sample_data/瓦赫什流域孪生数据/`：轻量样例数据树，演示 baseline/raw/processed 目录结构。
- 插件加载顺序：按 M01/M06/M02/M03/M09/M08/M07/M04/M05 的业务链路排序。

## 2. 全局数据规范

### 2.1 空间规范

- 统一投影坐标系：`WGS84 UTM 42N`，EPSG 编码为 `EPSG:32642`。
- 所有栅格、矢量和三维展示成果必须裁剪到瓦赫什河流域边界。
- 基础 DEM 只保留一份，放在 `baseline/DEM.tif`，所有模块共用，不允许模块各自携带不同底图。

### 2.2 时间规范

- 计算时间步长：逐日。
- 统一研究时段：`2005-2017`。
- 时间字段统一命名为 `date`。
- 日期格式统一为 `YYYY-MM-DD`。

### 2.3 文件格式规范

- 栅格成果：GeoTIFF，扩展名 `.tif`，必须带 CRS 和仿射变换。
- 水文时序表：CSV，扩展名 `.csv`，时间字段为 `date`。
- 边界和河网：正式数据使用 Shapefile，样例数据同时保留 `.prj` 文件。
- 每个模型成果建议同时生成 `*.meta.json` 旁路元数据，记录 CRS、单位、来源模块和来源文件。
- 每个方案时段目录计算完成后写入 `finish.tag`，供 Qt 或三维展示模块监听。

### 2.4 正式数据与参考数据隔离

- `baseline/`：固定基准数据，只读，不带时间和工况。
- `raw/`：瓦赫什流域原始观测数据，按年份和时段组织。
- `processed/`：各模块成果数据，按方案和时段组织。
- 演示参考数据不得混入正式计算目录。若需要保留参考数据，应单独放在 `reference/` 或在文件名、元数据中标明 `demo_only=true`。

## 3. 标准目录结构

```text
瓦赫什流域孪生数据/
├─ baseline/
│  ├─ 流域边界.shp
│  ├─ 河网.shp
│  ├─ 水库边界.shp
│  └─ DEM.tif
├─ raw/
│  ├─ 200503_融雪期/
│  │  ├─ 200503_哨兵影像.tif
│  │  ├─ 200503_SAR影像.tif
│  │  ├─ 200503_逐日气象.csv
│  │  ├─ 200503_水库参数.csv
│  │  └─ 200503_河道视频.mp4
│  └─ 201707_汛期/
│     ├─ 201707_哨兵影像.tif
│     ├─ 201707_逐日气象.csv
│     └─ 201707_水库参数.csv
└─ processed/
   ├─ scheme01_常规调度工况/
   │  ├─ 200503_融雪模拟/
   │  │  ├─ raster/
   │  │  ├─ table/
   │  │  └─ finish.tag
   │  └─ 201707_汛期模拟/
   │     ├─ raster/
   │     ├─ table/
   │     └─ finish.tag
   └─ scheme02_优化分水工况/
      └─ 200503_融雪模拟/
         ├─ raster/
         ├─ table/
         └─ finish.tag
```

## 4. 字段与单位

| 指标名称 | 字段名 | 统一单位 |
| --- | --- | --- |
| 积雪深度 | `snow_depth` | `m` |
| 积雪覆盖率 | `snow_cover` | `0-1` |
| 雪密度 | `snow_density` | `g/cm3` |
| 雪水当量 | `swe` | `mm` |
| 径流深度 | `runoff` | `mm` |
| 洪水水深 | `flood_depth` | `m` |
| 河道流量 | `discharge` | `m3/s` |
| 库容 | `storage` | `万m3` |
| 下泄流量 | `outflow` | `m3/s` |

## 5. 模块调用关系

### 5.1 主线链路

```text
baseline/DEM.tif + baseline/流域边界.shp
        ↓
M01 积雪水体识别
        ↓
M06 积雪状态分类
        ↓
M02 雪水当量计算
        ↓
M03 洪水演进汇流
        ├─→ M09 水库库容计算 → M08 水资源分配
        └─→ M07 洪涝风险评估
```

### 5.2 校核支路

```text
M04 SAR卫星淹没提取 → 实测淹没范围 → 校核 M07
M05 视频流速监测     → 实测流速数据 → 校准 M03
```

### 5.3 标准输入输出

| 编号 | 模块 | 输入 | 输出 | 下游 |
| --- | --- | --- | --- | --- |
| M01 | 积雪水体识别 | DEM、流域边界、哨兵影像 | `M01_snow_depth_m.tif`、`M01_snow_cover.tif`、积雪面积统计表 | M06、M02 |
| M06 | 积雪状态分类 | M01 积雪成果、DEM | `M06_snow_type.tif`、`M06_snow_density_gcm3.tif` | M02 |
| M02 | 雪水当量计算 | M01、M06、逐日气象 | `M02_swe_mm.tif`、`M02_runoff_mm.tif` | M03 |
| M03 | 洪水演进汇流 | `M02_runoff_mm.tif` | `M03_discharge.csv`、`M03_flood_depth_m.tif`、`M03_inundation.tif` | M07、M09 |
| M09 | 水库库容计算 | `M03_discharge.csv` | `M09_storage.csv`、`M09_outflow.csv` | M08 |
| M08 | 水资源分配 | M09 库容、下泄流量 | `M08_分水方案统计表.csv` | 无 |
| M07 | 洪涝风险评估 | M03 洪水成果、M04 校核成果 | `M07_洪涝风险分区图.tif` | 无 |
| M04 | SAR卫星淹没提取 | SAR 影像 | `实测_淹没范围.tif`、淹没面积统计报表 | 校核 M07 |
| M05 | 视频流速监测 | 河道视频 | `实测_流速数据.csv` | 校准 M03 |

## 6. 页面板块与插件顺序

主程序当前按以下业务顺序加载插件：

1. SegFormer 专题识别（M01）
2. 积雪状态识别（M06）
3. 雪水当量估算（M02）
4. 洪水演进与汇流模拟（M03）
5. 库区水量估算（M09）
6. 水资源分配（M08）
7. 洪涝风险评估（M07）
8. 淹没区监测（M04 校核）
9. RAFT 光流测速（M05 校核）

展示逻辑应按“基础数据标准化 -> 遥感解译 -> SWE 与融雪径流 -> 洪水汇流 -> 水库调度与风险评估 -> 监测对比 -> 三维成果展示”展开。

## 7. 模块负责人对接要求

- 输出文件必须写入 `processed/{scheme}_{工况}/{period}_{模拟时段}/raster` 或 `table`。
- 不允许模块直接读取其他模块的临时输出目录；只能读取 `processed/` 中的正式成果。
- 输出字段必须使用本文档统一字段名和单位。
- 栅格必须使用 `EPSG:32642`，并裁剪到 `baseline/流域边界.shp`。
- 每个输出文件建议调用 `app.digital_twin_standard.write_metadata_sidecar()` 写入旁路元数据。
- 每个方案时段完成后调用 `app.digital_twin_standard.write_finish_tag()` 写入完成标记。
- 模块插件应优先调用 `app.digital_twin_standard.module_output_path()`、`write_standard_csv()` 和 `mark_module_complete()`，避免各模块自行拼接目录导致成果无法互通。

标准成果写入示例：

```python
from app.digital_twin_standard import (
    infer_run_context_from_path,
    mark_module_complete,
    module_output_path,
    period_to_date,
    write_metadata_sidecar,
    write_standard_csv,
)

context = infer_run_context_from_path(input_file)
output_csv = module_output_path("M05", context=context)
write_standard_csv(
    output_csv,
    fieldnames=["date", "period", "module_code", "velocity_m_s"],
    rows=[{"date": period_to_date(context.period), "period": context.period, "module_code": "M05", "velocity_m_s": 1.23}],
)
write_metadata_sidecar(output_csv, module_code="M05", field="velocity", source_files=[input_file])
mark_module_complete(context, "M05")
```

当前已完成标准成果自动导出的插件：

- `M04 SAR/遥感淹没区监测`：输入 GeoTIFF 时自动写入 `raster/{period}_实测_淹没范围.tif` 和 `table/{period}_淹没面积统计报表.xlsx`，并写入旁路元数据与 `finish.tag`。普通 png/jpg 因缺少 CRS，只保留界面预览结果，不进入正式 GIS 成果链路。
- `M05 RAFT 光流测速`：自动写入 `table/{period}_实测_流速数据.csv`，字段包含 `date`、`period`、`scheme`、`module_code`、`method`、`velocity_m_s`、`mean_flow_direction_deg`、`fps`、`frame_count`、`valid_pairs`。
- `M07 洪涝风险评估`：自动写入 `raster/{period}_M07_洪涝风险分区图.tif`，优先采用五级风险等级栅格，必要时重投影到 `EPSG:32642`，并写入旁路元数据与 `finish.tag`。
- `M08 水资源分配`：自动写入 `table/{period}_M08_分水方案统计表.csv`，字段包含 `date`、`period`、`scheme`、`module_code`、`time_scale`、`sector`、`demand_million_m3`、`surface_release_million_m3`、`groundwater_million_m3`、`received_million_m3`、`shortage_million_m3`、`satisfaction_ratio_pct`。

## 8. 数据目录校验

仓库提供了轻量校验脚本，用于检查样例或正式目录是否满足本规范：

```bash
python tools/validate_twin_data.py
python tools/validate_twin_data.py "D:/path/to/瓦赫什流域孪生数据"
```

主程序也提供了界面校验入口：

1. 打开 `数据整理与流程总览` 标签页。
2. 在“数据根目录”中选择正式数据目录。
3. 点击“校验数据目录”。
4. 若出现错误，先修正目录结构、CRS、CSV 字段或 `finish.tag`，再进入业务模块。

正式数据根目录可在 `config.py` 中设置：

```python
TWIN_DATA_ROOT = "D:/path/to/瓦赫什流域孪生数据"
```

也可在启动前通过环境变量覆盖：

```powershell
$env:VAKHSH_TWIN_DATA_ROOT = "D:/path/to/瓦赫什流域孪生数据"
python main.py
```

当前校验内容包括：

- `baseline/raw/processed` 三类目录是否存在。
- baseline 是否包含流域边界、河网、水库边界和 DEM。
- raw 是否包含示例时段的遥感、气象、水库参数和河道视频。
- processed 是否按“方案 -> 时段 -> raster/table”组织。
- GeoTIFF 坐标系是否为 `EPSG:32642`。
- CSV 是否包含统一时间字段 `date`。
- `finish.tag` 是否可读并包含完成模块列表。

## 9. 待刘老师或模块负责人确认事项

- M01 当前由 SegFormer 专题识别承担，是否需要同时输出水体识别成果进入 M03 或 M07。
- M06 积雪状态识别当前 GEE 输出 `Snow_State` 与 `Runoff_Probability`，是否需要在标准成果中转换为 `snow_type` 和 `snow_density` 两个文件。
- M03 洪水演进与 Unity 三维展示之间的数据接口是否只读 `processed/`，还是需要额外三维场景缓存目录。
- M09 库区水量估算当前已有库容/库水量估算，但尚无可追溯下泄流量 `M09_outflow.csv` 计算逻辑；需要确认出库流量来源、单位和时间步长后再接入 M08。
- M04/M05 是否只做人工校核，还是要在界面中自动参与参数率定。
