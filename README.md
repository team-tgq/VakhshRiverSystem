# VakhshRiverSystem
瓦赫什河流域水文综合系统（Vakhsh River System）是一个基于 **Python + PyQt5** 构建的插件式桌面应用系统，面向流域水文、水资源、水灾害与遥感智能识别等业务场景，支持多模块集成、统一界面调度与专题功能扩展。

---

# 一、项目特点

- 基于 **PyQt5** 构建统一桌面界面
- 采用 **插件式架构**，各专题模块可独立开发
- 支持传统水文模型与 AI 推理模块融合
- 支持遥感影像、GIS、专题识别、优化配置等多类型任务
- 支持独立 Python 环境运行 AI 推理服务
- 便于扩展新的流域分析模块

---

# 二、项目目录结构

```
VakhshRiverSystem/
│
├─ algorithms/                         # 各业务算法模块
│  ├─ flood/                           # 洪涝灾害风险等级评估
│  ├─ inundation_monitoring/           # 淹没区监测（SegFormerNet）
│  ├─ monitoring/                      # 旧水文监测算法，当前不加载
│  ├─ raft/                            # RAFT 光流测速算法与 raft-sintel.pth 权重
│  ├─ reservoir_estimation/            # 库区水量估算
│  ├─ routing/                         # Unity 洪水演进三维可视化启动封装
│  ├─ segformer_service/               # SegFormer 推理服务（水体/积雪识别）
│  ├─ snow_state/                      # 积雪状态识别与融雪径流概率（GEE）
│  ├─ swe/                             # 雪水当量估算
│  ├─ warning/                         # 旧预警算法，当前不加载
│  ├─ water_allocation/                # 水资源动态优化配置
│  └─ __init__.py
│
├─ app/                               # 主程序框架
│  ├─ __init__.py
│  ├─ base_plugin.py
│  ├─ digital_twin_standard.py         # 数字孪生统一数据规范、字段单位和模块链路
│  ├─ main_window.py
│  └─ plugin_manager.py
│
├─ plugins/                           # 功能插件
│  ├─ flood_plugin/
│  ├─ integration_overview_plugin/      # 数据整理与流程总览
│  ├─ inundation_monitoring_plugin/
│  ├─ monitoring_plugin/               # enabled=false，旧水文监测入口已禁用
│  ├─ raft_plugin/
│  ├─ reservoir_estimation_plugin/
│  ├─ routing_plugin/
│  ├─ segformer_plugin/
│  ├─ snow_state_plugin/
│  ├─ swe_plugin/
│  ├─ warning_plugin/                  # enabled=false，预警模块当前不加载
│  ├─ water_allocation_plugin/
│  └─ __init__.py
│
├─ data/                              # 正式 baseline/raw/processed 数据根目录
├─ reports/                           # 迁移、运行、验证与完成报告
├─ docs/                              # 集成说明与模块文档
├─ sample_data/                       # 历史演示样例，不是正式运行入口
├─ config.py                          # 全局配置
├─ main.py                            # 程序入口
└─ README.md
```

---

# 三、系统架构说明

系统采用 **主系统 + 插件 + 算法模块** 的分层结构。

## 1. 主程序层

主程序位于：

```
main.py
app/
```

主要功能：

- 启动 Qt 应用
- 创建主窗口
- 初始化插件管理器
- 加载插件
- 管理标签页界面

---

## 2. 插件层

插件位于：

```
plugins/
```

每个插件对应一个系统功能模块。

插件负责：

- 构建界面
- 获取用户输入
- 调用算法模块
- 显示结果

插件结构示例：

```
plugins/example_plugin/
├─ plugin.py
└─ example_widget.py
```

插件接口示例：

```python
class ExamplePlugin:

    def name(self):
        return "模块名称"

    def widget(self):
        return ExampleWidget()
```

---

## 3. 算法层

算法模块位于：

```
algorithms/
```

主要负责：

- 数据处理
- 计算模型
- AI 推理
- 优化算法
- 输出结果

示例结构：

```
algorithms/module_name/
├─ __init__.py
├─ core.py
├─ model.py
└─ utils.py
```

算法层与 GUI 完全解耦。

---

## 4. AI 推理服务层

部分 AI 模型使用 **独立推理服务**。

例如：

```
algorithms/segformer_service/
```

主要特点：

- 独立 Python 环境
- GPU 推理
- 通过 subprocess 调用

---

## 5. GEE 云端识别层

新增的积雪状态识别模块采用“插件界面 + 算法封装 + GEE 云端导出”的接入方式：

```
plugins/snow_state_plugin/
algorithms/snow_state/
```

其中：

- 插件层负责日期、区域、数据源、GEE Project ID 和导出参数录入
- 算法层负责 Earth Engine 初始化、积雪状态计算、融雪径流概率计算和导出任务提交
- 结果以 GeoTIFF 导出到 Google Drive，包含 `Snow_State` 与 `Runoff_Probability` 两个波段
- `Snow_State` 取值为 `1=无雪/裸地`、`2=干雪/稳定积雪`、`3=湿雪/融雪活跃区`

---

# 四、统一数据架构与模块现状

本阶段只处理八个可独立运行模块：`M01`、`M02`、`M04`、`M05`、`M06`、`M07`、`M08`、`M09`。不建立上下游依赖，不按年份、工况或方案划分顶层目录。`M03` 洪水演进源码和 Unity 程序保留，但不纳入本阶段数据管线，不生成任何 M03 成果。

## 1. 唯一正式数据根目录

```text
data/瓦赫什流域孪生数据/
├─ baseline/                         # 固定公共基准，只读
│  ├─ DEM.tif
│  ├─ 流域边界.*
│  ├─ 河网.*
│  └─ 水库边界.*
├─ raw/                              # 按数据类型组织的正式输入
│  ├─ remote_sensing/
│  │  ├─ optical_rgb/                # SegFormer snow/water val/test
│  │  ├─ sentinel1_sar/              # M04 SAR 原生验证样例
│  │  ├─ sentinel2_multispectral/     # M04 多光谱原生验证样例
│  │  └─ gee/snow_state/             # M06 本地 GEE 产品
│  ├─ meteorology/                   # 气温、降水、固态降水、daily forcing
│  ├─ land_surface/                  # 土壤湿度、土地覆盖
│  ├─ snow_hydrology/                # 存在真实数据时保存前期 SWE/雪盖/雪密度
│  ├─ reservoir/                     # 参数、观测、水位-面积-库容曲线
│  ├─ socioeconomic/                 # 人口、产业、农业、需水资料
│  ├─ configuration/water_allocation/# M08 可恢复界面的配置 CSV
│  └─ video/river_velocity/          # 2026-05 清水河道流速测试视频
└─ processed/                        # 只能保存模块本次实际运行结果
   ├─ M01_segformer/
   ├─ M02_swe/
   ├─ M04_inundation/
   ├─ M05_raft/
   ├─ M06_snow_state/
   ├─ M07_flood_risk/
   ├─ M08_water_allocation/
   └─ M09_reservoir_estimation/
```

`sample_data/` 不是正式运行入口。`algorithms/*/data` 只作为历史来源或算法自带验证集备份，插件和运行工具必须从统一 `raw` 读取。`processed` 不得保存 raw 副本、模型权重、缓存、日志、临时帧或占位成果。

## 2. 数据真实性分类

- **模块测试数据**：`raw/video/river_velocity/202605_清水河道流速测试视频.mp4` 为清水河道测试视频，不是现场或现地实测数据。
- **公开或官方产品**：气象、降水、土壤湿度、土地覆盖和 GEE 产品，按各自 `.meta.json` 的来源记录解释。
- **模块原生验证数据**：SegFormer 的 snow/water 验证集，以及 M04 的 Bolivia Sentinel-1、India Sentinel-2 样例。它们可验证算法，但不是瓦赫什流域业务成果。
- **用户配置**：M08、M09 的参数 CSV。配置不是观测数据，不得表述为实测。
- **来源未完全可核实**：必须在 `.meta.json` 和报告中明确说明，不能补写成“真实观测”。

每个 raw 和 processed 文件都有同名 `.meta.json`，记录来源文件、消费模块、模型权重/配置、尺寸、类型、CRS、SHA-256 和生成时间。每个成功模块在自己的目录写入 `finish.tag`。

## 3. 八模块输入输出

| 模块 | 统一输入 | 统一输出 | 当前独立运行结果 |
| --- | --- | --- | --- |
| M01 SegFormer | `raw/remote_sensing/optical_rgb/segformer_{snow,water}/` | `M01_segformer/{snow,water}/{val,test}/{masks,overlays}`、验证指标 | 11 张积雪、10 张水体均生成 mask/overlay；13 对有效标签计算指标 |
| M06 积雪状态 | `raw/remote_sensing/gee/snow_state/` | 雪类型、确定性雪密度、统计表 | 两份本地 GEE 产品运行完成；GEE 产品不是现地实测 |
| M02 SWE | `raw/meteorology/daily_forcing/`、`baseline/{DEM.tif,流域边界.shp}` | SWE、Snowmelt、日/统计表 | 使用 2026-05-25 GFS+VIIRS forcing 运行完成；当前算法不生成 runoff |
| M09 库容估算 | `raw/reservoir/{parameters,observations,hypsometry}/` | `reservoir_storage.csv`、`estimation_summary.json` | 独立运行完成；当前不生成 outflow |
| M08 水资源分配 | `raw/configuration/water_allocation/` | `allocation_plan.csv`、`allocation_summary.csv` | NSGA-II `pop=200/gen=400`，输出 12 月 × 5 部门共 60 行；不依赖 M09 |
| M07 洪涝风险 | 降水、土壤湿度、土地覆盖及 baseline | 风险指数、五级图、权重、统计和 HTML | 使用 2026-05-28 六因子独立运行完成，不读取 M03/M04 |
| M04 淹没识别 | Sentinel-1 SAR 弱标注、Sentinel-2 多光谱手工标注样例 | 两类数据各自的 mask/overlay | 3 幅原生验证影像均完成；结果不能表述为瓦赫什实测淹没范围 |
| M05 RAFT | `raw/video/river_velocity/` | `frame_pair_velocity.csv` | 911 帧全部处理，生成 910 帧对记录，895 个有效流场 |

M05 没有可信空间标定，因此 `velocity_m_s` 全部留空，只输出 `velocity_px_frame`。不得使用默认相机参数伪造米每秒。

## 4. M08 与 M09 配置 CSV

M08 插件提供“加载统一配置”和“回写统一配置”，以下文件共同恢复界面：

- `supply/global_supply_config.csv`：`start_year,start_month,end_year,end_month,time_scale,initial_surface_supply_mcm,inflow_source_year`
- `supply/monthly_inflow.csv`：`year,month,inflow_m3_s,source,quality_status`
- `demand/demand_parameters.csv`：`parameter,value,unit,ui_key`
- `crops/crops.csv`：`crop_type,growth_stage,area_km2,yield_kg_km2,price_cny_kg,kc`
- `decision_weights/decision_weights.csv`：`group,name,value`

M08 缺少显式 `monthly_inflow.csv` 时直接报错，不再使用固定 `500 m3/s` 回退值。

M09 使用：

- `reservoir_parameters.csv`：`reservoir_id,reservoir_name,parameter,value,unit,source`
- `reservoir_observations.csv`：`date,reservoir_id,sensor,water_level_m,surface_area_km2,source,quality_status`
- `reservoir_hypsometry.csv`：`reservoir_id,elevation_m,area_km2,storage_mcm`

插件可从这些 CSV 加载参数、观测和曲线，也可把当前界面配置回写到统一 raw。没有实际观测的字段不得补造。

## 5. 统一路径接口

模块负责人必须使用 `app/digital_twin_standard.py`，不得硬编码 `data` 绝对路径：

```python
from app.digital_twin_standard import (
    ensure_raw_source_path,
    module_processed_dir,
    raw_data_dir,
    write_metadata_sidecar,
)

video = ensure_raw_source_path(
    raw_data_dir("video", "river_velocity") / "202605_清水河道流速测试视频.mp4"
)
output = module_processed_dir("M05", "tables", create=True) / "frame_pair_velocity.csv"

# 算法实际运行并生成 output 后，再记录真实来源。
write_metadata_sidecar(
    output,
    module_code="M05",
    field="frame_pair_pixel_velocity",
    source_files=[video],
)
```

## 6. 准备、运行和验证

以下命令均只使用本地已有数据，不会联网下载：

```bash
# 查看迁移/删除候选，不改文件
python tools/prepare_real_twin_data.py --audit-only

# 将仓库内原生数据迁移到类型化 raw，并清理已确认的旧目录
python tools/prepare_real_twin_data.py --migrate

# 独立运行八个模块；M03 不在列表中
python tools/prepare_real_twin_data.py --run-modules --modules "M01,M06,M02,M09,M08,M07,M04,M05"

# 第一轮：baseline/raw 结构、元数据、哈希和文件可读性
python tools/validate_twin_data.py --stage baseline-raw

# 第二、三轮：模块成果、来源链、栅格/图片/CSV/视频质量
python tools/validate_twin_data.py --stage full
```

运行证据保存在：

- `reports/data_architecture_migration_audit.json`
- `reports/independent_module_run_report.json`
- `reports/twin_data_validation_report.json`

正式数据根目录可通过 `config.py` 的 `TWIN_DATA_ROOT` 或环境变量 `VAKHSH_TWIN_DATA_ROOT` 指定。

## 7. 当前已知限制

- M02 的 `forcing_20260525.npz` 是带真实 GFS 经纬度和地形参考的 `9×21` 网格；运行时将 baseline DEM/流域边界投影到 forcing 网格，裁剪后输出 `7×19`、51 个流域内有效像元，不做无空间参考的静默拉伸。
- 当前统一 raw 只保留 2026-05-25 forcing；M02 以冷启动方式重放该日数据，SWE 均值约 `1.153 mm`，Snowmelt 均值约 `0.049 mm/day`。
- M04 两幅 Sentinel-1 样例的淹没比例约为 `99.98%`，说明当前 7 通道模型对该类 2 波段 SAR 样例适配质量较差。
- M04 的 Sentinel-2 样例淹没比例约为 `4.18%`，相对 S1 更符合该光学模型的输入分布；但仓库没有与 `India_73419_S2Hand.tif` 配套的独立真值掩膜，因此不能据此报告 IoU、F1 或“精度更高”。
- M09 的 2022 面积记录缺少底层原始影像，只能标记为模块原生验证记录，不能证明为现地实测。
- SegFormer water 验证集中 8 个原标签为零字节文件，迁移时没有补造标签；对应影像只做推理，不计算指标。

## 8. 当前禁用或不处理模块

- `plugins/monitoring_plugin/`：旧水文监测模块禁用，RAFT 已迁移到独立 M05 插件。
- `plugins/warning_plugin/`：预警模块通过 `enabled=false` 禁用。
- `M03`：本阶段不参与统一数据运行，不生成 discharge、flood depth 或 inundation 成果。

---

# 五、运行环境

本项目实际使用两个 Conda 环境：

- `VakhshRiverSystem`：主程序环境，用于启动 PyQt5 桌面系统和大部分业务插件。
- `segformer`：SegFormer 独立推理环境，用于 `plugins/segformer_plugin` 调用水体/积雪语义分割服务。

团队成员从 GitHub clone 仓库后不会自动得到本机 Conda 环境，需要按下面步骤创建或更新环境。不要直接使用 `base` 环境运行本项目。

依赖文件已导出到：

```text
requirements/
├─ environment-vakhsh.yml          # 主程序 Conda 环境
├─ environment-segformer.yml       # SegFormer Conda 环境
├─ requirements-vakhsh.txt         # 主程序 pip 依赖清单
└─ requirements-segformer.txt      # SegFormer pip 依赖清单
```

推荐优先使用 `environment-*.yml` 创建环境，因为它包含 Python 版本、Conda 包和 pip 包信息；`requirements-*.txt` 主要用于排查缺包或在已有环境中补装 pip 依赖。

当前已随仓库保留的关键资源：

- `algorithms/raft/raft-sintel.pth`：RAFT 光流测速默认权重
- `algorithms/water_allocation/resources/models/best.pth`：水资源分配 LSTM 入库径流预测权重
- `algorithms/water_allocation/resources/scalers/`：水资源分配 LSTM 归一化器
- `algorithms/segformer_service/`：SegFormer 推理服务代码、运行时和本地 wheel

当前未随仓库强制提供的可选资源：

- `algorithms/water_allocation/resources/models/3_Class_FULL_FTW_Pretrained_v2.ckpt`：FTW 遥感耕地识别权重。缺少该文件或缺少 `segmentation_models_pytorch` 时，水资源分配核心优化、LSTM 预测和手工输入模式仍可运行，仅本地遥感智能估算农业面积不可用。

## 1. 创建主程序环境

在项目根目录执行：

```bash
conda env create -f requirements/environment-vakhsh.yml
conda activate VakhshRiverSystem
python main.py
```

如果本机已经存在 `VakhshRiverSystem` 环境，可用下面命令更新：

```bash
conda activate VakhshRiverSystem
conda env update -n VakhshRiverSystem -f requirements/environment-vakhsh.yml
```

如果只是提示缺少某个 pip 包，可在项目根目录补装：

```bash
conda activate VakhshRiverSystem
pip install -r requirements/requirements-vakhsh.txt
```

## 2. 创建 SegFormer 独立环境

SegFormer 模块依赖旧版 `python=3.8`、`pytorch=1.10`、`mmcv-full=1.6.0` 和本项目内置的 MMSeg runtime，建议单独创建环境：

```bash
conda env create -f requirements/environment-segformer.yml
conda activate segformer
```

如果 Conda 创建过程中 `mmcv-full` 安装失败，可在项目根目录用本地 wheel 补装：

```bash
conda activate segformer
pip install algorithms/segformer_service/mmcv_full-1.6.0-cp38-cp38-win_amd64.whl
pip install -e algorithms/segformer_service/segformer_runtime
```

已有 `segformer` 环境时可更新：

```bash
conda activate segformer
conda env update -n segformer -f requirements/environment-segformer.yml
```

## 3. PyCharm 解释器设置

团队成员 clone 仓库后，需要在 PyCharm 中手动选择解释器：

- 主程序运行配置选择 `VakhshRiverSystem` 环境的 `python.exe`
- SegFormer 插件确认 `algorithms/segformer_service/service_config.py` 中的 `SEGFORMER_PYTHON` 指向本机 `segformer` 环境的 `python.exe`
- RAFT 光流测速使用主程序环境，默认从 `algorithms/raft/raft-sintel.pth` 加载权重
- 不要用 `base` 环境运行本项目，否则可能出现 `cv2`、`geopandas`、`pymoo`、`torch`、`mmcv`、`earthengine-api` 等模块缺失

可用下面命令查看本机 Conda 环境路径：

```bash
conda info --envs
```

## 4. 维护者更新依赖文件

如果模块负责人新增了依赖，先在对应环境中安装并验证，再更新依赖文件。Windows PowerShell 中建议显式使用 UTF-8 输出：

```powershell
$OutputEncoding = [Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
conda activate VakhshRiverSystem
pip freeze | Out-File -FilePath requirements/requirements-vakhsh.txt -Encoding utf8

conda activate segformer
pip freeze | Out-File -FilePath requirements/requirements-segformer.txt -Encoding utf8
```

Conda YAML 需要由项目维护者统一导出，并过滤本机 `prefix:` 路径，避免把个人电脑路径写进仓库。

```powershell
$OutputEncoding = [Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
conda env export -n VakhshRiverSystem --no-builds | Where-Object { $_ -notmatch '^prefix:' } | Out-File -FilePath requirements/environment-vakhsh.yml -Encoding utf8
conda env export -n segformer --no-builds | Where-Object { $_ -notmatch '^prefix:' } | Out-File -FilePath requirements/environment-segformer.yml -Encoding utf8
```

---

# 六、系统启动

进入项目目录后执行：

```
python main.py
```

系统启动流程：

1. 初始化 Qt
2. 创建主窗口
3. 扫描插件
4. 加载模块
5. 启动系统

---

# 七、插件加载机制

插件加载流程：

1. main.py 启动程序
2. MainWindow 创建界面
3. PluginManager 扫描 plugins
4. 读取每个插件目录下的 `plugin.json`
5. 如果 `plugin.json` 中存在 `"enabled": false`，该插件会被跳过
6. 导入 `plugin.py` 中的 `Plugin` 类
7. 创建 widget
8. 按 `order()` 返回值排序后添加到主界面标签页

核心文件：

```
app/plugin_manager.py
app/main_window.py
```

当前被禁用的插件：

```text
plugins/monitoring_plugin/plugin.json
plugins/warning_plugin/plugin.json
```

启用或禁用插件时，只需要修改对应 `plugin.json`：

```json
{
  "name": "模块名称",
  "version": "1.0",
  "author": "负责人",
  "enabled": false
}
```

---

# 八、新模块接入

新增模块步骤：

### 1 新建算法模块

```
algorithms/new_module/
```

### 2 新建插件模块

```
plugins/new_module_plugin/
```

### 3 添加 plugin.json

```json
{
  "name": "新模块名称",
  "version": "1.0",
  "author": "负责人",
  "enabled": true
}
```

如果模块暂时不想显示在主界面，把 `enabled` 改成 `false`。

### 4 实现插件接口

```python
from app.base_plugin import BasePlugin
from .new_module_widget import NewModuleWidget


class Plugin(BasePlugin):
    def name(self):
        return "新模块名称"

    def order(self):
        return 999

    def widget(self):
        return NewModuleWidget()
```

---

# 九、GitHub 多人协同开发流程

本项目采用“稳定主线 + 集成分支 + 个人功能分支”的协作方式。所有模块负责人都应先在自己的分支完成算法和界面修改，再通过 GitHub Pull Request 合并，避免多人直接改同一条分支导致冲突。

## 1. 分支职责

- `main`：稳定版本分支，只放已经验证通过、可以演示或交付的代码。
- `dev/next`：日常集成分支，各模块功能先合并到这里统一联调。
- `feature/模块名-功能名`：模块负责人自己的开发分支，例如 `feature/swe-load-results`、`feature/flood-risk-levels`。
- `fix/模块名-问题名`：紧急修复分支，例如 `fix/water-allocation-pymoo-error`。

## 2. 第一次克隆项目

```bash
git clone <GitHub仓库地址>
cd VakhshRiverSystem
git checkout dev/next
conda activate VakhshRiverSystem
python main.py
```

如果本地已经有项目，不要重新复制文件夹，直接在原仓库里同步：

```bash
git fetch origin
git checkout dev/next
git pull origin dev/next
```

## 3. 每次开始开发前

先确认自己当前在哪个分支，并把 `dev/next` 更新到最新：

```bash
git status
git checkout dev/next
git pull origin dev/next
```

再从最新的 `dev/next` 创建自己的分支：

```bash
git checkout -b feature/模块名-功能名
```

示例：

```bash
git checkout -b feature/swe-result-view
```

## 4. 开发过程中提交代码

查看改动：

```bash
git status
git diff
```

只提交和自己模块相关的文件：

```bash
git add algorithms/模块名 plugins/模块名_plugin README.md
git commit -m "feat(模块名): 更新算法与界面显示"
```

常用提交信息前缀：

- `feat`：新增功能
- `fix`：修复问题
- `docs`：更新文档
- `refactor`：重构，不改变功能
- `chore`：配置、依赖、清理类修改

## 5. 推送到 GitHub 并发起 Pull Request

```bash
git push -u origin feature/模块名-功能名
```

然后在 GitHub 页面创建 Pull Request：

- base 分支选择 `dev/next`
- compare 分支选择自己的 `feature/...` 或 `fix/...`
- PR 标题写清楚模块和目的
- PR 描述里说明改了哪些算法、哪些界面、怎么验证

PR 合并前至少确认：

- `python main.py` 可以启动
- 自己负责的插件可以打开
- 按钮、输入框、图表、结果展示可以正常工作
- 没有提交 `__pycache__`、临时输出、大模型权重、个人路径配置

## 6. 开发中同步别人最新修改

如果自己开发了一段时间，别人已经合并了新代码，要把 `dev/next` 的最新内容同步到自己的分支：

```bash
git fetch origin
git checkout feature/模块名-功能名
git merge origin/dev/next
```

如果出现冲突，先打开冲突文件，保留双方需要的内容，确认程序能运行后再提交：

```bash
git status
git add 冲突文件路径
git commit -m "chore(模块名): resolve merge conflicts"
```

## 7. 合并后的本地清理

PR 合并后，本地切回集成分支并更新：

```bash
git checkout dev/next
git pull origin dev/next
```

确认自己的功能分支已经合并后，可以删除本地旧分支：

```bash
git branch -d feature/模块名-功能名
```

## 8. 不要直接提交的内容

以下内容通常不要提交到 GitHub：

- `output/` 下的临时运行结果
- `__pycache__/`、`.pyc`、`.ipynb_checkpoints/`
- 本机绝对路径配置，例如只在某台电脑存在的 `E:/...`
- 大体积模型权重、遥感原始数据、临时下载数据
- 账号、token、Google Earth Engine 私钥、云服务密钥

如果模块必须依赖大文件，请在 README 或模块说明中写清楚下载位置、文件名、放置目录，不要直接把大文件提交进仓库。

---

# 十、模块负责人更新算法与显示界面规范

每个业务模块通常由两部分组成：

- 算法代码：放在 `algorithms/模块名/`
- 插件界面：放在 `plugins/模块名_plugin/`

模块负责人更新功能时，优先只改自己负责的算法目录和插件目录。除非确实需要公共能力，否则不要随意修改 `main.py`、`app/main_window.py`、`app/plugin_manager.py`。

## 1. 推荐更新流程

1. 先在 `algorithms/模块名/` 中完成纯算法函数。
2. 用简单脚本或 Python 交互命令验证算法能独立运行。
3. 再到 `plugins/模块名_plugin/` 中更新按钮、输入框、结果展示。
4. 在插件里调用算法函数，把异常显示到界面日志或弹窗。
5. 启动 `python main.py`，验证主界面、插件标签页和结果展示。
6. 提交 PR，并在 PR 描述中写清验证步骤。

## 2. 算法模块怎么写

算法模块应尽量保持“可独立调用”，不要依赖界面控件。推荐结构：

```text
algorithms/example_module/
├─ __init__.py
├─ core.py
├─ config.py
└─ utils.py
```

`core.py` 示例：

```python
def run_example_analysis(input_path: str, threshold: float = 0.5) -> dict:
    if not input_path:
        raise ValueError("input_path 不能为空")

    # 在这里执行数据读取、模型推理或计算分析
    result_path = "output/example/result.png"

    return {
        "status": "success",
        "result_path": result_path,
        "summary": "分析完成",
    }
```

算法函数建议返回 `dict`，至少包含：

- `status`：运行状态
- `summary`：结果摘要
- `result_path` 或 `output_files`：输出文件路径
- `metrics`：关键指标
- `message`：给界面显示的说明

不要在算法层直接创建复杂 PyQt 控件；界面显示应由插件层负责。

## 3. 插件界面怎么写

每个插件目录至少包含：

```text
plugins/example_plugin/
├─ plugin.py
├─ plugin.json
└─ example_widget.py
```

`plugin.py` 示例：

```python
from app.base_plugin import BasePlugin
from .example_widget import ExampleWidget


class Plugin(BasePlugin):
    def name(self):
        return "示例模块"

    def order(self):
        return 999

    def widget(self):
        return ExampleWidget()
```

`example_widget.py` 中负责：

- 创建输入控件，例如文件选择、日期、阈值、下拉框
- 调用 `algorithms/模块名/` 中的算法函数
- 在 `QTextEdit`、`QLabel`、图表或图片区域展示结果
- 捕获异常并提示用户，不要让整个主程序崩溃

界面调用示例：

```python
from PyQt5.QtWidgets import QMessageBox, QPushButton, QTextEdit, QVBoxLayout, QWidget

from algorithms.example_module.core import run_example_analysis


class ExampleWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.log = QTextEdit()
        self.log.setReadOnly(True)

        self.run_btn = QPushButton("运行分析")
        self.run_btn.clicked.connect(self.run_analysis)

        layout = QVBoxLayout(self)
        layout.addWidget(self.run_btn)
        layout.addWidget(self.log)

    def run_analysis(self):
        try:
            result = run_example_analysis(input_path="data/input.tif")
            self.log.append(result["summary"])
        except Exception as exc:
            self.log.append(f"[ERROR] {exc}")
            QMessageBox.critical(self, "运行失败", str(exc))
```

## 4. 算法和界面的连接原则

- 算法层只关心输入、计算、输出，不关心按钮和窗口。
- 插件层只负责收集用户输入、调用算法、展示结果。
- 长时间运行的任务应使用后台线程，避免界面卡死。
- 文件路径应来自用户选择、配置文件或模块默认目录，不要写死个人电脑路径。
- 输出结果优先放到模块自己的 `output/` 子目录，并避免提交临时结果。

## 5. 更新依赖时必须说明

如果模块新增第三方库，要同步更新文档，写清：

- 包名
- 推荐安装方式
- 是否需要独立 conda 环境
- 是否依赖 GPU、CUDA、GDAL、MMCV 等特殊组件

示例：

```bash
conda activate VakhshRiverSystem
pip install package-name
```

如果是 SegFormer 这类独立推理服务，应说明使用的独立环境，例如：

```text
algorithms/segformer_service/environment.yaml
```

并确认插件中的解释器路径配置正确。

## 6. 更新前后的自检清单

提交 PR 前请逐项检查：

- 当前分支不是 `main`
- 已从最新 `dev/next` 创建功能分支
- 只修改了自己负责的模块文件
- 新增算法可以单独运行
- 插件界面可以打开并显示结果
- 异常情况会在界面中提示
- `python main.py` 可以启动主程序
- `git status` 中没有无关临时文件
- PR 描述写明了测试步骤和已知限制

---

# 十一、项目说明

**VakhshRiverSystem**

瓦赫什河流域水文综合系统。

该系统整合：

- 洪涝风险与淹没区识别
- 洪水演进三维可视化
- 水资源调度
- 遥感识别
- AI 模型分析

用于构建流域级综合分析平台。

---

# 十二、界面输入提示与格式说明（2026-06-25）

说明：

- 已在有输入项的界面中新增 `i` 说明符号或文字提示。
- 鼠标悬停在提示符号或输入控件上，可查看“输入内容 + 数据格式 + 示例”。
- `monitoring_plugin` 和 `warning_plugin` 当前不加载，相关旧输入说明不再作为主程序使用说明。

## 0 数据整理与流程总览（`plugins/integration_overview_plugin`）

- 显示统一 `baseline/raw/processed` 数据根目录和八个独立模块状态
- 正式数据根目录：`data/瓦赫什流域孪生数据/`
- 可打开整改说明：`docs/digital_twin_integration_standard.md`
- 用途：作为演示首页，先说明数据规范和计算顺序，再进入业务模块

## 1 淹没区监测（`plugins/inundation_monitoring_plugin`）

- 输入影像：`tif/tiff`，建议使用与训练数据一致的遥感影像格式
- 模型权重：需要与当前 `SegFormerNet` 结构匹配
- 输出：淹没区 mask、叠加显示结果和运行日志

## 2 库区水量估算（`plugins/reservoir_estimation_plugin`）

- 水位/面积估算：输入日期、水位 `m` 或水面面积 `km2`
- 影像面积估算：选择 `tif/tiff/png/jpg/jpeg/bmp`，GeoTIFF 自动读取像元面积，普通图片需要手动输入像元大小 `m`
- 输出：库容估算文本、库容曲线图、`reservoir_storage.csv` 与 `estimation_summary.json`
- 注意：当前库容模块不生成下泄流量占位文件；只有算法真实计算出库流量后才允许增加 outflow 成果

## 3 洪涝灾害风险等级评估（`plugins/flood_plugin`）

- 输入：模块界面内选择的灾害风险因子、数据文件或默认样例数据
- 输出：风险等级评估结果、图表或专题图

## 4 RAFT 光流测速（`plugins/raft_plugin`）

- 输入视频：`mp4/avi/mov`，建议固定机位、画面稳定、河面纹理清晰
- 默认权重路径：`algorithms/raft/raft-sintel.pth`
- 运行范围：RAFT 模式处理完整视频的全部相邻帧对，不使用“仅选若干帧”的旧测试逻辑
- 输出：`frame_pair_velocity.csv`，包含像素速度、有效像元、置信度、时间戳和来源视频
- 标定限制：没有可信空间标定时 `velocity_m_s` 留空，不将默认相机参数换算结果当作实测流速

## 5 洪水演进与汇流模拟（`plugins/routing_plugin`）

- 当前入口：点击“启动三维可视化”
- 数据状态：M03 当前不纳入统一数据运行，不生成 discharge、洪水深度或淹没成果
- Unity 文件：优先读取 `algorithms/routing/tjk/` 或 `algorithms/routing/unity_build/` 下的完整 Unity build，同时要求存在 `.exe`、`*_Data/` 与 `UnityPlayer.dll`
- 说明：界面不显示可视化程序路径，只展示启动状态和异常日志

## 6 SegFormer 专题识别（`plugins/segformer_plugin`）

- 任务：下拉选择水体识别或积雪识别
- 设备：`cpu` 或 `cuda:0`
- 图片路径：图像文件路径，常见格式为 `png/jpg/jpeg/bmp`
- 环境：需要 `segformer` 独立 Conda 环境和 `service_config.py` 中的解释器路径配置正确
- 标准成果：每张统一 raw 验证图片直接生成 mask 和 overlay 到 `processed/M01_segformer/{snow,water}/{val,test}/`
- 注意：当前模型不输出真实雪深，不得根据积雪覆盖伪造雪深；带 CRS 的业务 GeoTIFF 可走可选 GIS 导出流程

## 7 积雪状态识别（`plugins/snow_state_plugin`）

- 目标日期范围：`yyyy-MM-dd`
- SAR 融雪期与参考期：`yyyy-MM-dd`
- 研究区范围：`west,south,east,north`，示例 `70.0,36.0,76.5,40.0`
- GEE Project ID：Google Earth Engine 项目 ID
- Drive 文件夹：Google Drive 导出目录
- 导出分辨率：正整数，单位 `m`
- 输出波段：`Snow_State` 与 `Runoff_Probability`
- 标准成果：本地 GEE GeoTIFF 输出到 `processed/M06_snow_state/rasters/` 和 `tables/`
- 说明：`M06_snow_density_gcm3.tif` 当前由状态类别映射生成，无雪=0、干雪=0.25、湿雪=0.40，便于 M02 使用统一输入；真实雪密度模型确认后可替换

## 8 雪水当量估算（`plugins/swe_plugin`）

- 回算最近天数：正整数
- 可选操作：更新最新 SWE、加载已有结果、重新训练模型
- 输出：SWE GeoTIFF、Snowmelt、QA 与界面图层展示
- 标准成果：输出 `SWE_mm_*.tif` 与 `Snowmelt_mm_day_*.tif`，并写入元数据和完成标记
- 注意：当前 M02 没有径流算法，因此不生成 runoff
- 注意：加载已有结果依赖 `algorithms/swe/swe_assessment.py` 中的已有结果加载接口

## 9 水资源分配（`plugins/water_allocation_plugin`）

- 起始时间/结束时间：年和月组合
- 时间粒度：`daily/monthly/yearly`
- 人口：浮点数，单位 `万人`
- 城镇化率：`0~100` 浮点数，单位 `%`
- 当地 GDP：浮点数
- 工业重复利用率：`0~100` 浮点数，单位 `%`
- 灌溉利用系数：`0~1` 浮点数
- 传输损耗率：`0~100` 浮点数，单位 `%`
- 生态保底水：浮点数，单位 `百万m³`
- 水电参数：单机最大功率 `MW`、单机最大流量 `m³/s`、上网电价 `元/kWh`
- 农业模式：可选择人工精细输入或遥感图像智能估算模式
- 作物参数：作物类型、生育期、面积 `km²`、产量 `kg/km²`、市价 `元/kg`
- 决策偏好权重：整体经济、降低缺水、部门公平，建议总和接近 1
- 部门收益权重：生活、生态、农业、工业、下游国家
- 正式配置：从 `raw/configuration/water_allocation/` 加载并可回写；缺少 12 个月显式入流时直接报错
- LSTM 训练参数：训练数据、序列长度、预测步长、训练轮数等
- 输出：`allocation_plan.csv`、`allocation_summary.csv`、经济效益、缺水量、公平性指标和结果图表；不依赖 M09

---

# 十三、第二次模块更新说明

本次完整系统实现对应第二次模块集成，重点更新如下：

- README 与当前主程序真实模块同步，明确 9 个启用模块和 2 个禁用模块
- 积雪状态识别升级为 GEE 双波段产品：`Snow_State` + `Runoff_Probability`
- RAFT 光流测速独立为 `plugins/raft_plugin`，默认权重统一放在 `algorithms/raft/raft-sintel.pth`
- 水文监测旧模块和洪水预警模块通过 `enabled=false` 禁用
- 洪水演进与汇流模块调整为 Qt 启动 Unity 三维可视化程序
- 水资源分配模块更新为 v2.0 多时间尺度动态优化配置，并修复分组标题显示被裁剪的问题
