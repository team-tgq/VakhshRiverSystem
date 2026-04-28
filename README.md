# VakhshRiverSystem

## 洪涝风险评估模块更新（2026-04）

本次对“洪涝灾害风险评估”模块做的是输入协议和运行流程重构，不改变原有六因子风险评估主框架，重点解决“月尺度动态数据不够及时、静态数据重复录入、无有效日期时仍继续出图”这三个问题。

### 1. 改造目标

- 将动态气象输入从原来的**月平均尺度**调整为**日尺度**
- 将静态地理数据改为**自动读取**，不再要求每次运行手动重复输入
- 当目标日期没有有效动态数据时，系统**停止计算并给出友好提示**，不再生成不可靠结果

### 2. 当前动态数据内容

洪涝风险评估当前使用的动态数据包括：

- 逐日降雨栅格
- 表层土壤湿度栅格（0-0.1 m）

界面中已新增“目标日期”输入控件，系统会严格按所选日期匹配这两类逐日数据。

### 3. 当前静态数据内容

静态地理数据默认由系统自动读取，包括：

- 研究区边界 `study_area.shp`
- DEM 裁剪栅格 `dem_clip.tif`
- 土地覆盖栅格 `landcover_demgrid.tif`
- 河网矢量 `hydrorivers.gpkg`

如果静态数据不完整，但已提供原始 `dem.tif`，系统会优先自动补齐下列基础数据：

- `dem_clip.tif`
- `landcover_demgrid.tif`
- `hydrorivers.gpkg`

### 4. 动态数据解析与自动获取逻辑

洪涝模块新增了统一输入解析器：

- 入口文件：`algorithms/flood/input_resolver.py`
- 风险评估主入口：`algorithms/flood/risk_assessment_6factors.py`
- 熵权版本入口：`algorithms/flood/risk_assessment_6factors_entropy.py`

解析策略如下：

- 优先在 `algorithms/flood/data/processed` 及其子目录中自动发现逐日文件
- 推荐文件命名：
  - `rain_mm_demgrid_YYYY-MM-DD.tif`
  - `soil_moist_demgrid_YYYY-MM-DD.tif`
- 若本地没有逐日数据，系统会尝试自动获取近实时逐日数据
- 当前近实时数据源已接入 **NOAA GFS**
- 近实时数据会自动处理并对齐到 DEM 网格后参与评估
- 未来日期或尚无完整逐日数据的日期不会再强行生成结果，而是提示用户改选可用日期

### 5. 界面与交互更新

洪涝风险评估插件界面已同步更新：

- 新增“目标日期”输入控件
- 明确展示动态数据项说明
- 明确说明静态数据为自动读取
- 运行日志中显示：
  - 实际使用日期
  - 动态数据来源
  - 静态数据自动准备动作
  - 动态数据自动获取动作
- 当数据缺失或日期无效时，使用更接近真实软件的友好提示文案反馈问题

### 6. 输出成果

成功运行后，系统会生成：

- 风险栅格：`outputs/risk_6factors.tif`
- 交互地图：`outputs/flood_risk_map.html`
- 权重报告：`outputs/final_weights.txt`

### 7. 汇报建议表述

如果用于阶段汇报，可概括为：

- “洪涝风险评估模块已完成从月尺度向日尺度动态输入的升级”
- “静态地理底图实现自动读取与缺失自动补齐”
- “新增按日期驱动的评估流程，避免无数据日期误生成结果”
- “已具备近实时逐日降雨与表层土壤湿度自动获取能力”

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
│  ├─ flood/                          # 洪涝风险评估
│  ├─ inundation_monitoring/          # 淹没区监测（UNet）
│  ├─ monitoring/                     # 水文监测
│  ├─ reservoir_estimation/           # 库区水量估算
│  ├─ routing/                        # 洪水演进与汇流
│  ├─ segformer_service/              # SegFormer推理服务（水体/积雪识别）
│  ├─ snow_state/                     # 积雪状态识别（GEE）
│  ├─ swe/                            # 雪水当量估算
│  ├─ warning/                        # 洪水预警监控
│  ├─ water_allocation/               # 水资源分配优化
│  └─ __init__.py
│
├─ app/                               # 主程序框架
│  ├─ __init__.py
│  ├─ base_plugin.py
│  ├─ main_window.py
│  └─ plugin_manager.py
│
├─ plugins/                           # 功能插件
│  ├─ flood_plugin/
│  ├─ inundation_monitoring_plugin/
│  ├─ monitoring_plugin/
│  ├─ reservoir_estimation_plugin/
│  ├─ routing_plugin/
│  ├─ segformer_plugin/
│  ├─ snow_state_plugin/
│  ├─ swe_plugin/
│  ├─ warning_plugin/
│  ├─ water_allocation_plugin/
│  └─ __init__.py
│
├─ output/                            # 输出目录
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

- 插件层负责日期、区域、数据源和导出参数录入
- 算法层负责 Earth Engine 初始化、积雪状态计算和导出任务提交
- 结果以 1-4 类积雪状态 GeoTIFF 导出到 Google Drive

---

# 四、系统模块

系统目前包含以下模块。

---

## 1 洪涝风险评估

插件目录：

```
plugins/flood_plugin/
```

算法目录：

```
algorithms/flood/
```

功能：

- 洪水风险识别
- GIS 分析
- 风险可视化

---

## 2 水文监测系统

插件目录：

```
plugins/monitoring_plugin/
```

算法目录：

```
algorithms/monitoring/
```

功能：

- 水位识别
- 光流测速
- 水文监测

---

## 3 库区水量估算

插件目录：

```
plugins/reservoir_estimation_plugin/
```

算法目录：

```
algorithms/reservoir_estimation/
```

功能：

- 库区面积估算
- 水库体积估算
- 结果 CSV 输出

---

## 4 洪水演进与汇流

插件目录：

```
plugins/routing_plugin/
```

算法目录：

```
algorithms/routing/
```

功能：

- 洪水传播模拟
- 汇流计算
- 河道分析

---

## 5 SegFormer 专题识别

插件目录：

```
plugins/segformer_plugin/
```

算法目录：

```
algorithms/segformer_service/
```

功能：

- 水体识别
- 积雪识别
- 语义分割

特点：

- 使用 SegFormer 模型
- 独立 AI 推理环境

---

## 6 积雪状态识别

插件目录：

```
plugins/snow_state_plugin/
```

算法目录：

```
algorithms/snow_state/
```

功能：

- 基于 GEE 的积雪物理状态识别
- 融合 Sentinel-1、Sentinel-2、MODIS 与 DEM 数据
- 支持 Google Drive 导出 GeoTIFF 结果
- 支持替换默认遥感数据源 ID

---

## 7 雪水当量估算

插件目录：

```
plugins/swe_plugin/
```

算法目录：

```
algorithms/swe/
```

功能：

- SWE 计算
- 雪区监测

---

## 8 淹没区监测

插件目录：

```
plugins/inundation_monitoring_plugin/
```

算法目录：

```
algorithms/inundation_monitoring/
```

功能：

- SAR 淹没区识别
- UNet 模型
- mask 叠加显示

---

## 9 水资源分配

插件目录：

```
plugins/water_allocation_plugin/
```

算法目录：

```
algorithms/water_allocation/
```

功能：

- NSGA-II 多目标优化
- 水资源调度
- 经济效益优化
- 公平性分析

---

## 10 洪水预警监控

插件目录：

```
plugins/warning_plugin/
```

算法目录：

```
algorithms/warning/
```

功能：

- 洪水预警
- 监控分析
- 决策辅助

---

# 五、运行环境

推荐环境：

- Python 3.10+
- PyQt5
- numpy
- pandas
- matplotlib
- earthengine-api
- rasterio
- opencv-python
- torch
- pymoo

安装示例：

```
pip install pyqt5 numpy pandas matplotlib earthengine-api rasterio opencv-python torch pymoo
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
4. 导入插件
5. 创建 widget
6. 添加标签页

核心文件：

```
app/plugin_manager.py
app/main_window.py
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

### 3 实现插件接口

```python
class ModulePlugin:

    class Plugin(BasePlugin):
        return "模块名称"

    def widget(self):
        return ModuleWidget()
```

---

# 九、项目说明

**VakhshRiverSystem**

瓦赫什河流域水文综合系统。

该系统整合：

- 水文监测
- 洪水模拟
- 水资源调度
- 遥感识别
- AI 模型分析

用于构建流域级综合分析平台。

---

# 十、界面输入提示与格式说明（2026-04-07）

说明：

- 已在有输入项的界面中新增 `i` 说明符号（暗提示）。
- 鼠标悬停在 `i` 或输入控件上，可查看“输入内容 + 数据格式 + 示例”。
- 洪涝风险评估、洪水演进与汇流、雪水当量评估、洪水预警监控等模块当前以按钮/勾选操作为主，不涉及结构化数值录入格式。

## 1 水文监测（`plugins/monitoring_plugin`）

- 水位识别输入图像：单张 `jpg/png/bmp`，建议包含完整水尺刻度和清晰水线
- 流速识别输入视频：`mp4/avi/mov`，建议固定机位、画面稳定、河面纹理清晰
- 实测水面宽度：浮点数，单位 `m`，示例 `15.5`
- 实测河道水深：浮点数，单位 `dm`，示例 `2.3`
- 表面中心流速：浮点数，单位 `m/s`，示例 `1.28`

## 2 淹没区监测（`plugins/inundation_monitoring_plugin`）

- SAR 输入影像：`tif/tiff`（SAR=合成孔径雷达影像，非可见光照片），建议单波段灰度强度图
- 淹没区识别阈值：`0~1` 浮点数，示例 `0.5`

## 3 库区水量估算（`plugins/reservoir_estimation_plugin`）

- 水库名称：下拉选择
- 起始日期：日期格式 `yyyy-MM-dd`，示例 `2022-06-01`
- 结束日期：日期格式 `yyyy-MM-dd`，示例 `2022-06-07`

## 4 SegFormer 专题识别（`plugins/segformer_plugin`）

- 任务：下拉选择（`water` / `snow`）
- 设备：下拉选择（`cpu` / `cuda:0`）
- 图片路径：图像文件路径（`png/jpg/jpeg/bmp`，通过“选择输入图片”填写）

## 5 水资源分配（`plugins/water_allocation_plugin`）

- 选择月份：`1~12` 整数（下拉）
- 大坝当月可供水量：浮点数，单位 `百万m³`
- 区域当月可供其他水：浮点数，单位 `百万m³`
- 人口：浮点数，单位 `万人`
- 城镇化率：`0~100` 浮点数（百分比）
- 当地GDP：浮点数，单位 `亿元`
- 工业重复利用率：`0~100` 浮点数（百分比）
- 灌溉利用系数：`0~1` 浮点数
- 传输损耗率：`0~100` 浮点数（百分比）
- 生态保底水：浮点数，单位 `百万m³`
- 日ET0：浮点数，单位 `mm/天`（自动计算）
- 单机最大功率：浮点数，单位 `MW`
- 单机最大流量：浮点数，单位 `m³/s`
- 上网电价：浮点数，单位 `元/kWh`
- 作物类型：下拉选择
- 生育期：下拉选择
- 作物面积：浮点数，单位 `万亩`
- 作物产量：浮点数，单位 `kg/亩`
- 作物单价：浮点数，单位 `元/kg`
- 部门需水量：浮点数，单位 `百万m³`
- 整体经济权重：`0~1` 浮点数
- 降低缺水权重：`0~1` 浮点数
- 部门公平权重：`0~1` 浮点数
- 气象/水文数据源：`csv/nc` 文件路径或目录路径（可留空）
- 月初初始蓄水量：浮点数，单位 `亿m³`
- 种群规模：正整数
- 迭代代数：正整数
- ET0 气象参数（Rn/G/T/u2/es/ea/delta/gamma）：浮点数
