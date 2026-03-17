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
│  ├─ flood/                          # 洪涝风险评估
│  ├─ inundation_monitoring/          # 淹没区监测（UNet）
│  ├─ monitoring/                     # 水文监测
│  ├─ reservoir_estimation/           # 库区水量估算
│  ├─ routing/                        # 洪水演进与汇流
│  ├─ segformer_service/              # SegFormer推理服务
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

## 6 雪水当量估算

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

## 7 淹没区监测

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

## 8 水资源分配

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

## 9 洪水预警监控

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
- rasterio
- opencv-python
- torch
- pymoo

安装示例：

```
pip install pyqt5 numpy pandas matplotlib rasterio opencv-python torch pymoo
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

    def name(self):
        return "模块名称"

    def widget(self):
        return ModuleWidget()
```

---

# 九、注意事项

开发建议：

- 插件统一使用包路径导入
- 不要创建多个 QApplication
- AI 模块建议独立环境
- 路径建议使用绝对路径
- GUI 与算法分离

---

# 十二、项目说明

**VakhshRiverSystem**

瓦赫什河流域水文综合系统。

该系统整合：

- 水文监测
- 洪水模拟
- 水资源调度
- 遥感识别
- AI 模型分析

用于构建流域级综合分析平台。