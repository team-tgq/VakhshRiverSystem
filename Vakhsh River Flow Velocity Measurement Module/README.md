# Vakhsh River Flow Velocity Measurement Module

本项目已整理为插件式结构，可通过 `python main.py` 启动主程序，并在“水文监测”插件中使用原有 LK/RAFT 河流表面流速测量功能。

## 项目结构

```text
algorithms/
  monitoring/
    flow_velocity_app.py
    core/
    alt_cuda_corr/
app/
  base_plugin.py
  main_window.py
  plugin_manager.py
plugins/
  monitoring_plugin/
    plugin.py
    plugin.json
    flow_velocity_widget.py
data/
  sample/
models/
output/
requirements/
config.py
main.py
README.md
```

## 运行

```bash
conda env create -f requirements/environment-vakhsh.yml
conda activate VakhshRiverSystem
python main.py
```

或在已有 Python 环境中安装 pip 依赖：

```bash
pip install -r requirements/requirements-vakhsh.txt
python main.py
```

RAFT 权重文件位于 `models/raft-sintel.pth`，示例视频位于 `data/sample/清水河.mp4`。运行结果 CSV 默认写入 `output/flow_measurements.csv`。
