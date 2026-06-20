# RAFT 光流测速模块

## 模块概述

基于 **RAFT (Recurrent All-Pairs Field Transforms)** 密集光流算法的河流表面流速测量模块。通过分析固定机位拍摄的河流视频，逐帧计算像素级光流场，结合相机安装参数（高度、视场角、俯角）将像素位移转换为物理流速（m/s），并绘制流向玫瑰图。

同时保留 **Lucas-Kanade (LK) 稀疏光流** 作为轻量备选方案。

---

## 目录结构

```
algorithms/raft/
├── __init__.py          # 模块导出
├── core.py              # 纯算法入口 — run_raft_analysis()
├── raft_model.py        # RAFT 神经网络模型定义
├── extractor.py         # 特征编码器 (BasicEncoder / SmallEncoder)
├── update.py            # 迭代更新模块 (ConvGRU + FlowHead)
├── corr.py              # 全对相关性计算 + 可选 CUDA 加速
├── datasets.py          # 训练数据集加载器 (Sintel / KITTI / HD1K 等)
└── utils/
    ├── __init__.py
    ├── augmentor.py     # 数据增强 (训练用)
    ├── flow_viz.py      # 光流可视化 (颜色编码)
    ├── frame_utils.py   # 帧文件读写 (.flo / .pfm / .png)
    └── utils.py         # InputPadder, bilinear_sampler, coords_grid

plugins/raft_plugin/
├── __init__.py
├── plugin.json          # 插件清单
├── plugin.py            # BasePlugin 注册
└── raft_widget.py       # PyQt5 界面控件

raft-sintel.pth          # 预训练模型权重 (Sintel 数据集)
alt_cuda_corr/           # 可选 CUDA 加速扩展
```

---

## 分层说明

本模块严格遵循 VakhshRiverSystem 的三层架构：

| 层 | 位置 | 职责 |
|---|---|---|
| **算法层** | `algorithms/raft/core.py` | `run_raft_analysis()` 纯函数，接收视频路径和参数，返回结果 dict。**不依赖任何 GUI**。 |
| **插件层** | `plugins/raft_plugin/raft_widget.py` | QWidget 界面：视频选择 → 方法切换 → 参数输入 → 调用算法 → 结果展示。 |
| **注册层** | `plugins/raft_plugin/plugin.py` | 继承 `BasePlugin`，暴露 name() 和 widget()。 |

### 算法层 API

```python
from algorithms.raft.core import run_raft_analysis, load_raft_model

result = run_raft_analysis(
    video_path="path/to/video.mp4",   # 输入视频
    height_m=4.0,                     # 相机安装高度 (m)
    fov_deg=60.0,                     # 水平视场角 (°)
    tilt_deg=35.0,                    # 相机俯角, 0=水平 90=垂直向下 (°)
    start_frame=2,                    # 起始帧 (1-based)
    total_frames=10,                  # 提取帧数
    model_path="raft-sintel.pth",     # 模型权重路径
    device=None,                      # None=自动检测 CPU/GPU
    progress_callback=None,           # 可选进度回调 fn(i, total, msg)
)

# 返回格式
{
    "status": "success",            # 或 "error"
    "velocity": 1.284,             # 中值流速 (m/s)
    "all_angles": [45.2, 47.1, ...],  # 有效流向角度列表
    "flow_rgb": np.ndarray,        # 光流可视化图 (H, W, 3)
    "fps": 30.0,                   # 视频帧率
    "device": "cuda",              # 推理设备
    "frame_count": 10,             # 实际提取帧数
    "valid_pairs": 8,              # 有效帧对数量
    "summary": "RAFT 测速完成 1.2840 m/s (基于8个有效帧对)",
}
```

---

## 环境依赖

- `torch` ≥ 1.6（推荐 CUDA 版本以启用 GPU 推理）
- `torchvision`
- `opencv-python` (cv2)
- `numpy`
- `scipy`
- `matplotlib`
- `pillow`

以上均在 VakhshRiverSystem 主环境中提供。无额外第三方依赖。

### 可选 CUDA 加速

`alt_cuda_corr/` 提供 Alternate Correlation 的 CUDA 实现，仅在设置 `alternate_corr=True` 时需要：

```bash
cd alt_cuda_corr
python setup.py install
```

默认不使用此路径，模块会自动回退到纯 PyTorch 实现。

---

## 界面参数说明

插件标签页中提供以下输入项：

| 参数 | 控件 | 单位 | 默认值 | 说明 |
|---|---|---|---|---|
| 上传视频 | 按钮 | — | — | 选择 mp4/avi/mov |
| 方法 | 下拉 | — | RAFT | RAFT (密集) / LK (稀疏) |
| 选用帧数 | 数字框 | 帧 | 10 | 从起始帧开始提取 |
| 相机高度 | 浮点框 | m | 4.0 | 相机离水面垂直高度 |
| 视场角 FOV | 浮点框 | ° | 60.0 | 相机水平视场角 |
| 相机俯角 | 浮点框 | ° | 35.0 | 0=水平, 90=垂直向下 |

输出区域：
- **左侧**：RAFT 模式下显示密集光流可视化图，LK 模式下显示特征点匹配连线。
- **右侧**：综合玫瑰流向图（极坐标直方图），展示水流方向的分布。

---

## 工作原理简述

### RAFT 密集光流

1. **特征提取**：BasicEncoder 对两帧图像分别提取 1/8 分辨率特征图。
2. **相关性计算**：对特征图的所有像素对计算相关性，构建 4 层相关性金字塔。
3. **迭代更新**：通过 ConvGRU 迭代优化光流场（默认 20 次），每次从相关性金字塔中查找匹配。
4. **上采样**：将 1/8 分辨率的光流凸组合上采样回原始分辨率。

### 像素到物理流速转换

使用针孔相机模型：

```
focal_length = (width / 2) / tan(FOV / 2)
gamma = tilt - arctan(y_offset / focal_length)
Z = height / tan(gamma)
meters_per_pixel = Z / focal_length
velocity = pixel_distance × meters_per_pixel / (1 / fps)
```

### 异常值过滤

- 像素位移 < 0.2 px 的点直接丢弃
- 角度偏差 > 45° (相对中值) 的点视为噪声丢弃
- 有效点数 < 1000 的帧对跳过
- 最终流速取所有有效帧对的中值

---

## 参考

- RAFT 原始论文：Teed & Deng, "RAFT: Recurrent All-Pairs Field Transforms for Optical Flow" (ECCV 2020)
- 官方实现：https://github.com/princeton-vl/RAFT
- 光流可视化：Baker et al., "A Database and Evaluation Methodology for Optical Flow" (ICCV 2007)

---

## 维护说明

- 如需更新模型权重，替换根目录下的 `raft-sintel.pth` 并在 `core.py` 中验证 `state_dict` 键名兼容性。
- 算法层 (`core.py`) 和插件层 (`raft_widget.py`) 通过 dict 接口解耦，修改算法逻辑时不影响界面。
- 插件通过 `PluginManager` 自动发现，无需修改 `main.py` 或 `app/` 中任何文件。
