import sys
import os
import cv2
import numpy as np
import torch
import math
import csv
from datetime import datetime
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, QFrame,
                             QMessageBox, QSpinBox, QDoubleSpinBox, QGroupBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORE_DIR = Path(__file__).resolve().parent / "core"
sys.path.insert(0, str(CORE_DIR))
try:
    from raft import RAFT
    from utils.utils import InputPadder
    from utils import flow_viz
except ImportError:
    print("[警告] 未找到 RAFT core 模块，RAFT 测速功能可能无法运行。")

# =====================================================================
# 全局基本配置
# =====================================================================
CSV_FILE_PATH = str(PROJECT_ROOT / "output" / "flow_measurements.csv")
DEVICE_ID = 'DEV_CAM_001'
MODEL_PATH = str(PROJECT_ROOT / "models" / "raft-sintel.pth")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Args:
    """模拟 RAFT 原版的命令行参数"""

    def __init__(self):
        self.small = False
        self.mixed_precision = False
        self.alternate_corr = False
        self.dropout = 0.0

    def __contains__(self, key):
        return hasattr(self, key)


# =====================================================================
# 算法核心与后台线程 (防止阻塞UI)
# =====================================================================
class VideoAnalyzerWorker(QThread):
    finished_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)
    status_signal = pyqtSignal(str)

    def __init__(self, video_path, method, start_frame, total_frames, height_m, fov_deg, tilt_deg):
        super().__init__()
        self.video_path = video_path
        self.method = method
        self.start_frame = start_frame  # 此时这里接收到的永远是 2
        self.total_frames = total_frames
        self.height_m = height_m
        self.fov_deg = fov_deg
        self.tilt_deg = tilt_deg
        self.fps = 30.0  # 默认帧率，后续会动态从视频读取

    # ------------------ LK 核心逻辑 ------------------
    def process_lk_pair(self, frame1, frame2):
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        feature_params = dict(maxCorners=1000, qualityLevel=0.05, minDistance=10, blockSize=7)
        p0 = cv2.goodFeaturesToTrack(gray1, mask=None, **feature_params)

        if p0 is None:
            return None

        lk_params = dict(winSize=(21, 21), maxLevel=3,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        p1, st, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, p0, None, **lk_params)

        good_new = p1[st == 1]
        good_old = p0[st == 1]

        dx = good_new[:, 0] - good_old[:, 0]
        dy = good_new[:, 1] - good_old[:, 1]

        distances = np.sqrt(dx ** 2 + dy ** 2)
        angles = np.mod(np.degrees(np.arctan2(dy, dx)), 360)

        dist_mask = distances > 0.2
        angles_filtered = angles[dist_mask]

        if len(angles_filtered) == 0:
            return None

        median_angle = np.median(angles_filtered)
        angle_diffs = np.abs(angles_filtered - median_angle)
        angle_diffs = np.minimum(angle_diffs, 360 - angle_diffs)
        angle_mask = angle_diffs < 45.0

        final_old = good_old[dist_mask][angle_mask]
        final_new = good_new[dist_mask][angle_mask]
        final_angles = angles_filtered[angle_mask]
        valid_points_count = len(final_old)

        if valid_points_count == 0:
            return None

        avg_physical_dist = self.calculate_physical_distance(final_old, final_new, frame1.shape)
        velocity_m_s = avg_physical_dist / (1.0 / self.fps)

        return {
            "velocity": velocity_m_s,
            "pts_count": valid_points_count,
            "old_pts": final_old,
            "new_pts": final_new,
            "angles": final_angles
        }

    # ------------------ RAFT 核心逻辑 ------------------
    def process_raft_pair(self, model, frame1, frame2):
        def prepare_image(frame):
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
            return img_tensor.unsqueeze(0).to(DEVICE)

        img1 = prepare_image(frame1)
        img2 = prepare_image(frame2)

        padder = InputPadder(img1.shape)
        img1_pad, img2_pad = padder.pad(img1, img2)

        with torch.no_grad():
            _, flow_up = model(img1_pad, img2_pad, iters=20, test_mode=True)

        flow_up = padder.unpad(flow_up)
        flow_np = flow_up[0].permute(1, 2, 0).cpu().numpy()

        flow_rgb = flow_viz.flow_to_image(flow_np)

        height, width = frame1.shape[:2]
        flow_u = flow_np[..., 0]
        flow_v = flow_np[..., 1]

        U, V = np.meshgrid(np.arange(width), np.arange(height))
        pixel_distances = np.sqrt(flow_u ** 2 + flow_v ** 2)
        angles = np.mod(np.degrees(np.arctan2(flow_v, flow_u)), 360)

        dist_mask = pixel_distances > 0.2
        valid_angles = angles[dist_mask]

        if len(valid_angles) == 0:
            return None

        median_angle = np.median(valid_angles)
        angle_diffs = np.abs(angles - median_angle)
        angle_diffs = np.minimum(angle_diffs, 360 - angle_diffs)
        angle_mask = angle_diffs < 45.0

        final_mask = dist_mask & angle_mask
        valid_points_count = np.sum(final_mask)

        if valid_points_count == 0:
            return None

        U1_valid = U[final_mask]
        V1_valid = V[final_mask]
        U2_valid = U1_valid + flow_u[final_mask]
        V2_valid = V1_valid + flow_v[final_mask]

        old_pts_all = np.column_stack((U1_valid, V1_valid))
        new_pts_all = np.column_stack((U2_valid, V2_valid))

        avg_physical_dist = self.calculate_physical_distance(old_pts_all, new_pts_all, frame1.shape)
        velocity_m_s = avg_physical_dist / (1.0 / self.fps)

        return {
            "velocity": velocity_m_s,
            "pts_count": valid_points_count,
            "flow_rgb": flow_rgb,
            "angles": angles[final_mask]
        }

    # ------------------ 公共物理测算逻辑 ------------------
    def calculate_physical_distance(self, old_pts, new_pts, frame_shape):
        frame_height, frame_width = frame_shape[:2]
        focal_length = (frame_width / 2.0) / math.tan(math.radians(self.fov_deg / 2.0))
        pitch_rad = math.radians(self.tilt_deg)

        def pixel_to_meters(x, y):
            y_offset = (frame_height / 2.0) - y
            alpha_y = np.arctan(y_offset / focal_length)
            gamma = pitch_rad - alpha_y
            gamma = np.maximum(gamma, 0.05)
            Z = self.height_m / np.tan(gamma)
            x_offset = x - (frame_width / 2.0)
            X = (self.height_m / np.sin(gamma)) * (x_offset / focal_length)
            return X, Z

        X1, Z1 = pixel_to_meters(old_pts[:, 0], old_pts[:, 1])
        X2, Z2 = pixel_to_meters(new_pts[:, 0], new_pts[:, 1])
        distances_m = np.sqrt((X2 - X1) ** 2 + (Z2 - Z1) ** 2)
        return np.mean(distances_m)

    def append_to_csv(self, final_vel):
        file_exists = os.path.isfile(CSV_FILE_PATH)
        next_id = 1
        if file_exists:
            try:
                with open(CSV_FILE_PATH, 'r', encoding='utf-8') as f:
                    lines = list(csv.reader(f))
                    if len(lines) > 1 and lines[-1][0].isdigit():
                        next_id = int(lines[-1][0]) + 1
            except Exception:
                pass

        now = datetime.now().astimezone()
        timestamp_str = now.isoformat(timespec='seconds')

        try:
            with open(CSV_FILE_PATH, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(['ID', 'Device_ID', 'Measurement_Result', 'Unit', 'Timestamp', 'Algorithm'])
                writer.writerow([next_id, DEVICE_ID, round(final_vel, 4), 'm/s', timestamp_str, self.method])
        except Exception as e:
            self.error_signal.emit(f"CSV写入失败: {str(e)}")

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.error_signal.emit("无法打开视频文件。")
            return

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps and video_fps > 0:
            self.fps = video_fps
        else:
            self.fps = 30.0  # 异常回退机制

        self.status_signal.emit(f"成功识别视频帧率: {self.fps:.2f} FPS")

        # 始终从第二帧开始 (OpenCV帧索引从0开始，所以第2帧是 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame - 1)
        frames = []

        # 🚀 根据总帧数提取数据
        for _ in range(self.total_frames):
            ret, frame = cap.read()
            if not ret: break
            frames.append(frame)
        cap.release()

        if len(frames) < 2:
            self.error_signal.emit("提取的有效帧数不足，无法分析。")
            return

        velocities = []
        all_angles = []
        first_pair_data = None
        model = None

        if self.method == 'RAFT':
            self.status_signal.emit("正在加载 RAFT 模型...")
            try:
                args = Args()
                model = RAFT(args)
                state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
                new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                model.load_state_dict(new_state_dict)
                model.to(DEVICE)
                model.eval()
            except Exception as e:
                self.error_signal.emit(f"RAFT模型加载失败: {str(e)}")
                return

        MIN_POINTS_THRESHOLD = 150 if self.method == 'LK' else 1000

        for i in range(len(frames) - 1):
            self.status_signal.emit(f"正在处理帧对比序列 {i + 1}/{len(frames) - 1} ...")
            if self.method == 'LK':
                res = self.process_lk_pair(frames[i], frames[i + 1])
            else:
                res = self.process_raft_pair(model, frames[i], frames[i + 1])

            if res and res["pts_count"] >= MIN_POINTS_THRESHOLD:
                velocities.append(res["velocity"])
                all_angles.extend(res["angles"])

                if first_pair_data is None:
                    if self.method == 'LK':
                        first_pair_data = (res["old_pts"], res["new_pts"])
                    elif self.method == 'RAFT':
                        first_pair_data = res["flow_rgb"]

        if not velocities:
            self.error_signal.emit(f"({self.method}) 未提取到有效数据，特征点数可能低于阈值。")
            return

        final_vel = np.median(velocities)
        self.append_to_csv(final_vel)

        self.finished_signal.emit({
            "velocity": final_vel,
            "first_pair_data": first_pair_data,
            "all_angles": all_angles,
            "method": self.method
        })


# =====================================================================
# UI 界面与绘图组件
# =====================================================================
class MatplotlibCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100, is_polar=False):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.patch.set_facecolor('#ffffff')

        # 🚀 优化图表留白边距 (减少白边)
        if is_polar:
            self.axes = self.fig.add_subplot(111, polar=True)
            self.fig.subplots_adjust(left=0.12, right=0.92, top=0.92, bottom=0.08)
        else:
            self.axes = self.fig.add_subplot(111)
            self.fig.subplots_adjust(left=0.05, right=0.98, top=0.98, bottom=0.02)

        super(MatplotlibCanvas, self).__init__(self.fig)


class FlowAnalysisApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("综合水流表面测速与特征分析系统")
        self.resize(1100, 700)
        self.video_path = None
        self.initUI()
        self.apply_stylesheet()

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 顶部控制区
        control_layout = QHBoxLayout()

        self.btn_upload = QPushButton("1. 上传视频")
        self.btn_upload.clicked.connect(self.select_video)

        self.btn_lk = QPushButton("2. LK稀疏光流测速")
        self.btn_lk.clicked.connect(self.run_lk)
        self.btn_lk.setEnabled(False)

        self.btn_raft = QPushButton("3. RAFT密集光流测速")
        self.btn_raft.clicked.connect(self.run_raft)
        self.btn_raft.setEnabled(False)

        self.lbl_result = QLabel("请先上传视频文件")
        self.lbl_result.setFont(QFont("Arial", 16, QFont.Bold))  # 🚀 修复字号：由13改为16
        self.lbl_result.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        control_layout.addWidget(self.btn_upload)
        control_layout.addWidget(self.btn_lk)
        control_layout.addWidget(self.btn_raft)
        control_layout.addStretch()
        control_layout.addWidget(self.lbl_result)

        main_layout.addLayout(control_layout)

        # 参数设置面板
        self.param_group = QGroupBox("算法及相机环境参数设置")
        param_layout = QHBoxLayout(self.param_group)
        param_layout.setSpacing(12)  # 🚀 优化间距：为内部元素设置合理的间距

        # 🚀 已删除起始帧输入框，保留提取总帧数和其他物理参数
        param_layout.addWidget(QLabel("选用帧数:"))
        self.spin_total = QSpinBox()
        self.spin_total.setRange(2, 99999)
        self.spin_total.setValue(10)
        param_layout.addWidget(self.spin_total)

        # 可选：在元素对之间加上一点固定间距让视觉更清晰
        param_layout.addSpacing(10)

        param_layout.addWidget(QLabel("相机高度(m):"))
        self.spin_height = QDoubleSpinBox()
        self.spin_height.setRange(0.1, 100.0)
        self.spin_height.setSingleStep(0.5)
        self.spin_height.setValue(4.0)
        param_layout.addWidget(self.spin_height)

        param_layout.addSpacing(10)

        param_layout.addWidget(QLabel("视场角 FOV(°):"))
        self.spin_fov = QDoubleSpinBox()
        self.spin_fov.setRange(1.0, 179.0)
        self.spin_fov.setValue(60.0)
        param_layout.addWidget(self.spin_fov)

        param_layout.addSpacing(10)

        param_layout.addWidget(QLabel("相机俯角(°):"))
        self.spin_tilt = QDoubleSpinBox()
        self.spin_tilt.setRange(0.0, 89.9)
        self.spin_tilt.setValue(35.0)
        param_layout.addWidget(self.spin_tilt)

        # 🚀 核心修复：在此处加上弹簧，吸收右侧多余空间，防止左侧控件被强制拉开！
        param_layout.addStretch()

        main_layout.addWidget(self.param_group)

        # 分割线
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(line)

        # 底部图表区
        chart_layout = QHBoxLayout()

        plot_vbox1 = QVBoxLayout()
        self.lbl_plot1 = QLabel("特征运动情况概览")
        self.lbl_plot1.setAlignment(Qt.AlignCenter)
        self.canvas_match = MatplotlibCanvas(self)
        plot_vbox1.addWidget(self.lbl_plot1)
        plot_vbox1.addWidget(self.canvas_match)

        plot_vbox2 = QVBoxLayout()
        self.lbl_plot2 = QLabel("综合玫瑰流向特征图")
        self.lbl_plot2.setAlignment(Qt.AlignCenter)
        self.canvas_rose = MatplotlibCanvas(self, is_polar=True)
        plot_vbox2.addWidget(self.lbl_plot2)
        plot_vbox2.addWidget(self.canvas_rose)

        chart_layout.addLayout(plot_vbox1, 6)
        chart_layout.addLayout(plot_vbox2, 4)
        main_layout.addLayout(chart_layout)

    def apply_stylesheet(self):
        # 🚀 全局字号上调修复：修改了下方的 font-size 参数
        qss = """
        QMainWindow {
            background-color: #F0F4F8;
        }
        QGroupBox {
            font-size: 16px;
            font-weight: bold;
            color: #0D47A1;
            border: 1px solid #90CAF9;
            border-radius: 6px;
            margin-top: 15px;
            padding-top: 15px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 3px 0 3px;
        }
        QPushButton {
            background-color: #1E88E5;
            color: white;
            border-radius: 4px;
            padding: 10px 15px;
            font-size: 17px;  /* 从 14px 改为 17px */
            font-weight: bold;
            border: none;
        }
        QPushButton:hover {
            background-color: #1565C0;
        }
        QPushButton:disabled {
            background-color: #90CAF9;
            color: #E3F2FD;
        }
        QLabel {
            color: #0D47A1;
            font-size: 16px;  /* 从 13px 改为 16px */
        }
        QSpinBox, QDoubleSpinBox {
            border: 1px solid #90CAF9;
            padding: 4px;
            border-radius: 4px;
            background: white;
            color: #0D47A1;
            font-size: 16px;  /* 新增输入框字号设定 */
        }
        """
        self.setStyleSheet(qss)

    def select_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "Video Files (*.mp4 *.avi *.mov)")
        if file_path:
            self.video_path = file_path
            self.lbl_result.setText(f"已选中视频: {os.path.basename(file_path)}")
            self.btn_lk.setEnabled(True)
            self.btn_raft.setEnabled(True)

    def run_lk(self):
        self.start_analysis('LK')

    def run_raft(self):
        self.start_analysis('RAFT')

    def start_analysis(self, method):
        # 🚀 强制锁定起始帧为 2
        start_frame = 2
        total_frames = self.spin_total.value()
        height_m = self.spin_height.value()
        fov_deg = self.spin_fov.value()
        tilt_deg = self.spin_tilt.value()

        self.btn_upload.setEnabled(False)
        self.btn_lk.setEnabled(False)
        self.btn_raft.setEnabled(False)
        self.param_group.setEnabled(False)
        self.lbl_result.setText(f"[{method}] 分析中，请稍候...")

        self.worker = VideoAnalyzerWorker(
            self.video_path, method, start_frame, total_frames, height_m, fov_deg, tilt_deg
        )
        self.worker.status_signal.connect(self.on_analysis_status)
        self.worker.finished_signal.connect(self.on_analysis_finished)
        self.worker.error_signal.connect(self.on_analysis_error)
        self.worker.start()

    def on_analysis_status(self, msg):
        self.lbl_result.setText(msg)

    def on_analysis_error(self, err_msg):
        self.btn_upload.setEnabled(True)
        self.btn_lk.setEnabled(True)
        self.btn_raft.setEnabled(True)
        self.param_group.setEnabled(True)
        self.lbl_result.setText("分析失败")
        QMessageBox.critical(self, "错误", err_msg)

    def on_analysis_finished(self, results):
        self.btn_upload.setEnabled(True)
        self.btn_lk.setEnabled(True)
        self.btn_raft.setEnabled(True)
        self.param_group.setEnabled(True)

        method = results["method"]
        vel = results["velocity"]
        self.lbl_result.setText(f"[{method}] 流速测算完毕: {vel:.4f} m/s")
        self.lbl_plot2.setText(f"[{method}] 综合玫瑰流向特征图")

        # 1. 绘制图像展示区
        ax1 = self.canvas_match.axes
        ax1.clear()

        first_pair = results.get("first_pair_data")
        if first_pair is not None:
            if method == 'LK':
                ax1.set_facecolor('black')
                old_pts, new_pts = first_pair
                ax1.invert_yaxis()
                for pt1, pt2 in zip(old_pts, new_pts):
                    ax1.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color='white', linewidth=0.5, alpha=0.7)
                ax1.scatter(old_pts[:, 0], old_pts[:, 1], c='#42A5F5', s=15, zorder=5, label='Frame 1')
                ax1.scatter(new_pts[:, 0], new_pts[:, 1], c='#EF5350', s=15, zorder=5, label='Frame 2')
                self.lbl_plot1.setText(f"[{method}] 前两帧特征匹配情况")

            elif method == 'RAFT':
                ax1.imshow(first_pair)
                self.lbl_plot1.setText(f"[{method}] 前两帧稠密光流图")

            ax1.set_xticks([])
            ax1.set_yticks([])

        self.canvas_match.draw()

        # 2. 绘制玫瑰流向图
        ax2 = self.canvas_rose.axes
        ax2.clear()
        angles = results.get("all_angles", [])
        if len(angles) > 0:
            angles_rad = np.radians(angles)
            bins = np.linspace(0.0, 2 * np.pi, 37)
            counts, _ = np.histogram(angles_rad, bins)
            width = 2 * np.pi / 36
            bars = ax2.bar(bins[:-1], counts, width=width, bottom=0.0)

            for bar in bars:
                bar.set_facecolor('#1E88E5')
                bar.set_edgecolor('white')
                bar.set_alpha(0.8)

            ax2.set_theta_zero_location("N")
            ax2.set_theta_direction(-1)
            ax2.set_yticklabels([])

        self.canvas_rose.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FlowAnalysisApp()
    window.show()
    sys.exit(app.exec_())
