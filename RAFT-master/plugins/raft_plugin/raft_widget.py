"""
RAFT / LK river surface velocity measurement plugin widget.

Provides a QWidget interface for optical-flow-based surface velocity estimation
on river video footage, supporting both Lucas-Kanade (LK) sparse and RAFT dense methods.
"""

import os
import sys
import csv
import math
import numpy as np
import cv2
import torch
from datetime import datetime

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog,
    QFrame, QMessageBox, QSpinBox, QDoubleSpinBox, QGroupBox, QComboBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from algorithms.raft.core import run_raft_analysis, load_raft_model


# ---------------------------------------------------------------------------
# Matplotlib canvas helper
# ---------------------------------------------------------------------------
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100, polar=False):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.patch.set_facecolor("#ffffff")
        if polar:
            self.axes = self.fig.add_subplot(111, polar=True)
        else:
            self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)


# ---------------------------------------------------------------------------
# Background analysis worker (QThread)
# ---------------------------------------------------------------------------
class AnalysisWorker(QThread):
    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, video_path, method, height_m, fov_deg, tilt_deg,
                 start_frame, total_frames, model_path):
        super().__init__()
        self.video_path = video_path
        self.method = method
        self.height_m = height_m
        self.fov_deg = fov_deg
        self.tilt_deg = tilt_deg
        self.start_frame = start_frame
        self.total_frames = total_frames
        self.model_path = model_path

    def _process_lk(self):
        """Lucas-Kanade sparse optical flow analysis."""
        cap = cv2.VideoCapture(self.video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame - 1)

        frames = []
        for _ in range(self.total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        if len(frames) < 2:
            raise RuntimeError("提取的有效帧数不足")

        velocities = []
        all_angles = []
        first_pair_old = None
        first_pair_new = None

        for i in range(len(frames) - 1):
            self.progress.emit(i + 1, len(frames) - 1,
                               f"[LK] 处理帧对 {i+1}/{len(frames)-1} ...")

            gray1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)

            p0 = cv2.goodFeaturesToTrack(gray1, maxCorners=1000,
                                         qualityLevel=0.05, minDistance=10,
                                         blockSize=7)
            if p0 is None:
                continue

            p1, st, err = cv2.calcOpticalFlowPyrLK(
                gray1, gray2, p0, None,
                winSize=(21, 21), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
            )
            good_new = p1[st == 1]
            good_old = p0[st == 1]
            if len(good_new) < 150:
                continue

            dx = good_new[:, 0] - good_old[:, 0]
            dy = good_new[:, 1] - good_old[:, 1]
            distances = np.sqrt(dx**2 + dy**2)
            angles_px = np.mod(np.degrees(np.arctan2(dy, dx)), 360)

            dist_mask = distances > 0.2
            angles_filt = angles_px[dist_mask]
            if len(angles_filt) == 0:
                continue

            median_a = np.median(angles_filt)
            angle_diffs = np.abs(angles_filt - median_a)
            angle_diffs = np.minimum(angle_diffs, 360 - angle_diffs)
            angle_mask = angle_diffs < 45.0

            final_old = good_old[dist_mask][angle_mask]
            final_new = good_new[dist_mask][angle_mask]
            final_angles = angles_filt[angle_mask]

            if len(final_old) < 150:
                continue

            # Physical velocity
            frame_h, frame_w = frames[i].shape[:2]
            focal_len = (frame_w / 2.0) / math.tan(math.radians(self.fov_deg / 2.0))
            pitch_rad = math.radians(self.tilt_deg)
            center_y = frame_h / 2.0
            alpha_y = np.arctan(center_y / focal_len)
            gamma = np.maximum(pitch_rad - alpha_y, 0.05)
            Z = self.height_m / np.tan(gamma)
            mpp = Z / focal_len
            avg_dist = float(np.mean(np.sqrt(
                (final_new[:, 0] - final_old[:, 0])**2 +
                (final_new[:, 1] - final_old[:, 1])**2
            )))
            vel = avg_dist * mpp / (1.0 / video_fps)
            velocities.append(vel)
            all_angles.extend(final_angles.tolist())

            if first_pair_old is None:
                first_pair_old = final_old
                first_pair_new = final_new

        if not velocities:
            raise RuntimeError("(LK) 未提取到有效数据")

        return {
            "velocity": float(np.median(velocities)),
            "all_angles": all_angles,
            "lk_old_pts": first_pair_old,
            "lk_new_pts": first_pair_new,
            "fps": video_fps,
        }

    def run(self):
        try:
            if self.method == "RAFT":
                result = run_raft_analysis(
                    video_path=self.video_path,
                    height_m=self.height_m,
                    fov_deg=self.fov_deg,
                    tilt_deg=self.tilt_deg,
                    start_frame=self.start_frame,
                    total_frames=self.total_frames,
                    model_path=self.model_path,
                    progress_callback=lambda i, t, msg: self.progress.emit(i, t, msg),
                )
                if result["status"] == "error":
                    self.error.emit(result["message"])
                    return
                result["method"] = "RAFT"
                self.finished.emit(result)
            else:
                result = self._process_lk()
                result["method"] = "LK"
                self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


# ---------------------------------------------------------------------------
# Main plugin widget
# ---------------------------------------------------------------------------
class RaftWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.video_path = None
        self._current_model_path = "raft-sintel.pth"
        self._worker = None
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # ---- Top control bar ----
        ctrl_row = QHBoxLayout()

        self.btn_upload = QPushButton("1. 上传视频")
        self.btn_upload.clicked.connect(self._select_video)
        ctrl_row.addWidget(self.btn_upload)

        self.cmb_method = QComboBox()
        self.cmb_method.addItems(["RAFT", "LK"])
        ctrl_row.addWidget(QLabel("  方法:"))
        ctrl_row.addWidget(self.cmb_method)

        self.btn_run = QPushButton("2. 开始测速")
        self.btn_run.clicked.connect(self._start_analysis)
        self.btn_run.setEnabled(False)
        ctrl_row.addWidget(self.btn_run)

        ctrl_row.addStretch()

        self.lbl_status = QLabel("请先上传视频文件")
        self.lbl_status.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        ctrl_row.addWidget(self.lbl_status)

        layout.addLayout(ctrl_row)

        # ---- Parameter panel ----
        param_group = QGroupBox("算法及相机环境参数设置")
        param_layout = QHBoxLayout(param_group)
        param_layout.setSpacing(10)

        param_layout.addWidget(QLabel("选用帧数:"))
        self.spin_total = QSpinBox()
        self.spin_total.setRange(2, 99999)
        self.spin_total.setValue(10)
        param_layout.addWidget(self.spin_total)

        param_layout.addWidget(QLabel("相机高度(m):"))
        self.spin_height = QDoubleSpinBox()
        self.spin_height.setRange(0.1, 100.0)
        self.spin_height.setSingleStep(0.5)
        self.spin_height.setValue(4.0)
        param_layout.addWidget(self.spin_height)

        param_layout.addWidget(QLabel("视场角 FOV(°):"))
        self.spin_fov = QDoubleSpinBox()
        self.spin_fov.setRange(1.0, 179.0)
        self.spin_fov.setValue(60.0)
        param_layout.addWidget(self.spin_fov)

        param_layout.addWidget(QLabel("相机俯角(°):"))
        self.spin_tilt = QDoubleSpinBox()
        self.spin_tilt.setRange(0.0, 89.9)
        self.spin_tilt.setValue(35.0)
        param_layout.addWidget(self.spin_tilt)

        param_layout.addStretch()
        layout.addWidget(param_group)

        # ---- Separator ----
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line)

        # ---- Chart area ----
        chart_row = QHBoxLayout()

        chart_left = QVBoxLayout()
        self.lbl_chart_left = QLabel("特征运动情况概览")
        self.lbl_chart_left.setAlignment(Qt.AlignCenter)
        self.canvas_left = MplCanvas(self)
        chart_left.addWidget(self.lbl_chart_left)
        chart_left.addWidget(self.canvas_left)

        chart_right = QVBoxLayout()
        self.lbl_chart_right = QLabel("综合玫瑰流向特征图")
        self.lbl_chart_right.setAlignment(Qt.AlignCenter)
        self.canvas_right = MplCanvas(self, polar=True)
        chart_right.addWidget(self.lbl_chart_right)
        chart_right.addWidget(self.canvas_right)

        chart_row.addLayout(chart_left, 6)
        chart_row.addLayout(chart_right, 4)
        layout.addLayout(chart_row)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------
    def _select_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "",
            "Video Files (*.mp4 *.avi *.mov)")
        if path:
            self.video_path = path
            self.lbl_status.setText(f"已选中视频: {os.path.basename(path)}")
            self.btn_run.setEnabled(True)

    def _start_analysis(self):
        if not self.video_path:
            QMessageBox.warning(self, "提示", "请先选择视频文件")
            return

        method = self.cmb_method.currentText()
        start_frame = 2
        total_frames = self.spin_total.value()
        height_m = self.spin_height.value()
        fov_deg = self.spin_fov.value()
        tilt_deg = self.spin_tilt.value()

        self.btn_upload.setEnabled(False)
        self.btn_run.setEnabled(False)
        self.cmb_method.setEnabled(False)
        self.lbl_status.setText(f"[{method}] 分析中，请稍候...")

        self._worker = AnalysisWorker(
            self.video_path, method, height_m, fov_deg, tilt_deg,
            start_frame, total_frames, self._current_model_path
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_progress(self, current, total, msg):
        self.lbl_status.setText(msg)

    def _on_error(self, err_msg):
        self._reenable_controls()
        self.lbl_status.setText("分析失败")
        QMessageBox.critical(self, "错误", err_msg)

    def _on_finished(self, results):
        self._reenable_controls()

        method = results.get("method", "?")
        vel = results.get("velocity", 0)
        self.lbl_status.setText(f"[{method}] 流速测算完毕 {vel:.4f} m/s")

        # ---- Left chart: flow visualization ----
        ax1 = self.canvas_left.axes
        ax1.clear()

        if method == "RAFT":
            flow_rgb = results.get("flow_rgb")
            if flow_rgb is not None:
                ax1.imshow(flow_rgb)
                ax1.set_xticks([])
                ax1.set_yticks([])
                self.lbl_chart_left.setText(f"[RAFT] 密集光流图")
            else:
                ax1.text(0.5, 0.5, "无光流数据", transform=ax1.transAxes,
                         ha="center", va="center")
        else:
            old_pts = results.get("lk_old_pts")
            new_pts = results.get("lk_new_pts")
            if old_pts is not None and new_pts is not None:
                ax1.set_facecolor("black")
                ax1.invert_yaxis()
                for pt1, pt2 in zip(old_pts, new_pts):
                    ax1.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]],
                             color="white", linewidth=0.5, alpha=0.7)
                ax1.scatter(old_pts[:, 0], old_pts[:, 1],
                            c="#42A5F5", s=15, zorder=5, label="Frame 1")
                ax1.scatter(new_pts[:, 0], new_pts[:, 1],
                            c="#EF5350", s=15, zorder=5, label="Frame 2")
                ax1.set_xticks([])
                ax1.set_yticks([])
                self.lbl_chart_left.setText(f"[LK] 前两帧特征匹配情况")

        self.canvas_left.draw()

        # ---- Right chart: rose diagram ----
        ax2 = self.canvas_right.axes
        ax2.clear()
        angles = results.get("all_angles", [])
        if len(angles) > 0:
            angles_rad = np.radians(angles)
            bins = np.linspace(0.0, 2 * np.pi, 37)
            counts, _ = np.histogram(angles_rad, bins)
            width = 2 * np.pi / 36
            bars = ax2.bar(bins[:-1], counts, width=width, bottom=0.0)
            for bar in bars:
                bar.set_facecolor("#1E88E5")
                bar.set_edgecolor("white")
                bar.set_alpha(0.8)
            ax2.set_theta_zero_location("N")
            ax2.set_theta_direction(-1)
            ax2.set_yticklabels([])
            self.lbl_chart_right.setText(f"[{method}] 综合玫瑰流向特征图")
        else:
            ax2.text(0.5, 0.5, "无角度数据", transform=ax2.transAxes,
                     ha="center", va="center")

        self.canvas_right.draw()

    def _reenable_controls(self):
        self.btn_upload.setEnabled(True)
        self.btn_run.setEnabled(True)
        self.cmb_method.setEnabled(True)
