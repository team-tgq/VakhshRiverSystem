# plugins/monitoring_plugin/monitoring_widget.py
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
MONITORING_DIR = os.path.join(PROJECT_ROOT, "algorithms", "monitoring")

if MONITORING_DIR not in sys.path:
    # Use project's customized ultralytics before site-packages.
    sys.path.insert(0, MONITORING_DIR)
import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QFileDialog, QMessageBox, QGroupBox, QTableWidget,
    QTableWidgetItem, QHeaderView, QProgressBar, QDialog,
    QTextEdit, QStackedWidget, QFormLayout, QGridLayout, QSizePolicy
)
from PyQt5.QtGui import QPixmap, QFont, QDoubleValidator
from PyQt5.QtCore import Qt

from algorithms.monitoring.engine import (
    math_waterlevel as calculator,
    enhanced as preprocessor,
    predict_seg,
    predict_number
)
from algorithms.monitoring.engine.waterSpeed import calculate_velocity_for_ui
from app.ui_hints import attach_hint, create_hint_badge, label_with_hint


THEME_COLOR = "#0078d7"
BG_COLOR = "#f4f7fc"
PANEL_BG = "#ffffff"
TEXT_COLOR = "#333333"

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

STYLESHEET = f"""
    QWidget {{ background-color: {BG_COLOR}; }}
    QGroupBox {{
        font-weight: bold; border: 1px solid #d1d9e6; border-radius: 6px;
        margin-top: 12px; background-color: {PANEL_BG}; color: {THEME_COLOR};
    }}
    QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 5px; }}
    QPushButton {{
        background-color: {THEME_COLOR}; color: white; border-radius: 4px; padding: 8px; font-weight: bold;
    }}
    QPushButton:hover {{ background-color: #005a9e; }}
    QPushButton:pressed {{ background-color: #004275; }}
    QPushButton.NavBtn {{
        background-color: transparent; color: #555555; text-align: left;
        padding: 15px 20px; font-size: 16px; border-radius: 0px; border-left: 5px solid transparent;
    }}
    QPushButton.NavBtn:hover {{ background-color: #e6f0fa; }}
    QPushButton.NavBtn[active="true"] {{
        background-color: #e6f0fa; color: {THEME_COLOR}; border-left: 5px solid {THEME_COLOR}; font-weight: bold;
    }}
    QTableWidget {{
        border: 1px solid #e0e0e0; gridline-color: #f0f0f0; background-color: white;
        selection-background-color: #e6f7ff; selection-color: black;
    }}
    QHeaderView::section {{
        background-color: {THEME_COLOR}; color: white; padding: 4px; border: none; font-weight: bold;
    }}
    QProgressBar {{ border: 1px solid #ccc; border-radius: 4px; text-align: center; background-color: #eee; }}
    QProgressBar::chunk {{ background-color: {THEME_COLOR}; border-radius: 3px; }}
    QTextEdit {{
        background-color: #2b2b2b; color: #00ff00; font-family: 'Consolas'; border-radius: 6px; border: 1px solid #555;
    }}
"""


class AnalysisDialog(QDialog):
    def __init__(self, current_level, data_file):
        super().__init__()
        self.setWindowTitle("智能水位分析报告")
        self.resize(1000, 700)
        self.setStyleSheet(f"background-color: {BG_COLOR};")
        self.current_level = current_level
        self.data_file = data_file

        main_layout = QVBoxLayout()
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)

        self.init_evaluation_panel(main_layout)
        self.init_trend_panel(main_layout)
        self.setLayout(main_layout)

    def init_evaluation_panel(self, parent_layout):
        gb_eval = QGroupBox("当前水位智能评价")
        gb_eval.setMinimumHeight(220)
        layout = QHBoxLayout()

        status_widget = QWidget()
        status_widget.setStyleSheet("background-color: white; border-radius: 8px;")
        v_layout = QVBoxLayout(status_widget)

        if self.current_level is None:
            title, color, desc, level_str = "未知状态", "#999999", "未获取当前水位，请先在主界面运行监测。", "--"
        else:
            level = self.current_level
            level_str = f"{level:.2f} dm"
            if 0 <= level < 0.1:
                title, color, desc = "干涸 (Dry)", "#d9534f", "当前处于极度缺水状态，建议停止取水并进行生态补水。"
            elif 0.1 <= level < 1:
                title, color, desc = "低水位干涸预警", "#f0ad4e", "水位显著低于正常水平，可能影响航运及生态。"
            elif 1 <= level < 3:
                title, color, desc = "正常水位 (Normal)", "#5cb85c", "水位处于标准运行区间，水文状况良好。"
            elif 3 <= level < 5:
                title, color, desc = "高水位预警", "#f0ad4e", "水位较高，接近警戒线，请注意防汛安全。"
            elif level >= 5:
                title, color, desc = "洪涝 (Flood)", "#d9534f", "水位严重超标，存在漫堤风险，请立即启动应急预案！"
            else:
                title, color, desc = "数据异常", "#333", "检测到的数值不在合理范围内。"

        lbl_val = QLabel(level_str)
        lbl_val.setAlignment(Qt.AlignCenter)
        lbl_val.setFont(QFont("Arial", 36, QFont.Bold))
        lbl_val.setStyleSheet(f"color: {color};")

        lbl_title = QLabel(title)
        lbl_title.setAlignment(Qt.AlignCenter)
        lbl_title.setFont(QFont("Microsoft YaHei", 18, QFont.Bold))
        lbl_title.setStyleSheet(f"background-color: {color}; color: white; border-radius: 4px; padding: 5px;")

        lbl_desc = QLabel(desc)
        lbl_desc.setWordWrap(True)
        lbl_desc.setAlignment(Qt.AlignCenter)
        lbl_desc.setFont(QFont("Microsoft YaHei", 12))
        lbl_desc.setStyleSheet("color: #666; margin-top: 10px;")

        v_layout.addWidget(lbl_title)
        v_layout.addWidget(lbl_val)
        v_layout.addWidget(lbl_desc)

        color_block = QLabel()
        color_block.setFixedSize(20, 150)
        color_block.setStyleSheet(f"background-color: {color}; border-radius: 10px;")

        layout.addWidget(status_widget, 4)
        layout.addWidget(color_block, 0)
        gb_eval.setLayout(layout)
        parent_layout.addWidget(gb_eval)

    def init_trend_panel(self, parent_layout):
        gb_trend = QGroupBox("历史水位变化趋势 (Trend Analysis)")
        layout = QHBoxLayout()

        dates, levels = [], []
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split(": ")
                        if len(parts) >= 2:
                            time_str = parts[0]
                            val_str = parts[1].replace(" dm", "")
                            dt = datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
                            dates.append(dt.strftime("%m-%d %H:%M"))
                            levels.append(float(val_str))
        except Exception as e:
            print(f"读取历史数据失败: {e}")

        chart_widget = QWidget()
        chart_layout = QVBoxLayout(chart_widget)

        if len(levels) > 0:
            fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
            fig.patch.set_facecolor(PANEL_BG)
            ax.set_facecolor("#f9f9f9")
            ax.plot(range(len(levels)), levels, marker='o', linestyle='-', color=THEME_COLOR, linewidth=2, label='水位 (dm)')
            ax.fill_between(range(len(levels)), levels, color=THEME_COLOR, alpha=0.1)
            ax.set_xticks(range(len(dates)))
            ax.set_xticklabels(dates, rotation=30, fontsize=8)
            ax.set_ylabel("水位 (dm)")
            ax.set_title("历史水位波动图")
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend()
            plt.tight_layout()
            canvas = FigureCanvas(fig)
            chart_layout.addWidget(canvas)
            max_lvl, min_lvl, avg_lvl = max(levels), min(levels), sum(levels) / len(levels)
        else:
            lbl_no_data = QLabel("暂无历史数据，请保存数据后查看。")
            lbl_no_data.setAlignment(Qt.AlignCenter)
            chart_layout.addWidget(lbl_no_data)
            max_lvl, min_lvl, avg_lvl = 0, 0, 0

        stats_panel = QWidget()
        stats_panel.setMinimumWidth(180)
        stats_layout = QVBoxLayout(stats_panel)
        stats_layout.setSpacing(15)

        def create_stat_box(title, value, color):
            box = QGroupBox(title)
            box.setStyleSheet(
                f"QGroupBox{{border: 1px solid {color}; border-radius: 5px;}} QGroupBox::title{{color: {color};}}"
            )
            l = QVBoxLayout()
            lbl = QLabel(f"{value:.2f} dm")
            lbl.setFont(QFont("Arial", 16, QFont.Bold))
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("color: #333; background: transparent;")
            l.addWidget(lbl)
            box.setLayout(l)
            return box

        stats_layout.addWidget(create_stat_box("历史最高 (Max)", max_lvl, "#d9534f"))
        stats_layout.addWidget(create_stat_box("历史最低 (Min)", min_lvl, "#28a745"))
        stats_layout.addWidget(create_stat_box("平均水位 (Avg)", avg_lvl, "#0078d7"))
        stats_layout.addStretch()

        layout.addWidget(chart_widget, 1)
        layout.addWidget(stats_panel)
        gb_trend.setLayout(layout)
        parent_layout.addWidget(gb_trend)


class MonitoringWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setStyleSheet(STYLESHEET)

        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.monitoring_alg_dir = os.path.join(self.base_dir, "algorithms", "monitoring")
        self.output_dir = os.path.join(self.monitoring_alg_dir, "outputs")
        self.final_result_dir = os.path.join(self.output_dir, "final_result")
        self.log_dir = os.path.join(self.output_dir, "logs")

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.final_result_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        self.current_water_level = None
        self.current_water_speed = None
        self.data_file = os.path.join(self.log_dir, "water_level_history.txt")

        self.master_layout = QHBoxLayout(self)
        self.master_layout.setContentsMargins(0, 0, 0, 0)
        self.master_layout.setSpacing(0)

        self.init_sidebar()
        self.stack = QStackedWidget()
        self.master_layout.addWidget(self.stack, 1)

        self.init_level_page()
        self.init_speed_page()
        self.init_flow_page()

        self.switch_page(0)

    def init_sidebar(self):
        sidebar = QWidget()
        sidebar.setMinimumWidth(180)
        sidebar.setStyleSheet("background-color: #ffffff; border-right: 1px solid #d1d9e6;")
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(0, 20, 0, 0)

        self.nav_btns = []
        btn_data = [("🌊 水位实时监测", 0), ("🎥 表面流速测算", 1), ("📊 综合流量推算", 2)]

        for text, index in btn_data:
            btn = QPushButton(text)
            btn.setProperty("class", "NavBtn")
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda checked, idx=index: self.switch_page(idx))
            layout.addWidget(btn)
            self.nav_btns.append(btn)

        layout.addStretch()
        self.master_layout.addWidget(sidebar)

    def switch_page(self, index):
        self.stack.setCurrentIndex(index)
        for i, btn in enumerate(self.nav_btns):
            btn.setProperty("active", "true" if i == index else "false")
            btn.style().unpolish(btn)
            btn.style().polish(btn)

        if index == 2:
            if self.current_water_level is not None:
                self.input_depth.setText(f"{self.current_water_level:.3f}")
            if self.current_water_speed is not None:
                self.input_velocity.setText(f"{self.current_water_speed:.3f}")

    def init_level_page(self):
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)

        control_panel = QGroupBox("系统控制")
        control_panel.setMinimumWidth(230)
        c_layout = QVBoxLayout(control_panel)
        c_layout.setContentsMargins(15, 25, 15, 15)
        c_layout.setSpacing(15)

        self.btn_upload = QPushButton("🚀 开始监测")
        self.btn_upload.setMinimumHeight(45)
        self.btn_upload.setStyleSheet("background-color: #28a745; font-size: 14px; border-radius: 6px;")
        self.btn_upload.clicked.connect(self.start_level_process)
        level_image_hint = (
            "内容：用于水位识别的原始水尺图像。\n"
            "格式：jpg/png/bmp 单张图像；建议包含完整刻度与当前水线，画面清晰、无遮挡。"
        )
        attach_hint(self.btn_upload, level_image_hint)

        self.btn_download = QPushButton("💾 保存数据")
        self.btn_download.setMinimumHeight(40)
        self.btn_download.clicked.connect(self.save_level_data)

        self.btn_analysis = QPushButton("📊 水位分析")
        self.btn_analysis.setMinimumHeight(40)
        self.btn_analysis.setStyleSheet("background-color: #17a2b8;")
        self.btn_analysis.clicked.connect(self.open_analysis_window)

        self.lbl_level_value = QLabel("当前水位\n-- dm")
        self.lbl_level_value.setFont(QFont("Arial", 20, QFont.Bold))
        self.lbl_level_value.setStyleSheet(
            "color: #d9534f; background: #fff; border: 2px solid #d9534f; border-radius: 8px; padding: 15px;"
        )
        self.lbl_level_value.setAlignment(Qt.AlignCenter)

        self.info_log_level = QTextEdit()
        self.info_log_level.setReadOnly(True)
        self.info_log_level.setText("系统就绪...")

        upload_row = QHBoxLayout()
        upload_row.addWidget(self.btn_upload)
        upload_row.addWidget(create_hint_badge(level_image_hint))
        upload_row.addStretch(1)
        c_layout.addLayout(upload_row)
        level_tip = QLabel("输入图像：请上传包含水尺刻度与当前水线的清晰单张图（jpg/png/bmp）。")
        level_tip.setWordWrap(True)
        level_tip.setStyleSheet("color:#5f6b7a;font-size:12px;")
        c_layout.addWidget(level_tip)
        c_layout.addWidget(self.btn_download)
        c_layout.addWidget(self.btn_analysis)
        c_layout.addSpacing(15)
        c_layout.addWidget(self.lbl_level_value)
        c_layout.addStretch()
        c_layout.addWidget(QLabel("运行日志:"))
        c_layout.addWidget(self.info_log_level, 1)

        display_panel = QWidget()
        display_v_layout = QVBoxLayout(display_panel)
        display_v_layout.setContentsMargins(0, 0, 0, 0)
        display_v_layout.setSpacing(10)

        top_row_layout = QHBoxLayout()
        top_row_layout.setSpacing(10)

        box_origin, self.lbl_origin = self.create_image_box("原始图像 (Input)", min_height=320)
        box_seg, self.lbl_seg = self.create_image_box("分割结果 (Mask)", min_height=320)
        top_row_layout.addWidget(box_origin, 1)
        top_row_layout.addWidget(box_seg, 1)
        display_v_layout.addLayout(top_row_layout)

        gb_process = QGroupBox("图像处理流水线")
        gb_process.setStyleSheet(
            f"QGroupBox {{ border: 1px solid {THEME_COLOR}; border-radius: 6px; margin-top: 1.2em; }} "
            f"QGroupBox::title {{ color: {THEME_COLOR}; subcontrol-origin: margin; left: 10px; }}"
        )

        bottom_row_layout = QHBoxLayout()
        bottom_row_layout.setSpacing(10)
        bottom_row_layout.setContentsMargins(10, 25, 10, 10)

        box_str, self.lbl_straight = self.create_image_box("1. 旋正", min_height=240)
        box_enh, self.lbl_enhance = self.create_image_box("2. 增强", min_height=240)
        box_den, self.lbl_denoise = self.create_image_box("3. 去噪", min_height=240)
        box_det, self.lbl_detect = self.create_image_box("4. 识别", min_height=240)
        box_fin, self.lbl_final = self.create_image_box("5. 结果", min_height=240)

        bottom_row_layout.addWidget(box_str, 1)
        bottom_row_layout.addWidget(box_enh, 1)
        bottom_row_layout.addWidget(box_den, 1)
        bottom_row_layout.addWidget(box_det, 1)
        bottom_row_layout.addWidget(box_fin, 1)
        gb_process.setLayout(bottom_row_layout)
        display_v_layout.addWidget(gb_process)

        data_panel = QGroupBox("数据分析中心")
        data_panel.setMinimumWidth(320)
        d_layout = QVBoxLayout(data_panel)
        d_layout.setContentsMargins(15, 25, 15, 15)
        d_layout.setSpacing(20)

        gb_seg = QGroupBox("1. 分割置信度")
        gb_seg_layout = QVBoxLayout()
        self.lbl_seg_conf = QLabel("等待数据...")
        self.lbl_seg_conf.setFont(QFont("Arial", 12))
        self.progress_seg = QProgressBar()
        self.progress_seg.setRange(0, 100)
        self.progress_seg.setFixedHeight(20)
        gb_seg_layout.addWidget(self.lbl_seg_conf)
        gb_seg_layout.addWidget(self.progress_seg)
        gb_seg.setLayout(gb_seg_layout)
        d_layout.addWidget(gb_seg)

        gb_char = QGroupBox("2. 刻度识别详情")
        gb_char_layout = QVBoxLayout()
        gb_char_layout.setContentsMargins(5, 15, 5, 5)
        self.table_chars = QTableWidget()
        self.table_chars.setColumnCount(4)
        self.table_chars.setHorizontalHeaderLabels(["Number", "Conf", "X", "Y"])
        self.table_chars.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_chars.verticalHeader().setVisible(False)
        self.table_chars.setAlternatingRowColors(True)
        gb_char_layout.addWidget(self.table_chars)
        gb_char.setLayout(gb_char_layout)
        d_layout.addWidget(gb_char)

        gb_calc = QGroupBox("3. 实时计算逻辑")
        gb_calc_layout = QVBoxLayout()
        self.lbl_calc_details = QLabel("等待计算...")
        self.lbl_calc_details.setWordWrap(True)
        self.lbl_calc_details.setStyleSheet(
            "font-family: Consolas; font-size: 11pt; color: #333; background-color: #f8f9fa; padding: 10px; border-radius: 4px;"
        )
        self.lbl_calc_details.setTextFormat(Qt.RichText)
        gb_calc_layout.addWidget(self.lbl_calc_details)
        gb_calc.setLayout(gb_calc_layout)
        d_layout.addWidget(gb_calc)

        layout.addWidget(control_panel, 2)
        layout.addWidget(display_panel, 6)
        layout.addWidget(data_panel, 3)
        self.stack.addWidget(page)

    def init_speed_page(self):
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)

        control_panel = QGroupBox("流速系统控制")
        control_panel.setMinimumWidth(230)
        c_layout = QVBoxLayout(control_panel)
        c_layout.setContentsMargins(15, 25, 15, 15)

        self.btn_speed_upload = QPushButton("🎥 上传河道视频")
        self.btn_speed_upload.setMinimumHeight(45)
        self.btn_speed_upload.setStyleSheet("background-color: #17a2b8; font-size: 14px; border-radius: 6px;")
        self.btn_speed_upload.clicked.connect(self.start_speed_process)
        speed_video_hint = (
            "内容：用于表面流速计算的河道视频。\n"
            "格式：mp4/avi/mov；建议固定机位、画面稳定、河面纹理清晰。"
        )
        attach_hint(self.btn_speed_upload, speed_video_hint)

        self.lbl_speed_value = QLabel("表面流速\n-- m/s")
        self.lbl_speed_value.setFont(QFont("Arial", 20, QFont.Bold))
        self.lbl_speed_value.setStyleSheet(
            "color: #D97706; background: #FEF3C7; border: 2px solid #D97706; border-radius: 8px; padding: 15px; margin-top: 20px;"
        )
        self.lbl_speed_value.setAlignment(Qt.AlignCenter)

        self.info_log_speed = QTextEdit()
        self.info_log_speed.setReadOnly(True)
        self.info_log_speed.setText("光流法引擎已就绪...\n等待输入视频数据。")

        speed_upload_row = QHBoxLayout()
        speed_upload_row.addWidget(self.btn_speed_upload)
        speed_upload_row.addWidget(create_hint_badge(speed_video_hint))
        speed_upload_row.addStretch(1)
        c_layout.addLayout(speed_upload_row)
        c_layout.addWidget(self.lbl_speed_value)
        c_layout.addStretch()
        c_layout.addWidget(QLabel("运算日志:"))
        c_layout.addWidget(self.info_log_speed, 1)

        display_panel = QGroupBox("光流法特征追踪与分析视图")
        grid_layout = QGridLayout(display_panel)
        grid_layout.setContentsMargins(10, 20, 10, 10)

        self.img_labels_speed = []
        titles = ["图1: 前一帧特征提取 (红点)", "图2: 后一帧特征追踪 (蓝点)", "图3: 黑底特征匹配轨迹", "图4: 运动方向玫瑰图"]

        for i in range(4):
            vbox = QVBoxLayout()
            desc = QLabel(titles[i])
            desc.setAlignment(Qt.AlignCenter)
            desc.setStyleSheet("font-size: 14px; color: #4B5563; font-weight: bold;")
            img_lbl = QLabel("暂无图像")
            img_lbl.setAlignment(Qt.AlignCenter)
            img_lbl.setStyleSheet("background-color: white; border: 2px dashed #93C5FD; border-radius: 8px;")
            vbox.addWidget(desc)
            vbox.addWidget(img_lbl, 1)
            self.img_labels_speed.append(img_lbl)
            grid_layout.addLayout(vbox, i // 2, i % 2)

        layout.addWidget(control_panel, 2)
        layout.addWidget(display_panel, 6)
        self.stack.addWidget(page)

    def init_flow_page(self):
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(30, 30, 30, 30)

        center_card = QGroupBox("🌊 水文综合流量推算中心")
        center_card.setStyleSheet(
            f"QGroupBox {{ background-color: white; border: 2px solid #cbd5e1; border-radius: 12px; margin-top: 20px;}}"
            f"QGroupBox::title {{ color: {THEME_COLOR}; font-size: 22px; font-weight: bold; left: 20px; padding: 0 10px;}}"
        )
        c_layout = QVBoxLayout(center_card)
        c_layout.setContentsMargins(50, 40, 50, 40)

        tip = QLabel("✅ 系统已将您测得的水位和流速自动同步。默认采用 U形/梯形河槽模型(0.80) 及 表面流速折算系数(0.85)。")
        tip.setStyleSheet(
            "font-size: 16px; color: #6B7280; font-weight: bold; background-color: #F3F4F6; padding: 15px; border-radius: 8px;"
        )
        tip.setAlignment(Qt.AlignCenter)
        c_layout.addWidget(tip)
        c_layout.addSpacing(30)

        form_layout = QFormLayout()
        form_layout.setSpacing(30)

        validator = QDoubleValidator(0.0, 9999.99, 4)
        validator.setNotation(QDoubleValidator.StandardNotation)
        input_style = "padding: 15px; font-size: 20px; font-weight: bold; border: 2px solid #93C5FD; border-radius: 8px; color: #1D4ED8; background-color: #F8FAFC;"

        self.input_width = QLineEdit()
        self.input_width.setPlaceholderText("在此输入实测河宽 (如: 15.5)")
        self.input_width.setValidator(validator)
        self.input_width.setMinimumHeight(55)
        self.input_width.setStyleSheet(input_style)

        self.input_depth = QLineEdit()
        self.input_depth.setPlaceholderText("系统自动同步 (dm)")
        self.input_depth.setValidator(validator)
        self.input_depth.setMinimumHeight(55)
        self.input_depth.setStyleSheet(input_style)

        self.input_velocity = QLineEdit()
        self.input_velocity.setPlaceholderText("系统自动同步 (m/s)")
        self.input_velocity.setValidator(validator)
        self.input_velocity.setMinimumHeight(55)
        self.input_velocity.setStyleSheet(input_style)

        width_hint = "内容：实测河道水面宽度。\n格式：浮点数，单位 m，例如 15.5。"
        depth_hint = "内容：实测河道水深。\n格式：浮点数，单位 dm，例如 2.3。"
        velocity_hint = "内容：表面中心流速。\n格式：浮点数，单位 m/s，例如 1.28。"
        attach_hint(self.input_width, width_hint)
        attach_hint(self.input_depth, depth_hint)
        attach_hint(self.input_velocity, velocity_hint)

        font_lbl = QFont("Microsoft YaHei", 16, QFont.Bold)
        lbl_w = QLabel("实测水面宽度 (m) :")
        lbl_w.setFont(font_lbl)
        lbl_d = QLabel("实测河道水深 (dm):")
        lbl_d.setFont(font_lbl)
        lbl_v = QLabel("表面中心流速 (m/s):")
        lbl_v.setFont(font_lbl)

        form_layout.addRow(label_with_hint(lbl_w, width_hint), self.input_width)
        form_layout.addRow(label_with_hint(lbl_d, depth_hint), self.input_depth)
        form_layout.addRow(label_with_hint(lbl_v, velocity_hint), self.input_velocity)

        c_layout.addLayout(form_layout)
        c_layout.addSpacing(40)

        self.btn_calc_flow = QPushButton("⚡ 开 始 核 算 流 量 ⚡")
        self.btn_calc_flow.setMinimumHeight(70)
        self.btn_calc_flow.setCursor(Qt.PointingHandCursor)
        self.btn_calc_flow.setStyleSheet(
            "QPushButton { background-color: #2563EB; color: white; border-radius: 10px; font-size: 24px; font-weight: bold; letter-spacing: 2px; }"
            "QPushButton:hover { background-color: #1D4ED8; }"
        )
        self.btn_calc_flow.clicked.connect(self.calculate_flow)
        c_layout.addWidget(self.btn_calc_flow)
        c_layout.addSpacing(30)

        self.lbl_flow_value = QLabel("请录入参数后进行计算")
        self.lbl_flow_value.setAlignment(Qt.AlignCenter)
        self.lbl_flow_value.setStyleSheet(
            "background-color: #ECFDF5; border: 3px dashed #34D399; border-radius: 12px; padding: 30px; color: #065F46; font-size: 20px;"
        )
        c_layout.addWidget(self.lbl_flow_value)
        c_layout.addStretch()

        layout.addWidget(center_card)
        self.stack.addWidget(page)

    def level_log(self, text):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.info_log_level.append(f"[{timestamp}] {text}")

    def parse_yolo_conf(self, image_path, task_type='segment'):
        try:
            dirname = os.path.dirname(image_path)
            basename = os.path.basename(image_path)
            name_no_ext = os.path.splitext(basename)[0]
            label_dir = os.path.join(dirname, 'labels')
            txt_path = os.path.join(label_dir, name_no_ext + ".txt")

            if not os.path.exists(txt_path):
                label_dir = os.path.join(os.path.dirname(dirname), 'labels')
                txt_path = os.path.join(label_dir, name_no_ext + ".txt")

            if os.path.exists(txt_path):
                max_conf = 0.0
                lines_data = []
                with open(txt_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split()
                        try:
                            conf = float(parts[-1])
                            if conf > 1.0:
                                conf = 1.0
                            if conf > max_conf:
                                max_conf = conf
                            if task_type == 'detect':
                                cls_id = int(parts[0])
                                lines_data.append({'cls': cls_id, 'conf': conf})
                        except Exception:
                            pass
                return max_conf, lines_data
            return 0.0, []
        except Exception:
            return 0.0, []

    def start_level_process(self):
        fname, _ = QFileDialog.getOpenFileName(self, "打开原图", "", "Images (*.jpg *.png *.bmp)")
        if not fname:
            return

        self.table_chars.setRowCount(0)
        self.lbl_calc_details.setText("计算中...")
        self.lbl_seg_conf.setText("--")
        self.progress_seg.setValue(0)

        self.info_log_level.append("\n--- 新任务开始 ---")
        self.level_log(f"加载图片: {os.path.basename(fname)}")

        try:
            self.update_image(self.lbl_origin, fname)

            self.level_log("Step 1: 运行分割模型")
            seg_path = predict_seg.run_segmentation(fname)
            if not seg_path or not os.path.exists(seg_path):
                raise RuntimeError("分割失败：未生成有效输出图像。")
            self.update_image(self.lbl_seg, seg_path)
            seg_conf, _ = self.parse_yolo_conf(seg_path, 'segment')
            self.lbl_seg_conf.setText(f"置信度: {seg_conf:.4f}")
            self.progress_seg.setValue(int(seg_conf * 100))

            self.level_log("Step 2: 图像预处理流水线")
            p_str, p_enh, p_den = preprocessor.run_preprocessing_pipeline(seg_path)
            if not p_den or not os.path.exists(p_den):
                raise RuntimeError("预处理失败：未生成去噪结果图像。")
            self.update_image(self.lbl_straight, p_str)
            self.update_image(self.lbl_enhance, p_enh)
            self.update_image(self.lbl_denoise, p_den)

            self.level_log("Step 3: 刻度数字识别")
            det_img, label_txt = predict_number.run_number_detection(p_den)
            if not det_img or not os.path.exists(det_img):
                raise RuntimeError("数字识别失败：未生成检测结果图像。")
            if not label_txt or not os.path.exists(label_txt):
                raise RuntimeError("数字识别失败：未生成标签文件。")
            self.update_image(self.lbl_detect, det_img)

            self.level_log("Step 4: 水位几何计算")
            final_save_path = os.path.join(self.final_result_dir, f"final_{os.path.basename(p_den)}")

            centers, level = calculator.advanced_draw_center(p_den, label_txt, final_save_path)
            _, det_data = self.parse_yolo_conf(det_img, 'detect')
            conf_map = {item['cls']: item['conf'] for item in det_data}

            self.table_chars.setRowCount(len(centers))
            for i, item in enumerate(centers):
                cls_id = item['class_id']
                number_display = cls_id + 1
                conf_val = conf_map.get(cls_id, 0.0)
                cx, cy = item['center']

                items = [
                    QTableWidgetItem(str(number_display)),
                    QTableWidgetItem(f"{conf_val:.2f}"),
                    QTableWidgetItem(str(cx)),
                    QTableWidgetItem(str(cy)),
                ]
                for j, cell in enumerate(items):
                    cell.setTextAlignment(Qt.AlignCenter)
                    self.table_chars.setItem(i, j, cell)

            self.update_image(self.lbl_final, final_save_path)

            if level is not None and len(centers) >= 2:
                self.current_water_level = level
                self.lbl_level_value.setText(f"当前水位\n{level:.2f} dm")

                sorted_centers = sorted(centers, key=lambda x: x['class_id'])
                min_pt = sorted_centers[0]
                min_number = sorted_centers[0]['class_id'] + 1
                p1, p2 = min_pt['center'], sorted_centers[1]['center']
                dist_scale = abs(p1[1] - p2[1])
                bottom_y = calculator.find_non_transparent_bottom_below_point(p_den, p1)
                dist_bottom = bottom_y - p1[1]

                html_text = f"""
                <html><body>
                <p><b>[变量参数]</b></p>
                <p>基准点 Number: {min_number} (ID:{min_pt['class_id']})</p>
                <p>刻度像素间距: <font color='{THEME_COLOR}'>{dist_scale} px</font></p>
                <p>至水底像素距离: <font color='#6f42c1'>{dist_bottom} px</font></p>
                <hr style="border: 1px dashed #ccc;">
                <p><b>[计算公式]</b></p>
                <p>H = (Number + 0.25) - (d_bottom / d_scale)</p>
                <p>H = ({min_number} + 0.25) - ({dist_bottom} / {dist_scale})</p>
                <p><b>结果 = <font color='#d9534f' size='5'>{level:.3f} dm</font></b></p>
                </body></html>
                """
                self.lbl_calc_details.setText(html_text)
            else:
                self.lbl_calc_details.setText("无法计算 (需至少2个点)")
                self.lbl_level_value.setText("计算失败")

            self.level_log("所有任务执行完毕")
        except Exception as e:
            self.level_log(f"Error: {e}")
            import traceback
            print(traceback.format_exc())
            QMessageBox.critical(self, "错误", str(e))

    def save_level_data(self):
        if self.current_water_level is not None:
            with open(self.data_file, "a", encoding="utf-8") as f:
                f.write(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: {self.current_water_level:.2f} dm\n")
            QMessageBox.information(self, "保存成功", f"水位数据已追加至\n{self.data_file}")

    def open_analysis_window(self):
        dialog = AnalysisDialog(self.current_water_level, self.data_file)
        dialog.exec_()

    def start_speed_process(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "选择河道俯拍视频", "", "Video Files (*.mp4 *.avi *.mov)")
        if not file_name:
            return

        self.info_log_speed.append("\n[流速] 正在调用光流引擎解析视频帧...")
        self.lbl_speed_value.setText("计算中...")

        success, result, paths = calculate_velocity_for_ui(file_name)
        if success:
            self.current_water_speed = result
            self.lbl_speed_value.setText(f"表面流速\n{result:.3f} m/s")
            self.info_log_speed.append(f"[流速] 计算完成: {result:.3f} m/s")
            for i, path in enumerate(paths):
                if os.path.exists(path):
                    pixmap = QPixmap(path)
                    scaled = pixmap.scaled(self.img_labels_speed[i].size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    self.img_labels_speed[i].setPixmap(scaled)
        else:
            self.info_log_speed.append(f"[流速] 错误: {result}")
            self.lbl_speed_value.setText("计算失败")

    def calculate_flow(self):
        w_text = self.input_width.text()
        d_text = self.input_depth.text()
        v_text = self.input_velocity.text()

        if not w_text or not d_text or not v_text:
            QMessageBox.warning(self, "参数缺失", "请确保宽度、水位、流速三项数据完整填写。")
            return

        try:
            width = float(w_text)
            depth_dm = float(d_text)
            depth_m = depth_dm / 10.0
            v_surf = float(v_text)

            alpha = 0.80
            k_coef = 0.85

            area = alpha * width * depth_m
            v_mean = k_coef * v_surf
            flow_rate = area * v_mean

            html_result = f"""
            <div style="font-weight:bold; color: #047857;">
                <div style="font-size: 20px; margin-bottom: 10px; color: #10B981;">
                    核算详情：换算水深 {depth_m:.3f} m | 截面积 {area:.2f} m² | 均流速 {v_mean:.2f} m/s
                </div>
                <div>
                    综合流量 Q = <span style='font-size: 52px; font-weight: 900;'>{flow_rate:.2f}</span> m³/s
                </div>
            </div>
            """
            self.lbl_flow_value.setText(html_result)
        except ValueError:
            QMessageBox.warning(self, "数据错误", "请输入有效数字。")

    def create_image_box(self, title, min_height=220):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        lbl_title = QLabel(title)
        lbl_title.setAlignment(Qt.AlignCenter)
        lbl_title.setStyleSheet(f"font-weight: bold; color: {TEXT_COLOR}; font-size: 13px;")

        lbl_img = QLabel("No Data")
        lbl_img.setAlignment(Qt.AlignCenter)
        lbl_img.setMinimumHeight(min_height)
        lbl_img.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lbl_img.setStyleSheet("background-color: #e9ecef; border: 1px solid #ced4da; border-radius: 4px; color: #adb5bd;")

        layout.addWidget(lbl_title)
        layout.addWidget(lbl_img, 1)
        return container, lbl_img

    def update_image(self, label_obj, path):
        final_path = path
        if not path or not os.path.exists(path):
            if path:
                base = os.path.splitext(path)[0]
                for ext in [".png", ".jpg", ".jpeg"]:
                    candidate = base + ext
                    if os.path.exists(candidate):
                        final_path = candidate
                        break

        if final_path and os.path.exists(final_path):
            pixmap = QPixmap(final_path)
            if not pixmap.isNull():
                scaled = pixmap.scaled(label_obj.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                label_obj.setPixmap(scaled)
            else:
                label_obj.setText("Error")
        else:
            label_obj.setText("N/A")
