from __future__ import annotations

from datetime import datetime

from PyQt5.QtCore import QDate
from PyQt5.QtWidgets import (
    QDateEdit,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from algorithms.snow_state import (
    BAND_DESCRIPTIONS,
    DEFAULT_BBOX,
    DEFAULT_DRIVE_FOLDER,
    DEFAULT_PROJECT_ID,
    DEFAULT_SOURCES,
    DEFAULT_TASK_PREFIX,
    STATE_LABELS,
    ensure_earth_engine,
    parse_bbox_text,
    submit_runoff_warning_export,
)
from app.ui_hints import attach_hint, label_with_hint


class SnowStateWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.source_edits = {}
        self.last_task_info = None
        self.init_ui()
        self.reset_defaults()

    def init_ui(self):
        layout = QVBoxLayout(self)

        title = QLabel("积雪状态识别与融雪径流预警")
        title.setStyleSheet("font-size:18px;font-weight:bold;")
        layout.addWidget(title)

        intro = QLabel(
            "基于 Google Earth Engine 的帕米尔高原积雪状态识别模块。"
            "输出 GeoTIFF 包含 Snow_State 和 Runoff_Probability 两个波段，"
            "结果提交到 Google Drive 后供后续预警展示使用。"
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        layout.addWidget(self._build_parameter_group())
        layout.addWidget(self._build_source_group())
        layout.addLayout(self._build_button_row())
        layout.addWidget(self._build_legend_group())

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log, 1)

    def _build_parameter_group(self):
        group = QGroupBox("识别与导出参数")
        form = QFormLayout(group)

        self.target_start_edit = self._create_date_edit()
        self.target_end_edit = self._create_date_edit()
        self.sar_melt_start_edit = self._create_date_edit()
        self.sar_melt_end_edit = self._create_date_edit()
        self.sar_ref_start_edit = self._create_date_edit()
        self.sar_ref_end_edit = self._create_date_edit()

        form.addRow(
            label_with_hint("目标光学时段:", "用于光学 SNOMAP 积雪范围识别的日期窗口。"),
            self._build_date_row(self.target_start_edit, self.target_end_edit),
        )
        form.addRow(
            label_with_hint("SAR 融雪期:", "用于 Sentinel-1 湿雪信号提取的融雪期窗口。"),
            self._build_date_row(self.sar_melt_start_edit, self.sar_melt_end_edit),
        )
        form.addRow(
            label_with_hint("SAR 参考期:", "夏季稳定期后向散射基准，默认 2022-07-05 至 2022-07-30。"),
            self._build_date_row(self.sar_ref_start_edit, self.sar_ref_end_edit),
        )

        self.bbox_edit = QLineEdit()
        bbox_hint = "研究区范围，格式 west,south,east,north，例如 70.0, 36.0, 76.5, 40.0。"
        attach_hint(self.bbox_edit, bbox_hint)
        form.addRow(label_with_hint("区域范围:", bbox_hint), self.bbox_edit)

        self.project_edit = QLineEdit()
        project_hint = "Google Earth Engine Cloud Project ID，留空则使用默认初始化。"
        attach_hint(self.project_edit, project_hint)
        form.addRow(label_with_hint("GEE Project:", project_hint), self.project_edit)

        self.drive_folder_edit = QLineEdit()
        folder_hint = "Google Drive 输出目录名，导出 GeoTIFF 后可在对应文件夹查看。"
        attach_hint(self.drive_folder_edit, folder_hint)
        form.addRow(label_with_hint("导出目录:", folder_hint), self.drive_folder_edit)

        self.task_prefix_edit = QLineEdit()
        prefix_hint = "任务名前缀，仅保留字母、数字、下划线和中划线。"
        attach_hint(self.task_prefix_edit, prefix_hint)
        form.addRow(label_with_hint("任务前缀:", prefix_hint), self.task_prefix_edit)

        self.scale_spin = QSpinBox()
        self.scale_spin.setRange(1, 10000)
        attach_hint(self.scale_spin, "导出分辨率，单位米，默认 30。")
        form.addRow(label_with_hint("导出分辨率:", "导出分辨率，单位米，默认 30。"), self.scale_spin)

        return group

    def _build_source_group(self):
        group = QGroupBox("数据源配置（可选）")
        form = QFormLayout(group)

        hints = {
            "dem_source": "DEM 数据源 ID，默认使用 USGS/SRTMGL1_003。",
            "eco_source": "生态区边界数据源 ID，默认使用 RESOLVE/ECOREGIONS/2017。",
            "opt_s2_source": "Sentinel-2 光学影像数据源 ID。",
            "opt_l8_source": "Landsat 8 TOA 光学影像数据源 ID。",
            "opt_l9_source": "Landsat 9 TOA 光学影像数据源 ID。",
            "modis_source": "MODIS 雪盖数据源 ID。",
            "sar_source": "Sentinel-1 SAR 数据源 ID。",
            "river_source": "河流矢量数据源 ID，用于融雪径流概率 AHP 因子。",
        }
        labels = {
            "dem_source": "DEM:",
            "eco_source": "生态区:",
            "opt_s2_source": "Sentinel-2:",
            "opt_l8_source": "Landsat 8:",
            "opt_l9_source": "Landsat 9:",
            "modis_source": "MODIS 雪盖:",
            "sar_source": "SAR:",
            "river_source": "河流:",
        }

        for key, default_value in DEFAULT_SOURCES.items():
            edit = QLineEdit()
            edit.setPlaceholderText(default_value)
            attach_hint(edit, hints[key])
            self.source_edits[key] = edit
            form.addRow(label_with_hint(labels[key], hints[key]), edit)

        return group

    def _build_button_row(self):
        row = QHBoxLayout()

        self.init_btn = QPushButton("初始化 GEE")
        self.run_btn = QPushButton("提交预警产品任务")
        self.reset_btn = QPushButton("恢复默认参数")

        self.init_btn.clicked.connect(self.initialize_gee)
        self.run_btn.clicked.connect(self.run_task)
        self.reset_btn.clicked.connect(self.reset_defaults)

        row.addWidget(self.init_btn)
        row.addWidget(self.run_btn)
        row.addWidget(self.reset_btn)
        row.addStretch()

        return row

    def _build_legend_group(self):
        group = QGroupBox("输出说明")
        layout = QVBoxLayout(group)

        state_lines = [f"{code}: {label}" for code, label in STATE_LABELS.items()]
        band_lines = [f"{name}: {desc}" for name, desc in BAND_DESCRIPTIONS.items()]
        legend = QLabel(
            "Snow_State 像元值:\n"
            + "\n".join(state_lines)
            + "\n\n输出波段:\n"
            + "\n".join(band_lines)
            + "\n\n说明: 当前模块提交云端异步导出任务，计算结果不在本标签页直接渲染。"
        )
        legend.setWordWrap(True)
        layout.addWidget(legend)

        return group

    def _create_date_edit(self):
        edit = QDateEdit()
        edit.setCalendarPopup(True)
        edit.setDisplayFormat("yyyy-MM-dd")
        return edit

    def _build_date_row(self, start_edit, end_edit):
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addWidget(start_edit)
        row_layout.addWidget(QLabel("至"))
        row_layout.addWidget(end_edit)
        return row_widget

    def reset_defaults(self):
        self.target_start_edit.setDate(QDate(2023, 5, 10))
        self.target_end_edit.setDate(QDate(2023, 5, 15))
        self.sar_melt_start_edit.setDate(QDate(2023, 5, 5))
        self.sar_melt_end_edit.setDate(QDate(2023, 5, 20))
        self.sar_ref_start_edit.setDate(QDate(2022, 7, 5))
        self.sar_ref_end_edit.setDate(QDate(2022, 7, 30))

        self.bbox_edit.setText(", ".join(str(value) for value in DEFAULT_BBOX))
        self.project_edit.setText(DEFAULT_PROJECT_ID)
        self.drive_folder_edit.setText(DEFAULT_DRIVE_FOLDER)
        self.task_prefix_edit.setText(DEFAULT_TASK_PREFIX)
        self.scale_spin.setValue(30)

        for key, edit in self.source_edits.items():
            edit.setText(DEFAULT_SOURCES[key])

        if hasattr(self, "log"):
            self.log.clear()
            self.append_log("已恢复积雪状态识别与融雪径流预警默认参数。")

    def initialize_gee(self):
        try:
            message = ensure_earth_engine(
                authenticate=True,
                project_id=self.project_edit.text().strip(),
            )
            self.append_log(message)
            QMessageBox.information(self, "初始化完成", message)
        except Exception as exc:
            self.append_log(f"[ERROR] {exc}")
            QMessageBox.critical(self, "初始化失败", str(exc))

    def run_task(self):
        try:
            self._validate_date_range(self.target_start_edit, self.target_end_edit, "目标光学时段")
            self._validate_date_range(self.sar_melt_start_edit, self.sar_melt_end_edit, "SAR 融雪期")
            self._validate_date_range(self.sar_ref_start_edit, self.sar_ref_end_edit, "SAR 参考期")

            target_start = self.target_start_edit.date().toString("yyyy-MM-dd")
            target_end = self.target_end_edit.date().toString("yyyy-MM-dd")
            sar_melt_start = self.sar_melt_start_edit.date().toString("yyyy-MM-dd")
            sar_melt_end = self.sar_melt_end_edit.date().toString("yyyy-MM-dd")
            sar_ref_start = self.sar_ref_start_edit.date().toString("yyyy-MM-dd")
            sar_ref_end = self.sar_ref_end_edit.date().toString("yyyy-MM-dd")
            bbox_coords = parse_bbox_text(self.bbox_edit.text())

            self.append_log("开始提交积雪状态与融雪径流预警任务...")
            self.append_log(f"目标时段: {target_start} 至 {target_end}")
            self.append_log(f"SAR 融雪期: {sar_melt_start} 至 {sar_melt_end}")
            self.append_log(f"区域范围: {bbox_coords}")

            result = submit_runoff_warning_export(
                target_start=target_start,
                target_end=target_end,
                sar_melt_start=sar_melt_start,
                sar_melt_end=sar_melt_end,
                sar_ref_start=sar_ref_start,
                sar_ref_end=sar_ref_end,
                bbox_coords=bbox_coords,
                drive_folder=self.drive_folder_edit.text().strip(),
                task_prefix=self.task_prefix_edit.text().strip(),
                scale=self.scale_spin.value(),
                authenticate=False,
                project_id=self.project_edit.text().strip(),
                **self._collect_sources(),
            )

            self.last_task_info = result
            self.append_log("任务已提交到 Google Earth Engine。")
            self.append_log(f"任务名称: {result['description']}")
            self.append_log(f"Drive 目录: {result['drive_folder']}")
            self.append_log(f"任务状态: {result['task_state']}")
            if result.get("task_id"):
                self.append_log(f"任务 ID: {result['task_id']}")

            QMessageBox.information(
                self,
                "任务已提交",
                f"任务 {result['description']} 已提交，当前状态: {result['task_state']}",
            )
        except Exception as exc:
            self.append_log(f"[ERROR] {exc}")
            QMessageBox.critical(self, "任务提交失败", str(exc))

    def _validate_date_range(self, start_edit, end_edit, label):
        if start_edit.date() > end_edit.date():
            raise ValueError(f"{label}开始日期不能晚于结束日期")

    def _collect_sources(self):
        return {
            key: edit.text().strip() or DEFAULT_SOURCES[key]
            for key, edit in self.source_edits.items()
        }

    def append_log(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log.append(f"[{timestamp}] {message}")
