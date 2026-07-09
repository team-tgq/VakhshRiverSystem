from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
import rasterio.warp
from PyQt5.QtCore import QDate
from PyQt5.QtWidgets import (
    QDateEdit,
    QFileDialog,
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
from app.digital_twin_standard import (
    DEFAULT_PERIOD,
    TARGET_CRS,
    default_run_context,
    mark_module_complete,
    module_output_path,
    standard_dialog_dir,
    write_metadata_sidecar,
)
from app.ui_hints import attach_hint, label_with_hint
from rasterio.enums import Resampling


SNOW_DENSITY_BY_TYPE_GCM3 = {
    1: 0.0,
    2: 0.25,
    3: 0.40,
}


def _context_from_target_date(date_text: str | None):
    text = str(date_text or "").strip()
    if len(text) >= 7 and text[4] == "-":
        period = text[:7].replace("-", "")
        month = int(text[5:7])
    elif len(text) >= 6 and text[:6].isdigit():
        period = text[:6]
        month = int(period[4:6])
    else:
        period = DEFAULT_PERIOD
        month = int(period[4:6])

    if 3 <= month <= 5:
        period_name = "融雪模拟"
    elif 6 <= month <= 9:
        period_name = "汛期模拟"
    else:
        period_name = "积雪状态模拟"
    return default_run_context(period=period, period_name=period_name)


def _read_state_band_as_target(src_path: str | Path):
    path = Path(src_path)
    if not path.exists():
        raise FileNotFoundError(path)

    with rasterio.open(path) as src:
        band_count = src.count
        if src.count < 1:
            raise ValueError("GeoTIFF 至少需要包含 Snow_State 波段。")
        if src.crs is None:
            raise ValueError(f"源 GeoTIFF 缺少 CRS，不能写入标准成果: {path}")

        src_crs = src.crs
        if src_crs.to_string() == TARGET_CRS:
            state = src.read(1)
            transform = src.transform
            width = src.width
            height = src.height
        else:
            transform, width, height = rasterio.warp.calculate_default_transform(
                src_crs,
                TARGET_CRS,
                src.width,
                src.height,
                *src.bounds,
            )
            state = np.zeros((height, width), dtype=np.float32)
            rasterio.warp.reproject(
                source=rasterio.band(src, 1),
                destination=state,
                src_transform=src.transform,
                src_crs=src_crs,
                dst_transform=transform,
                dst_crs=TARGET_CRS,
                resampling=Resampling.nearest,
                src_nodata=src.nodata,
                dst_nodata=0,
            )

        nodata = src.nodata

    state = np.asarray(np.rint(state), dtype=np.int16)
    if nodata is not None and np.isfinite(nodata):
        state = np.where(state == int(round(nodata)), 0, state)
    valid_classes = np.array(list(SNOW_DENSITY_BY_TYPE_GCM3), dtype=np.int16)
    state = np.where(np.isin(state, valid_classes), state, 0)
    return state.astype(np.uint8), transform, width, height, band_count


def _density_from_state(state: np.ndarray) -> np.ndarray:
    density = np.full(state.shape, -9999.0, dtype=np.float32)
    for class_value, density_value in SNOW_DENSITY_BY_TYPE_GCM3.items():
        density[state == class_value] = density_value
    return density


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
        self.sync_btn = QPushButton("同步已下载GeoTIFF")
        self.reset_btn = QPushButton("恢复默认参数")

        self.init_btn.clicked.connect(self.initialize_gee)
        self.run_btn.clicked.connect(self.run_task)
        self.sync_btn.clicked.connect(self.sync_downloaded_geotiff)
        self.reset_btn.clicked.connect(self.reset_defaults)

        row.addWidget(self.init_btn)
        row.addWidget(self.run_btn)
        row.addWidget(self.sync_btn)
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
            self.append_log("Drive GeoTIFF 下载到本地后，可点击“同步已下载GeoTIFF”写入标准成果目录。")

            QMessageBox.information(
                self,
                "任务已提交",
                f"任务 {result['description']} 已提交，当前状态: {result['task_state']}",
            )
        except Exception as exc:
            self.append_log(f"[ERROR] {exc}")
            QMessageBox.critical(self, "任务提交失败", str(exc))

    def sync_downloaded_geotiff(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择已下载的 GEE GeoTIFF",
            standard_dialog_dir("processed", module_code="M06"),
            "GeoTIFF (*.tif *.tiff);;All files (*.*)",
        )
        if not path:
            return

        try:
            target_start = self.target_start_edit.date().toString("yyyy-MM-dd")
            outputs = self._export_standard_geotiff(Path(path), target_start)
            self.append_log("M06 标准成果已同步。")
            self.append_log(f"雪状态: {outputs['snow_type']}")
            self.append_log(f"雪密度: {outputs['snow_density']}")
            QMessageBox.information(self, "同步完成", "已写入 M06 标准成果目录。")
        except Exception as exc:
            self.append_log(f"[ERROR] {exc}")
            QMessageBox.critical(self, "同步失败", str(exc))

    def _export_standard_geotiff(self, geotiff_path: str | Path, target_date: str | None = None) -> dict[str, Path]:
        context = _context_from_target_date(target_date)
        snow_type_tif = module_output_path("M06", context=context, output_index=0)
        snow_density_tif = module_output_path("M06", context=context, output_index=1)

        state, transform, width, height, band_count = _read_state_band_as_target(geotiff_path)
        density = _density_from_state(state)

        snow_type_tif.parent.mkdir(parents=True, exist_ok=True)
        state_profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 1,
            "dtype": "uint8",
            "crs": TARGET_CRS,
            "transform": transform,
            "nodata": 0,
            "compress": "lzw",
        }
        with rasterio.open(snow_type_tif, "w", **state_profile) as dst:
            dst.write(state, 1)

        density_profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 1,
            "dtype": "float32",
            "crs": TARGET_CRS,
            "transform": transform,
            "nodata": -9999.0,
            "compress": "lzw",
        }
        with rasterio.open(snow_density_tif, "w", **density_profile) as dst:
            dst.write(density, 1)

        source_files = [geotiff_path]
        common_extra = {
            "scheme": context.scheme,
            "scheme_name": context.scheme_name,
            "period": context.period,
            "period_name": context.period_name,
            "source_product": "GEE Snow_State + Runoff_Probability GeoTIFF",
            "source_band_count": band_count,
            "source_band_1": "Snow_State",
        }
        if band_count >= 2:
            common_extra["source_band_2"] = "Runoff_Probability"
        write_metadata_sidecar(
            snow_type_tif,
            module_code="M06",
            field="snow_type",
            source_files=source_files,
            extra={
                **common_extra,
                "standard_field": "snow_type",
                "class_labels": STATE_LABELS,
            },
        )
        write_metadata_sidecar(
            snow_density_tif,
            module_code="M06",
            field="snow_density",
            source_files=source_files,
            extra={
                **common_extra,
                "density_mapping_gcm3": SNOW_DENSITY_BY_TYPE_GCM3,
                "density_mapping_note": "当前由 Snow_State 类别确定性映射：无雪=0，干雪=0.25，湿雪=0.40；后续可替换为实测或模型雪密度。",
            },
        )
        mark_module_complete(context, "M06")
        return {"snow_type": snow_type_tif, "snow_density": snow_density_tif}

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
