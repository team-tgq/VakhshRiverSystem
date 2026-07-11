import csv
import json
from pathlib import Path

from PyQt5.QtCore import QDate, Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QComboBox,
    QDateEdit,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from algorithms.reservoir_estimation.reservoir_core import NurekReservoirEstimator, save_curve_plot
from app.digital_twin_standard import (
    DEFAULT_PERIOD,
    default_run_context,
    ensure_raw_source_path,
    infer_run_context_from_path,
    mark_module_complete,
    module_output_path,
    module_processed_dir,
    period_to_date,
    raw_data_dir,
    standard_dialog_dir,
    write_metadata_sidecar,
    write_raw_metadata_sidecar,
    write_standard_csv,
)
from app.ui_hints import attach_hint


PLUGIN_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PLUGIN_DIR.parent.parent / "algorithms" / "reservoir_estimation" / "output" / "Nurek"
PLOT_PATH = OUTPUT_DIR / "last_estimate_plot.png"


def _reservoir_raw_paths() -> dict[str, Path]:
    return {
        "parameters": raw_data_dir("reservoir", "parameters") / "reservoir_parameters.csv",
        "observations": raw_data_dir("reservoir", "observations") / "reservoir_observations.csv",
        "curve": raw_data_dir("reservoir", "hypsometry") / "reservoir_hypsometry.csv",
    }


def _configured_estimator() -> NurekReservoirEstimator:
    paths = _reservoir_raw_paths()
    for path in paths.values():
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    with paths["parameters"].open("r", encoding="utf-8-sig", newline="") as file:
        values = {row["parameter"]: row["value"] for row in csv.DictReader(file)}
    return NurekReservoirEstimator(
        curve_path=paths["curve"],
        total_capacity_km3=float(values["total_capacity"]),
        active_storage_km3=float(values["active_storage"]),
    )


def _context_from_result_date(date_text: str | None):
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
        period_name = "水库调度"
    return default_run_context(period=period, period_name=period_name)


class ReservoirEstimationWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.estimator = _configured_estimator()
        self.image_path: Path | None = None
        self.init_ui()

    def init_ui(self):
        root = QVBoxLayout(self)
        tabs = QTabWidget()
        tabs.addTab(self.build_manual_tab(), "水位/面积估算")
        tabs.addTab(self.build_image_tab(), "影像面积估算")
        root.addWidget(tabs)

    def build_manual_tab(self) -> QWidget:
        page = QWidget()
        root = QGridLayout(page)
        root.setContentsMargins(12, 12, 12, 12)
        root.setHorizontalSpacing(14)

        form_box = QFrame()
        form_box.setFrameShape(QFrame.StyledPanel)
        form = QFormLayout(form_box)
        form.setLabelAlignment(Qt.AlignRight)

        self.manual_date = QDateEdit()
        self.manual_date.setCalendarPopup(True)
        self.manual_date.setDisplayFormat("yyyy-MM-dd")
        self.manual_date.setDate(QDate.currentDate())

        self.level_input = QLineEdit()
        self.level_input.setPlaceholderText("例如 900.0")
        attach_hint(self.level_input, "输入努列克水库水位，单位 m；留空时可用水面面积反推库容。")

        self.area_input = QLineEdit()
        self.area_input.setPlaceholderText("可选，例如 68.2")
        attach_hint(self.area_input, "输入水面面积，单位 km2；与水位同时输入时，库容以水位插值为准。")

        compute = QPushButton("开始估算")
        compute.clicked.connect(self.estimate_manual)
        config_row = QWidget()
        config_layout = QHBoxLayout(config_row)
        config_layout.setContentsMargins(0, 0, 0, 0)
        load_config = QPushButton("加载观测CSV")
        save_config = QPushButton("回写当前观测CSV")
        load_config.clicked.connect(self.load_observation_config)
        save_config.clicked.connect(self.save_observation_config)
        config_layout.addWidget(load_config)
        config_layout.addWidget(save_config)

        form.addRow("日期", self.manual_date)
        form.addRow("水位 (m)", self.level_input)
        form.addRow("水面面积 (km2)", self.area_input)
        form.addRow("配置", config_row)
        form.addRow("", compute)

        self.manual_result = QTextEdit()
        self.manual_result.setReadOnly(True)
        self.manual_result.setMinimumWidth(380)

        self.manual_plot = PlotLabel("库容曲线图")

        root.addWidget(form_box, 0, 0)
        root.addWidget(self.manual_result, 1, 0)
        root.addWidget(self.manual_plot, 0, 1, 2, 1)
        root.setColumnStretch(0, 1)
        root.setColumnStretch(1, 2)
        return page

    def build_image_tab(self) -> QWidget:
        page = QWidget()
        root = QGridLayout(page)
        root.setContentsMargins(12, 12, 12, 12)
        root.setHorizontalSpacing(14)

        form_box = QFrame()
        form_box.setFrameShape(QFrame.StyledPanel)
        form = QFormLayout(form_box)
        form.setLabelAlignment(Qt.AlignRight)

        self.image_date = QDateEdit()
        self.image_date.setCalendarPopup(True)
        self.image_date.setDisplayFormat("yyyy-MM-dd")
        self.image_date.setDate(QDate.currentDate())

        file_row = QWidget()
        file_layout = QHBoxLayout(file_row)
        file_layout.setContentsMargins(0, 0, 0, 0)
        self.image_file_label = QLabel("未选择文件")
        choose_file = QPushButton("选择影像")
        choose_file.clicked.connect(self.choose_image)
        file_layout.addWidget(self.image_file_label, 1)
        file_layout.addWidget(choose_file)

        self.pixel_size_input = QLineEdit()
        self.pixel_size_input.setPlaceholderText("GeoTIFF 可留空；普通图片填像元大小，如 10")
        attach_hint(self.pixel_size_input, "非地理参考影像必须填写像元大小，单位 m。")

        self.threshold_input = QLineEdit()
        self.threshold_input.setPlaceholderText("留空使用 Otsu 自动阈值")
        attach_hint(self.threshold_input, "灰度水体分割阈值，范围通常为 0 到 1；留空时自动计算。")

        self.water_mode = QComboBox()
        self.water_mode.addItems(["dark", "bright"])
        attach_hint(self.water_mode, "dark 表示水体在灰度图中更暗；bright 表示水体更亮。")

        compute = QPushButton("提取面积并估算")
        compute.clicked.connect(self.estimate_image)

        form.addRow("日期", self.image_date)
        form.addRow("影像", file_row)
        form.addRow("像元大小 (m)", self.pixel_size_input)
        form.addRow("阈值", self.threshold_input)
        form.addRow("水体模式", self.water_mode)
        form.addRow("", compute)

        self.image_result = QTextEdit()
        self.image_result.setReadOnly(True)
        self.image_result.setMinimumWidth(380)

        self.image_plot = PlotLabel("影像估算曲线图")

        root.addWidget(form_box, 0, 0)
        root.addWidget(self.image_result, 1, 0)
        root.addWidget(self.image_plot, 0, 1, 2, 1)
        root.setColumnStretch(0, 1)
        root.setColumnStretch(1, 2)
        return page

    def choose_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择 raw 中的遥感影像",
            standard_dialog_dir("raw"),
            "Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp);;All files (*.*)",
        )
        if path:
            try:
                self.image_path = ensure_raw_source_path(path)
                self.image_file_label.setText(self.image_path.name)
            except Exception as exc:
                QMessageBox.warning(self, "输入路径错误", str(exc))

    def load_observation_config(self):
        path = _reservoir_raw_paths()["observations"]
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as file:
                rows = list(csv.DictReader(file))
            if not rows:
                raise ValueError(f"观测 CSV 没有数据行: {path}")
            row = rows[-1]
            date_value = QDate.fromString(str(row.get("date", "")), "yyyy-MM-dd")
            if date_value.isValid():
                self.manual_date.setDate(date_value)
            self.level_input.setText(str(row.get("water_level_m", "")))
            self.area_input.setText(str(row.get("surface_area_km2", "")))
            QMessageBox.information(self, "加载完成", f"已加载: {path}")
        except Exception as exc:
            QMessageBox.critical(self, "加载失败", str(exc))

    def _record_manual_observation(self, result=None) -> Path:
        path = _reservoir_raw_paths()["observations"]
        rows: list[dict] = []
        if path.exists():
            with path.open("r", encoding="utf-8-sig", newline="") as file:
                rows = list(csv.DictReader(file))
        date_text = self.manual_date.date().toString("yyyy-MM-dd")
        level_text = self.level_input.text().strip()
        area_text = self.area_input.text().strip()
        current_row = {
            "date": date_text,
            "reservoir_id": "nurek",
            "sensor": "manual_ui",
            "water_level_m": level_text,
            "surface_area_km2": area_text,
            "source": "reservoir_estimation_plugin_ui",
            "quality_status": "user_entered; not independently verified",
        }
        if not rows or any(str(rows[-1].get(key, "")) != str(value) for key, value in current_row.items()):
            rows.append(current_row)
        write_standard_csv(
            path,
            fieldnames=[
                "date",
                "reservoir_id",
                "sensor",
                "water_level_m",
                "surface_area_km2",
                "source",
                "quality_status",
            ],
            rows=rows,
        )
        write_raw_metadata_sidecar(
            path,
            data_role="reservoir_observation_record",
            data_status="user_entered",
            source_name="M09 界面观测配置",
            source_files=[Path(__file__).resolve()],
            extra={
                "data_type": "reservoir_observation_record",
                "dataset_name": "M09 reservoir observations",
                "source_origin": "reservoir_estimation_plugin UI",
                "consumer_modules": ["M09"],
                "date": date_text,
                "is_module_native": True,
            },
        )
        return path

    def save_observation_config(self):
        try:
            if not self.level_input.text().strip() and not self.area_input.text().strip():
                raise ValueError("水位和水面面积不能同时为空。")
            path = self._record_manual_observation()
            QMessageBox.information(self, "回写完成", f"已写入: {path}")
        except Exception as exc:
            QMessageBox.critical(self, "回写失败", str(exc))

    def estimate_manual(self):
        try:
            date = self.manual_date.date().toString("yyyy-MM-dd")
            level = parse_optional_float(self.level_input.text())
            area = parse_optional_float(self.area_input.text())
            result = self.estimator.estimate_manual(date=date, water_level_m=level, area_km2=area)
            self.manual_result.setPlainText(result.summary_text())
            self.show_plot(result, self.manual_plot)
            standard_outputs = self._export_standard_result(result, source_files=[])
            self.manual_result.append(
                "\n标准成果已同步:\n"
                f"- {standard_outputs['storage'].name}\n"
                "下泄流量未写入：当前模块没有真实观测或调度模型输入。"
            )
        except Exception as exc:
            QMessageBox.critical(self, "估算失败", str(exc))

    def estimate_image(self):
        try:
            if self.image_path is None:
                raise ValueError("请先选择影像文件。")
            date = self.image_date.date().toString("yyyy-MM-dd")
            pixel_size = parse_optional_float(self.pixel_size_input.text())
            threshold = parse_optional_float(self.threshold_input.text())
            area_result, estimate = self.estimator.estimate_from_image(
                image_path=self.image_path,
                date=date,
                pixel_size_m=pixel_size,
                threshold=threshold,
                water_mode=self.water_mode.currentText(),
            )
            lines = [
                f"Image: {area_result.image_path}",
                f"Detected water area: {area_result.area_km2:.3f} km2",
                f"Water pixels: {area_result.water_pixels} / {area_result.total_pixels}",
                f"Pixel area: {area_result.pixel_area_m2:.3f} m2",
                f"Threshold: {area_result.threshold:.4f}",
                f"Water mode: {area_result.water_mode}",
                "",
                estimate.summary_text(),
            ]
            self.image_result.setPlainText("\n".join(lines))
            self.show_plot(estimate, self.image_plot)
            standard_outputs = self._export_standard_result(estimate, source_files=[self.image_path])
            self.image_result.append(
                "\n标准成果已同步:\n"
                f"- {standard_outputs['storage'].name}\n"
                "下泄流量未写入：当前模块没有真实观测或调度模型输入。"
            )
        except Exception as exc:
            QMessageBox.critical(self, "影像估算失败", str(exc))

    def show_plot(self, result, label: "PlotLabel"):
        save_curve_plot(result, PLOT_PATH)
        label.set_plot(PLOT_PATH)

    def _export_standard_result(self, result, *, source_files: list[Path]) -> dict[str, Path]:
        source_files = [ensure_raw_source_path(path) for path in source_files]
        context = infer_run_context_from_path(source_files[0]) if source_files else _context_from_result_date(result.date)
        raw_paths = _reservoir_raw_paths()
        observation_source = self._record_manual_observation(result) if not source_files else None
        storage_csv = module_output_path("M09", context=context, output_index=0)
        summary_json = module_output_path("M09", context=context, output_index=1)
        source_items = [raw_paths["parameters"], raw_paths["curve"], *source_files]
        if observation_source is not None:
            source_items.append(observation_source)

        storage_million_m3 = float(result.estimated_volume_mcm)
        storage_rows = [
            {
                "date": result.date or period_to_date(context.period),
                "period": context.period,
                "period_name": context.period_name,
                "scheme": context.scheme,
                "scheme_name": context.scheme_name,
                "module_code": "M09",
                "reservoir_name": "Nurek",
                "input_type": result.input_type,
                "water_level_m": result.water_level_m,
                "area_km2": result.area_km2,
                "storage_million_m3": storage_million_m3,
                "storage_10k_m3": storage_million_m3 * 100.0,
                "storage_km3": result.estimated_volume_km3,
                "total_capacity_percent": result.total_capacity_percent,
                "active_storage_percent": result.active_storage_percent,
                "method": result.method,
                "warning": "; ".join(result.warnings),
            }
        ]
        write_standard_csv(
            storage_csv,
            fieldnames=[
                "date",
                "period",
                "period_name",
                "scheme",
                "scheme_name",
                "module_code",
                "reservoir_name",
                "input_type",
                "water_level_m",
                "area_km2",
                "storage_million_m3",
                "storage_10k_m3",
                "storage_km3",
                "total_capacity_percent",
                "active_storage_percent",
                "method",
                "warning",
            ],
            rows=storage_rows,
        )

        write_metadata_sidecar(
            storage_csv,
            module_code="M09",
            field="storage",
            source_files=source_items,
            extra={
                "scheme": context.scheme,
                "scheme_name": context.scheme_name,
                "period": context.period,
                "period_name": context.period_name,
                "reservoir_name": "Nurek",
                "storage_unit_columns": {
                    "storage_million_m3": "百万m3",
                    "storage_10k_m3": "万m3",
                    "storage_km3": "km3",
                },
                "threshold_or_config": {
                    "curve_source": str(raw_paths["curve"]),
                    "total_capacity_km3": self.estimator.total_capacity_km3,
                    "active_storage_km3": self.estimator.active_storage_km3,
                },
            },
        )
        summary_payload = {
            "date": result.date or period_to_date(context.period),
            "reservoir_name": "Nurek",
            "input_type": result.input_type,
            "estimated_storage_mcm": storage_million_m3,
            "estimated_storage_km3": result.estimated_volume_km3,
            "curve_source": str(raw_paths["curve"]),
            "outflow_generated": False,
            "warnings": result.warnings,
        }
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary_json.write_text(
            json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        write_metadata_sidecar(
            summary_json,
            module_code="M09",
            field="reservoir_estimation_summary",
            source_files=source_items,
            extra={
                "output_type": "reservoir_estimation_summary",
                "threshold_or_config": {"curve_source": str(raw_paths["curve"])},
                "outflow_generated": False,
            },
        )
        mark_module_complete(context, "M09")
        return {"storage": storage_csv, "summary": summary_json}


class PlotLabel(QLabel):
    def __init__(self, text: str):
        super().__init__(text)
        self._pixmap = None
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumHeight(430)
        self.setStyleSheet("background:#f8fafc;border:1px solid #cbd5e1;color:#334155;")

    def set_plot(self, path: Path):
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            self._pixmap = None
            self.setText("曲线图生成失败。")
            return
        self._pixmap = pixmap
        self._refresh()

    def resizeEvent(self, event):
        self._refresh()
        super().resizeEvent(event)

    def _refresh(self):
        if self._pixmap is not None:
            self.setPixmap(self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


def parse_optional_float(text: str) -> float | None:
    cleaned = text.strip()
    if cleaned == "":
        return None
    return float(cleaned)
