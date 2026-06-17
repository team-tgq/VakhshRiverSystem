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
from app.ui_hints import attach_hint


PLUGIN_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PLUGIN_DIR.parent.parent / "algorithms" / "reservoir_estimation" / "output" / "Nurek"
PLOT_PATH = OUTPUT_DIR / "last_estimate_plot.png"


class ReservoirEstimationWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.estimator = NurekReservoirEstimator()
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

        form.addRow("日期", self.manual_date)
        form.addRow("水位 (m)", self.level_input)
        form.addRow("水面面积 (km2)", self.area_input)
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
            "选择遥感影像",
            "",
            "Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp);;All files (*.*)",
        )
        if path:
            self.image_path = Path(path)
            self.image_file_label.setText(self.image_path.name)

    def estimate_manual(self):
        try:
            date = self.manual_date.date().toString("yyyy-MM-dd")
            level = parse_optional_float(self.level_input.text())
            area = parse_optional_float(self.area_input.text())
            result = self.estimator.estimate_manual(date=date, water_level_m=level, area_km2=area)
            self.manual_result.setPlainText(result.summary_text())
            self.show_plot(result, self.manual_plot)
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
        except Exception as exc:
            QMessageBox.critical(self, "影像估算失败", str(exc))

    def show_plot(self, result, label: "PlotLabel"):
        save_curve_plot(result, PLOT_PATH)
        label.set_plot(PLOT_PATH)


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
