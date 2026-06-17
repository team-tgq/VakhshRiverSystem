from __future__ import annotations

import sys
from pathlib import Path

try:
    from PyQt5.QtCore import QDate, Qt
    from PyQt5.QtGui import QPixmap
    from PyQt5.QtWidgets import (
        QApplication,
        QComboBox,
        QDateEdit,
        QFileDialog,
        QFormLayout,
        QFrame,
        QGridLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QTabWidget,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
except ImportError:
    from PySide6.QtCore import QDate, Qt
    from PySide6.QtGui import QPixmap
    from PySide6.QtWidgets import (
        QApplication,
        QComboBox,
        QDateEdit,
        QFileDialog,
        QFormLayout,
        QFrame,
        QGridLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QTabWidget,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )

from reservoir_core import NurekReservoirEstimator, save_curve_plot


APP_DIR = Path(__file__).resolve().parent
PLOT_PATH = APP_DIR / "last_estimate_plot.png"


class NurekEstimatorWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.estimator = NurekReservoirEstimator()
        self.image_path: Path | None = None

        self.setWindowTitle("Nurek Reservoir Basic Estimation")
        self.resize(1080, 720)

        tabs = QTabWidget()
        tabs.addTab(self._build_manual_tab(), "Manual input")
        tabs.addTab(self._build_image_tab(), "Upload image")
        self.setCentralWidget(tabs)

    def _build_manual_tab(self) -> QWidget:
        page = QWidget()
        root = QGridLayout(page)
        root.setContentsMargins(18, 18, 18, 18)
        root.setHorizontalSpacing(18)

        form_box = QFrame()
        form_box.setFrameShape(QFrame.StyledPanel)
        form = QFormLayout(form_box)
        form.setLabelAlignment(Qt.AlignRight)

        self.manual_date = QDateEdit()
        self.manual_date.setCalendarPopup(True)
        self.manual_date.setDate(QDate.currentDate())

        self.level_input = QLineEdit()
        self.level_input.setPlaceholderText("e.g. 900.0")

        self.area_input = QLineEdit()
        self.area_input.setPlaceholderText("optional, e.g. 68.2")

        compute = QPushButton("Estimate")
        compute.clicked.connect(self.estimate_manual)

        form.addRow("Date", self.manual_date)
        form.addRow("Water level (m)", self.level_input)
        form.addRow("Water area (km2)", self.area_input)
        form.addRow("", compute)

        self.manual_result = QTextEdit()
        self.manual_result.setReadOnly(True)
        self.manual_result.setMinimumWidth(390)

        self.manual_plot = QLabel("Curve plot will appear here.")
        self.manual_plot.setAlignment(Qt.AlignCenter)
        self.manual_plot.setMinimumHeight(430)
        self.manual_plot.setStyleSheet("QLabel { background: #f6f7f8; border: 1px solid #d5d8dc; }")

        root.addWidget(form_box, 0, 0)
        root.addWidget(self.manual_result, 1, 0)
        root.addWidget(self.manual_plot, 0, 1, 2, 1)
        root.setColumnStretch(0, 1)
        root.setColumnStretch(1, 2)
        return page

    def _build_image_tab(self) -> QWidget:
        page = QWidget()
        root = QGridLayout(page)
        root.setContentsMargins(18, 18, 18, 18)
        root.setHorizontalSpacing(18)

        form_box = QFrame()
        form_box.setFrameShape(QFrame.StyledPanel)
        form = QFormLayout(form_box)
        form.setLabelAlignment(Qt.AlignRight)

        self.image_date = QDateEdit()
        self.image_date.setCalendarPopup(True)
        self.image_date.setDate(QDate.currentDate())

        file_row = QWidget()
        file_layout = QHBoxLayout(file_row)
        file_layout.setContentsMargins(0, 0, 0, 0)
        self.image_file_label = QLabel("No file selected")
        choose_file = QPushButton("Choose image")
        choose_file.clicked.connect(self.choose_image)
        file_layout.addWidget(self.image_file_label, 1)
        file_layout.addWidget(choose_file)

        self.pixel_size_input = QLineEdit()
        self.pixel_size_input.setPlaceholderText("10 for Sentinel-2; optional for GeoTIFF")

        self.threshold_input = QLineEdit()
        self.threshold_input.setPlaceholderText("blank = Otsu automatic threshold")

        self.water_mode = QComboBox()
        self.water_mode.addItems(["dark", "bright"])

        compute = QPushButton("Extract area and estimate")
        compute.clicked.connect(self.estimate_image)

        form.addRow("Date", self.image_date)
        form.addRow("Image", file_row)
        form.addRow("Pixel size (m)", self.pixel_size_input)
        form.addRow("Threshold 0-1", self.threshold_input)
        form.addRow("Water mode", self.water_mode)
        form.addRow("", compute)

        self.image_result = QTextEdit()
        self.image_result.setReadOnly(True)
        self.image_result.setMinimumWidth(390)

        self.image_plot = QLabel("Image-based curve estimate will appear here.")
        self.image_plot.setAlignment(Qt.AlignCenter)
        self.image_plot.setMinimumHeight(430)
        self.image_plot.setStyleSheet("QLabel { background: #f6f7f8; border: 1px solid #d5d8dc; }")

        root.addWidget(form_box, 0, 0)
        root.addWidget(self.image_result, 1, 0)
        root.addWidget(self.image_plot, 0, 1, 2, 1)
        root.setColumnStretch(0, 1)
        root.setColumnStretch(1, 2)
        return page

    def choose_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose remote-sensing image",
            str(APP_DIR),
            "Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp);;All files (*.*)",
        )
        if path:
            self.image_path = Path(path)
            self.image_file_label.setText(self.image_path.name)

    def estimate_manual(self) -> None:
        try:
            date = self.manual_date.date().toString("yyyy-MM-dd")
            level = parse_optional_float(self.level_input.text())
            area = parse_optional_float(self.area_input.text())
            result = self.estimator.estimate_manual(date=date, water_level_m=level, area_km2=area)
            self.manual_result.setPlainText(result.summary_text())
            self._show_plot(result, self.manual_plot)
        except Exception as exc:
            QMessageBox.critical(self, "Estimation failed", str(exc))

    def estimate_image(self) -> None:
        try:
            if self.image_path is None:
                raise ValueError("Choose an image first.")
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

            area_lines = [
                f"Image: {area_result.image_path}",
                f"Detected water area: {area_result.area_km2:.3f} km2",
                f"Water pixels: {area_result.water_pixels} / {area_result.total_pixels}",
                f"Pixel area: {area_result.pixel_area_m2:.3f} m2",
                f"Threshold: {area_result.threshold:.4f}",
                f"Water mode: {area_result.water_mode}",
                "",
                estimate.summary_text(),
            ]
            self.image_result.setPlainText("\n".join(area_lines))
            self._show_plot(estimate, self.image_plot)
        except Exception as exc:
            QMessageBox.critical(self, "Image estimation failed", str(exc))

    def _show_plot(self, result, label: QLabel) -> None:
        save_curve_plot(result, PLOT_PATH)
        pixmap = QPixmap(str(PLOT_PATH))
        if pixmap.isNull():
            label.setText("Plot generation failed.")
            return
        label.setPixmap(pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


def parse_optional_float(text: str) -> float | None:
    cleaned = text.strip()
    if cleaned == "":
        return None
    return float(cleaned)


def main() -> int:
    app = QApplication(sys.argv)
    window = NurekEstimatorWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
