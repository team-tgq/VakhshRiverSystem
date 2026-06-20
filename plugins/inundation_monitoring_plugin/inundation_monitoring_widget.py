import os

import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from algorithms.inundation_monitoring.predictor import FloodPredictor
from app.ui_hints import attach_hint, create_hint_badge, label_with_hint


class ImageLabel(QLabel):
    def __init__(self, text="No image"):
        super().__init__(text)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(300, 220)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background:#f5f7fa;border:1px solid #cbd5e1;color:#334155;")
        self._pix = None

    def set_qimage(self, qimg: QImage):
        self._pix = QPixmap.fromImage(qimg)
        self._refresh()

    def _refresh(self):
        if self._pix is not None:
            self.setPixmap(self._pix.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        self._refresh()
        super().resizeEvent(event)


class InundationMonitoringWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.predictor = FloodPredictor()
        self.current_image_path = ""
        self.last_result = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        threshold_hint = "SegFormer 淹没概率阈值，范围 0 到 1；值越大，识别结果越保守。"
        threshold_row = QHBoxLayout()
        self.thresh_input = QLineEdit("0.50")
        attach_hint(self.thresh_input, threshold_hint)
        threshold_row.addWidget(label_with_hint("淹没识别阈值", threshold_hint, stretch=False))
        threshold_row.addWidget(self.thresh_input)
        layout.addLayout(threshold_row)

        input_hint = (
            "选择遥感影像文件。优先使用多波段 GeoTIFF；模型会构建 6 个可见/近红外/SWIR "
            "通道加 MNDWI 的 7 通道特征，普通图片会自动退化为 RGB/灰度特征。"
        )
        button_row = QHBoxLayout()
        self.btn_select = QPushButton("选择遥感影像")
        self.btn_select.clicked.connect(self.select_image)
        attach_hint(self.btn_select, input_hint)

        self.btn_open_overlay = QPushButton("打开叠加图")
        self.btn_open_overlay.clicked.connect(self.open_overlay_file)

        self.btn_open_mask = QPushButton("打开掩膜图")
        self.btn_open_mask.clicked.connect(self.open_mask_file)

        button_row.addWidget(self.btn_select)
        button_row.addWidget(create_hint_badge(input_hint))
        button_row.addWidget(self.btn_open_overlay)
        button_row.addWidget(self.btn_open_mask)
        layout.addLayout(button_row)

        image_row = QHBoxLayout()
        self.label_orig = ImageLabel("原始影像")
        self.label_result = ImageLabel("淹没识别结果")
        image_row.addWidget(self.label_orig, 1)
        image_row.addWidget(self.label_result, 1)
        layout.addLayout(image_row, 1)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(130)
        layout.addWidget(self.log)

    def select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择遥感影像",
            "",
            "Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp);;All Files (*)",
        )
        if file_path:
            self.current_image_path = file_path
            self.run_prediction(file_path)

    def run_prediction(self, img_path: str):
        try:
            thresh = float(self.thresh_input.text().strip())
            if not (0.0 <= thresh <= 1.0):
                raise ValueError("阈值必须在 0 到 1 之间。")

            self.log.append(f"开始 SegFormer 7 通道淹没识别: {img_path}")
            self.log.append(f"阈值: {thresh:.2f}")

            result = self.predictor.predict(img_path, thresh=thresh)
            self.last_result = result

            self.show_rgb_image(self.label_orig, result["original"])
            self.show_rgb_image(self.label_result, result["overlay"])

            self.log.append(f"设备: {result['device']}")
            self.log.append(f"淹没占比: {result['water_ratio'] * 100:.2f}%")
            self.log.append(f"掩膜输出: {result['mask_path']}")
            self.log.append(f"叠加图输出: {result['overlay_path']}")
            self.log.append("识别完成\n")
        except Exception as e:
            self.log.append(f"[ERROR] {e}\n")
            QMessageBox.critical(self, "识别失败", str(e))

    def open_overlay_file(self):
        self.open_result_file("overlay_path")

    def open_mask_file(self):
        self.open_result_file("mask_path")

    def open_result_file(self, key: str):
        if not self.last_result:
            QMessageBox.warning(self, "提示", "请先运行识别。")
            return
        path = self.last_result[key]
        if os.path.exists(path):
            os.startfile(path)
        else:
            QMessageBox.warning(self, "文件不存在", path)

    def show_rgb_image(self, label: ImageLabel, img: np.ndarray):
        arr = np.ascontiguousarray(img.astype(np.uint8))
        qimg = QImage(arr.data, arr.shape[1], arr.shape[0], arr.shape[1] * 3, QImage.Format_RGB888).copy()
        label.set_qimage(qimg)
