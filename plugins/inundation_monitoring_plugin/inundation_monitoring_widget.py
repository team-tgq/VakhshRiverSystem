import os
import cv2
import numpy as np

from PyQt5.QtWidgets import (
    QWidget,
    QLabel,
    QPushButton,
    QFileDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLineEdit,
    QMessageBox,
    QTextEdit,
    QSizePolicy,
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

from algorithms.inundation_monitoring.predictor import FloodPredictor
from app.ui_hints import attach_hint, create_hint_badge, label_with_hint


class ImageLabel(QLabel):
    def __init__(self, text="No Image"):
        super().__init__(text)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(260, 200)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background:#f0f0f0;border:1px solid #ccc;")
        self._pix = None

    def set_qimage(self, qimg: QImage):
        pix = QPixmap.fromImage(qimg)
        self._pix = pix
        self._refresh()

    def clear_image(self, text="No Image"):
        self._pix = None
        self.clear()
        self.setText(text)

    def _refresh(self):
        if self._pix is not None:
            self.setPixmap(
                self._pix.scaled(
                    self.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
            )

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

        # 阈值输入
        thresh_layout = QHBoxLayout()
        self.thresh_input = QLineEdit("0.5")
        thresh_hint = "内容：淹没区概率阈值。\n格式：0 到 1 之间浮点数，例如 0.5。"
        attach_hint(self.thresh_input, thresh_hint)

        thresh_layout.addWidget(label_with_hint("淹没区识别阈值", thresh_hint, stretch=False))
        thresh_layout.addWidget(self.thresh_input)
        layout.addLayout(thresh_layout)

        # 文件选择按钮
        btn_layout = QHBoxLayout()

        self.btn_select = QPushButton("选择SAR影像")
        self.btn_select.clicked.connect(self.select_image)
        sar_hint = (
            "内容：用于淹没识别的 SAR 雷达影像（合成孔径雷达）。\n"
            "格式：.tif/.tiff；建议为单波段灰度强度图，影像已做基础几何校正。"
        )
        attach_hint(self.btn_select, sar_hint)

        self.btn_open_overlay = QPushButton("打开结果图")
        self.btn_open_overlay.clicked.connect(self.open_overlay_file)

        self.btn_open_mask = QPushButton("打开掩码图")
        self.btn_open_mask.clicked.connect(self.open_mask_file)

        btn_layout.addWidget(self.btn_select)
        btn_layout.addWidget(create_hint_badge(sar_hint))
        btn_layout.addWidget(self.btn_open_overlay)
        btn_layout.addWidget(self.btn_open_mask)

        layout.addLayout(btn_layout)

        sar_desc = QLabel("SAR 影像说明：SAR 是雷达遥感影像（非普通可见光照片），请优先选择 .tif/.tiff 数据。")
        sar_desc.setWordWrap(True)
        sar_desc.setStyleSheet("color:#555;font-size:12px;")
        layout.addWidget(sar_desc)

        # 图像显示
        img_layout = QHBoxLayout()

        self.label_orig = ImageLabel("原图")
        self.label_result = ImageLabel("识别结果图")

        img_layout.addWidget(self.label_orig, 1)
        img_layout.addWidget(self.label_result, 1)

        layout.addLayout(img_layout)

        # 日志
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log)

    def select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择SAR影像",
            "",
            "TIF Files (*.tif *.tiff)"
        )

        if file_path:
            self.current_image_path = file_path
            self.run_prediction(file_path)

    def run_prediction(self, img_path: str):
        try:
            thresh = float(self.thresh_input.text().strip())

            if not (0.0 <= thresh <= 1.0):
                raise ValueError("阈值必须在 0 到 1 之间")

            self.log.append(f"开始推理: {img_path}")
            self.log.append(f"阈值: {thresh}")

            result = self.predictor.predict(img_path, thresh=thresh)
            self.last_result = result

            self.show_gray_image(self.label_orig, result["original"])
            self.show_color_image(self.label_result, result["overlay"])

            self.log.append(f"设备: {result['device']}")
            self.log.append(f"掩码输出: {result['mask_path']}")
            self.log.append(f"结果图输出: {result['overlay_path']}")
            self.log.append("推理完成\n")

        except Exception as e:
            self.log.append(f"[ERROR] {str(e)}\n")
            QMessageBox.critical(self, "错误", str(e))

    def open_overlay_file(self):
        if not self.last_result:
            QMessageBox.warning(self, "提示", "请先运行识别")
            return

        path = self.last_result["overlay_path"]
        if os.path.exists(path):
            os.startfile(path)
        else:
            QMessageBox.warning(self, "错误", f"文件不存在:\n{path}")

    def open_mask_file(self):
        if not self.last_result:
            QMessageBox.warning(self, "提示", "请先运行识别")
            return

        path = self.last_result["mask_path"]
        if os.path.exists(path):
            os.startfile(path)
        else:
            QMessageBox.warning(self, "错误", f"文件不存在:\n{path}")

    def show_gray_image(self, label: ImageLabel, img: np.ndarray):
        img = (img * 255).astype(np.uint8)

        qimg = QImage(
            img.data,
            img.shape[1],
            img.shape[0],
            img.shape[1],
            QImage.Format_Grayscale8
        ).copy()

        label.set_qimage(qimg)

    def show_color_image(self, label: ImageLabel, img: np.ndarray):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        qimg = QImage(
            img_rgb.data,
            img_rgb.shape[1],
            img_rgb.shape[0],
            img_rgb.shape[1] * 3,
            QImage.Format_RGB888
        ).copy()

        label.set_qimage(qimg)
