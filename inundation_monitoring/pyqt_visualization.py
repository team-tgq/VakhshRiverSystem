import sys
from pathlib import Path

import numpy as np
from PIL import Image
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from inundation_inference import DEFAULT_WEIGHT, load_model, predict_mask


def array_to_pixmap(arr):
    arr = np.ascontiguousarray(arr.astype(np.uint8))
    height, width, channels = arr.shape
    qimg = QImage(arr.data, width, height, channels * width, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


class ImageView(QLabel):
    def __init__(self, title):
        super().__init__(title)
        self._pixmap = None
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(420, 360)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(
            "QLabel { background: #101418; color: #cdd6df; border: 1px solid #27313a; font-size: 16px; }"
        )

    def set_image(self, arr):
        self._pixmap = array_to_pixmap(arr)
        self._rescale()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self):
        if self._pixmap is None:
            return
        self.setPixmap(
            self._pixmap.scaled(
                self.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )


class PredictWorker(QThread):
    finished = pyqtSignal(object, object, object)
    failed = pyqtSignal(str)

    def __init__(self, image_path, threshold, model, device):
        super().__init__()
        self.image_path = image_path
        self.threshold = threshold
        self.model = model
        self.device = device

    def run(self):
        try:
            rgb, mask, overlay, _ = predict_mask(
                self.image_path,
                model=self.model,
                device=self.device,
                threshold=self.threshold,
                weight_path=DEFAULT_WEIGHT,
            )
            self.finished.emit(rgb, mask, overlay)
        except Exception as exc:
            self.failed.emit(str(exc))


class InundationWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("遥感图像淹没区监测系统")
        self.resize(1120, 720)
        self.image_path = None
        self.model = None
        self.device = None
        self.worker = None
        self.last_overlay = None

        self.open_button = QPushButton("上传遥感图像")
        self.open_button.clicked.connect(self.open_image)
        self.open_button.setMinimumHeight(38)

        self.save_button = QPushButton("保存识别结果")
        self.save_button.clicked.connect(self.save_overlay)
        self.save_button.setEnabled(False)
        self.save_button.setMinimumHeight(38)

        self.threshold_label = QLabel("阈值: 0.50")
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(10, 90)
        self.threshold_slider.setValue(50)
        self.threshold_slider.valueChanged.connect(self.update_threshold_label)

        self.original_view = ImageView("原图")
        self.result_view = ImageView("识别结果")

        controls = QHBoxLayout()
        controls.addWidget(self.open_button)
        controls.addWidget(self.save_button)
        controls.addSpacing(20)
        controls.addWidget(self.threshold_label)
        controls.addWidget(self.threshold_slider, 1)

        images = QHBoxLayout()
        images.addWidget(self.original_view, 1)
        images.addWidget(self.result_view, 1)

        root = QVBoxLayout()
        root.addLayout(controls)
        root.addLayout(images, 1)

        container = QWidget()
        container.setLayout(root)
        container.setStyleSheet(
            """
            QWidget { background: #161b22; color: #e6edf3; font-family: Microsoft YaHei, Arial; }
            QPushButton { background: #238636; color: white; border: 0; border-radius: 4px; padding: 8px 18px; font-size: 15px; }
            QPushButton:disabled { background: #3d444d; color: #8b949e; }
            QPushButton:hover:enabled { background: #2ea043; }
            QLabel { font-size: 14px; }
            QSlider::groove:horizontal { height: 6px; background: #30363d; border-radius: 3px; }
            QSlider::handle:horizontal { background: #58a6ff; width: 16px; margin: -5px 0; border-radius: 8px; }
            """
        )
        self.setCentralWidget(container)
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("准备就绪")

    def update_threshold_label(self):
        self.threshold_label.setText(f"阈值: {self.threshold_slider.value() / 100:.2f}")

    def ensure_model(self):
        if self.model is None:
            self.statusBar().showMessage("正在加载模型权重...")
            QApplication.processEvents()
            self.model, self.device = load_model(DEFAULT_WEIGHT)

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择遥感图像",
            "",
            "Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp);;All Files (*)",
        )
        if not path:
            return
        self.image_path = path
        self.run_prediction()

    def run_prediction(self):
        try:
            self.ensure_model()
        except Exception as exc:
            QMessageBox.critical(self, "模型加载失败", str(exc))
            self.statusBar().showMessage("模型加载失败")
            return

        self.open_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.statusBar().showMessage("正在识别淹没区...")
        threshold = self.threshold_slider.value() / 100
        self.worker = PredictWorker(self.image_path, threshold, self.model, self.device)
        self.worker.finished.connect(self.on_prediction_finished)
        self.worker.failed.connect(self.on_prediction_failed)
        self.worker.start()

    def on_prediction_finished(self, rgb, mask, overlay):
        self.original_view.set_image(rgb)
        self.result_view.set_image(overlay)
        self.last_overlay = overlay
        self.open_button.setEnabled(True)
        self.save_button.setEnabled(True)
        ratio = float(mask.sum()) / max(mask.size, 1) * 100
        self.statusBar().showMessage(f"识别完成，淹没区占比 {ratio:.2f}%")

    def on_prediction_failed(self, message):
        self.open_button.setEnabled(True)
        self.statusBar().showMessage("识别失败")
        QMessageBox.critical(self, "识别失败", message)

    def save_overlay(self):
        if self.last_overlay is None:
            return
        default_name = str(Path(self.image_path).with_name(Path(self.image_path).stem + "_inundation_overlay.png"))
        path, _ = QFileDialog.getSaveFileName(self, "保存识别结果", default_name, "PNG Image (*.png)")
        if path:
            Image.fromarray(self.last_overlay).save(path)
            self.statusBar().showMessage(f"已保存: {path}")


def main():
    app = QApplication(sys.argv)
    window = InundationWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
