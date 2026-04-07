import os

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QTextEdit, QMessageBox, QComboBox, QLineEdit, QSizePolicy
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

from algorithms.segformer_service.service_config import TASKS
from algorithms.segformer_service.service_runner import run_segformer_service
from app.ui_hints import attach_hint, label_with_hint


class ImageLabel(QLabel):
    def __init__(self, text="No Image"):
        super().__init__(text)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(260, 200)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background:#f0f0f0;border:1px solid #ccc;")
        self._pix = None

    def set_image(self, path):
        if path and os.path.exists(path):
            pix = QPixmap(path)
            if not pix.isNull():
                self._pix = pix
                self._refresh()
                return
        self._pix = None
        self.clear()
        self.setText("Image Not Found")

    def _refresh(self):
        if self._pix is not None:
            self.setPixmap(self._pix.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        self._refresh()
        super().resizeEvent(event)


class SegFormerWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.current_task = "water"
        self.image_path = ""
        self.result_path = ""
        self.init_ui()
        self.update_defaults()

    def init_ui(self):
        layout = QVBoxLayout(self)

        title = QLabel("SegFormer 专题识别服务")
        title.setStyleSheet("font-size:18px;font-weight:bold;")
        layout.addWidget(title)

        task_row = QHBoxLayout()
        task_hint = "内容：分割任务类型。\n格式：下拉选择，当前支持 water / snow。"
        task_row.addWidget(label_with_hint("任务:", task_hint, stretch=False))

        self.task_combo = QComboBox()
        self.task_combo.addItem("水体识别", "water")
        self.task_combo.addItem("积雪识别", "snow")
        self.task_combo.currentIndexChanged.connect(self.on_task_changed)
        attach_hint(self.task_combo, task_hint)

        self.device_combo = QComboBox()
        self.device_combo.addItems(["cpu", "cuda:0"])
        device_hint = "内容：推理设备。\n格式：下拉选择，cpu 或 cuda:0。"
        attach_hint(self.device_combo, device_hint)

        task_row.addWidget(self.task_combo)
        task_row.addWidget(label_with_hint("设备:", device_hint, stretch=False))
        task_row.addWidget(self.device_combo)
        task_row.addStretch()

        layout.addLayout(task_row)

        path_row = QHBoxLayout()
        self.image_edit = QLineEdit()
        self.image_edit.setReadOnly(True)
        image_hint = "内容：输入影像文件路径。\n格式：png/jpg/jpeg/bmp 文件路径（通过“选择输入图片”填写）。"
        attach_hint(self.image_edit, image_hint)
        self.image_btn = QPushButton("选择输入图片")
        self.image_btn.clicked.connect(self.select_image)
        path_row.addWidget(label_with_hint("图片:", image_hint, stretch=False))
        path_row.addWidget(self.image_edit, 1)
        path_row.addWidget(self.image_btn)
        layout.addLayout(path_row)

        btn_row = QHBoxLayout()
        self.run_btn = QPushButton("运行分割")
        self.load_btn = QPushButton("加载已有结果")
        self.run_btn.clicked.connect(self.run_task)
        self.load_btn.clicked.connect(self.load_existing_result)
        btn_row.addWidget(self.run_btn)
        btn_row.addWidget(self.load_btn)
        layout.addLayout(btn_row)

        img_row = QHBoxLayout()
        self.input_label = ImageLabel("输入图")
        self.result_label = ImageLabel("结果图")
        img_row.addWidget(self.input_label, 1)
        img_row.addWidget(self.result_label, 1)
        layout.addLayout(img_row)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log)

    def on_task_changed(self):
        self.current_task = self.task_combo.currentData()
        self.update_defaults()

    def update_defaults(self):
        task = TASKS[self.current_task]
        if os.path.exists(task["input_dir"]):
            files = [
                os.path.join(task["input_dir"], f)
                for f in os.listdir(task["input_dir"])
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
            ]
            if files:
                self.image_path = files[0]
                self.image_edit.setText(self.image_path)
                self.input_label.set_image(self.image_path)

    def select_image(self):
        task = TASKS[self.current_task]
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择输入图片",
            task["input_dir"],
            "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_path:
            self.image_path = file_path
            self.image_edit.setText(file_path)
            self.input_label.set_image(file_path)

    def run_task(self):
        try:
            if not self.image_path:
                raise ValueError("请先选择输入图片")

            self.log.append(f"开始运行任务: {TASKS[self.current_task]['name']}")

            result = run_segformer_service(
                task_key=self.current_task,
                image_path=self.image_path,
                device=self.device_combo.currentText()
            )

            self.log.append("执行命令:")
            self.log.append(" ".join(result["command"]))
            self.log.append(f"返回码: {result['returncode']}")

            if result["stdout"]:
                self.log.append("STDOUT:")
                self.log.append(result["stdout"])
            if result["stderr"]:
                self.log.append("STDERR:")
                self.log.append(result["stderr"])

            if result["returncode"] != 0:
                raise RuntimeError("SegFormer 推理服务执行失败，请查看日志。")

            if not os.path.exists(result["overlay_path"]):
                raise FileNotFoundError(f"结果图未生成: {result['overlay_path']}")

            self.result_path = result["overlay_path"]
            self.result_label.set_image(self.result_path)
            self.log.append(f"结果图: {self.result_path}")

        except Exception as e:
            self.log.append(f"[ERROR] {str(e)}")
            QMessageBox.critical(self, "错误", str(e))

    def load_existing_result(self):
        task = TASKS[self.current_task]
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "加载已有结果",
            task["output_dir"],
            "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_path:
            self.result_path = file_path
            self.result_label.set_image(file_path)
            self.log.append(f"已加载结果: {file_path}")
