import traceback

from PyQt5.QtWidgets import (
    QLabel,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from algorithms.routing.unity_visualization import launch_unity_visualization


class RoutingWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        title = QLabel("洪水演进与汇流三维可视化模块")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")

        description = QLabel(
            "该模块用于启动 Unity 洪水演进三维可视化程序。"
            "Qt 负责入口、状态提示和异常提示，Unity 负责三维场景展示。"
        )
        description.setWordWrap(True)

        self.run_btn = QPushButton("启动三维可视化")
        self.status_label = QLabel("状态: 未启动")
        self.log = QTextEdit()
        self.log.setReadOnly(True)

        layout.addWidget(title)
        layout.addWidget(description)
        layout.addWidget(self.run_btn)
        layout.addWidget(self.status_label)
        layout.addWidget(self.log)

        self.run_btn.clicked.connect(self.launch_unity_module)

    def launch_unity_module(self):
        try:
            self.log.append("正在启动三维可视化程序...")
            result = launch_unity_visualization()
            self.status_label.setText(f"状态: 已启动，PID={result.pid}")
            self.log.append("三维可视化程序已启动。")
        except Exception as e:
            self.status_label.setText("状态: 启动失败")
            self.log.append(str(e))
            self.log.append(traceback.format_exc())
            QMessageBox.critical(self, "启动失败", str(e))
