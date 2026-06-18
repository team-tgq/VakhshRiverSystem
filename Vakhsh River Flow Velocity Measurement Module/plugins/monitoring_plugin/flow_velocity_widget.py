from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QVBoxLayout, QWidget

from algorithms.monitoring.flow_velocity_app import FlowAnalysisApp


class FlowVelocityWidget(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.flow_window = FlowAnalysisApp()
        self.flow_window.setWindowFlags(Qt.Widget)
        layout.addWidget(self.flow_window)

