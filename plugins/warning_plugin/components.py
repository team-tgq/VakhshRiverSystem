from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Qt5Agg')
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QGridLayout, QLabel, QPushButton,
                             QCheckBox, QSlider, QComboBox, QTableWidget,
                             QTableWidgetItem, QGroupBox, QSplitter, QTabWidget,
                             QScrollArea, QFrame, QProgressBar, QMessageBox)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QColor, QPalette, QIcon
class MetricCard(QFrame):
    def __init__(self, title, value, unit="", parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Box | QFrame.Raised)
        self.setLineWidth(2)
        layout = QVBoxLayout()
        self.title_label = QLabel(title)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setFont(QFont("Arial", 10, QFont.Bold))
        self.value_label = QLabel(str(value))
        self.value_label.setAlignment(Qt.AlignCenter)
        self.value_label.setFont(QFont("Arial", 16, QFont.Bold))
        self.unit_label = QLabel(unit)
        self.unit_label.setAlignment(Qt.AlignCenter)
        self.unit_label.setFont(QFont("Arial", 8))
        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)
        layout.addWidget(self.unit_label)
        self.setLayout(layout)
        self.setMinimumHeight(100)

    def update_value(self, value, unit=""):
        self.value_label.setText(str(value))
        self.unit_label.setText(unit)

class ControlPanel(QWidget):
    refresh_clicked = pyqtSignal()
    filter_changed = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        title = QLabel("⚙️ 控制面板")
        title.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(title)

        # 刷新控制
        refresh_group = QGroupBox("刷新控制")
        refresh_layout = QVBoxLayout()
        self.auto_refresh_check = QCheckBox("自动刷新")
        self.auto_refresh_check.setChecked(True)
        refresh_layout.addWidget(self.auto_refresh_check)
        refresh_layout.addWidget(QLabel("刷新间隔(秒):"))
        self.refresh_slider = QSlider(Qt.Horizontal)
        self.refresh_slider.setRange(5,60)
        self.refresh_slider.setValue(30)
        self.refresh_slider.setTickPosition(QSlider.TicksBelow)
        self.refresh_slider.setTickInterval(5)
        refresh_layout.addWidget(self.refresh_slider)
        self.refresh_interval_label = QLabel(f"当前间隔: {self.refresh_slider.value()}秒")
        refresh_layout.addWidget(self.refresh_interval_label)
        self.refresh_slider.valueChanged.connect(self.update_refresh_label)
        self.refresh_btn = QPushButton("🔄 手动刷新")
        self.refresh_btn.clicked.connect(self.refresh_clicked.emit)
        refresh_layout.addWidget(self.refresh_btn)
        refresh_group.setLayout(refresh_layout)
        layout.addWidget(refresh_group)


        # 预警筛选
        filter_group = QGroupBox("预警筛选")
        filter_layout = QVBoxLayout()
        warning_levels = ['蓝','黄','橙','红']
        self.filter_checks = []
        for level in warning_levels:
            check = QCheckBox(level)
            check.setChecked(True)
            check.stateChanged.connect(self.on_filter_changed)
            filter_layout.addWidget(check)
            self.filter_checks.append(check)
        filter_group.setLayout(filter_layout)
        layout.addWidget(filter_group)

        # 系统信息
        info_group = QGroupBox("系统信息")
        info_layout = QVBoxLayout()
        info_labels = [
            "版本: v2.0",
            "数据源: 模拟数据",
            f"最后更新: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "监控区域: 目标流域"
        ]
        for info in info_labels:
            label = QLabel(info)
            label.setFont(QFont("Arial", 9))
            info_layout.addWidget(label)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        layout.addStretch()
        self.setLayout(layout)

    def update_refresh_label(self):
        interval = self.refresh_slider.value()
        self.refresh_interval_label.setText(f"当前间隔: {interval}秒")

    def on_filter_changed(self):
        selected = []
        for check, level in zip(self.filter_checks, ['蓝','黄','橙','红']):
            if check.isChecked():
                selected.append(level)
        self.filter_changed.emit(selected)

class StationTable(QTableWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setColumnCount(6)
        self.setHorizontalHeaderLabels(['站点名称','预警等级','水位(m)','流量(m³/s)','置信度','状态'])
        self.setColumnWidth(0,100); self.setColumnWidth(1,100); self.setColumnWidth(2,80)
        self.setColumnWidth(3,100); self.setColumnWidth(4,80); self.setColumnWidth(5,80)
        self.setAlternatingRowColors(True)
        self.setSortingEnabled(True)

    def update_table(self, station_df, filter_levels=None):
        self.clearContents()
        if filter_levels:
            filtered_df = station_df[station_df['预警等级'].isin(filter_levels)]
        else:
            filtered_df = station_df
        self.setRowCount(len(filtered_df))
        for i, (_, row) in enumerate(filtered_df.iterrows()):
            name_item = QTableWidgetItem(row['站点名称'])
            name_item.setTextAlignment(Qt.AlignCenter)
            self.setItem(i,0,name_item)
            level_item = QTableWidgetItem(row['预警等级'])
            level_item.setTextAlignment(Qt.AlignCenter)
            color_map = {'蓝':QColor('#1f77b4'), '黄':QColor('yellow'), '橙':QColor('#ff9900'), '红':QColor('#d62728')}
            level_item.setBackground(color_map.get(row['预警等级'], QColor('white')))
            self.setItem(i,1,level_item)
            water_item = QTableWidgetItem(f"{row['水位(m)']:.2f}")
            water_item.setTextAlignment(Qt.AlignCenter)
            self.setItem(i,2,water_item)
            flow_item = QTableWidgetItem(f"{row['流量(m³/s)']:.1f}")
            flow_item.setTextAlignment(Qt.AlignCenter)
            self.setItem(i,3,flow_item)
            conf_item = QTableWidgetItem(f"{row['置信度']:.1%}")
            conf_item.setTextAlignment(Qt.AlignCenter)
            self.setItem(i,4,conf_item)
            status = "正常" if row['预警等级']=='蓝' else "预警"
            status_item = QTableWidgetItem(status)
            status_item.setTextAlignment(Qt.AlignCenter)
            self.setItem(i,5,status_item)