# ==================== 主窗口 ====================
from datetime import datetime
import warnings

from plugins.warning_plugin.charts import WarningBarChart, TimeSeriesChart, SpatialMapChart
from plugins.warning_plugin.components import ControlPanel, MetricCard, StationTable

warnings.filterwarnings('ignore')
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QGridLayout, QLabel, QPushButton,
                             QCheckBox, QSlider, QComboBox, QTableWidget,
                             QTableWidgetItem, QGroupBox, QSplitter, QTabWidget,
                             QScrollArea, QFrame, QProgressBar, QMessageBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont
from algorithms.warning.core import build_warning_system, FloodEarlyWarningSystem
from algorithms.warning.generator import FloodDataGenerator
class WarningWidget(QWidget):
    def __init__(self, warning_system: FloodEarlyWarningSystem):
        super().__init__()
        self.warning_system = build_warning_system()
        self.data_gen = FloodDataGenerator(self.warning_system)
        self.current_data = self.data_gen.generate_all_data()
        self.filter_levels = ['蓝','黄','橙','红']

        self.setWindowTitle("🌊 洪水智能预警监控模块")
        self.setGeometry(100,100,1400,900)

        main_layout = QHBoxLayout(self)

        self.control_panel = ControlPanel()
        self.control_panel.refresh_clicked.connect(self.refresh_data)
        self.control_panel.filter_changed.connect(self.on_filter_changed)
        main_layout.addWidget(self.control_panel, 1)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        title_label = QLabel("🌊 实时洪水预警监控模块")
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #1f77b4; padding: 10px;")
        right_layout.addWidget(title_label)

        self.metrics_panel = self.create_metrics_panel()
        right_layout.addWidget(self.metrics_panel)

        self.charts_panel = self.create_charts_panel()
        right_layout.addWidget(self.charts_panel, 1)

        self.table_panel = self.create_table_panel()
        right_layout.addWidget(self.table_panel, 1)

        main_layout.addWidget(right_widget, 4)

        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_data)

        self.update_display()
        self.start_auto_refresh()

    def create_metrics_panel(self):
        panel = QWidget()
        layout = QGridLayout(panel)
        self.metric_cards = []
        metric_defs = [("总站点数","sites","个"),("预警站点数","warning_sites","个"),
                       ("高风险站点","high_risk","个"),("平均置信度","avg_confidence","%"),
                       ("最高水位","max_water","m"),("最大流量","max_flow","m³/s")]
        for i, (title,key,unit) in enumerate(metric_defs):
            row = i//3; col = i%3
            card = MetricCard(title,"",unit)
            self.metric_cards.append((card,key))
            layout.addWidget(card, row, col)
        return panel

    def create_charts_panel(self):
        panel = QTabWidget()
        self.warning_chart = WarningBarChart()
        panel.addTab(self.warning_chart, "📊 预警分布")
        self.time_chart = TimeSeriesChart()
        panel.addTab(self.time_chart, "📈 趋势分析")
        self.map_chart = SpatialMapChart()
        panel.addTab(self.map_chart, "🗺️ 站点分布")
        return panel

    def create_table_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        table_title = QLabel("📋 站点详情")
        table_title.setFont(QFont("Arial",12,QFont.Bold))
        layout.addWidget(table_title)
        self.station_table = StationTable()
        layout.addWidget(self.station_table)
        status_bar = QWidget()
        status_layout = QHBoxLayout(status_bar)
        self.status_label = QLabel("就绪")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0,100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.update_time_label = QLabel("")
        status_layout.addWidget(self.status_label,1)
        status_layout.addWidget(self.progress_bar,2)
        status_layout.addWidget(self.update_time_label,1)
        layout.addWidget(status_bar)
        return panel

    def refresh_data(self):
        self.progress_bar.setValue(30)
        self.status_label.setText("正在更新数据...")
        self.current_data = self.data_gen.generate_all_data()
        self.progress_bar.setValue(60)
        self.status_label.setText("正在更新显示...")
        self.update_display()
        self.progress_bar.setValue(100)
        self.status_label.setText("数据更新完成")
        self.update_time_label.setText(f"最后更新: {datetime.now().strftime('%H:%M:%S')}")
        QTimer.singleShot(1000, lambda: self.progress_bar.setValue(0))


    def update_display(self):
        station_df = self.current_data['stations']
        metric_values = {
            'sites': len(station_df),
            'warning_sites': len(station_df[station_df['预警等级']!='蓝']),
            'high_risk': len(station_df[station_df['预警等级']=='红']),
            'avg_confidence': f"{station_df['置信度'].mean()*100:.1f}",
            'max_water': f"{station_df['水位(m)'].max():.1f}",
            'max_flow': f"{station_df['流量(m³/s)'].max():.0f}"
        }
        for card, key in self.metric_cards:
            card.update_value(metric_values[key])

        self.warning_chart.update_chart(station_df)
        self.time_chart.update_chart(self.current_data['weather'], self.current_data['hydro'])
        self.map_chart.update_chart(station_df)
        self.station_table.update_table(station_df, self.filter_levels)

    def on_filter_changed(self, levels):
        self.filter_levels = levels
        self.update_display()   # 刷新所有可视化组件

    def start_auto_refresh(self):
        if self.control_panel.auto_refresh_check.isChecked():
            interval = self.control_panel.refresh_slider.value() * 1000
            self.refresh_timer.start(interval)

    def stop_auto_refresh(self):
        self.refresh_timer.stop()

    def closeEvent(self, event):
        reply = QMessageBox.question(self, '确认退出', '确定要退出吗？', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.stop_auto_refresh()
            event.accept()
        else:
            event.ignore()