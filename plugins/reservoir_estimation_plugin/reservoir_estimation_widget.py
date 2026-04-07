import os
import geopandas as gpd

from PyQt5.QtWidgets import (
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QMessageBox,
    QComboBox,
    QDateEdit
)
from PyQt5.QtCore import QDate

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from algorithms.reservoir_estimation.main_v import run_estimation
from algorithms.reservoir_estimation.reservoir_config import RESERVOIR_CONFIGS
from app.ui_hints import attach_hint, label_with_hint


class MapCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)

    def plot_region(self, reservoir_name, lon, lat):
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        region_path = os.path.join(
            project_root,
            "algorithms",
            "reservoir_estimation",
            "data",
            "region.shp"
        )

        self.ax.clear()

        if os.path.exists(region_path):
            try:
                gdf = gpd.read_file(region_path)
                gdf.plot(
                    ax=self.ax,
                    color="lightblue",
                    edgecolor="black",
                    alpha=0.5
                )
            except Exception as e:
                print(f"读取 region.shp 失败: {e}")

        self.ax.scatter(
            lon,
            lat,
            color="red",
            s=80,
            zorder=5
        )

        self.ax.text(
            lon,
            lat,
            f" {reservoir_name}",
            fontsize=12,
            color="red",
            weight="bold"
        )

        self.ax.set_title("Reservoir Region Map")
        self.ax.set_xlabel("Longitude")
        self.ax.set_ylabel("Latitude")
        self.draw()


class ReservoirEstimationWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # 水库选择
        name_layout = QHBoxLayout()
        name_label = QLabel("水库名称")

        self.name_combo = QComboBox()
        self.name_combo.addItems(list(RESERVOIR_CONFIGS.keys()))
        name_hint = "内容：待估算水库名称。\n格式：从下拉列表选择。"
        attach_hint(self.name_combo, name_hint)

        name_layout.addWidget(label_with_hint(name_label, name_hint, stretch=False))
        name_layout.addWidget(self.name_combo)
        layout.addLayout(name_layout)

        # 起始日期
        start_layout = QHBoxLayout()
        start_label = QLabel("起始日期")

        self.start_input = QDateEdit()
        self.start_input.setCalendarPopup(True)
        self.start_input.setDisplayFormat("yyyy-MM-dd")
        self.start_input.setDate(QDate(2022, 6, 1))
        start_hint = "内容：估算起始日期。\n格式：yyyy-MM-dd，例如 2022-06-01。"
        attach_hint(self.start_input, start_hint)

        start_layout.addWidget(label_with_hint(start_label, start_hint, stretch=False))
        start_layout.addWidget(self.start_input)
        layout.addLayout(start_layout)

        # 结束日期
        end_layout = QHBoxLayout()
        end_label = QLabel("结束日期")

        self.end_input = QDateEdit()
        self.end_input.setCalendarPopup(True)
        self.end_input.setDisplayFormat("yyyy-MM-dd")
        self.end_input.setDate(QDate(2022, 6, 7))
        end_hint = "内容：估算结束日期。\n格式：yyyy-MM-dd，例如 2022-06-07。"
        attach_hint(self.end_input, end_hint)

        end_layout.addWidget(label_with_hint(end_label, end_hint, stretch=False))
        end_layout.addWidget(self.end_input)
        layout.addLayout(end_layout)

        # 运行按钮
        self.run_button = QPushButton("运行计算")
        self.run_button.clicked.connect(self.run_estimation_task)
        layout.addWidget(self.run_button)

        # 地图
        self.canvas = MapCanvas()
        layout.addWidget(self.canvas)

        # CSV 按钮
        csv_layout = QVBoxLayout()
        self.csv_files = [
            "estimation_area.csv",
            "estimation_level0_product.csv",
            "estimation_level1_product.csv",
            "estimation_level2_product.csv",
            "estimation_level3_product.csv",
            "estimation_level4_product.csv",
            "reservoir_hypsometry.csv"
        ]

        for file_name in self.csv_files:
            btn = QPushButton(file_name)
            btn.clicked.connect(self.open_csv)
            csv_layout.addWidget(btn)

        layout.addLayout(csv_layout)

        # 初始化地图
        self.plot_current_reservoir()
        self.name_combo.currentIndexChanged.connect(self.plot_current_reservoir)

    def get_current_reservoir_name(self):
        return self.name_combo.currentText().strip()

    def get_current_reservoir_config(self):
        reservoir_name = self.get_current_reservoir_name()
        if reservoir_name not in RESERVOIR_CONFIGS:
            raise KeyError(f"未找到水库配置: {reservoir_name}")
        return RESERVOIR_CONFIGS[reservoir_name]

    def plot_current_reservoir(self):
        try:
            reservoir_name = self.get_current_reservoir_name()
            cfg = self.get_current_reservoir_config()
            self.canvas.plot_region(
                reservoir_name,
                cfg["lon"],
                cfg["lat"]
            )
        except Exception as e:
            print(f"绘图失败: {e}")

    def run_estimation_task(self):
        reservoir_name = self.get_current_reservoir_name()
        start_date = self.start_input.date().toString("yyyy-MM-dd")
        end_date = self.end_input.date().toString("yyyy-MM-dd")

        if not reservoir_name:
            QMessageBox.warning(self, "错误", "请选择水库名称")
            return

        if self.start_input.date() > self.end_input.date():
            QMessageBox.warning(self, "错误", "起始日期不能晚于结束日期")
            return

        try:
            cfg = self.get_current_reservoir_config()

            self.run_button.setEnabled(False)
            self.run_button.setText("计算中...")

            run_estimation(
                reservoir_name,
                cfg["lon"],
                cfg["lat"],
                cfg["bbox"],
                start_date,
                end_date,
                cfg["elevation"],
                cfg["param_1"],
                cfg["year"]
            )

            QMessageBox.information(self, "完成", "计算完成")
            self.canvas.plot_region(
                reservoir_name,
                cfg["lon"],
                cfg["lat"]
            )

        except Exception as e:
            QMessageBox.warning(self, "错误", str(e))

        finally:
            self.run_button.setEnabled(True)
            self.run_button.setText("运行计算")

    def open_csv(self):
        button = self.sender()
        file_name = button.text()
        reservoir_name = self.get_current_reservoir_name()

        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )

        path = os.path.join(
            project_root,
            "algorithms",
            "reservoir_estimation",
            "output",
            reservoir_name,
            file_name
        )

        if os.path.exists(path):
            os.startfile(path)
        else:
            QMessageBox.warning(self, "错误", f"文件不存在:\n{path}")
