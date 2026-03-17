import sys
import os
import subprocess
import geopandas as gpd
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit,
    QPushButton, QVBoxLayout, QHBoxLayout,
    QMessageBox
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from main_v import run_estimation


# dam 坐标
LON = 69.348056
LAT = 38.371667


class MapCanvas(FigureCanvas):

    def __init__(self):

        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)

        super().__init__(self.fig)

    def plot_region(self, reservoir_name):

        region_path = "Estimation_v-reservoir_vakhsh_yf/data/region.shp"

        if not os.path.exists(region_path):
            print("region.shp not found")
            return

        gdf = gpd.read_file(region_path)

        self.ax.clear()

        # 读取 shapefile
        gdf = gpd.read_file(region_path)

        self.ax.clear()

        # 显示 polygon 区域
        gdf.plot(
            ax=self.ax,
            color="lightblue",
            edgecolor="black",
            alpha=0.5
        )

        # 红点标注 dam
        self.ax.scatter(LON, LAT,
                        color="red",
                        s=80,
                        zorder=5)

        # 标注水库名称
        self.ax.text(
            LON,
            LAT,
            f" {reservoir_name}",
            fontsize=12,
            color="red",
            weight="bold"
        )

        self.ax.set_title("Reservoir Region Map")

        self.ax.set_xlabel("Longitude")
        self.ax.set_ylabel("Latitude")

        self.draw()


class InfeResGUI(QWidget):

    def __init__(self):

        super().__init__()

        self.init_ui()

    def init_ui(self):

        self.setWindowTitle("瓦赫什河流域库区水量估算")

        layout = QVBoxLayout()

        # -------------------------
        # 输入框
        # -------------------------

        name_layout = QHBoxLayout()
        name_label = QLabel("请输入需要查询的水库名称")
        self.name_input = QLineEdit()

        name_layout.addWidget(name_label)
        name_layout.addWidget(self.name_input)

        layout.addLayout(name_layout)

        start_layout = QHBoxLayout()
        start_label = QLabel("起始日期")
        self.start_input = QLineEdit()

        start_layout.addWidget(start_label)
        start_layout.addWidget(self.start_input)

        layout.addLayout(start_layout)

        end_layout = QHBoxLayout()
        end_label = QLabel("结束日期")
        self.end_input = QLineEdit()

        end_layout.addWidget(end_label)
        end_layout.addWidget(self.end_input)

        layout.addLayout(end_layout)

        # -------------------------
        # 运行按钮
        # -------------------------

        self.run_button = QPushButton("运行计算")

        self.run_button.clicked.connect(self.run_inferes)

        layout.addWidget(self.run_button)

        # -------------------------
        # 地图显示
        # -------------------------

        self.canvas = MapCanvas()

        layout.addWidget(self.canvas)

        # -------------------------
        # CSV按钮
        # -------------------------

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

        for file in self.csv_files:

            btn = QPushButton(file)

            btn.clicked.connect(self.open_csv)

            csv_layout.addWidget(btn)

        layout.addLayout(csv_layout)

        self.setLayout(layout)

    # -------------------------
    # 运行主程序
    # -------------------------

    def run_inferes(self):

        reservoir_name = self.name_input.text()
        start_date = self.start_input.text()
        end_date = self.end_input.text()

        if reservoir_name == "":
            QMessageBox.warning(self, "错误", "请输入水库名称")
            return

        try:

            print("Running Estimation...")

            run_estimation("Nurek" , 69.348056, 38.371667, [69.10, 38.20, 69.60, 38.50], "2022-06-01", "2022-06-07", 910, 98, 1983 )

            QMessageBox.information(self, "完成", "计算完成")

            # 画图
            self.canvas.plot_region(reservoir_name)

        except Exception as e:

            QMessageBox.warning(self, "错误", str(e))

    # -------------------------
    # 打开CSV
    # -------------------------

    def open_csv(self):

        button = self.sender()

        file_name = button.text()

        path = os.path.join("output", self.name_input.text(), file_name)

        if os.path.exists(path):

            os.startfile(path)

        else:

            QMessageBox.warning(self, "错误", "文件不存在")


# -------------------------
# 主程序
# -------------------------

if __name__ == "__main__":

    app = QApplication(sys.argv)

    window = InfeResGUI()

    window.resize(800, 700)

    window.show()

    sys.exit(app.exec_())