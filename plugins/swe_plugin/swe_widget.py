import os

import geopandas as gpd
import rasterio
from matplotlib import ticker
from rasterio.plot import plotting_extent

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit,
    QLabel, QMessageBox, QListWidget, QListWidgetItem, QCheckBox,
    QSizePolicy
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from algorithms.swe import swe_assessment
import numpy as np  # 记得在文件顶部添加这个导入
import matplotlib.colors as colors # 记得添加这个导入


class SWEMapCanvas(FigureCanvas):
    def __init__(self, parent=None):
        # 使用 tight_layout 自动调整布局，防止标签被遮挡
        self.figure = Figure(tight_layout=True)
        super().__init__(self.figure)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()

    def plot_tif(self, tif_path, study_area_shp=None, use_basemap=False):
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        # 1. 读取数据
        with rasterio.open(tif_path) as src:
            arr = src.read(1).astype("float32")
            nodata = src.nodata
            if nodata is not None:
                arr[arr == nodata] = np.nan
            extent = plotting_extent(src)
            tif_crs = src.crs

        # 2. 数据清洗：积雪数据中 0 或负值会干扰对数拉伸，设为极小值或 NaN
        arr_plot = np.where(arr <= 0.1, np.nan, arr)
        valid_data = arr_plot[~np.isnan(arr_plot)]

        if valid_data.size > 0:
            # 自动计算显示范围：取 1% 到 99% 分位数，彻底干掉极端异常值
            vmin, vmax = np.percentile(valid_data, [1, 99])
            # 确保 vmin 为正数以适配 LogNorm
            vmin = max(0.5, vmin)
        else:
            vmin, vmax = 1, 100

        # 3. 核心：使用对数归一化 (LogNorm) 增强低值区的颜色对比
        # 如果你觉得对数拉伸太夸张，可以换回 colors.Normalize(vmin=vmin, vmax=vmax)
        norm = colors.LogNorm(vmin=vmin, vmax=vmax)

        # 选择对比度极高的色带：'Spectral_r' (红黄蓝) 或 'RdYlBu_r'
        cmap = 'Spectral_r'

        # 4. 绘图
        im = ax.imshow(arr_plot, extent=extent, origin="upper",
                       cmap=cmap, norm=norm, alpha=1.0, interpolation='nearest')

        # 5. 优化 Colorbar (对数刻度显示)
        # 使用 LogLocator 自动生成 1, 10, 100, 1000 这样的科学刻度
        cbar = self.figure.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
        cbar.set_label("SWE (cm) - Log Scaled", fontsize=10, fontweight='bold')
        cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))

        # 6. 叠加研究区矢量边界
        if study_area_shp and os.path.exists(study_area_shp):
            try:
                gdf = gpd.read_file(study_area_shp)
                if gdf.crs is not None and tif_crs is not None and gdf.crs != tif_crs:
                    gdf = gdf.to_crs(tif_crs)
                # 使用鲜艳的红色边界，线宽适中
                gdf.boundary.plot(ax=ax, color='#e41a1c', linewidth=1.5, zorder=5)
            except Exception as e:
                print(f"Vector plot error: {e}")

        # 7. 叠加在线底图
        if use_basemap:
            try:
                import contextily as ctx
                # 换一个更轻量的底图源，或者直接不传 CRS 试试（让它自动对齐）
                ctx.add_basemap(ax, crs=src.crs, source=ctx.providers.CartoDB.Positron)
                print("底图加载成功")
            except Exception as e:
                print(f"底图加载失败：{e}")  # 看看控制台报错是网络问题还是坐标系问题

        # 8. 图形细节美化
        ax.set_title(f"Vakhsh Basin SWE Distribution\n({os.path.basename(tif_path)})",
                     fontsize=12, pad=10)
        ax.set_xlabel("Longitude (°E)", fontsize=9)
        ax.set_ylabel("Latitude (°N)", fontsize=9)

        # 移除默认网格，改用淡色虚线
        ax.grid(True, which='both', linestyle='--', alpha=0.2, color='gray')

        self.draw()


class SWEWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.result = None
        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout(self)

        # 左侧控制区
        left_layout = QVBoxLayout()

        self.title_label = QLabel("雪水当量评估模块")
        self.run_btn = QPushButton("运行雪水当量评估")
        self.load_btn = QPushButton("加载已有TIF结果")
        self.basemap_check = QCheckBox("叠加在线底图")
        self.tif_list_widget = QListWidget()
        self.log = QTextEdit()

        self.log.setReadOnly(True)

        left_layout.addWidget(self.title_label)
        left_layout.addWidget(self.run_btn)
        left_layout.addWidget(self.load_btn)
        left_layout.addWidget(self.basemap_check)
        left_layout.addWidget(QLabel("输出TIF列表"))
        left_layout.addWidget(self.tif_list_widget, 1)
        left_layout.addWidget(QLabel("运行日志"))
        left_layout.addWidget(self.log, 2)

        # 右侧地图显示区
        right_layout = QVBoxLayout()
        self.map_canvas = SWEMapCanvas()
        right_layout.addWidget(self.map_canvas)

        main_layout.addLayout(left_layout, 1)
        main_layout.addLayout(right_layout, 3)

        self.run_btn.clicked.connect(self.run_assessment)
        self.load_btn.clicked.connect(self.load_existing_results)
        self.tif_list_widget.itemClicked.connect(self.on_tif_selected)
        self.basemap_check.stateChanged.connect(self.refresh_current_tif)

    def run_assessment(self):
        try:
            self.log.append("开始运行雪水当量评估...")
            self.result = swe_assessment.run_swe_assessment()

            tif_list = self.result.get("tif_list", [])
            self.populate_tif_list(tif_list)

            self.log.append(f"预测CSV已生成: {self.result['csv']}")
            self.log.append(f"TIF数量: {len(tif_list)}")

            latest_tif = self.result.get("latest_tif")
            if latest_tif:
                self.display_tif(latest_tif)
                self.log.append(f"当前显示: {latest_tif}")

            self.log.append("雪水当量评估完成。")

        except Exception as e:
            self.log.append(f"[ERROR] {str(e)}")
            QMessageBox.critical(self, "错误", str(e))

    def load_existing_results(self):
        try:
            base_dir = os.path.dirname(swe_assessment.__file__)
            tif_dir = os.path.join(base_dir, "output", "tif")
            study_area_shp = os.path.join(base_dir, "study_area.shp")
            csv_path = os.path.join(base_dir, "output", "swe_ml_distribution.csv")

            if not os.path.exists(tif_dir):
                raise FileNotFoundError(f"未找到目录: {tif_dir}")

            tif_list = sorted([
                os.path.join(tif_dir, f)
                for f in os.listdir(tif_dir)
                if f.lower().endswith(".tif")
            ])

            if not tif_list:
                raise FileNotFoundError("未找到任何 SWE tif 文件。")

            self.result = {
                "csv": csv_path,
                "tif_list": tif_list,
                "latest_tif": tif_list[-1],
                "study_area_shp": study_area_shp
            }

            self.populate_tif_list(tif_list)
            self.display_tif(tif_list[-1])
            self.log.append("已加载已有 SWE 结果。")

        except Exception as e:
            self.log.append(f"[ERROR] {str(e)}")
            QMessageBox.critical(self, "错误", str(e))

    def populate_tif_list(self, tif_list):
        self.tif_list_widget.clear()
        for tif_path in tif_list:
            item = QListWidgetItem(os.path.basename(tif_path))
            item.setData(256, tif_path)
            self.tif_list_widget.addItem(item)

    def on_tif_selected(self, item):
        tif_path = item.data(256)
        self.display_tif(tif_path)
        self.log.append(f"切换显示: {tif_path}")

    def refresh_current_tif(self):
        item = self.tif_list_widget.currentItem()
        if item:
            tif_path = item.data(256)
            self.display_tif(tif_path)

    def display_tif(self, tif_path):
        if not self.result:
            return

        self.map_canvas.plot_tif(
            tif_path=tif_path,
            study_area_shp=self.result.get("study_area_shp"),
            use_basemap=self.basemap_check.isChecked()
        )