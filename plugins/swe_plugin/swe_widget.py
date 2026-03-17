import os

import geopandas as gpd
import rasterio
from rasterio.plot import plotting_extent

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit,
    QLabel, QMessageBox, QListWidget, QListWidgetItem, QCheckBox,
    QSizePolicy
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from algorithms.swe import swe_assessment


class SWEMapCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.figure = Figure()
        super().__init__(self.figure)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()

    def plot_tif(self, tif_path, study_area_shp=None, use_basemap=False):
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        with rasterio.open(tif_path) as src:
            arr = src.read(1).astype("float32")
            nodata = src.nodata
            if nodata is not None:
                arr[arr == nodata] = float("nan")
            extent = plotting_extent(src)
            tif_crs = src.crs

        im = ax.imshow(arr, extent=extent, origin="upper", alpha=0.78)
        self.figure.colorbar(im, ax=ax, fraction=0.036, pad=0.04, label="SWE (cm)")

        if study_area_shp and os.path.exists(study_area_shp):
            gdf = gpd.read_file(study_area_shp)
            if gdf.crs is not None and tif_crs is not None and gdf.crs != tif_crs:
                gdf = gdf.to_crs(tif_crs)

            try:
                gdf.boundary.plot(ax=ax, linewidth=1.8)
            except Exception:
                pass

        # 可选在线底图
        if use_basemap:
            try:
                import contextily as ctx
                if tif_crs is not None:
                    ctx.add_basemap(ax, crs=tif_crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)
            except Exception:
                pass

        ax.set_title(f"SWE Raster: {os.path.basename(tif_path)}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(False)

        self.figure.tight_layout()
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