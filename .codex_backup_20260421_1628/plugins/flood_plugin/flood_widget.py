# plugins/flood_plugin/flood_widget.py
import os

import geopandas as gpd
import rasterio
from rasterio.plot import plotting_extent

from PyQt5.QtCore import QUrl
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QTextEdit,
    QLabel,
    QTabWidget,
    QSizePolicy,
    QMessageBox,
    QLineEdit,
    QFileDialog,
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWebEngineWidgets import QWebEngineView

from algorithms.flood import risk_assessment_6factors_entropy


class RasterCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.figure = Figure()
        super().__init__(self.figure)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()

    def plot_risk_tif(self, tif_path, study_area_shp=None):
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        with rasterio.open(tif_path) as src:
            arr = src.read(1).astype("float32")
            nodata = src.nodata
            if nodata is not None:
                arr[arr == nodata] = float("nan")
            extent = plotting_extent(src)

        im = ax.imshow(arr, extent=extent, origin="upper")
        self.figure.colorbar(im, ax=ax, fraction=0.036, pad=0.04, label="Flood Risk")

        if study_area_shp and os.path.exists(study_area_shp):
            gdf = gpd.read_file(study_area_shp)
            with rasterio.open(tif_path) as src:
                tif_crs = src.crs
            if gdf.crs is not None and tif_crs is not None and gdf.crs != tif_crs:
                gdf = gdf.to_crs(tif_crs)
            gdf.boundary.plot(ax=ax, linewidth=1.5)

        ax.set_title("Flood Risk Raster")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(False)
        self.figure.tight_layout()
        self.draw()


class FloodWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.result_paths = None
        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout(self)
        left_layout = QVBoxLayout()
        default_cfg = risk_assessment_6factors_entropy.CFG

        self.title_label = QLabel("洪涝风险评估模块")

        self.study_area_input = QLineEdit(default_cfg["study_area_shp"])
        self.proc_dir_input = QLineEdit(default_cfg["proc_dir"])
        self.raw_dir_input = QLineEdit(default_cfg["raw_dir"])
        self.study_area_btn = QPushButton("选择研究区 Shapefile")
        self.proc_dir_btn = QPushButton("选择处理数据目录")
        self.raw_dir_btn = QPushButton("选择河流数据目录")

        self.input_hint = QLabel(
            "输入数据格式说明:\n"
            "1) 研究区边界: .shp（需同名 .dbf/.shx/.prj）\n"
            "2) 处理数据目录内需包含:\n"
            "   dem_clip.tif, rain_mm_demgrid.tif,\n"
            "   soil_moist_demgrid.tif, landcover_demgrid.tif\n"
            "3) 河流数据目录内需包含: hydrorivers.gpkg\n"
            "不选择时默认使用系统当前内置路径。"
        )
        self.input_hint.setWordWrap(True)

        self.run_btn = QPushButton("运行洪涝风险评估")
        self.load_btn = QPushButton("加载已有结果")
        self.log = QTextEdit()
        self.log.setReadOnly(True)

        left_layout.addWidget(self.title_label)
        left_layout.addWidget(QLabel("研究区 Shapefile (.shp)"))
        left_layout.addWidget(self.study_area_input)
        left_layout.addWidget(self.study_area_btn)
        left_layout.addWidget(QLabel("处理数据目录"))
        left_layout.addWidget(self.proc_dir_input)
        left_layout.addWidget(self.proc_dir_btn)
        left_layout.addWidget(QLabel("河流数据目录"))
        left_layout.addWidget(self.raw_dir_input)
        left_layout.addWidget(self.raw_dir_btn)
        left_layout.addWidget(self.input_hint)
        left_layout.addWidget(self.run_btn)
        left_layout.addWidget(self.load_btn)
        left_layout.addWidget(self.log)

        right_tabs = QTabWidget()
        self.raster_canvas = RasterCanvas()
        self.map_view = QWebEngineView()

        raster_tab = QWidget()
        raster_layout = QVBoxLayout(raster_tab)
        raster_layout.addWidget(self.raster_canvas)

        map_tab = QWidget()
        map_layout = QVBoxLayout(map_tab)
        map_layout.addWidget(self.map_view)

        right_tabs.addTab(raster_tab, "风险栅格视图")
        right_tabs.addTab(map_tab, "交互地图视图")

        main_layout.addLayout(left_layout, 1)
        main_layout.addWidget(right_tabs, 3)

        self.run_btn.clicked.connect(self.run_analysis)
        self.load_btn.clicked.connect(self.load_existing_results)
        self.study_area_btn.clicked.connect(self.select_study_area)
        self.proc_dir_btn.clicked.connect(self.select_proc_dir)
        self.raw_dir_btn.clicked.connect(self.select_raw_dir)

    def select_study_area(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择研究区 Shapefile",
            self.study_area_input.text().strip() or os.getcwd(),
            "Shapefile (*.shp)",
        )
        if path:
            self.study_area_input.setText(path)

    def select_proc_dir(self):
        path = QFileDialog.getExistingDirectory(
            self,
            "选择处理数据目录",
            self.proc_dir_input.text().strip() or os.getcwd(),
        )
        if path:
            self.proc_dir_input.setText(path)

    def select_raw_dir(self):
        path = QFileDialog.getExistingDirectory(
            self,
            "选择河流数据目录",
            self.raw_dir_input.text().strip() or os.getcwd(),
        )
        if path:
            self.raw_dir_input.setText(path)

    def run_analysis(self):
        try:
            self.log.append("开始运行洪涝风险评估...")
            study_area_shp = self.study_area_input.text().strip() or None
            proc_dir = self.proc_dir_input.text().strip() or None
            raw_dir = self.raw_dir_input.text().strip() or None

            self.log.append(
                f"输入数据 -> study_area: {study_area_shp}, proc_dir: {proc_dir}, raw_dir: {raw_dir}"
            )

            result = risk_assessment_6factors_entropy.run_risk_assessment(
                study_area_shp=study_area_shp,
                proc_dir=proc_dir,
                raw_dir=raw_dir,
            )
            self.result_paths = result

            self.log.append(f"风险栅格已生成: {result['risk_tif']}")
            self.log.append(f"地图已生成: {result['map_html']}")
            self.display_results()
            self.log.append("洪涝风险评估完成。")

        except Exception as e:
            self.log.append(f"[ERROR] {str(e)}")
            QMessageBox.critical(self, "错误", str(e))

    def load_existing_results(self):
        try:
            base_dir = os.path.dirname(risk_assessment_6factors_entropy.__file__)
            risk_tif = os.path.join(base_dir, "outputs", "risk_6factors.tif")
            map_html = os.path.join(base_dir, "outputs", "flood_risk_map.html")
            study_area_shp = os.path.join(base_dir, "study_area.shp")

            if not os.path.exists(risk_tif):
                raise FileNotFoundError(f"未找到结果栅格: {risk_tif}")
            if not os.path.exists(map_html):
                raise FileNotFoundError(f"未找到结果地图: {map_html}")

            self.result_paths = {
                "risk_tif": risk_tif,
                "map_html": map_html,
                "study_area_shp": study_area_shp,
            }

            self.display_results()
            self.log.append("已加载已有结果。")

        except Exception as e:
            self.log.append(f"[ERROR] {str(e)}")
            QMessageBox.critical(self, "错误", str(e))

    def display_results(self):
        if not self.result_paths:
            return

        risk_tif = self.result_paths["risk_tif"]
        map_html = self.result_paths["map_html"]
        study_area_shp = self.result_paths.get("study_area_shp")

        self.raster_canvas.plot_risk_tif(
            tif_path=risk_tif,
            study_area_shp=study_area_shp,
        )

        if os.path.exists(map_html):
            self.map_view.load(QUrl.fromLocalFile(os.path.abspath(map_html)))
