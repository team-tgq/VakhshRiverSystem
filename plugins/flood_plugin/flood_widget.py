import os
import re

import geopandas as gpd
import rasterio
from rasterio.plot import plotting_extent

from PyQt5.QtCore import QDate, QUrl
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWidgets import (
    QDateEdit,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from algorithms.flood import risk_assessment_6factors_entropy
from app.ui_hints import attach_hint, label_with_hint


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

        title_label = QLabel("洪涝灾害风险评估")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold;")

        intro_label = QLabel(
            "动态气象因子包括逐日降水和表层土壤湿度，按日尺度输入；静态地理因子默认自动读取。"
            "如果缺少静态基础数据，系统会优先尝试自动补齐。"
        )
        intro_label.setWordWrap(True)

        self.run_btn = QPushButton("运行风险评估")
        self.load_btn = QPushButton("加载已有结果")
        self.log = QTextEdit()
        self.log.setReadOnly(True)

        left_layout.addWidget(title_label)
        left_layout.addWidget(intro_label)
        left_layout.addWidget(self._build_input_group())
        left_layout.addWidget(self.run_btn)
        left_layout.addWidget(self.load_btn)
        left_layout.addWidget(self.log, 1)

        right_tabs = QTabWidget()
        self.raster_canvas = RasterCanvas()
        self.map_view = QWebEngineView()

        raster_tab = QWidget()
        raster_layout = QVBoxLayout(raster_tab)
        raster_layout.addWidget(self.raster_canvas)

        map_tab = QWidget()
        map_layout = QVBoxLayout(map_tab)
        map_layout.addWidget(self.map_view)

        right_tabs.addTab(raster_tab, "风险栅格")
        right_tabs.addTab(map_tab, "交互地图")

        main_layout.addLayout(left_layout, 1)
        main_layout.addWidget(right_tabs, 3)

        self.run_btn.clicked.connect(self.run_analysis)
        self.load_btn.clicked.connect(self.load_existing_results)

    def _build_input_group(self):
        group = QGroupBox("输入设置")
        form = QFormLayout(group)

        self.date_input = QDateEdit()
        self.date_input.setCalendarPopup(True)
        self.date_input.setDisplayFormat("yyyy-MM-dd")
        self.date_input.setDate(QDate.currentDate())

        date_hint = "选择需要评估的日期。系统会严格按这一天匹配逐日降雨和土壤湿度栅格。"
        attach_hint(self.date_input, date_hint)
        form.addRow(label_with_hint("目标日期:", date_hint), self.date_input)

        dynamic_info = QLabel(
            "本模块当前使用的动态数据包括：\n"
            "1. 逐日降水\n"
            "2. 表层土壤湿度（0-0.1 m）\n\n"
            "系统会优先按所选日期自动获取并匹配这两类逐日数据；"
            "如果所选日期还没有完整日数据，系统会明确提示，"
            "不会再直接生成不可靠的结果。"
        )
        dynamic_info.setWordWrap(True)

        dynamic_hint = (
            "动态数据要求为日尺度。当前评估必须同时具备“逐日降水”和"
            "“表层土壤湿度（0-0.1 m）”两项输入。"
        )
        attach_hint(dynamic_info, dynamic_hint)
        form.addRow(label_with_hint("动态数据:", dynamic_hint), dynamic_info)

        static_info = QLabel(
            "DEM、土地覆盖、研究区边界和河网默认从模块目录自动读取，"
            "不需要每次重复输入。"
        )
        static_info.setWordWrap(True)

        static_hint = "静态数据固定读取；若缺失，系统会优先尝试自动准备。"
        attach_hint(static_info, static_hint)
        form.addRow(label_with_hint("静态数据:", static_hint), static_info)

        return group

    def _extract_missing_date(self, text):
        match = re.search(r"(\d{4}-\d{2}-\d{2})", text)
        if match:
            return match.group(1)
        return None

    def _format_user_error(self, exc, target_date=None, loading_existing=False):
        detail = str(exc).strip()

        if loading_existing:
            return (
                "还没有找到可显示的结果。\n\n请先运行一次洪涝风险评估，再回来查看结果。",
                detail,
            )

        if "近实时数据源当前只支持截至" in detail:
            return (
                detail + "\n\n请把目标日期调整到已经结束并完成同步的那一天后再试。",
                detail,
            )

        if "近实时数据源暂未提供" in detail:
            return (
                detail + "\n\n你可以稍后重试，或换一个已经存在逐日数据的日期继续运行。",
                detail,
            )

        if "自动获取近实时逐日气象数据时失败" in detail:
            return (
                "系统已经尝试自动获取当天的逐日降水和土壤湿度数据，但这次没有成功。\n\n"
                "请检查网络连接后重试；如果问题持续存在，再看一下数据源服务是否可访问。",
                detail,
            )

        if "No daily dynamic inputs found for" in detail:
            missing_date = self._extract_missing_date(detail) or target_date or "所选日期"
            return (
                f"未找到 {missing_date} 的逐日气象数据，暂时无法生成这一天的洪涝风险结果。\n\n"
                "请先准备当天的降雨和土壤湿度栅格数据后再运行。",
                detail,
            )

        if "Daily dynamic inputs for" in detail and "Available dates:" in detail:
            missing_date = self._extract_missing_date(detail) or target_date or "所选日期"
            available_dates = detail.split("Available dates:", 1)[1].strip()
            return (
                f"未找到 {missing_date} 的逐日气象数据，暂时无法生成结果。\n\n"
                f"当前可用日期：{available_dates}",
                detail,
            )

        if target_date and target_date in detail:
            return (
                f"未找到 {target_date} 的逐日气象数据，暂时无法生成结果。\n\n"
                "请确认当天的降雨和土壤湿度栅格已经准备完成后再运行。",
                detail,
            )

        if "No daily dynamic inputs were found" in detail:
            return (
                "当前还没有可用的逐日气象数据，暂时无法运行洪涝风险评估。\n\n"
                "请先准备逐日降雨和土壤湿度栅格数据。",
                detail,
            )

        if "静态地理数据缺失" in detail or "study_area.shp" in detail:
            return (
                "基础地理数据还没有准备完整，当前无法运行洪涝风险评估。\n\n"
                "请检查研究区边界、DEM、土地覆盖和河网数据是否齐全。",
                detail,
            )

        if "有效像元过少" in detail:
            return (
                "这一天的数据覆盖范围不足，当前无法完成风险计算。\n\n"
                "请检查研究区范围以及当天栅格数据是否有效。",
                detail,
            )

        return (
            "本次洪涝风险评估没有成功完成。\n\n请检查输入数据是否完整后再试。",
            detail,
        )

    def _show_user_error(self, title, message):
        QMessageBox.critical(self, title, message)

    def run_analysis(self):
        target_date = self.date_input.date().toString("yyyy-MM-dd")
        try:
            self.log.append(f"开始运行洪涝风险评估，目标日期：{target_date}")
            result = risk_assessment_6factors_entropy.run_risk_assessment(
                target_date=target_date,
                allow_legacy_dynamic=False,
            )
            self.result_paths = result

            self.log.append(f"动态数据来源：{result.get('dynamic_scale', 'unknown')}")
            if result.get("resolved_target_date"):
                self.log.append(f"实际使用日期：{result['resolved_target_date']}")

            self.log.append(f"降雨数据：{result['rain_path']}")
            self.log.append(f"土壤湿度数据：{result['soil_path']}")
            if result.get("static_actions"):
                self.log.append("静态数据处理：" + "；".join(result["static_actions"]))
            if result.get("dynamic_actions"):
                self.log.append("动态数据处理：" + "；".join(result["dynamic_actions"]))

            self.log.append(f"风险栅格已生成：{result['risk_tif']}")
            self.log.append(f"交互地图已生成：{result['map_html']}")

            self.display_results()
            self.log.append("洪涝风险评估完成。")
        except Exception as exc:
            user_message, detail = self._format_user_error(exc, target_date=target_date)
            self.log.append("本次运行未完成。")
            self.log.append(user_message.replace("\n\n", " "))
            if detail and detail != user_message:
                self.log.append(f"详细原因：{detail}")
            self._show_user_error("无法完成洪涝风险评估", user_message)

    def load_existing_results(self):
        try:
            base_dir = os.path.dirname(risk_assessment_6factors_entropy.__file__)
            risk_tif = os.path.join(base_dir, "outputs", "risk_6factors.tif")
            map_html = os.path.join(base_dir, "outputs", "flood_risk_map.html")
            study_area_shp = os.path.join(base_dir, "study_area.shp")

            if not os.path.exists(risk_tif):
                raise FileNotFoundError(f"Result raster not found: {risk_tif}")
            if not os.path.exists(map_html):
                raise FileNotFoundError(f"Result map not found: {map_html}")

            self.result_paths = {
                "risk_tif": risk_tif,
                "map_html": map_html,
                "study_area_shp": study_area_shp,
            }

            self.display_results()
            self.log.append("已加载已有结果。")
        except Exception as exc:
            user_message, detail = self._format_user_error(exc, loading_existing=True)
            self.log.append(user_message.replace("\n\n", " "))
            if detail and detail != user_message:
                self.log.append(f"详细原因：{detail}")
            self._show_user_error("无法加载结果", user_message)

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
