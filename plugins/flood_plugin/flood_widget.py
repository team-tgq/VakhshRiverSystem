import csv
import os
import re

import geopandas as gpd
import numpy as np
import rasterio
from matplotlib import colors as mcolors
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from rasterio.plot import plotting_extent

from PyQt5.QtCore import QDate, QUrl
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QDateEdit,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from algorithms.flood import risk_assessment_6factors_entropy
from app.ui_hints import attach_hint, label_with_hint


class RasterCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.figure = Figure()
        super().__init__(self.figure)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()

    def _plot_boundary(self, ax, tif_path, study_area_shp):
        if not study_area_shp or not os.path.exists(study_area_shp):
            return

        gdf = gpd.read_file(study_area_shp)
        with rasterio.open(tif_path) as src:
            tif_crs = src.crs

        if gdf.crs is not None and tif_crs is not None and gdf.crs != tif_crs:
            gdf = gdf.to_crs(tif_crs)

        gdf.boundary.plot(ax=ax, linewidth=1.5, color="#1E40AF")

    def show_message(self, message):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=12)
        ax.axis("off")
        self.figure.tight_layout()
        self.draw()

    def plot_risk_tif(self, tif_path, study_area_shp=None):
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        with rasterio.open(tif_path) as src:
            arr = src.read(1).astype("float32")
            nodata = src.nodata
            if nodata is not None:
                arr[arr == nodata] = float("nan")
            extent = plotting_extent(src)

        im = ax.imshow(arr, extent=extent, origin="upper", cmap="YlOrRd")
        self.figure.colorbar(im, ax=ax, fraction=0.036, pad=0.04, label="Flood Risk")
        self._plot_boundary(ax, tif_path, study_area_shp)

        ax.set_title("洪涝风险栅格")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(False)

        self.figure.tight_layout()
        self.draw()

    def plot_landcover_tif(self, tif_path, study_area_shp=None):
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        with rasterio.open(tif_path) as src:
            arr = src.read(1).astype("float32")
            nodata = src.nodata
            if nodata is not None:
                arr[arr == nodata] = float("nan")
            extent = plotting_extent(src)

        finite_values = arr[np.isfinite(arr)]
        if finite_values.size == 0:
            self.show_message("土地利用底图暂无可显示数据。")
            return

        known_codes = list(risk_assessment_6factors_entropy.LANDCOVER_CLASSES.keys())
        present_codes = sorted({int(round(float(value))) for value in np.unique(finite_values)})
        ordered_codes = [code for code in known_codes if code in present_codes]
        ordered_codes.extend(code for code in present_codes if code not in known_codes)

        display = np.full(arr.shape, np.nan, dtype=np.float32)
        labels = []
        colors = []
        for index, code in enumerate(ordered_codes):
            info = risk_assessment_6factors_entropy.landcover_class_info(code)
            mask = np.isfinite(arr) & (np.round(arr) == code)
            display[mask] = float(index)
            labels.append(info["name"])
            colors.append(info["color"])

        cmap = mcolors.ListedColormap(colors)
        cmap.set_bad(alpha=0.0)
        im = ax.imshow(
            np.ma.masked_invalid(display),
            extent=extent,
            origin="upper",
            cmap=cmap,
            vmin=-0.5,
            vmax=max(len(labels) - 0.5, 0.5),
        )
        cbar = self.figure.colorbar(
            im,
            ax=ax,
            fraction=0.036,
            pad=0.04,
            ticks=list(range(len(labels))),
        )
        cbar.ax.set_yticklabels(labels)
        cbar.ax.tick_params(labelsize=8)

        self._plot_boundary(ax, tif_path, study_area_shp)

        ax.set_title("土地利用类型底图")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(False)

        self.figure.tight_layout()
        self.draw()


class FloodWidget(QWidget):
    STATS_HEADER_LABELS = {
        "landcover_code": "地类编码",
        "landcover_name": "土地利用类型",
        "included_in_ranking": "参与排名",
        "pixel_count": "像元数",
        "area_km2": "面积(km²)",
        "mean_risk": "平均风险",
        "p90_risk": "P90 风险",
        "high_risk_ratio": "高风险占比",
        "dominant_risk_level": "主导风险等级",
    }

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
        self.risk_canvas = RasterCanvas()
        self.landcover_canvas = RasterCanvas()
        self.map_view = QWebEngineView()
        self.stats_table = QTableWidget()
        self.stats_table.setAlternatingRowColors(True)
        self.stats_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.stats_table.verticalHeader().setVisible(False)
        self.stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.stats_table.horizontalHeader().setStretchLastSection(True)
        self.stats_summary = QTextEdit()
        self.stats_summary.setReadOnly(True)

        risk_tab = QWidget()
        risk_layout = QVBoxLayout(risk_tab)
        risk_layout.addWidget(self.risk_canvas)

        landcover_tab = QWidget()
        landcover_layout = QVBoxLayout(landcover_tab)
        landcover_layout.addWidget(self.landcover_canvas)

        stats_tab = QWidget()
        stats_layout = QVBoxLayout(stats_tab)
        stats_layout.addWidget(self.stats_table, 3)
        stats_layout.addWidget(QLabel("解释摘要"))
        stats_layout.addWidget(self.stats_summary, 1)

        map_tab = QWidget()
        map_layout = QVBoxLayout(map_tab)
        map_layout.addWidget(self.map_view)

        right_tabs.addTab(risk_tab, "风险栅格")
        right_tabs.addTab(landcover_tab, "土地利用")
        right_tabs.addTab(stats_tab, "类型统计")
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

    def _format_top_landuse(self, entries, metric_key, percentage=False):
        if not entries:
            return "暂无可用统计"

        parts = []
        for item in entries[:3]:
            value = item.get(metric_key)
            if value is None:
                continue
            metric_text = f"{value:.1%}" if percentage else f"{value:.3f}"
            parts.append(f"{item.get('landcover_name', '-') } ({metric_text})")
        return "；".join(parts) if parts else "暂无可用统计"

    def _clear_stats_table(self):
        self.stats_table.clear()
        self.stats_table.setRowCount(0)
        self.stats_table.setColumnCount(0)

    def _load_stats_table(self, csv_path):
        if not csv_path or not os.path.exists(csv_path):
            self._clear_stats_table()
            return

        with open(csv_path, "r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            rows = list(reader)
            columns = reader.fieldnames or []

        self.stats_table.clear()
        self.stats_table.setColumnCount(len(columns))
        self.stats_table.setRowCount(len(rows))
        self.stats_table.setHorizontalHeaderLabels(
            [self.STATS_HEADER_LABELS.get(column, column) for column in columns]
        )

        for row_index, row in enumerate(rows):
            for column_index, column in enumerate(columns):
                item = QTableWidgetItem(row.get(column, ""))
                self.stats_table.setItem(row_index, column_index, item)

        self.stats_table.resizeColumnsToContents()

    def _load_summary_text(self, summary_path):
        if summary_path and os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as file:
                self.stats_summary.setPlainText(file.read())
        else:
            self.stats_summary.setPlainText("还没有可显示的土地利用解释摘要。")

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
            requested_date = result.get("requested_target_date")
            resolved_date = result.get("resolved_target_date")
            if requested_date and resolved_date and requested_date != resolved_date:
                self.log.append(
                    f"{requested_date} 的实时逐日数据暂时不可用，系统已自动使用最近可用日期 {resolved_date} 继续计算。"
                )
            if result.get("resolved_target_date"):
                self.log.append(f"实际使用日期：{result['resolved_target_date']}")

            self.log.append(f"降雨数据：{result['rain_path']}")
            self.log.append(f"土壤湿度数据：{result['soil_path']}")
            self.log.append(f"土地利用数据：{result['landcover_path']}")
            if result.get("static_actions"):
                self.log.append("静态数据处理：" + "；".join(result["static_actions"]))
            if result.get("dynamic_actions"):
                self.log.append("动态数据处理：" + "；".join(result["dynamic_actions"]))

            self.log.append(f"风险栅格已生成：{result['risk_tif']}")
            self.log.append(f"交互地图已生成：{result['map_html']}")
            self.log.append(f"类型统计表已生成：{result['landuse_stats_csv']}")
            self.log.append(f"解释摘要已生成：{result['landuse_summary_txt']}")
            self.log.append(
                "土地利用高风险占比前3："
                + self._format_top_landuse(result.get("top_high_risk_landuse", []), "high_risk_ratio", percentage=True)
            )
            self.log.append(
                "土地利用平均风险前3："
                + self._format_top_landuse(result.get("top_mean_risk_landuse", []), "mean_risk", percentage=False)
            )

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
            landcover_path = os.path.join(base_dir, "data", "processed", "landcover_demgrid.tif")
            landuse_stats_csv = os.path.join(base_dir, "outputs", "landuse_risk_stats.csv")
            landuse_summary_txt = os.path.join(base_dir, "outputs", "landuse_risk_summary.txt")

            if not os.path.exists(risk_tif):
                raise FileNotFoundError(f"Result raster not found: {risk_tif}")
            if not os.path.exists(map_html):
                raise FileNotFoundError(f"Result map not found: {map_html}")

            self.result_paths = {
                "risk_tif": risk_tif,
                "map_html": map_html,
                "study_area_shp": study_area_shp,
                "landcover_path": landcover_path,
                "landuse_stats_csv": landuse_stats_csv,
                "landuse_summary_txt": landuse_summary_txt,
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
        landcover_path = self.result_paths.get("landcover_path")
        landuse_stats_csv = self.result_paths.get("landuse_stats_csv")
        landuse_summary_txt = self.result_paths.get("landuse_summary_txt")

        self.risk_canvas.plot_risk_tif(
            tif_path=risk_tif,
            study_area_shp=study_area_shp,
        )

        if landcover_path and os.path.exists(landcover_path):
            self.landcover_canvas.plot_landcover_tif(
                tif_path=landcover_path,
                study_area_shp=study_area_shp,
            )
        else:
            self.landcover_canvas.show_message("未找到土地利用底图。")

        self._load_stats_table(landuse_stats_csv)
        self._load_summary_text(landuse_summary_txt)

        if os.path.exists(map_html):
            self.map_view.load(QUrl.fromLocalFile(os.path.abspath(map_html)))
