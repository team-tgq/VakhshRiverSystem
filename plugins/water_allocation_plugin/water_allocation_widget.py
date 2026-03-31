from __future__ import annotations

import os

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QFileDialog,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QComboBox,
)

from algorithms.water_allocation.core import (
    SECTOR_AGR,
    calculate_et0,
    calculate_monthly_demands,
    estimate_economic_params,
    format_dam_result_text,
    format_result_text,
    run_dam_scheduling_optimization,
    run_water_allocation_optimization,
)

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False


class PlotDialog(QDialog):
    def __init__(self, title: str, figure, parent=None):
        super().__init__(parent)
        self._figure = figure
        self.setWindowTitle(title)
        self.resize(900, 540)

        layout = QVBoxLayout(self)
        canvas = FigureCanvas(figure)
        layout.addWidget(canvas)

        btn_row = QHBoxLayout()
        save_btn = QPushButton("导出 PNG")
        close_btn = QPushButton("关闭")
        btn_row.addStretch()
        btn_row.addWidget(save_btn)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

        save_btn.clicked.connect(self._save_png)
        close_btn.clicked.connect(self.close)

    def _save_png(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "保存图表", "", "PNG Image (*.png)")
        if file_path:
            self._figure.savefig(file_path, dpi=180, bbox_inches="tight")

    def closeEvent(self, event):
        try:
            self._figure.clf()
            plt.close(self._figure)
        except Exception:
            pass
        super().closeEvent(event)


class CropRowWidget(QWidget):
    def __init__(self, crop_types, stages, remove_callback, parent=None):
        super().__init__(parent)
        self._remove_callback = remove_callback
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.crop_type = QComboBox()
        self.crop_type.addItems(crop_types)
        if "细绒棉" in crop_types:
            self.crop_type.setCurrentText("细绒棉")

        self.crop_stage = QComboBox()
        self.crop_stage.addItems(stages)
        if "中期" in stages:
            self.crop_stage.setCurrentText("中期")

        self.area_edit = QLineEdit("150")
        self.yield_edit = QLineEdit("300")
        self.price_edit = QLineEdit("7.5")

        remove_btn = QPushButton("删除")
        remove_btn.clicked.connect(lambda: self._remove_callback(self))

        layout.addWidget(QLabel("作物"))
        layout.addWidget(self.crop_type)
        layout.addWidget(QLabel("生育期"))
        layout.addWidget(self.crop_stage)
        layout.addWidget(QLabel("面积(万亩)"))
        layout.addWidget(self.area_edit)
        layout.addWidget(QLabel("产量(kg/亩)"))
        layout.addWidget(self.yield_edit)
        layout.addWidget(QLabel("单价(元/kg)"))
        layout.addWidget(self.price_edit)
        layout.addWidget(remove_btn)

    def collect(self, strict: bool, row_index: int) -> dict | None:
        area_raw = self.area_edit.text().strip()
        if not area_raw:
            return None
        try:
            return {
                "type": self.crop_type.currentText(),
                "stage": self.crop_stage.currentText(),
                "area": float(area_raw),
                "yield": float(self.yield_edit.text().strip()),
                "price": float(self.price_edit.text().strip()),
            }
        except Exception:
            if strict:
                raise ValueError(f"作物第 {row_index} 行参数有无效数字。")
            return None


class MeteoDialog(QDialog):
    def __init__(self, params: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("配置气象参数")
        self.resize(420, 360)
        self._entries = {}

        form = QFormLayout(self)
        rows = [
            ("Rn", "太阳净辐射 Rn (mm/d)"),
            ("G", "土壤热通量 G (MJ/m²)"),
            ("T", "地表日平均气温 T (℃)"),
            ("u2", "2m 风速 u2 (m/s)"),
            ("es", "饱和水汽压 es (hPa)"),
            ("ea", "实际水汽压 ea (hPa)"),
            ("delta", "水汽压变化率 δ"),
            ("gamma", "湿度计常数 γ"),
        ]

        for key, label in rows:
            edit = QLineEdit(str(params[key]))
            self._entries[key] = edit
            form.addRow(label, edit)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def collect(self) -> dict:
        out = {}
        for key, edit in self._entries.items():
            out[key] = float(edit.text().strip())
        return out


class WaterAllocationWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.sectors = ["生活", "生态", "农业", "工业"]
        self.stages = ["初期", "发育期", "中期", "后期"]
        self.fao_kc = {
            "冬小麦": {"初期": 0.40, "发育期": 0.8, "中期": 1.15, "后期": 0.60},
            "细绒棉": {"初期": 0.3, "发育期": 0.7, "中期": 1.15, "后期": 0.70},
            "玉米": {"初期": 0.30, "发育期": 0.9, "中期": 1.10, "后期": 0.50},
            "水稻": {"初期": 1.05, "发育期": 1.15, "中期": 1.20, "后期": 0.90},
            "油菜": {"初期": 0.50, "发育期": 0.75, "中期": 1.05, "后期": 0.50},
        }
        self.meteo_params = {
            "Rn": 10.0,
            "G": 0.0,
            "T": 20.0,
            "u2": 2.0,
            "es": 23.4,
            "ea": 15.0,
            "delta": 1.45,
            "gamma": 0.66,
        }

        self.crop_rows = []
        self._last_nsga2_result = None
        self._last_nsga3_result = None

        if HAS_MATPLOTLIB:
            plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "sans-serif"]
            plt.rcParams["axes.unicode_minus"] = False

        self._init_ui()
        self.calculate_et0_and_demands(silent=True)

    def _init_ui(self):
        root = QVBoxLayout(self)

        tabs = QTabWidget()
        root.addWidget(tabs)

        self.tab_nsga2 = QWidget()
        self.tab_nsga3 = QWidget()
        tabs.addTab(self.tab_nsga2, "部门用水分配")
        tabs.addTab(self.tab_nsga3, "大坝水库调度")

        self._build_nsga2_tab()
        self._build_nsga3_tab()

    def _build_nsga2_tab(self):
        outer = QVBoxLayout(self.tab_nsga2)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll)

        container = QWidget()
        body = QVBoxLayout(container)
        scroll.setWidget(container)

        global_box = QGroupBox("全局供水配置")
        global_form = QFormLayout(global_box)
        self.month_combo = QComboBox()
        self.month_combo.addItems([str(i) for i in range(1, 13)])
        self.month_combo.setCurrentText("6")
        self.month_combo.currentIndexChanged.connect(lambda: self.calculate_et0_and_demands(silent=True))
        self.w_surface_edit = QLineEdit("850")
        self.w_ground_edit = QLineEdit("70")
        global_form.addRow("选择月份", self.month_combo)
        global_form.addRow("大坝当月可供水量(百万m³)", self.w_surface_edit)
        global_form.addRow("区域当月可供其他水(百万m³)", self.w_ground_edit)
        body.addWidget(global_box)

        base_box = QGroupBox("基础与水文参数")
        base_form = QFormLayout(base_box)
        self.pop_edit = QLineEdit("387")
        self.urban_edit = QLineEdit("23")
        self.gdp_edit = QLineEdit("82")
        self.reuse_edit = QLineEdit("25")
        self.eff_edit = QLineEdit("0.55")
        self.loss_edit = QLineEdit("12")
        self.eco_edit = QLineEdit("5")
        self.et0_edit = QLineEdit("0.0")
        self.et0_edit.setReadOnly(True)
        base_form.addRow("人口(万人)", self.pop_edit)
        base_form.addRow("城镇化率(%)", self.urban_edit)
        base_form.addRow("当地GDP(亿元)", self.gdp_edit)
        base_form.addRow("工业重复利用率(%)", self.reuse_edit)
        base_form.addRow("灌溉利用系数", self.eff_edit)
        base_form.addRow("传输损耗率(%)", self.loss_edit)
        base_form.addRow("生态保底水(百万m³)", self.eco_edit)
        base_form.addRow("日ET0(mm/天)", self.et0_edit)
        meteo_btn = QPushButton("配置气象参数并计算 ET0")
        meteo_btn.clicked.connect(self.open_meteo_config)
        base_form.addRow(meteo_btn)
        body.addWidget(base_box)

        hydro_box = QGroupBox("水电物理参数")
        hydro_form = QFormLayout(hydro_box)
        self.hydro_pmax_edit = QLineEdit("335")
        self.hydro_qmax_edit = QLineEdit("146")
        self.hydro_price_edit = QLineEdit("0.4")
        hydro_form.addRow("单机最大功率(MW)", self.hydro_pmax_edit)
        hydro_form.addRow("单机最大流量(m³/s)", self.hydro_qmax_edit)
        hydro_form.addRow("上网电价(元/kWh)", self.hydro_price_edit)
        body.addWidget(hydro_box)

        crop_box = QGroupBox("农业作物动态配置")
        crop_layout = QVBoxLayout(crop_box)

        header = QHBoxLayout()
        for label in ["作物类型", "生育期", "面积(万亩)", "产量(kg/亩)", "市价(元/kg)"]:
            head = QLabel(label)
            head.setAlignment(Qt.AlignCenter)
            header.addWidget(head)
        header.addWidget(QLabel(""))
        crop_layout.addLayout(header)

        self.crop_container = QVBoxLayout()
        crop_layout.addLayout(self.crop_container)

        add_crop_btn = QPushButton("添加作物")
        add_crop_btn.clicked.connect(self.add_crop_row)
        crop_layout.addWidget(add_crop_btn, alignment=Qt.AlignLeft)
        body.addWidget(crop_box)

        self.add_crop_row()

        demand_box = QGroupBox("按当月估算需水量")
        demand_form = QFormLayout(demand_box)
        self.demand_edits = {}
        for sec in self.sectors:
            edit = QLineEdit("0.0")
            self.demand_edits[sec] = edit
            demand_form.addRow(f"{sec}需水(百万m³)", edit)
        recalc_btn = QPushButton("重新估算需水")
        recalc_btn.clicked.connect(lambda: self.calculate_et0_and_demands(silent=False))
        demand_form.addRow(recalc_btn)
        body.addWidget(demand_box)

        weight_box = QGroupBox("决策偏好权重")
        weight_form = QFormLayout(weight_box)
        self.w_econ_edit = QLineEdit("0.33")
        self.w_short_edit = QLineEdit("0.33")
        self.w_gini_edit = QLineEdit("0.34")
        weight_form.addRow("整体经济权重", self.w_econ_edit)
        weight_form.addRow("降低缺水权重", self.w_short_edit)
        weight_form.addRow("部门公平(Gini)权重", self.w_gini_edit)
        body.addWidget(weight_box)

        run_btn = QPushButton("启动部门分配分析 (NSGA-II)")
        run_btn.clicked.connect(self.run_nsga2_optimization)
        body.addWidget(run_btn)

        result_box = QGroupBox("NSGA-II 结果")
        result_layout = QVBoxLayout(result_box)
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        result_layout.addWidget(self.result_text)

        result_btn_row = QHBoxLayout()
        chart_btn = QPushButton("查看分配图表")
        chart_btn.clicked.connect(self.show_nsga2_plot)
        export_btn = QPushButton("导出文本")
        export_btn.clicked.connect(self.export_nsga2_text)
        result_btn_row.addWidget(chart_btn)
        result_btn_row.addWidget(export_btn)
        result_btn_row.addStretch()
        result_layout.addLayout(result_btn_row)

        body.addWidget(result_box)
        body.addStretch()

    def _build_nsga3_tab(self):
        layout = QVBoxLayout(self.tab_nsga3)

        desc = QLabel(
            "该模块针对全年大坝调度过程进行多目标优化。\n"
            "目标1：综合经济收益最大化（发电+供水）\n"
            "目标2：洪水风险最小化\n"
            "目标3：生态缺水指数最小化"
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)

        path_box = QGroupBox("气象/水文数据源")
        path_layout = QHBoxLayout(path_box)
        self.data_path_edit = QLineEdit("")
        self.data_path_edit.setPlaceholderText("可选：csv/nc 文件或包含 nc 的目录，留空将使用内置数据")
        choose_file_btn = QPushButton("选择文件")
        choose_dir_btn = QPushButton("选择目录")
        clear_btn = QPushButton("清空")
        choose_file_btn.clicked.connect(self.select_data_file)
        choose_dir_btn.clicked.connect(self.select_data_dir)
        clear_btn.clicked.connect(lambda: self.data_path_edit.setText(""))
        path_layout.addWidget(self.data_path_edit)
        path_layout.addWidget(choose_file_btn)
        path_layout.addWidget(choose_dir_btn)
        path_layout.addWidget(clear_btn)
        layout.addWidget(path_box)

        params_box = QGroupBox("调度参数")
        params_form = QFormLayout(params_box)
        self.v_init_edit = QLineEdit("84.0")
        self.nsga3_pop_edit = QLineEdit("100")
        self.nsga3_gen_edit = QLineEdit("200")
        params_form.addRow("月初初始蓄水量(亿m³)", self.v_init_edit)
        params_form.addRow("种群规模", self.nsga3_pop_edit)
        params_form.addRow("迭代代数", self.nsga3_gen_edit)
        layout.addWidget(params_box)

        run_btn = QPushButton("开始运行 NSGA-III")
        run_btn.clicked.connect(self.run_nsga3_optimization)
        layout.addWidget(run_btn, alignment=Qt.AlignLeft)

        result_box = QGroupBox("NSGA-III 结果")
        result_layout = QVBoxLayout(result_box)
        self.dam_result_text = QTextEdit()
        self.dam_result_text.setReadOnly(True)
        result_layout.addWidget(self.dam_result_text)

        result_btn_row = QHBoxLayout()
        chart_btn = QPushButton("查看调度图表")
        chart_btn.clicked.connect(self.show_nsga3_plot)
        export_btn = QPushButton("导出文本")
        export_btn.clicked.connect(self.export_nsga3_text)
        result_btn_row.addWidget(chart_btn)
        result_btn_row.addWidget(export_btn)
        result_btn_row.addStretch()
        result_layout.addLayout(result_btn_row)
        layout.addWidget(result_box)
        layout.addStretch()

    def _to_float(self, edit: QLineEdit, name: str) -> float:
        try:
            return float(edit.text().strip())
        except Exception as e:
            raise ValueError(f"{name} 输入无效。") from e

    def _to_int(self, edit: QLineEdit, name: str) -> int:
        try:
            return int(edit.text().strip())
        except Exception as e:
            raise ValueError(f"{name} 输入无效。") from e

    def add_crop_row(self):
        row = CropRowWidget(list(self.fao_kc.keys()), self.stages, self.remove_crop_row)
        self.crop_rows.append(row)
        self.crop_container.addWidget(row)
        self.calculate_et0_and_demands(silent=True)

    def remove_crop_row(self, row_widget):
        if row_widget in self.crop_rows:
            self.crop_rows.remove(row_widget)
        row_widget.setParent(None)
        row_widget.deleteLater()
        self.calculate_et0_and_demands(silent=True)

    def collect_crop_data(self, strict: bool) -> list:
        rows = []
        for idx, row in enumerate(self.crop_rows, start=1):
            data = row.collect(strict=strict, row_index=idx)
            if data is not None:
                rows.append(data)
        return rows

    def open_meteo_config(self):
        dialog = MeteoDialog(self.meteo_params, self)
        if dialog.exec_() == QDialog.Accepted:
            try:
                self.meteo_params = dialog.collect()
                self.calculate_et0_and_demands(silent=False)
            except Exception as e:
                QMessageBox.critical(self, "参数错误", str(e))

    def calculate_et0_and_demands(self, silent: bool):
        try:
            et0 = calculate_et0(self.meteo_params)
            self.et0_edit.setText(f"{et0:.2f}")
            demands = calculate_monthly_demands(
                month=int(self.month_combo.currentText()),
                pop_wan=self._to_float(self.pop_edit, "人口"),
                urban_rate_percent=self._to_float(self.urban_edit, "城镇化率"),
                gdp_yi=self._to_float(self.gdp_edit, "GDP"),
                reuse_percent=self._to_float(self.reuse_edit, "工业重复利用率"),
                irrigation_eff=self._to_float(self.eff_edit, "灌溉利用系数"),
                eco_base=self._to_float(self.eco_edit, "生态保底水"),
                et0_daily=et0,
                crop_rows=self.collect_crop_data(strict=False),
                fao_kc=self.fao_kc,
            )
            for sec in self.sectors:
                self.demand_edits[sec].setText(f"{demands[sec]:.2f}")
        except Exception as e:
            if not silent:
                QMessageBox.critical(self, "计算失败", str(e))

    def collect_nsga2_input(self) -> dict:
        self.calculate_et0_and_demands(silent=True)
        return {
            "month": int(self.month_combo.currentText()),
            "w_surface": self._to_float(self.w_surface_edit, "当月地表可供水"),
            "w_ground": self._to_float(self.w_ground_edit, "当月地下可供水"),
            "loss_percent": self._to_float(self.loss_edit, "传输损耗率"),
            "demands": {sec: self._to_float(self.demand_edits[sec], f"{sec}需水") for sec in self.sectors},
            "crop_rows": self.collect_crop_data(strict=True),
            "hydro_pmax": self._to_float(self.hydro_pmax_edit, "单机最大功率"),
            "hydro_qmax": self._to_float(self.hydro_qmax_edit, "单机最大流量"),
            "hydro_price": self._to_float(self.hydro_price_edit, "上网电价"),
            "w_econ": self._to_float(self.w_econ_edit, "经济权重"),
            "w_short": self._to_float(self.w_short_edit, "缺水权重"),
            "w_gini": self._to_float(self.w_gini_edit, "公平权重"),
        }

    def run_nsga2_optimization(self):
        try:
            input_data = self.collect_nsga2_input()
            result = run_water_allocation_optimization(input_data)
            self._last_nsga2_result = result
            self.result_text.setPlainText(format_result_text(result, self.sectors))
        except Exception as e:
            QMessageBox.critical(self, "运行错误", str(e))

    def run_nsga3_optimization(self):
        try:
            input_data = self.collect_nsga2_input()
            _, _, _, a_agr = estimate_economic_params(
                crop_rows=input_data["crop_rows"],
                agr_water_demand_million_m3=float(input_data["demands"][SECTOR_AGR]),
                hydro_pmax=float(input_data["hydro_pmax"]),
                hydro_qmax=float(input_data["hydro_qmax"]),
                hydro_price=float(input_data["hydro_price"]),
            )

            dam_input = {
                "elec_price": float(input_data["hydro_price"]),
                "unit_water_margin": float(a_agr),
                "data_path": self.data_path_edit.text().strip(),
                "v_initial": self._to_float(self.v_init_edit, "初始蓄水量"),
                "nsga3_pop": self._to_int(self.nsga3_pop_edit, "种群规模"),
                "nsga3_gen": self._to_int(self.nsga3_gen_edit, "迭代代数"),
            }
            result = run_dam_scheduling_optimization(dam_input)
            self._last_nsga3_result = result
            self.dam_result_text.setPlainText(format_dam_result_text(result))
        except Exception as e:
            QMessageBox.critical(self, "运行错误", str(e))

    def select_data_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择数据文件", "", "Data Files (*.csv *.nc)")
        if file_path:
            self.data_path_edit.setText(file_path)

    def select_data_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择包含 nc 文件的目录")
        if dir_path:
            self.data_path_edit.setText(dir_path)

    def export_nsga2_text(self):
        text = self.result_text.toPlainText().strip()
        if not text:
            QMessageBox.information(self, "提示", "请先运行 NSGA-II。")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "保存 NSGA-II 文本", "", "Text File (*.txt)")
        if file_path:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(text)

    def export_nsga3_text(self):
        text = self.dam_result_text.toPlainText().strip()
        if not text:
            QMessageBox.information(self, "提示", "请先运行 NSGA-III。")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "保存 NSGA-III 文本", "", "Text File (*.txt)")
        if file_path:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(text)

    def show_nsga2_plot(self):
        if self._last_nsga2_result is None:
            QMessageBox.information(self, "提示", "请先运行 NSGA-II。")
            return
        if not HAS_MATPLOTLIB:
            QMessageBox.critical(self, "缺少依赖", "当前环境未安装 matplotlib，无法显示图表。")
            return

        result = self._last_nsga2_result
        demand = np.asarray(result["D_demand"], dtype=float)[0]
        x_opt = np.asarray(result["X_opt"], dtype=float)
        loss = float(result["loss_rates"][0])
        allocated = np.array([x_opt[0, 0, i] * (1 - loss) + x_opt[1, 0, i] for i in range(len(self.sectors))])

        x = np.arange(len(self.sectors))
        width = 0.35
        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.bar(x - width / 2, demand, width, label="需求水量", color="#ff9999")
        ax.bar(x + width / 2, allocated, width, label="实际分配", color="#66b3ff")
        ax.set_ylabel("水量 (百万m³)")
        ax.set_title(f"第 {result['month']} 月部门需水与分配对比")
        ax.set_xticks(x)
        ax.set_xticklabels(self.sectors)
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        dlg = PlotDialog("NSGA-II 分配图表", fig, self)
        dlg.exec_()

    def show_nsga3_plot(self):
        if self._last_nsga3_result is None:
            QMessageBox.information(self, "提示", "请先运行 NSGA-III。")
            return
        if not HAS_MATPLOTLIB:
            QMessageBox.critical(self, "缺少依赖", "当前环境未安装 matplotlib，无法显示图表。")
            return

        result = self._last_nsga3_result
        best_sol = np.asarray(result["best_release"], dtype=float)
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        q_eco = float(result["params"].Q_eco)
        q_safe = float(result["params"].Q_safe)

        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.bar(months, best_sol, color="#4CAF50", edgecolor="black")
        ax.axhline(y=q_eco, color="r", linestyle="--", label="生态保底下泄")
        ax.axhline(y=q_safe, color="orange", linestyle="--", label="安全下泄上限")
        ax.set_ylabel("下泄流量 (m³/s)")
        ax.set_title("NSGA-III 全年调度推荐流量")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        dlg = PlotDialog("NSGA-III 调度图表", fig, self)
        dlg.exec_()
