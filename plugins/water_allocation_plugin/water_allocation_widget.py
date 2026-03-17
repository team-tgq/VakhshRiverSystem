from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLineEdit, QTextEdit, QMessageBox, QComboBox, QGroupBox, QScrollArea,
    QFrame
)
from PyQt5.QtCore import Qt

from algorithms.water_allocation.core import (
    calculate_et0,
    calculate_monthly_demands,
    run_water_allocation_optimization,
    format_result_text,
)


class CropRowWidget(QWidget):
    def __init__(self, crop_types, stages, remove_callback, parent=None):
        super().__init__(parent)
        self.remove_callback = remove_callback
        self.init_ui(crop_types, stages)

    def init_ui(self, crop_types, stages):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.crop_type = QComboBox()
        self.crop_type.addItems(crop_types)
        self.crop_type.setCurrentText("细绒棉" if "细绒棉" in crop_types else crop_types[0])

        self.crop_stage = QComboBox()
        self.crop_stage.addItems(stages)
        self.crop_stage.setCurrentText("中期" if "中期" in stages else stages[0])

        self.area_edit = QLineEdit("150")
        self.yield_edit = QLineEdit("300")
        self.price_edit = QLineEdit("7.5")

        self.remove_btn = QPushButton("删除")
        self.remove_btn.clicked.connect(lambda: self.remove_callback(self))

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
        layout.addWidget(self.remove_btn)

    def get_data(self):
        return {
            "type": self.crop_type.currentText(),
            "stage": self.crop_stage.currentText(),
            "area": float(self.area_edit.text().strip()),
            "yield": float(self.yield_edit.text().strip()),
            "price": float(self.price_edit.text().strip()),
        }


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
            "油菜": {"初期": 0.50, "发育期": 0.75, "中期": 1.05, "后期": 0.50}
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

        self.init_ui()
        self.calculate_et0_and_demands()

    def init_ui(self):
        outer_layout = QVBoxLayout(self)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        outer_layout.addWidget(scroll)

        container = QWidget()
        scroll.setWidget(container)

        layout = QVBoxLayout(container)

        # 全局供水配置
        global_box = QGroupBox("月度供水配置")
        global_grid = QGridLayout(global_box)

        self.month_combo = QComboBox()
        self.month_combo.addItems([str(i) for i in range(1, 13)])
        self.month_combo.setCurrentText("6")
        self.month_combo.currentIndexChanged.connect(self.calculate_et0_and_demands)

        self.w_surface_edit = QLineEdit("850")
        self.w_ground_edit = QLineEdit("70")

        global_grid.addWidget(QLabel("选择月份"), 0, 0)
        global_grid.addWidget(self.month_combo, 0, 1)
        global_grid.addWidget(QLabel("大坝当月可供水量(百万m³)"), 0, 2)
        global_grid.addWidget(self.w_surface_edit, 0, 3)
        global_grid.addWidget(QLabel("区域当月可供使用其他水(百万m³)"), 0, 4)
        global_grid.addWidget(self.w_ground_edit, 0, 5)

        layout.addWidget(global_box)

        # 基础参数
        basic_box = QGroupBox("哈特隆州 基础与水文参数")
        basic_grid = QGridLayout(basic_box)

        self.pop_edit = QLineEdit("387")
        self.urban_edit = QLineEdit("23")
        self.gdp_edit = QLineEdit("82")
        self.reuse_edit = QLineEdit("25")
        self.eff_edit = QLineEdit("0.55")
        self.loss_edit = QLineEdit("12")
        self.eco_edit = QLineEdit("5")
        self.et0_edit = QLineEdit("0.0")
        self.et0_edit.setReadOnly(True)

        fields = [
            ("人口(万人)", self.pop_edit),
            ("城镇化率(%)", self.urban_edit),
            ("当地GDP(亿元)", self.gdp_edit),
            ("工业重复利用率(%)", self.reuse_edit),
            ("灌溉利用系数", self.eff_edit),
            ("传输损耗率(%)", self.loss_edit),
            ("生态保底(百万m³)", self.eco_edit),
            ("日ET0(mm/天)", self.et0_edit),
        ]

        for idx, (label, edit) in enumerate(fields):
            r = idx // 2
            c = (idx % 2) * 2
            basic_grid.addWidget(QLabel(label), r, c)
            basic_grid.addWidget(edit, r, c + 1)

        self.calc_et0_btn = QPushButton("更新 ET0 并估算需水")
        self.calc_et0_btn.clicked.connect(self.calculate_et0_and_demands)
        basic_grid.addWidget(self.calc_et0_btn, 4, 0, 1, 4)

        layout.addWidget(basic_box)

        # PM 参数
        pm_box = QGroupBox("PM 气象参数")
        pm_grid = QGridLayout(pm_box)

        self.Rn_edit = QLineEdit("10.0")
        self.G_edit = QLineEdit("0.0")
        self.T_edit = QLineEdit("20.0")
        self.u2_edit = QLineEdit("2.0")
        self.es_edit = QLineEdit("23.4")
        self.ea_edit = QLineEdit("15.0")
        self.delta_edit = QLineEdit("1.45")
        self.gamma_edit = QLineEdit("0.66")

        pm_fields = [
            ("太阳净辐射 Rn", self.Rn_edit),
            ("土壤热通量 G", self.G_edit),
            ("平均气温 T", self.T_edit),
            ("2m风速 u2", self.u2_edit),
            ("饱和水汽压 es", self.es_edit),
            ("实际水汽压 ea", self.ea_edit),
            ("水汽压变率 Δ", self.delta_edit),
            ("湿度计常数 γ", self.gamma_edit),
        ]

        for idx, (label, edit) in enumerate(pm_fields):
            r = idx // 2
            c = (idx % 2) * 2
            pm_grid.addWidget(QLabel(label), r, c)
            pm_grid.addWidget(edit, r, c + 1)

        layout.addWidget(pm_box)

        # 水电参数
        hydro_box = QGroupBox("努列克坝发电机组物理参数")
        hydro_grid = QGridLayout(hydro_box)

        self.hydro_pmax_edit = QLineEdit("335")
        self.hydro_qmax_edit = QLineEdit("146")
        self.hydro_price_edit = QLineEdit("0.4")

        hydro_grid.addWidget(QLabel("单机最大功率(MW)"), 0, 0)
        hydro_grid.addWidget(self.hydro_pmax_edit, 0, 1)
        hydro_grid.addWidget(QLabel("单机最大流量(m³/s)"), 0, 2)
        hydro_grid.addWidget(self.hydro_qmax_edit, 0, 3)
        hydro_grid.addWidget(QLabel("上网电价(元/kWh)"), 0, 4)
        hydro_grid.addWidget(self.hydro_price_edit, 0, 5)

        layout.addWidget(hydro_box)

        # 农业作物配置
        crop_box = QGroupBox("农业作物动态配置")
        crop_layout = QVBoxLayout(crop_box)

        self.crop_container = QVBoxLayout()
        crop_layout.addLayout(self.crop_container)

        self.add_crop_btn = QPushButton("添加作物")
        self.add_crop_btn.clicked.connect(self.add_crop_row)
        crop_layout.addWidget(self.add_crop_btn)

        layout.addWidget(crop_box)

        self.add_crop_row()

        # 需水量
        demand_box = QGroupBox("月需水量预测结果")
        demand_grid = QGridLayout(demand_box)

        self.demand_edits = {}
        for idx, sec in enumerate(self.sectors):
            edit = QLineEdit("0.0")
            self.demand_edits[sec] = edit
            demand_grid.addWidget(QLabel(f"{sec}"), 0, idx * 2)
            demand_grid.addWidget(edit, 0, idx * 2 + 1)

        layout.addWidget(demand_box)

        # 偏好权重
        weight_box = QGroupBox("决策偏好权重")
        weight_layout = QHBoxLayout(weight_box)

        self.w_econ_edit = QLineEdit("0.33")
        self.w_short_edit = QLineEdit("0.33")
        self.w_gini_edit = QLineEdit("0.34")

        weight_layout.addWidget(QLabel("整体经济权重"))
        weight_layout.addWidget(self.w_econ_edit)
        weight_layout.addWidget(QLabel("降低缺水权重"))
        weight_layout.addWidget(self.w_short_edit)
        weight_layout.addWidget(QLabel("部门公平(Gini)权重"))
        weight_layout.addWidget(self.w_gini_edit)

        layout.addWidget(weight_box)

        # 运行
        self.run_button = QPushButton("启动 NSGA-II 算法进行三目标协同优化")
        self.run_button.clicked.connect(self.run_optimization)
        layout.addWidget(self.run_button)

        # 结果
        result_box = QGroupBox("优化结果")
        result_layout = QVBoxLayout(result_box)

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        result_layout.addWidget(self.result_text)

        layout.addWidget(result_box)

        layout.addStretch()

    def add_crop_row(self):
        row = CropRowWidget(
            crop_types=list(self.fao_kc.keys()),
            stages=self.stages,
            remove_callback=self.remove_crop_row
        )
        self.crop_rows.append(row)
        self.crop_container.addWidget(row)

    def remove_crop_row(self, row_widget):
        self.crop_rows.remove(row_widget)
        row_widget.setParent(None)
        row_widget.deleteLater()
        self.calculate_et0_and_demands()

    def collect_crop_data(self):
        crop_data = []
        for row in self.crop_rows:
            crop_data.append(row.get_data())
        return crop_data

    def collect_meteo_params(self):
        return {
            "Rn": float(self.Rn_edit.text().strip()),
            "G": float(self.G_edit.text().strip()),
            "T": float(self.T_edit.text().strip()),
            "u2": float(self.u2_edit.text().strip()),
            "es": float(self.es_edit.text().strip()),
            "ea": float(self.ea_edit.text().strip()),
            "delta": float(self.delta_edit.text().strip()),
            "gamma": float(self.gamma_edit.text().strip()),
        }

    def calculate_et0_and_demands(self):
        try:
            self.meteo_params = self.collect_meteo_params()
            et0 = calculate_et0(self.meteo_params)

            self.et0_edit.setText(f"{et0:.2f}")

            demands = calculate_monthly_demands(
                month=int(self.month_combo.currentText()),
                pop_wan=float(self.pop_edit.text().strip()),
                urban_rate_percent=float(self.urban_edit.text().strip()),
                gdp_yi=float(self.gdp_edit.text().strip()),
                reuse_percent=float(self.reuse_edit.text().strip()),
                irrigation_eff=float(self.eff_edit.text().strip()),
                eco_base=float(self.eco_edit.text().strip()),
                et0_daily=et0,
                crop_rows=self.collect_crop_data(),
                fao_kc=self.fao_kc,
            )

            for sec in self.sectors:
                self.demand_edits[sec].setText(f"{demands[sec]:.2f}")

        except Exception:
            # 输入尚未完整时，静默即可
            pass

    def collect_input_data(self):
        self.calculate_et0_and_demands()

        return {
            "month": int(self.month_combo.currentText()),
            "w_surface": float(self.w_surface_edit.text().strip()),
            "w_ground": float(self.w_ground_edit.text().strip()),
            "loss_percent": float(self.loss_edit.text().strip()),
            "demands": {
                sec: float(self.demand_edits[sec].text().strip())
                for sec in self.sectors
            },
            "crop_rows": self.collect_crop_data(),
            "hydro_pmax": float(self.hydro_pmax_edit.text().strip()),
            "hydro_qmax": float(self.hydro_qmax_edit.text().strip()),
            "hydro_price": float(self.hydro_price_edit.text().strip()),
            "w_econ": float(self.w_econ_edit.text().strip()),
            "w_short": float(self.w_short_edit.text().strip()),
            "w_gini": float(self.w_gini_edit.text().strip()),
        }

    def run_optimization(self):
        try:
            self.run_button.setEnabled(False)
            self.run_button.setText("优化中...")

            input_data = self.collect_input_data()
            result = run_water_allocation_optimization(input_data)
            text = format_result_text(result, self.sectors)

            self.result_text.setPlainText(text)

        except Exception as e:
            QMessageBox.critical(self, "运行错误", str(e))

        finally:
            self.run_button.setEnabled(True)
            self.run_button.setText("启动 NSGA-II 算法进行三目标协同优化")