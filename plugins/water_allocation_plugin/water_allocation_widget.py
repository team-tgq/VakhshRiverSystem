from __future__ import annotations

import calendar
import datetime
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QFileDialog, QDialog, QDialogButtonBox, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QPushButton, QScrollArea, QTabWidget, QTextEdit,
    QVBoxLayout, QWidget, QComboBox, QRadioButton, QButtonGroup,
    QGridLayout, QFrame, QFontDialog,
)
from PyQt5.QtGui import QFont

from algorithms.water_allocation.core import (
    SECTOR_LIVE, SECTOR_ECO, SECTOR_AGR, SECTOR_IND, SECTOR_DOWN,
    SECTOR_ORDER_V2,
    NurekDamParameters,
    NurekWaterAllocation,
    run_nsga2_opt,
)
from algorithms.water_allocation.predict import predict_downstream_total

# 遥感模块为可选依赖
_HAS_REMOTE_SENSING = False
_RS_IMPORT_ERROR = None
try:
    from algorithms.water_allocation.remote_sensing.gee_service import get_cropland_area_km2
    from algorithms.water_allocation.remote_sensing.ftw_model import (
        create_ftw_model, calculate_cropland_area,
        load_geojson_mask, FTW_BAND_INDICES,
    )
    _HAS_REMOTE_SENSING = True
except ImportError as e:
    _RS_IMPORT_ERROR = str(e)

_RESOURCE_DIR = Path(__file__).resolve().parent.parent.parent / "algorithms" / "water_allocation" / "resources"

try:
    import matplotlib
    matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False


# ======================= 结果弹窗 =======================
class ResultDialog(QDialog):

    def __init__(self, result_data, sectors, parent=None):
        super().__init__(parent)
        self.sectors = sectors
        self.result_data = result_data
        self.setWindowTitle(f"NSGA-II 优化配置分析报告 ({result_data['time_scale']})")
        self.resize(950, 700)

        layout = QVBoxLayout(self)

        nb = QTabWidget()
        layout.addWidget(nb)

        # -- 文本报告页 --
        tab_txt = QWidget()
        txt_layout = QVBoxLayout(tab_txt)
        self.report_text = self._build_report()
        text_widget = QTextEdit()
        text_widget.setReadOnly(True)
        text_widget.setFont(QFont("Consolas", 11))
        text_widget.setStyleSheet("QTextEdit { background-color: #1E1E1E; color: #D4D4D4; border: none; }")
        text_widget.setPlainText(self.report_text)
        txt_layout.addWidget(text_widget)
        nb.addTab(tab_txt, "📄 文本报告")

        # -- 时序图或柱状图 --
        if result_data.get('time_series_X') is not None and result_data.get('date_labels'):
            tab_ts = QWidget()
            ts_layout = QVBoxLayout(tab_ts)
            fig = self._build_time_series_chart()
            if fig:
                canvas = FigureCanvas(fig)
                ts_layout.addWidget(canvas)
            nb.addTab(tab_ts, "📈 时序分配")
        else:
            tab_bar = QWidget()
            bar_layout = QVBoxLayout(tab_bar)
            fig = self._build_bar_chart()
            if fig:
                canvas = FigureCanvas(fig)
                bar_layout.addWidget(canvas)
            nb.addTab(tab_bar, "📊 柱状图")

        # -- 饼图页 --
        tab_pie = QWidget()
        pie_layout = QVBoxLayout(tab_pie)
        fig_pie = self._build_pie_chart()
        if fig_pie:
            canvas_pie = FigureCanvas(fig_pie)
            pie_layout.addWidget(canvas_pie)
        nb.addTab(tab_pie, "🥧 饼图")

        # -- 导出按钮 --
        btn_row = QHBoxLayout()
        export_txt_btn = QPushButton("💾 导出文本")
        export_txt_btn.clicked.connect(self._export_text)
        export_csv_btn = QPushButton("📥 导出 CSV")
        export_csv_btn.clicked.connect(self._export_csv)
        btn_row.addStretch()
        btn_row.addWidget(export_txt_btn)
        btn_row.addWidget(export_csv_btn)
        layout.addLayout(btn_row)

    def _build_report(self):
        r = self.result_data
        gini_d = r['gini']
        if gini_d < 0.1:
            gini_tag = "(完全公平)"
        elif gini_d < 0.2:
            gini_tag = "(满意度较平均)"
        elif gini_d < 0.3:
            gini_tag = "(分配偏向高产值部门)"
        else:
            gini_tag = "(偏科严重，存在明显受损部门)"

        n_periods = r.get('n_periods', 1)
        lines = [
            f"🎯 哈特隆州水资源配置方案 ({r['time_scale']}, {n_periods} 期)",
            "=" * 85,
            f"💰 系统总综合经济效益参考值 : {r['profit']:,.2f} 万元",
            f"📉 系统总缺水量       : {r['shortage']:,.2f} 百万m³",
            f"⚖️ 部门公平性 Gini   : {gini_d:.4f} {gini_tag}",
            "=" * 85,
            f"\n📍 地区：哈特隆州 (管网传输损耗率: {r['loss_rates'][0]*100:.1f}%)",
            f"{'部门':<10} | {'需水量':<10} | {'水库放水量':<10} | {'实收水量':<10} | {'满足率'}",
            "-" * 75,
        ]
        X_agg = r['X_agg']
        D = r['D_demand']
        loss = r['loss_rates'][0]
        received_data = []
        for j, sec in enumerate(self.sectors):
            demand = D[0, j]
            surf_out = X_agg[0, 0, j]
            received = surf_out * (1 - loss) + X_agg[1, 0, j]
            received_data.append(received)
            ratio = (received / demand * 100) if demand > 0 else 100
            lines.append(f"{sec:<10} | {demand:<13.2f} | {surf_out:<15.2f} | {received:<16.2f} | {ratio:.1f}%")
        total_surf = X_agg[0, 0, :].sum()
        lines.extend([
            "\n" + "=" * 85,
            f"🌊 水库放水总量: {total_surf:.2f} / {r['W_supply'][0]:.2f} 百万m³",
            f"🔌 水力发电贡献参考值: 约 {total_surf * r['a_hydro']:,.2f} 万元",
        ])
        self._received_data = received_data
        return "\n".join(lines)

    def _build_bar_chart(self):
        if not HAS_MATPLOTLIB:
            return None
        r = self.result_data
        D = r['D_demand']
        loss = r['loss_rates'][0]
        X_agg = r['X_agg']
        received = [X_agg[0, 0, j] * (1 - loss) + X_agg[1, 0, j] for j in range(len(self.sectors))]

        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(len(self.sectors))
        w = 0.35
        ax.bar(x - w / 2, D[0], w, label='需水量', color='#ff9999')
        ax.bar(x + w / 2, received, w, label='实际分配', color='#66b3ff')
        ax.set_ylabel('水量 (百万m³)')
        ax.set_title(f'各部门用水需求与实际分配 ({r["time_scale"]})')
        ax.set_xticks(x)
        ax.set_xticklabels(self.sectors)
        ax.legend()
        fig.tight_layout()
        return fig

    def _build_pie_chart(self):
        if not HAS_MATPLOTLIB:
            return None
        r = self.result_data
        loss = r['loss_rates'][0]
        X_agg = r['X_agg']
        received = [X_agg[0, 0, j] * (1 - loss) + X_agg[1, 0, j] for j in range(len(self.sectors))]

        fig, ax = plt.subplots(figsize=(6, 5))
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#cc99ff']
        ax.pie(received, labels=self.sectors, autopct='%1.1f%%', colors=colors, startangle=90)
        ax.set_title('各部门分配水量占比 (合计)')
        return fig

    def _build_time_series_chart(self):
        if not HAS_MATPLOTLIB:
            return None
        r = self.result_data
        X = r['time_series_X']
        labels = r['date_labels']
        loss = r['loss_rates'][0]

        fig, ax = plt.subplots(figsize=(9, 4.5))
        colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336']
        for j in range(X.shape[3]):
            series = X[:, 0, 0, j] * (1 - loss) + X[:, 1, 0, j]
            ax.plot(range(len(labels)), series, marker='.', label=self.sectors[j],
                   color=colors[j], linewidth=1.5, markersize=3)
        ax.set_ylabel('实收水量 (百万m³)')
        ax.set_title(f'各部门逐{r["time_scale"]}分配水量')
        step = max(1, len(labels) // 12)
        ax.set_xticks(range(0, len(labels), step))
        ax.set_xticklabels([labels[i] for i in range(0, len(labels), step)],
                          rotation=45, ha='right', fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    def _export_text(self):
        path, _ = QFileDialog.getSaveFileName(self, "保存报告", "", "Text (*.txt)")
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.report_text)

    def _export_csv(self):
        r = self.result_data
        path, _ = QFileDialog.getSaveFileName(self, "保存CSV", "", "CSV (*.csv)")
        if path:
            import csv
            D = r['D_demand']
            loss = r['loss_rates'][0]
            X_agg = r['X_agg']
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(["部门", "需水量(百万m³)", "实收水量(百万m³)", "满足率(%)"])
                for j, sec in enumerate(self.sectors):
                    demand = D[0, j]
                    received = X_agg[0, 0, j] * (1 - loss) + X_agg[1, 0, j]
                    ratio = (received / demand * 100) if demand > 0 else 100
                    w.writerow([sec, f"{demand:.2f}", f"{received:.2f}", f"{ratio:.1f}"])

    def closeEvent(self, event):
        plt.close('all')
        super().closeEvent(event)


# ======================= 辅助类 =======================
class LSTMWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, data_path, parent=None):
        super().__init__(parent)
        self.data_path = data_path

    def run(self):
        try:
            from algorithms.water_allocation.train import train_from_external_data
            hp = {'num_epochs': 100, 'learning_rate': 0.001}

            def progress_cb(ep, total, tl, vl, rmse, mae):
                self.progress.emit(f"Epoch {ep}/{total} | Train Loss: {tl:.2f} | Val RMSE: {rmse:.2f}")

            result = train_from_external_data(self.data_path, hyperparams=hp, progress_callback=progress_cb)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class FTWInferenceWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, params, parent=None):
        super().__init__(parent)
        self.params = params

    def run(self):
        try:
            import torch
            self.progress.emit("正在加载 FTW 模型权重…")
            device = self.params['device'] if torch.cuda.is_available() else 'cpu'
            model, num_classes = create_ftw_model(
                num_classes=3, encoder_name='efficientnet-b3',
                in_channels=8, pretrained_weights_path=self.params['weights'],
                device=device,
            )

            mask_geom = None
            if self.params.get('mask_path') and os.path.exists(self.params['mask_path']):
                self.progress.emit("正在加载 GeoJSON 掩膜…")
                mask_geom, _ = load_geojson_mask(self.params['mask_path'])

            total_area_km2 = 0.0
            total_pixels = 0
            per_file_info = []
            pixel_area_sqm = 0.0
            n_total = len(self.params['filepaths'])
            n_workers = min(self.params.get('workers', 2), n_total)
            completed = [0]

            def process_one(fpath, idx):
                result = calculate_cropland_area(
                    image_path=fpath, model=model, device=device,
                    window_size=self.params['window_size'],
                    overlap=self.params['overlap'],
                    num_classes=num_classes,
                    band_indices=FTW_BAND_INDICES,
                    mask_geometry=mask_geom,
                    ndvi_threshold=self.params['ndvi_threshold'],
                )
                return idx, os.path.basename(fpath), result

            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = {executor.submit(process_one, fp, i): i for i, fp in enumerate(self.params['filepaths'])}
                for future in as_completed(futures):
                    idx, fname, result = future.result()
                    per_file_info.append((idx, fname, result))
                    completed[0] += 1
                    self.progress.emit(f"[{completed[0]}/{n_total}] {fname} 完成…")

            per_file_info.sort(key=lambda x: x[0])
            for _, fname, result in per_file_info:
                total_area_km2 += result['area_hectares'] / 100.0
                total_pixels += result['field_pixels']
                pixel_area_sqm = result['pixel_area_sqm']

            self.finished.emit({
                'area_km2': total_area_km2,
                'pixels': total_pixels,
                'pixel_sqm': pixel_area_sqm,
                'per_file': [(n, r) for _, n, r in per_file_info],
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))


class CropRowWidget(QWidget):
    def __init__(self, crop_types, stages, remove_callback, parent=None):
        super().__init__(parent)
        self._remove_callback = remove_callback
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.crop_type = QComboBox()
        self.crop_type.addItems(crop_types)
        self.crop_type.setCurrentText("细绒棉")
        self.crop_stage = QComboBox()
        self.crop_stage.addItems(stages)
        self.crop_stage.setCurrentText("中期")
        self.area_edit = QLineEdit("50"); self.area_edit.setFixedWidth(60)
        self.yield_edit = QLineEdit("300"); self.yield_edit.setFixedWidth(60)
        self.price_edit = QLineEdit("7.5"); self.price_edit.setFixedWidth(60)

        remove_btn = QPushButton("❌")
        remove_btn.setFixedWidth(30)
        remove_btn.clicked.connect(lambda: self._remove_callback(self))

        layout.addWidget(QLabel("作物类型"))
        layout.addWidget(self.crop_type)
        layout.addWidget(QLabel("生育期"))
        layout.addWidget(self.crop_stage)
        layout.addWidget(QLabel("面积(km²)"))
        layout.addWidget(self.area_edit)
        layout.addWidget(QLabel("产量(kg/km²)"))
        layout.addWidget(self.yield_edit)
        layout.addWidget(QLabel("市价(元/kg)"))
        layout.addWidget(self.price_edit)
        layout.addWidget(remove_btn)
        layout.addStretch()

    def collect(self) -> dict | None:
        try:
            return {
                "type": self.crop_type.currentText(),
                "stage": self.crop_stage.currentText(),
                "area": float(self.area_edit.text().strip()),
                "yield": float(self.yield_edit.text().strip()),
                "price": float(self.price_edit.text().strip()),
            }
        except Exception:
            return None


class MeteoDialog(QDialog):
    def __init__(self, params: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("配置气象参数")
        self.resize(450, 380)
        self._entries = {}
        self._parent = parent
        form = QFormLayout(self)
        rows = [
            ("Rn", "太阳净辐射 (Rn) [mm/d]:"), ("G", "土壤热通量 (G) [MJ/m²]:"),
            ("T", "地表日平均气温 (T) [°C]:"), ("u2", "地表2m处风速 (u2) [m/s]:"),
            ("es", "饱和水汽压 (es) [hPa]:"), ("ea", "实际水汽压 (ea) [hPa]:"),
            ("delta", "水汽压变率 (Δ):"), ("gamma", "湿度计常数 (γ) [hPa/°C]:"),
        ]
        for key, label in rows:
            edit = QLineEdit(str(params[key]))
            self._entries[key] = edit
            form.addRow(QLabel(label), edit)

        btn_layout = QHBoxLayout()
        fetch_btn = QPushButton("🌐 联网获取当地气象")
        fetch_btn.clicked.connect(self._fetch_weather)
        save_btn = QPushButton("✅ 保存并计算 ET0")
        save_btn.clicked.connect(self.accept)
        btn_layout.addWidget(fetch_btn)
        btn_layout.addWidget(save_btn)
        form.addRow(btn_layout)

    def collect(self) -> dict:
        return {key: float(edit.text().strip()) for key, edit in self._entries.items()}

    def _fetch_weather(self):
        try:
            import urllib3
            import requests
            import math
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

            def fetch_backup():
                url = "https://uapis.cn/api/v1/misc/weather?city=Bokhtar&extended=false&forecast=true"
                headers = {'User-Agent': 'Mozilla/5.0'}
                r = requests.get(url, headers=headers, params={"province": "Khatlon", "city": "Bokhtar"},
                               timeout=15, verify=False)
                r.raise_for_status()
                d = r.json()['forecast'][0]
                T = (d['temp_max'] + d['temp_min']) / 2
                u2 = d['wind_speed_day'] * (1000 / 3600)
                rh = d['humidity']
                es_kPa = 0.6108 * math.exp((17.27 * T) / (T + 237.3))
                es, ea = es_kPa * 10, es_kPa * 10 * (rh / 100)
                delta = (4098 * es_kPa) / ((T + 237.3) ** 2) * 10
                return {'T': T, 'u2': u2, 'es': es, 'ea': ea, 'delta': delta,
                        'gamma': 0.61, 'Rn': d.get('uv_index', 5) * 1.5, 'G': 0.0}

            try:
                lat, lon = 37.8333, 69.0000
                now = datetime.date.today()
                url = "https://power.larc.nasa.gov/api/temporal/daily/point"
                resp = requests.get(url, params={
                    "parameters": "ALLSKY_SFC_SW_DWN,ALLSKY_SFC_LW_DWN,ALLSKY_SFC_SW_UP,ALLSKY_SFC_LW_UP,T2M,WS2M,RH2M,PS",
                    "community": "AG", "longitude": lon, "latitude": lat,
                    "start": f"{now.year}{now.month:02d}01",
                    "end": f"{now.year}{now.month:02d}{now.day:02d}",
                    "format": "JSON"
                }, headers={'User-Agent': 'Mozilla/5.0'}, timeout=30, verify=False)
                resp.raise_for_status()
                data = resp.json().get("properties", {}).get("parameter", {})

                def avg(k):
                    vs = [v for v in data.get(k, {}).values() if isinstance(v, (int, float)) and v != -999]
                    return sum(vs) / len(vs) if vs else 0

                sw_d, sw_u = avg("ALLSKY_SFC_SW_DWN"), avg("ALLSKY_SFC_SW_UP")
                lw_d, lw_u = avg("ALLSKY_SFC_LW_DWN"), avg("ALLSKY_SFC_LW_UP")
                T = avg("T2M"); u2 = avg("WS2M"); RH = avg("RH2M"); P = avg("PS")
                Rn = max(0, (sw_d - sw_u + lw_d - lw_u) / 2.45)
                es = 0.6108 * math.exp((17.27 * T) / (T + 237.3)) * 10
                ea = es * (RH / 100)
                delta = (4098 * 0.6108 * math.exp((17.27 * T) / (T + 237.3))) / ((T + 237.3) ** 2) * 10
                gamma = 0.665e-3 * P * 10
                result = {'T': T, 'u2': u2, 'es': es, 'ea': ea, 'delta': delta, 'gamma': gamma, 'Rn': Rn, 'G': 0.0}
            except Exception:
                result = fetch_backup()

            for k, v in result.items():
                if k in self._entries:
                    self._entries[k].setText(f"{v:.2f}")
            QMessageBox.information(self, "成功", "已成功获取当地气象数据！")
        except Exception as e:
            QMessageBox.critical(self, "联网失败", str(e))


# ======================= 主 Widget =======================

class WaterAllocationWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.sectors = SECTOR_ORDER_V2  # ["生活","生态","农业","工业","下游国家"]
        self.stages = ["初期", "发育期", "中期", "后期"]
        self.fao_kc = {
            "冬小麦": {"初期": 0.40, "发育期": 0.8, "中期": 1.15, "后期": 0.60},
            "细绒棉": {"初期": 0.3, "发育期": 0.7, "中期": 1.15, "后期": 0.70},
            "玉米": {"初期": 0.30, "发育期": 0.9, "中期": 1.10, "后期": 0.50},
            "水稻": {"初期": 1.05, "发育期": 1.15, "中期": 1.20, "后期": 0.90},
            "油菜": {"初期": 0.50, "发育期": 0.75, "中期": 1.05, "后期": 0.50},
        }
        self.meteo_params = {
            "Rn": 10.0, "G": 0.0, "T": 20.0, "u2": 2.0,
            "es": 23.4, "ea": 15.0, "delta": 1.45, "gamma": 0.66,
        }
        self.crop_rows = []
        self._last_nsga2_result = None
        self.current_monthly_inflow = None
        self.rs_extracted_area = 0.0

        self._init_ui()

    # ======================= UI 初始化 =======================
    def _init_ui(self):
        self.setStyleSheet("""
            QGroupBox {
                margin-top: 22px;
                padding-top: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 12px;
                padding: 0 6px;
                font-family: "Microsoft YaHei", "SimHei";
                font-size: 10pt;
                font-weight: 600;
            }
        """)
        root = QVBoxLayout(self)
        tabs = QTabWidget()
        root.addWidget(tabs)

        self.tab_nsga2 = QWidget()
        self.tab_hydro = QWidget()
        tabs.addTab(self.tab_nsga2, " 📊 部门用水分配")
        tabs.addTab(self.tab_hydro, " 📡 水文数据与预测")

        self._build_nsga2_tab()
        self._build_hydro_tab()

    def _build_nsga2_tab(self):
        outer = QVBoxLayout(self.tab_nsga2)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll)
        container = QWidget()
        body = QVBoxLayout(container)
        scroll.setWidget(container)

        # ── 时间范围配置 ──
        time_box = QGroupBox("全局供水配置")
        time_grid = QGridLayout(time_box)
        time_grid.setHorizontalSpacing(18)
        time_grid.setVerticalSpacing(10)
        time_grid.setColumnStretch(1, 1)
        time_grid.setColumnStretch(3, 1)
        cur_year = datetime.date.today().year
        years = [str(i) for i in range(2000, cur_year + 16)]
        months = [str(i) for i in range(1, 13)]

        self.start_year_cb = QComboBox()
        self.start_year_cb.addItems(years)
        self.start_year_cb.setCurrentText(str(cur_year - 1))
        self.start_year_cb.setMinimumWidth(90)
        self.start_month_cb = QComboBox()
        self.start_month_cb.addItems(months)
        self.start_month_cb.setCurrentText("1")
        self.start_month_cb.setMinimumWidth(70)
        start_period = QWidget()
        start_layout = QHBoxLayout(start_period)
        start_layout.setContentsMargins(0, 0, 0, 0)
        start_layout.setSpacing(6)
        start_layout.addWidget(self.start_year_cb)
        start_layout.addWidget(QLabel("年"))
        start_layout.addWidget(self.start_month_cb)
        start_layout.addWidget(QLabel("月"))
        start_layout.addStretch()

        self.end_year_cb = QComboBox()
        self.end_year_cb.addItems(years)
        self.end_year_cb.setCurrentText(str(cur_year))
        self.end_year_cb.setMinimumWidth(90)
        self.end_month_cb = QComboBox()
        self.end_month_cb.addItems(months)
        self.end_month_cb.setCurrentText("12")
        self.end_month_cb.setMinimumWidth(70)
        end_period = QWidget()
        end_layout = QHBoxLayout(end_period)
        end_layout.setContentsMargins(0, 0, 0, 0)
        end_layout.setSpacing(6)
        end_layout.addWidget(self.end_year_cb)
        end_layout.addWidget(QLabel("年"))
        end_layout.addWidget(self.end_month_cb)
        end_layout.addWidget(QLabel("月"))
        end_layout.addStretch()

        time_grid.addWidget(QLabel("起始时间:"), 0, 0)
        time_grid.addWidget(start_period, 0, 1)
        time_grid.addWidget(QLabel("结束时间:"), 0, 2)
        time_grid.addWidget(end_period, 0, 3)

        time_grid.addWidget(QLabel("时间粒度:"), 1, 0)
        self.time_scale_cb = QComboBox()
        self.time_scale_cb.addItems(["monthly", "daily", "yearly"])
        self.time_scale_cb.setCurrentText("monthly")
        self.time_scale_cb.setMinimumWidth(120)
        time_grid.addWidget(self.time_scale_cb, 1, 1)

        time_grid.addWidget(QLabel("大坝起始可供水量(百万m³):"), 1, 2)
        self.w_surface_edit = QLineEdit("850")
        self.w_surface_edit.setMinimumWidth(120)
        time_grid.addWidget(self.w_surface_edit, 1, 3)

        body.addWidget(time_box)

        # ── 基础水文参数 ──
        base_box = QGroupBox("哈特隆州 指标估算与需水量配置")
        base_grid = QGridLayout(base_box)
        base_grid.setHorizontalSpacing(18)
        base_grid.setVerticalSpacing(10)
        params_data = [
            ("人口(万人):", "pop", "387"),
            ("城镇化率(%):", "urban", "23"),
            ("人口净增长率(%):", "pop_growth", "1.8"),
            ("工业重复利用率(%):", "reuse", "25"),
            ("当地GDP(亿元):", "gdp", "82"),
            ("生活废水回用率(%):", "dom_reuse", "15"),
            ("灌溉利用系数:", "eff", "0.85"),
            ("传输损耗率(%):", "loss", "12"),
            ("生态保障用水(百万m³):", "eco", "50"),
        ]
        self._param_edits = {}
        for idx, (text, key, default) in enumerate(params_data):
            r = idx // 3
            c = (idx % 3) * 2
            base_grid.addWidget(QLabel(text), r, c)
            edit = QLineEdit(default)
            edit.setMinimumWidth(90)
            base_grid.addWidget(edit, r, c + 1)
            self._param_edits[key] = edit

        meteo_btn = QPushButton("🌡️ 配置气象参数计算 ET0")
        meteo_btn.clicked.connect(self.open_meteo_config)
        base_grid.addWidget(meteo_btn, 3, 0, 1, 6)
        body.addWidget(base_box)

        # ── 水电参数 ──
        hydro_box = QGroupBox("努列克坝发电机组物理参数")
        hydro_grid = QGridLayout(hydro_box)
        hydro_grid.setHorizontalSpacing(18)
        hydro_grid.setVerticalSpacing(10)
        hydro_grid.addWidget(QLabel("单机最大功率(MW):"), 0, 0)
        self.hydro_pmax_edit = QLineEdit("335")
        self.hydro_pmax_edit.setMinimumWidth(90)
        hydro_grid.addWidget(self.hydro_pmax_edit, 0, 1)
        hydro_grid.addWidget(QLabel("单机最大流量(m³/s):"), 0, 2)
        self.hydro_qmax_edit = QLineEdit("146")
        self.hydro_qmax_edit.setMinimumWidth(90)
        hydro_grid.addWidget(self.hydro_qmax_edit, 0, 3)
        hydro_grid.addWidget(QLabel("上网电价(元/kWh):"), 0, 4)
        self.hydro_price_edit = QLineEdit("0.4")
        self.hydro_price_edit.setMinimumWidth(90)
        hydro_grid.addWidget(self.hydro_price_edit, 0, 5)
        hydro_grid.setColumnStretch(1, 1)
        hydro_grid.setColumnStretch(3, 1)
        hydro_grid.setColumnStretch(5, 1)
        body.addWidget(hydro_box)

        # ── 农业作物配置 ──
        crop_box = QGroupBox("农业作物动态配置")
        crop_outer = QVBoxLayout(crop_box)

        mode_layout = QHBoxLayout()
        self.agr_mode_group = QButtonGroup(self)
        self.manual_rb = QRadioButton("✍️ 人工精细输入模式")
        self.rs_rb = QRadioButton("🛰️ 遥感图像智能估算模式")
        self.manual_rb.setChecked(True)
        self.agr_mode_group.addButton(self.manual_rb, 0)
        self.agr_mode_group.addButton(self.rs_rb, 1)
        self.agr_mode_group.buttonClicked.connect(self._toggle_agr_mode)
        mode_layout.addWidget(self.manual_rb)
        mode_layout.addWidget(self.rs_rb)
        mode_layout.addStretch()
        crop_outer.addLayout(mode_layout)

        # 遥感面板
        self.rs_panel = QWidget()
        rs_layout = QVBoxLayout(self.rs_panel)
        rs_layout.setContentsMargins(0, 0, 0, 0)

        gee_row = QHBoxLayout()
        gee_row.addWidget(QLabel("🛰️ GEE 在线数据源 | 项目ID:"))
        self.gee_project_edit = QLineEdit("skillful-source-494707-h7")
        self.gee_project_edit.setFixedWidth(180)
        gee_row.addWidget(self.gee_project_edit)
        self.fetch_gee_btn = QPushButton("🌐 联网获取耕地面积")
        self.fetch_gee_btn.clicked.connect(self._fetch_gee_cropland)
        gee_row.addWidget(self.fetch_gee_btn)
        self.gee_status_label = QLabel("")
        gee_row.addWidget(self.gee_status_label)
        gee_row.addStretch()
        rs_layout.addLayout(gee_row)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        rs_layout.addWidget(sep)

        ftw_box = QGroupBox("本地影像处理")
        ftw_layout = QVBoxLayout(ftw_box)
        mask_row = QHBoxLayout()
        mask_row.addWidget(QLabel("掩  膜:"))
        self.mask_path_edit = QLineEdit()
        self.mask_path_edit.setPlaceholderText("可选，留空使用全图")
        mask_row.addWidget(self.mask_path_edit)
        mask_browse = QPushButton("浏览")
        mask_browse.clicked.connect(lambda: self.mask_path_edit.setText(
            QFileDialog.getOpenFileName(self, "选择GeoJSON", "", "GeoJSON (*.geojson *.json)")[0]
            or self.mask_path_edit.text()))
        mask_row.addWidget(mask_browse)
        ftw_layout.addLayout(mask_row)
        mask_row2 = QHBoxLayout()
        mask_row2.addWidget(QLabel("像素面积(m²):"))
        self.rs_resolution_edit = QLineEdit("自动检测"); self.rs_resolution_edit.setFixedWidth(80)
        self.rs_resolution_edit.setReadOnly(True)
        mask_row2.addWidget(self.rs_resolution_edit)
        mask_row2.addStretch()
        ftw_layout.addLayout(mask_row2)

        ftw_btn_row = QHBoxLayout()
        self.ftw_run_btn = QPushButton("🚀 选择影像并提取耕地面积")
        self.ftw_run_btn.clicked.connect(self._run_ftw_inference)
        ftw_btn_row.addWidget(self.ftw_run_btn)
        self.ftw_status_label = QLabel("请选择 Sentinel-2 L2A GeoTIFF 影像 (支持多选)")
        self.ftw_status_label.setStyleSheet("color: gray;")
        ftw_btn_row.addWidget(self.ftw_status_label)
        ftw_btn_row.addStretch()
        ftw_layout.addLayout(ftw_btn_row)

        ftw_help = QGroupBox("输入数据要求")
        ftw_help_layout = QVBoxLayout(ftw_help)
        ftw_help_layout.addWidget(QLabel(
            "• 4波段(单日期): [B4红, B3绿, B2蓝, B8近红外]\n"
            "• 8波段(双日期): [B4_A,B3_A,B2_A,B8_A, B4_B,B3_B,B2_B,B8_B]\n"
            ))
        ftw_help_layout.setContentsMargins(5, 5, 5, 5)
        ftw_layout.addWidget(ftw_help)
        rs_layout.addWidget(ftw_box)
        self.rs_panel.setVisible(False)
        crop_outer.addWidget(self.rs_panel)

        # 人工面板
        self.manual_panel = QWidget()
        manual_layout = QVBoxLayout(self.manual_panel)
        manual_layout.setContentsMargins(0, 0, 0, 0)

        header = QHBoxLayout()
        for label in ["作物类型", "生育期", "面积(km²)", "产量(kg/km²)", "市价(元/kg)"]:
            header.addWidget(QLabel(label))
        header.addStretch()
        manual_layout.addLayout(header)

        self.crop_container = QVBoxLayout()
        manual_layout.addLayout(self.crop_container)
        add_btn = QPushButton("➕ 添加作物")
        add_btn.clicked.connect(self.add_crop_row)
        manual_layout.addWidget(add_btn, alignment=Qt.AlignLeft)
        crop_outer.addWidget(self.manual_panel)
        self.add_crop_row()
        body.addWidget(crop_box)

        # ── 权重配置 ──
        weight_box = QGroupBox("决策偏好权重")
        weight_layout = QHBoxLayout(weight_box)
        weight_layout.addWidget(QLabel("整体经济权重:"))
        self.w_econ_edit = QLineEdit("0.33"); self.w_econ_edit.setFixedWidth(50)
        weight_layout.addWidget(self.w_econ_edit)
        weight_layout.addWidget(QLabel("降低缺水权重:"))
        self.w_short_edit = QLineEdit("0.33"); self.w_short_edit.setFixedWidth(50)
        weight_layout.addWidget(self.w_short_edit)
        weight_layout.addWidget(QLabel("部门公平(Gini)权重:"))
        self.w_gini_edit = QLineEdit("0.34"); self.w_gini_edit.setFixedWidth(50)
        weight_layout.addWidget(self.w_gini_edit)
        weight_layout.addStretch()
        body.addWidget(weight_box)

        t_weight_box = QGroupBox("部门收益权重 (T)")
        t_layout = QHBoxLayout(t_weight_box)
        self.t_edits = []
        for sec in self.sectors:
            t_layout.addWidget(QLabel(f"{sec}:"))
            edit = QLineEdit("1.0"); edit.setFixedWidth(50)
            t_layout.addWidget(edit)
            self.t_edits.append(edit)
        t_layout.addStretch()
        body.addWidget(t_weight_box)

        # ── 运行按钮 ──
        run_btn = QPushButton("🚀 启动部门分配分析")
        run_btn.setStyleSheet("QPushButton { font-size: 14px; font-weight: bold; padding: 10px 30px; }")
        run_btn.clicked.connect(self.run_nsga2_optimization)
        body.addWidget(run_btn)
        body.addStretch()

    def _build_hydro_tab(self):
        layout = QVBoxLayout(self.tab_hydro)

        # 水情信息
        info_box = QGroupBox("实时水情信息")
        info_layout = QVBoxLayout(info_box)
        info_row = QHBoxLayout()
        info_row.addWidget(QLabel("🌊 年平均径流量:"))
        self.annual_avg_label = QLabel("--")
        self.annual_avg_label.setStyleSheet("color: #2196F3; font-weight: bold; font-size: 14px;")
        info_row.addWidget(self.annual_avg_label)
        info_row.addWidget(QLabel("m³/s"))
        info_row.addSpacing(30)
        info_row.addWidget(QLabel("📅 水年份类型:"))
        self.water_year_label = QLabel("--")
        self.water_year_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        info_row.addWidget(self.water_year_label)
        info_row.addStretch()
        info_layout.addLayout(info_row)
        layout.addWidget(info_box)

        # 数据源
        ds_box = QGroupBox("气象/水文数据源")
        ds_layout = QVBoxLayout(ds_box)
        ds_row = QHBoxLayout()
        ds_row.addWidget(QLabel("数据源:"))
        self.nc_data_path_edit = QLineEdit()
        self.nc_data_path_edit.setReadOnly(True)
        ds_row.addWidget(self.nc_data_path_edit)
        for text, slot in [("选择文件", self._select_nc_file), ("选择文件夹", self._select_nc_dir), ("清空", lambda: self.nc_data_path_edit.setText(""))]:
            btn = QPushButton(text)
            btn.clicked.connect(slot)
            ds_row.addWidget(btn)
        ds_layout.addLayout(ds_row)

        ds_row2 = QHBoxLayout()
        self.target_year_label = QLabel("目标年份:")
        ds_row2.addWidget(self.target_year_label)
        sync_btn = QPushButton("🔄 同步年份")
        sync_btn.clicked.connect(self._sync_target_year)
        ds_row2.addWidget(sync_btn)
        ds_row2.addStretch()
        ds_row2.addWidget(QLabel("初蓄(亿m³):"))
        self.v_init_edit = QLineEdit("84.0"); self.v_init_edit.setFixedWidth(80)
        ds_row2.addWidget(self.v_init_edit)
        preview_btn = QPushButton("🔍 预览径流预测")
        preview_btn.clicked.connect(self._preview_water_info)
        ds_row2.addWidget(preview_btn)
        self.preview_status = QLabel("")
        ds_row2.addWidget(self.preview_status)
        ds_row2.addStretch()
        ds_layout.addLayout(ds_row2)
        self._sync_target_year()
        layout.addWidget(ds_box)

        # LSTM 训练
        train_box = QGroupBox("LSTM 径流预测模型训练")
        train_layout = QVBoxLayout(train_box)
        train_row = QHBoxLayout()
        train_row.addWidget(QLabel("训练数据:"))
        self.train_data_path_edit = QLineEdit()
        train_row.addWidget(self.train_data_path_edit)
        train_browse = QPushButton("浏览")
        train_browse.clicked.connect(lambda: self.train_data_path_edit.setText(
            QFileDialog.getOpenFileName(self, "选择训练数据", "", "CSV/Excel (*.csv *.xlsx *.xls);;NetCDF (*.nc)")[0]
            or self.train_data_path_edit.text()))
        train_row.addWidget(train_browse)
        train_layout.addLayout(train_row)

        train_row2 = QHBoxLayout()
        self.train_btn = QPushButton("🚀 开始训练")
        self.train_btn.clicked.connect(self._run_lstm_training)
        train_row2.addWidget(self.train_btn)
        self.train_status = QLabel("等待训练...")
        train_row2.addWidget(self.train_status)
        train_row2.addStretch()
        train_layout.addLayout(train_row2)
        layout.addWidget(train_box)
        layout.addStretch()

    # ======================= 农业模式切换 =======================
    def _toggle_agr_mode(self, btn):
        if self.rs_rb.isChecked():
            self.manual_panel.setVisible(False)
            self.rs_panel.setVisible(True)
        else:
            self.rs_panel.setVisible(False)
            self.manual_panel.setVisible(True)

    # ======================= 遥感相关 =======================
    def _fetch_gee_cropland(self):
        if not _HAS_REMOTE_SENSING:
            QMessageBox.warning(self, "缺少依赖", f"遥感模块不可用: {_RS_IMPORT_ERROR}")
            return
        try:
            self.gee_status_label.setText("🔄 正在从 GEE 获取数据，请稍候...")
            self.gee_status_label.setStyleSheet("color: blue;")
            gee_project = self.gee_project_edit.text().strip()
            if not gee_project:
                QMessageBox.warning(self, "配置错误", "请输入 GEE 项目 ID")
                return
            area_kilo = get_cropland_area_km2(gee_project=gee_project)
            self.rs_extracted_area = area_kilo
            self.gee_status_label.setText(f"✅ GEE 数据获取成功！耕地面积: {area_kilo:,.2f} 平方公里")
            self.gee_status_label.setStyleSheet("color: green;")
            QMessageBox.information(self, "GEE 数据获取成功", f"耕地面积: {area_kilo:,.2f} 平方公里")
        except Exception as e:
            self.gee_status_label.setText(f"❌ 获取失败: {str(e)[:50]}...")
            self.gee_status_label.setStyleSheet("color: red;")

    def _run_ftw_inference(self):
        if not _HAS_REMOTE_SENSING:
            QMessageBox.warning(self, "缺少依赖",
                f"遥感模块不可用: {_RS_IMPORT_ERROR}\n")
            return
        filepaths = QFileDialog.getOpenFileNames(self, "选择影像", "", "GeoTIFF (*.tif *.tiff)")[0]
        if not filepaths:
            return
        weights_path = str(_RESOURCE_DIR / "models" / "3_Class_FULL_FTW_Pretrained_v2.ckpt")
        if not os.path.exists(weights_path):
            QMessageBox.warning(self, "权重文件", f"FTW 权重文件不存在:\n{weights_path}")
            return
        params = {
            'filepaths': list(filepaths), 'weights': weights_path, 'device': 'cuda',
            'mask_path': self.mask_path_edit.text().strip(),
            'window_size': 1024, 'overlap': 64, 'ndvi_threshold': 0.3, 'workers': 2,
        }
        self.ftw_run_btn.setEnabled(False)
        self.ftw_status_label.setText("🔄 正在加载模型并处理影像文件…")
        self.ftw_status_label.setStyleSheet("color: blue;")
        self._ftw_worker = FTWInferenceWorker(params)
        self._ftw_worker.progress.connect(lambda msg: self.ftw_status_label.setText(msg))
        self._ftw_worker.finished.connect(self._on_ftw_complete)
        self._ftw_worker.error.connect(self._on_ftw_error)
        self._ftw_worker.start()

    def _on_ftw_complete(self, result):
        self.ftw_run_btn.setEnabled(True)
        area = round(result['area_km2'], 4)
        self.rs_extracted_area = area
        detail = "\n".join(f"  • {n}: {r['area_hectares']:.2f} ha ({r['field_pixels']:,} px)"
                          for n, r in result.get('per_file', []))
        summary = (f"✅ 提取完成！耕地面积: {area:,.2f} km²\n"
                   f"耕地总像素: {result['pixels']:,}  |  单像素面积: {result['pixel_sqm']:.2f} m²\n{detail}")
        self.ftw_status_label.setText(summary)
        self.ftw_status_label.setStyleSheet("color: green;")

    def _on_ftw_error(self, msg):
        self.ftw_run_btn.setEnabled(True)
        self.ftw_status_label.setText(f"❌ {msg[:80]}")
        self.ftw_status_label.setStyleSheet("color: red;")

    # ======================= 作物管理 =======================
    def add_crop_row(self):
        row = CropRowWidget(list(self.fao_kc.keys()), self.stages, self.remove_crop_row)
        self.crop_rows.append(row)
        self.crop_container.addWidget(row)

    def remove_crop_row(self, row_widget):
        if row_widget in self.crop_rows:
            self.crop_rows.remove(row_widget)
        row_widget.setParent(None)
        row_widget.deleteLater()

    # ======================= 气象 ET0 =======================
    def open_meteo_config(self):
        dialog = MeteoDialog(self.meteo_params, self)
        if dialog.exec_() == QDialog.Accepted:
            try:
                self.meteo_params = dialog.collect()
            except Exception as e:
                QMessageBox.critical(self, "参数错误", str(e))

    def _calc_et0(self):
        p = self.meteo_params
        num = 0.408 * p['delta'] * (p['Rn'] - p['G']) + p['gamma'] * (900 / (p['T'] + 278)) * p['u2'] * (p['es'] - p['ea'])
        den = p['delta'] + p['gamma'] * (1 + 0.34 * p['u2'])
        return num / den if den != 0 else 0.0

    # ======================= 内部需水计算 =======================
    def _calc_demands(self):
        """内部计算各部门需求量，不显示在UI"""
        try:
            month = int(self.start_month_cb.currentText())
            days = calendar.monthrange(2026, month)[1]
            et0 = self._calc_et0()

            pop = float(self._param_edits["pop"].text()) * 10000
            urban_rate = float(self._param_edits["urban"].text()) / 100
            gdp = float(self._param_edits["gdp"].text())
            reuse = float(self._param_edits["reuse"].text()) / 100
            eff = float(self._param_edits["eff"].text())
            eco_base = float(self._param_edits["eco"].text())

            pop_urban = pop * urban_rate
            pop_rural = pop * (1 - urban_rate)
            live_m3 = (pop_urban * 145 / 1000 + pop_rural * 80 / 1000) * days
            live = live_m3 / 1_000_000
            eco = 0.1 * live + eco_base

            agr = 0.0
            if self.rs_rb.isChecked() and self.rs_extracted_area > 0:
                def_type, def_stage = self._get_default_crop()
                kc = self.fao_kc[def_type][def_stage]
                c_area = self.rs_extracted_area * 1_000_000 * 0.85
                etc_monthly = kc * et0 * days
                water_m3 = etc_monthly * 0.001 * c_area * 0.05
                agr = (water_m3 / 1_000_000) / eff if eff > 0 else 0
            else:
                for row in self.crop_rows:
                    data = row.collect()
                    if data is None:
                        continue
                    kc = self.fao_kc[data["type"]][data["stage"]]
                    c_area = float(data["area"]) * 1_000_000
                    etc_monthly = kc * et0 * days
                    water_m3 = etc_monthly * 0.001 * c_area
                    agr += (water_m3 / 1_000_000) / eff if eff > 0 else 0

            quota = 140
            annual_ind = gdp * 10000 * quota * (1 - reuse)
            season = [0.85, 0.80, 0.90, 0.95, 1.05, 1.10, 1.15, 1.15, 1.05, 0.95, 0.85, 0.80]
            ind = annual_ind * season[month - 1] / 12 / 1000000

            target_year = int(self.start_year_cb.currentText())
            try:
                result = predict_downstream_total(target_year)
                downstream = result['downstream_monthly'][month - 1] * 1000
            except Exception:
                downstream = 0.0

            return {
                SECTOR_LIVE: live, SECTOR_ECO: eco, SECTOR_AGR: agr,
                SECTOR_IND: ind, SECTOR_DOWN: downstream,
            }
        except Exception:
            return None

    def _get_default_crop(self):
        month = int(self.start_month_cb.currentText())
        if month in [1, 2, 3]:
            return "冬小麦", "发育期"
        elif month in [4, 5]:
            return "细绒棉", "初期"
        elif month in [6, 7, 8]:
            return "细绒棉", "中期"
        elif month in [9, 10]:
            return "细绒棉", "后期"
        return "冬小麦", "初期"

    # ======================= 经济效益估算 =======================
    def _estimate_economic(self):
        p_max = float(self.hydro_pmax_edit.text())
        q_max = float(self.hydro_qmax_edit.text())
        price = float(self.hydro_price_edit.text())
        a_hydro = ((p_max * 1000) / (q_max * 3600)) * price

        total_revenue = 0.0
        if self.rs_rb.isChecked() and self.rs_extracted_area > 0:
            area = self.rs_extracted_area
            month = int(self.start_month_cb.currentText())
            if month in [1, 2, 3]:
                total_revenue = area * 417300.0 * 2.5
            elif month in [4, 5]:
                total_revenue = area * 350000.0 * 7.5
            elif month in [6, 7, 8]:
                total_revenue = area * 35000000.0 * 7.5
            elif month in [9, 10]:
                total_revenue = area * 3500000.0 * 7.5
            else:
                total_revenue = area * 417300.0 * 2.5
        else:
            for row in self.crop_rows:
                data = row.collect()
                if data is None:
                    continue
                total_revenue += float(data["area"]) * float(data["yield"]) * float(data["price"])

        demands = self._calc_demands()
        agr_demand = (demands[SECTOR_AGR] if demands else 1.0) * 1_000_000
        alpha = 0.5
        a_agr = (total_revenue / agr_demand) * alpha if agr_demand > 0 else 0.8
        a_dom, a_eco, a_ind = 1.1, 1.0, 9.0
        a_down = 1e-9

        a_surface = [a_dom + a_hydro, a_eco + a_hydro, a_agr + a_hydro, a_ind + a_hydro, a_down + a_hydro]
        b_surface = [0.005, 0.105, 0.005, 1.505, 0.0]
        a_ground = [a_dom, a_eco, a_agr, a_ind, a_down]
        b_ground = [a_dom + 0.4, 0.1, a_agr + 0.3, a_ind + 0.5, 0.0]
        return np.array([a_surface, a_ground]), np.array([b_surface, b_ground]), a_hydro, a_agr

    # ======================= NSGA-II 优化 =======================
    def run_nsga2_optimization(self):
        try:
            demands = self._calc_demands()
            if demands is None:
                QMessageBox.critical(self, "错误", "需水计算失败，请检查参数。")
                return

            start_year = int(self.start_year_cb.currentText())
            start_month = int(self.start_month_cb.currentText())
            end_year = int(self.end_year_cb.currentText())
            end_month = int(self.end_month_cb.currentText())
            time_scale = self.time_scale_cb.currentText()

            if time_scale == "yearly":
                n_periods = end_year - start_year + 1
                days_per_period = 365.25
            elif time_scale == "monthly":
                n_periods = (end_year - start_year) * 12 + (end_month - start_month + 1)
                days_per_period = 365.25 / 12
            else:
                import pandas as pd
                start = pd.Timestamp(year=start_year, month=start_month, day=1)
                end = pd.Timestamp(year=end_year, month=end_month, day=1)
                if end_month == 12:
                    end = pd.Timestamp(year=end_year, month=12, day=31)
                else:
                    end = pd.Timestamp(year=end_year, month=end_month + 1, day=1) - pd.Timedelta(days=1)
                n_periods = (end - start).days + 1
                days_per_period = 1.0

            date_labels = self._build_labels(start_year, start_month, end_year, end_month, time_scale)

            base_w = float(self.w_surface_edit.text())
            base_w_per_period = base_w / max(n_periods, 1)
            seconds_per_period = days_per_period * 24 * 3600

            if self.current_monthly_inflow is not None:
                monthly_inflow = np.asarray(self.current_monthly_inflow)
            else:
                monthly_inflow = np.full(12, 300.0)

            W_supply = np.zeros((n_periods, 2))
            for t in range(n_periods):
                if time_scale == "yearly":
                    inflow_cms = np.mean(monthly_inflow)
                elif time_scale == "monthly":
                    month_idx = (start_month - 1 + t) % 12
                    inflow_cms = monthly_inflow[month_idx]
                else:
                    cum = t; m = start_month - 1
                    while cum >= calendar.monthrange(start_year + (m // 12), (m % 12) + 1)[1]:
                        cum -= calendar.monthrange(start_year + (m // 12), (m % 12) + 1)[1]
                        m += 1
                    month_idx = m % 12
                    inflow_cms = monthly_inflow[month_idx]
                period_inflow = inflow_cms * seconds_per_period / 1_000_000
                W_supply[t, 0] = base_w_per_period + period_inflow

            demand_scale = days_per_period / (365.25 / 12)
            D_demand = np.zeros((n_periods, 5))
            for s, sec in enumerate(self.sectors):
                monthly_val = demands[sec]
                D_demand[:, s] = monthly_val * demand_scale

            loss_val = float(self._param_edits["loss"].text()) / 100.0
            loss_rates = np.array([loss_val])

            a_matrix, b_matrix, a_hydro, a_agr = self._estimate_economic()
            T_weights = np.array([float(e.text()) for e in self.t_edits])

            problem_params = {
                "n_sources": 2, "n_regions": 1, "m_sectors": 5,
                "n_periods": n_periods, "time_scale": time_scale,
                "a": a_matrix, "b": b_matrix, "T": T_weights,
                "D": D_demand, "W": W_supply,
                "F_min": D_demand * 0.2, "F_max": D_demand * 2.5,
                "loss_rates": loss_rates,
            }

            if n_periods <= 12:
                pop_size, n_gen = 200, 400
            elif n_periods <= 31:
                pop_size, n_gen = 150, 300
            elif n_periods <= 90:
                pop_size, n_gen = 120, 200
            elif n_periods <= 180:
                pop_size, n_gen = 80, 120
            else:
                pop_size, n_gen = 60, 80

            if n_periods > 45:
                reply = QMessageBox.question(self, "大规模优化提示",
                    f"当前配置将产生 {n_periods} 个时段、{n_periods * 10} 个优化变量。\n"
                    f"将使用 pop={pop_size}, gen={n_gen} 进行优化，可能需要较长时间。\n\n"
                    f"建议: 对于日粒度超过 1.5 个月的区间，考虑使用月粒度。\n\n是否继续?",
                    QMessageBox.Yes | QMessageBox.No)
                if reply != QMessageBox.Yes:
                    return

            res = run_nsga2_opt(problem_params, pop_size=pop_size, n_gen=n_gen)
            if res is None or res.F is None:
                res = run_nsga2_opt(problem_params, pop_size=pop_size * 2, n_gen=n_gen * 2)
            if res is None or res.F is None:
                QMessageBox.critical(self, "优化失败",
                    "无法找到可行解。\n可能原因: 时间粒度太细导致约束过多。\n建议: 改用月粒度或年度粒度重试。")
                return

            pref_weights = np.array([
                float(self.w_econ_edit.text()),
                float(self.w_short_edit.text()),
                float(self.w_gini_edit.text()),
            ])
            F, F_min_norm, F_max_norm = res.F, res.F.min(axis=0), res.F.max(axis=0)
            F_range = np.where(F_max_norm - F_min_norm == 0, 1e-9, F_max_norm - F_min_norm)
            best_idx = np.argmin(np.linalg.norm(((F - F_min_norm) / F_range) * pref_weights, axis=1))

            best_X = res.X[best_idx].reshape((n_periods, 2, 1, 5))
            X_agg = best_X.sum(axis=0)

            # 弹窗显示结果
            result_data = {
                "time_scale": time_scale,
                "profit": float(-res.F[best_idx, 0]),
                "shortage": float(res.F[best_idx, 1]),
                "gini": float(res.F[best_idx, 2]),
                "X_agg": X_agg,
                "D_demand": D_demand.sum(axis=0).reshape(1, -1),
                "loss_rates": loss_rates,
                "W_supply": W_supply.sum(axis=0),
                "a_hydro": a_hydro,
                "n_periods": n_periods,
                "time_series_X": best_X if n_periods > 1 else None,
                "date_labels": date_labels if n_periods > 1 else None,
            }
            dlg = ResultDialog(result_data, self.sectors, self)
            dlg.exec_()
        except Exception as e:
            QMessageBox.critical(self, "运行错误", str(e))

    def _build_labels(self, sy, sm, ey, em, scale):
        import pandas as pd
        labels = []
        if scale == "yearly":
            for y in range(sy, ey + 1):
                labels.append(str(y))
        elif scale == "monthly":
            for y in range(sy, ey + 1):
                ms = sm if y == sy else 1
                me = em if y == ey else 12
                for m in range(ms, me + 1):
                    labels.append(f"{y}-{m:02d}")
        else:
            start = pd.Timestamp(year=sy, month=sm, day=1)
            if em == 12:
                end = pd.Timestamp(year=ey, month=12, day=31)
            else:
                end = pd.Timestamp(year=ey, month=em + 1, day=1) - pd.Timedelta(days=1)
            d = start
            while d <= end:
                labels.append(d.strftime("%m-%d"))
                d += pd.Timedelta(days=1)
        return labels

    # ======================= 水文数据 Tab =======================
    def _select_nc_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择数据", "", "Data (*.nc *.csv *.xlsx *.xls)")
        if path:
            self.nc_data_path_edit.setText(path)

    def _select_nc_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择目录")
        if path:
            self.nc_data_path_edit.setText(path)

    def _sync_target_year(self):
        sy = self.start_year_cb.currentText()
        ey = self.end_year_cb.currentText()
        if sy == ey:
            self.target_year_label.setText(f"目标年份: {sy}")
        else:
            self.target_year_label.setText(f"目标时段: {sy} → {ey}")

    def _preview_water_info(self):
        filepath = self.nc_data_path_edit.text().strip()
        try:
            self.preview_status.setText("🔄 预测中...")
            self.preview_status.setStyleSheet("color: blue;")
            target_year = int(self.start_year_cb.currentText())

            def on_updated(monthly_inflow):
                self.current_monthly_inflow = monthly_inflow
                avg = np.mean(monthly_inflow)
                self.annual_avg_label.setText(f"{avg:.2f}")
                if avg >= 730:
                    wt, color = "丰水年", "#4CAF50"
                elif avg <= 574:
                    wt, color = "枯水年", "#F44336"
                else:
                    wt, color = "平水年", "#FF9800"
                self.water_year_label.setText(wt)
                self.water_year_label.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 14px;")

            _ = NurekDamParameters(
                elec_price=float(self.hydro_price_edit.text()),
                unit_water_margin=1.6,
                data_path=filepath if filepath else "",
                v_initial=float(self.v_init_edit.text()),
                update_callback=on_updated,
                target_year=target_year,
            )
            self.preview_status.setText("✅ 预测完成")
            self.preview_status.setStyleSheet("color: green;")
        except Exception as e:
            self.preview_status.setText(f"预测失败: {e}")
            self.preview_status.setStyleSheet("color: red;")

    def _run_lstm_training(self):
        data_path = self.train_data_path_edit.text().strip()
        if not data_path or not os.path.exists(data_path):
            QMessageBox.warning(self, "数据错误", "请选择有效的训练数据文件")
            return
        self.train_btn.setEnabled(False)
        self.train_status.setText("🔄 训练中...")
        self.train_status.setStyleSheet("color: blue;")
        self._lstm_worker = LSTMWorker(data_path)
        self._lstm_worker.progress.connect(lambda msg: self.train_status.setText(msg))
        self._lstm_worker.finished.connect(self._on_train_done)
        self._lstm_worker.error.connect(self._on_train_error)
        self._lstm_worker.start()

    def _on_train_done(self, result):
        self.train_btn.setEnabled(True)
        self.train_status.setText(
            f"✅ 训练完成 | Test RMSE: {result['test_rmse']:.2f} MAE: {result['test_mae']:.2f} R²: {result['test_r2']:.4f}")
        self.train_status.setStyleSheet("color: green;")

    def _on_train_error(self, msg):
        self.train_btn.setEnabled(True)
        self.train_status.setText(f"❌ 训练失败: {msg[:60]}")
        self.train_status.setStyleSheet("color: red;")
