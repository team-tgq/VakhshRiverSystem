"""努列克坝水资源动态配置系统 — Tkinter 独立界面 v2.0"""
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from pathlib import Path
import numpy as np
import cv2
import os
import sys
import math
import calendar
import datetime
import matplotlib.pyplot as plt
import requests
import urllib3
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from concurrent.futures import ThreadPoolExecutor, as_completed

# 资源目录
_RESOURCE_DIR = Path(__file__).resolve().parent / "resources"

# 模块内部导入
from .remote_sensing.gee_service import get_cropland_area_km2
from .remote_sensing.ftw_model import (
    create_ftw_model,
    calculate_cropland_area,
    load_geojson_mask,
    FTW_BAND_INDICES,
)
from .core import (
    NurekDamParameters,
    run_nsga2_opt,
)
from .predict import predict_downstream_total

# 字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class WaterAllocationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("努列克坝水资源动态配置系统 v2.0")
        self.root.geometry("1100x880")
        self.current_monthly_inflow = None
        self.mask_path_var = tk.StringVar()
        self.ftw_model = None
        self.ftw_worker_thread = None
        self.rs_extracted_area = 0.0
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TLabel", font=("Microsoft YaHei", 9))
        style.configure("TButton", font=("Microsoft YaHei", 10, "bold"), background="#0078D7", foreground="white")
        style.map("TButton", background=[("active", "#005A9E")])
        style.configure("Header.TLabel", font=("Microsoft YaHei", 12, "bold"), foreground="#003366")
        style.configure("Small.TButton", font=("Microsoft YaHei", 8), padding=0)

        # 基础数据
        self.fao_kc = {
            "冬小麦": {"初期": 0.40, "发育期": 0.8, "中期": 1.15, "后期": 0.60},
            "细绒棉": {"初期": 0.3, "发育期": 0.7, "中期": 1.15, "后期": 0.70},
            "玉米": {"初期": 0.30, "发育期": 0.9, "中期": 1.10, "后期": 0.50},
            "水稻": {"初期": 1.05, "发育期": 1.15, "中期": 1.20, "后期": 0.90},
            "油菜": {"初期": 0.50, "发育期": 0.75, "中期": 1.05, "后期": 0.50}
        }
        self.stages = ["初期", "发育期", "中期", "后期"]
        self.meteo_params = {
            0: {'Rn': 10.0, 'G': 0.0, 'T': 20.0, 'u2': 2.0, 'es': 23.4, 'ea': 15.0, 'delta': 1.45, 'gamma': 0.66}
        }

        self.sectors = ["生活", "生态", "农业", "工业", "下游国家"]
        self.t_entries = []

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.tab_config = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_config, text=" 📊 部门用水分配")

        self.tab_data = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_data, text=" 📡 水文数据与预测")

        self.build_config_tab()
        self.build_data_tab()

    # ========================== NSGA-II 界面构建 ==========================
    def build_config_tab(self):
        canvas = tk.Canvas(self.tab_config, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.tab_config, orient="vertical", command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)

        scroll_frame.bind("<Configure>",
                          lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        canvas_frame_id = canvas.create_window((0, 0), window=scroll_frame, anchor="nw")

        def _on_canvas_configure(event):
            canvas.itemconfig(canvas_frame_id, width=event.width)

        canvas.bind("<Configure>", _on_canvas_configure)

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        canvas.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))
        canvas.bind_all("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        canvas.configure(yscrollcommand=scrollbar.set)

        top_frame = ttk.Frame(scroll_frame)
        top_frame.pack(fill=tk.X, padx=15, pady=5)

        ttk.Label(top_frame, text="📅 全局供水配置", style="Header.TLabel").grid(row=0, column=0, columnspan=10, sticky=tk.W, pady=(0, 5))

        current_year = datetime.date.today().year
        year_range = [str(i) for i in range(2000, current_year + 16)]
        month_range = [str(i) for i in range(1, 13)]

        ttk.Label(top_frame, text="起始:").grid(row=1, column=0, padx=(5, 2))
        self.start_year_var = tk.StringVar(value=str(current_year - 1))
        ttk.Combobox(top_frame, textvariable=self.start_year_var,
                     values=year_range, width=6, state="readonly").grid(row=1, column=1, padx=2)
        ttk.Label(top_frame, text="年").grid(row=1, column=2, padx=(0, 5))
        self.start_month_var = tk.StringVar(value="1")
        ttk.Combobox(top_frame, textvariable=self.start_month_var,
                     values=month_range, width=4, state="readonly").grid(row=1, column=3, padx=2)
        ttk.Label(top_frame, text="月").grid(row=1, column=4, padx=(0, 10))

        ttk.Label(top_frame, text="结束:").grid(row=1, column=5, padx=(10, 2))
        self.end_year_var = tk.StringVar(value=str(current_year))
        ttk.Combobox(top_frame, textvariable=self.end_year_var,
                     values=year_range, width=6, state="readonly").grid(row=1, column=6, padx=2)
        ttk.Label(top_frame, text="年").grid(row=1, column=7, padx=(0, 5))
        self.end_month_var = tk.StringVar(value="12")
        ttk.Combobox(top_frame, textvariable=self.end_month_var,
                     values=month_range, width=4, state="readonly").grid(row=1, column=8, padx=2)
        ttk.Label(top_frame, text="月").grid(row=1, column=9, padx=(0, 10))

        ttk.Label(top_frame, text="时间粒度:").grid(row=2, column=0, padx=5, pady=(5, 0))
        self.time_scale_var = tk.StringVar(value="monthly")
        ttk.Combobox(top_frame, textvariable=self.time_scale_var,
                     values=["daily", "monthly", "yearly"], width=7,
                     state="readonly").grid(row=2, column=1, padx=5, pady=(5, 0))

        ttk.Label(top_frame, text="大坝起始可供水量(百万m³):").grid(row=2, column=2, columnspan=3, padx=15, pady=(5, 0))
        self.w_surface = ttk.Entry(top_frame, width=12)
        self.w_surface.insert(0, "850")
        self.w_surface.grid(row=2, column=5, columnspan=2, padx=5, pady=(5, 0), sticky=tk.W)

        self.year_var = self.start_year_var
        self.month_var = self.start_month_var

        ttk.Label(scroll_frame, text="📍 哈特隆州 指标估算与需水量配置", style="Header.TLabel").pack(anchor=tk.W, padx=15, pady=5)

        main_region_frame = ttk.Frame(scroll_frame)
        main_region_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=5)

        self.demand_entries = {}
        self.loss_entries = {}
        self.vars = {0: {}}
        self.crop_vars = {0: []}

        lf = ttk.LabelFrame(main_region_frame, text="基础水文参数区")
        lf.pack(fill=tk.BOTH, expand=True, padx=10, pady=2)

        params_f = ttk.Frame(lf)
        params_f.pack(fill=tk.X, padx=5, pady=2)

        labels_keys = [
            ("人口(万人):", "pop", "387"),        ("城镇化率(%):", "urban", "23"),
            ("人口净增长率(%):", "pop_growth", "1.8"), ("工业重复利用率(%):", "reuse", "25"),
            ("当地GDP(亿元):", "gdp", "82"),     ("生活废水回用率(%):", "dom_reuse", "15"),
            ("灌溉利用系数:", "eff", "0.85"),     ("传输损耗率(%):", "loss", "12"),
            ("生态保障用水(百万m³):", "eco", "50"),
        ]

        for i, (text, key, default) in enumerate(labels_keys):
            r, c = i // 2, (i % 2) * 2
            ttk.Label(params_f, text=text).grid(row=r, column=c, sticky=tk.W, pady=2, padx=10)
            ent = ttk.Entry(params_f, width=10)
            ent.insert(0, default)
            ent.grid(row=r, column=c + 1, padx=10, pady=2)
            self.vars[0][key] = ent
            if key == "loss":
                self.loss_entries[0] = ent

        self.vars[0]["urban_quota"] = tk.StringVar(value="145")
        self.vars[0]["rural_quota"] = tk.StringVar(value="80")
        self.et0_value = tk.StringVar(value="0.0")

        ttk.Button(params_f, text="🌡️ 配置气象参数计算 ET0", style="Small.TButton",
                   command=lambda: self.open_meteo_config(0)).grid(row=6, column=0, columnspan=4, pady=5, sticky=tk.EW)

        hydro_f = ttk.LabelFrame(lf, text=" ⚡ 努列克坝发电机组物理参数")
        hydro_f.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(hydro_f, text="单机最大功率(MW):").grid(row=0, column=0, padx=5, pady=5)
        self.hydro_pmax = ttk.Entry(hydro_f, width=8)
        self.hydro_pmax.insert(0, "335")
        self.hydro_pmax.grid(row=0, column=1, padx=5)

        ttk.Label(hydro_f, text="单机最大流量(m³/s):").grid(row=0, column=2, padx=5, pady=5)
        self.hydro_qmax = ttk.Entry(hydro_f, width=8)
        self.hydro_qmax.insert(0, "146")
        self.hydro_qmax.grid(row=0, column=3, padx=5)

        ttk.Label(hydro_f, text="上网电价(元/kWh):").grid(row=0, column=4, padx=5, pady=5)
        self.hydro_price = ttk.Entry(hydro_f, width=8)
        self.hydro_price.insert(0, "0.4")
        self.hydro_price.grid(row=0, column=5, padx=5)

        crop_outer_f = ttk.LabelFrame(lf, text=" 🌾 农业作物动态配置 ")
        crop_outer_f.pack(fill=tk.X, padx=10, pady=5)

        mode_frame = ttk.Frame(crop_outer_f)
        mode_frame.pack(fill=tk.X, padx=5, pady=5)

        self.agr_calc_mode = tk.StringVar(value="manual")

        ttk.Radiobutton(mode_frame, text="✍️ 人工精细输入模式", variable=self.agr_calc_mode,
                        value="manual", command=lambda: self.toggle_agr_mode(0)).pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(mode_frame, text="🛰️ 遥感图像智能估算模式", variable=self.agr_calc_mode,
                        value="rs", command=lambda: self.toggle_agr_mode(0)).pack(side=tk.LEFT, padx=10)

        self.rs_frame = ttk.Frame(crop_outer_f)

        gee_row = ttk.Frame(self.rs_frame)
        gee_row.pack(fill=tk.X, pady=5)

        ttk.Label(gee_row, text="🛰️ GEE 在线数据源:", font=("Microsoft YaHei", 9, "bold")).pack(side=tk.LEFT, padx=5)
        ttk.Label(gee_row, text="项目ID:").pack(side=tk.LEFT, padx=(10, 2))
        self.gee_project_entry = ttk.Entry(gee_row, width=25)
        self.gee_project_entry.insert(0, "skillful-source-494707-h7")
        self.gee_project_entry.pack(side=tk.LEFT, padx=2)

        self.btn_fetch_gee = ttk.Button(gee_row, text="🌐 联网获取耕地面积", style="Small.TButton",
                                         command=lambda: self.fetch_gee_cropland_area(0))
        self.btn_fetch_gee.pack(side=tk.LEFT, padx=10)

        self.gee_result_label = ttk.Label(gee_row, text="点击按钮从 GEE 获取数据...", foreground="gray")
        self.gee_result_label.pack(side=tk.LEFT, padx=5)

        ttk.Separator(self.rs_frame, orient='horizontal').pack(fill=tk.X, pady=5, padx=5)

        local_frame = ttk.LabelFrame(self.rs_frame, text="📁 本地影像处理 (FTW 耕地提取模型)", padding=8)
        local_frame.pack(fill=tk.X, pady=5, padx=5)

        f2 = ttk.Frame(local_frame)
        f2.pack(fill=tk.X, pady=2)

        f3 = ttk.Frame(local_frame)
        f3.pack(fill=tk.X, pady=2)
        ttk.Label(f3, text="掩  膜:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(f3, textvariable=self.mask_path_var, width=38).pack(side=tk.LEFT, padx=5)
        ttk.Button(f3, text="浏览", style="Small.TButton",
                   command=lambda: self.mask_path_var.set(
                       filedialog.askopenfilename(title="选择GeoJSON区域掩膜",
                           filetypes=[("GeoJSON", "*.geojson"), ("JSON", "*.json"), ("All", "*.*")])
                       or self.mask_path_var.get())).pack(side=tk.LEFT, padx=2)
        ttk.Label(f3, text="像素面积(m²):").pack(side=tk.LEFT, padx=(15, 0))
        self.rs_resolution_entry = ttk.Entry(f3, width=7)
        self.rs_resolution_entry.insert(0, "自动检测")
        self.rs_resolution_entry.pack(side=tk.LEFT, padx=2)

        f4 = ttk.Frame(local_frame)
        f4.pack(fill=tk.X, pady=(5, 0))
        self.btn_upload_rs = ttk.Button(f4, text="🚀 选择影像并提取耕地面积",
                                        command=lambda: self.upload_rs_image(0))
        self.btn_upload_rs.pack(side=tk.LEFT, padx=5)
        self.rs_result_label = ttk.Label(f4, text="请选择 Sentinel-2 L2A GeoTIFF 影像 (支持多选)", foreground="gray")
        self.rs_result_label.pack(side=tk.LEFT, padx=10)

        help_frame = ttk.LabelFrame(local_frame, text="📋 输入数据要求", padding=5)
        help_frame.pack(fill=tk.X, pady=(5, 0), padx=5)
        help_lines = (
            '• 数据源: Sentinel-2 L2A (大气校正) 10m 分辨率\n'
            '• 文件格式: GeoTIFF (.tif), 支持多选 | 默认区域: GeoJSON 掩膜\n'
            '• 4波段(单日期): [B4红, B3绿, B2蓝, B8近红外]\n'
            '• 8波段(双日期): [B4_A,B3_A,B2_A,B8_A, B4_B,B3_B,B2_B,B8_B]\n'
            '• 自定义区域: 点击[浏览]选择 GeoJSON, 或留空使用全图\n'
            '• 模型: FTW U-Net+EfficientNet-B3 | 归一化/3000 | 像素面积自动检测'
        )
        ttk.Label(help_frame, text=help_lines, justify=tk.LEFT,
                  font=("Microsoft YaHei", 7), foreground="#666666").pack(anchor=tk.W, padx=5)

        self.manual_frame = ttk.Frame(crop_outer_f)
        self.manual_frame.pack(fill=tk.X)

        header_f = ttk.Frame(self.manual_frame)
        header_f.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(header_f, text="作物类型", width=10).pack(side=tk.LEFT, padx=5)
        ttk.Label(header_f, text="生育期", width=10).pack(side=tk.LEFT, padx=5)
        ttk.Label(header_f, text="面积(km²)", width=12).pack(side=tk.LEFT, padx=5)
        ttk.Label(header_f, text="产量(kg/km²)", width=13).pack(side=tk.LEFT, padx=5)
        ttk.Label(header_f, text="市价(元/kg)", width=10).pack(side=tk.LEFT, padx=5)

        self.vars[0]["crop_container"] = ttk.Frame(self.manual_frame)
        self.vars[0]["crop_container"].pack(fill=tk.X)
        self.add_crop_row(0)

        ttk.Button(self.manual_frame, text="➕ 添加作物", style="Small.TButton",
                   command=lambda: self.add_crop_row(0)).pack(pady=5)

        for s_idx, sec in enumerate(self.sectors):
            ent = ttk.Entry(lf)
            ent.insert(0, "0.0")
            self.demand_entries[(0, s_idx)] = ent

        self.calculate_et0_from_params(0)

        bottom_f = ttk.Frame(scroll_frame)
        bottom_f.pack(fill=tk.X, pady=5)

        weight_f = ttk.LabelFrame(bottom_f, text=" ⚖️ 决策偏好权重")
        weight_f.pack(fill=tk.X, padx=15, pady=5)

        ttk.Label(weight_f, text="整体经济权重:").pack(side=tk.LEFT, padx=(10, 2))
        self.w_econ = ttk.Entry(weight_f, width=5)
        self.w_econ.insert(0, "0.33")
        self.w_econ.pack(side=tk.LEFT, padx=2)

        ttk.Label(weight_f, text="降低缺水权重:").pack(side=tk.LEFT, padx=(15, 2))
        self.w_short = ttk.Entry(weight_f, width=5)
        self.w_short.insert(0, "0.33")
        self.w_short.pack(side=tk.LEFT, padx=2)

        ttk.Label(weight_f, text="部门公平(Gini)权重:").pack(side=tk.LEFT, padx=(15, 2))
        self.w_gini = ttk.Entry(weight_f, width=5)
        self.w_gini.insert(0, "0.34")
        self.w_gini.pack(side=tk.LEFT, padx=2)

        t_weight_f = ttk.LabelFrame(bottom_f, text=" 🎛️ 部门收益权重 (T)")
        t_weight_f.pack(fill=tk.X, padx=15, pady=5)

        for i, sec in enumerate(self.sectors):
            ttk.Label(t_weight_f, text=f"{sec}:").pack(side=tk.LEFT, padx=(10, 2))
            ent = ttk.Entry(t_weight_f, width=5)
            ent.insert(0, "1.0")
            ent.pack(side=tk.LEFT, padx=2)
            self.t_entries.append(ent)

        ttk.Button(bottom_f, text="🚀 启动部门分配分析", command=self.run_nsga2_optimization).pack(ipadx=20, ipady=8, pady=10)

    def toggle_agr_mode(self, r_idx):
        if self.agr_calc_mode.get() == "rs":
            self.manual_frame.pack_forget()
            self.rs_frame.pack(fill=tk.X, padx=5, pady=5)
        else:
            self.rs_frame.pack_forget()
            self.manual_frame.pack(fill=tk.X)
        self.calc_exact(r_idx)

    def upload_rs_image(self, r_idx):
        """选择 Sentinel-2 L2A GeoTIFF 影像"""
        filepaths = filedialog.askopenfilenames(
            title="选择 Sentinel-2 L2A GeoTIFF 影像 (支持多选)",
            filetypes=[("GeoTIFF", "*.tif *.tiff"), ("All Files", "*.*")]
        )
        if not filepaths:
            return

        default_weights = str(_RESOURCE_DIR / "models" / "3_Class_FULL_FTW_Pretrained_v2.ckpt")
        params = {
            'filepaths': list(filepaths),
            'weights': default_weights,
            'device': 'cuda',
            'mask_path': self.mask_path_var.get().strip(),
            'window_size': 1024,
            'overlap': 64,
            'ndvi_threshold': 0.3,
            'workers': 2,
        }

        if not params['weights'] or not os.path.exists(params['weights']):
            messagebox.showwarning("权重文件", f"模型权重文件不存在:\n{params['weights']}\n\n请先下载 FTW 预训练权重并放到 models/ 目录。")
            return

        self.btn_upload_rs.config(state=tk.DISABLED)
        self.rs_result_label.config(
            text=f"🔄 正在加载模型并处理 {len(filepaths)} 个影像文件…", foreground="blue")
        self.root.update()

        import threading
        self.ftw_worker_thread = threading.Thread(
            target=self._run_ftw_inference, args=(params,), daemon=True)
        self.ftw_worker_thread.start()

    def _run_ftw_inference(self, params):
        """后台线程: 加载模型 + 并行处理多个 TIFF 文件"""
        total_area_km2 = 0.0
        total_pixels = 0
        per_file_info = []
        error_msg = None
        pixel_area_sqm = 0.0

        try:
            import torch

            if self.ftw_model is None or getattr(self, '_last_weights', '') != params['weights']:
                self.root.after(0, lambda: self.rs_result_label.config(
                    text="🔄 正在加载 FTW 模型权重…", foreground="blue"))
                device = params['device'] if torch.cuda.is_available() or params['device'] == 'cpu' else 'cpu'
                model, num_classes = create_ftw_model(
                    num_classes=3,
                    encoder_name='efficientnet-b3',
                    in_channels=8,
                    pretrained_weights_path=params['weights'],
                    device=device,
                )
                self.ftw_model = (model, num_classes, device)
                self._last_weights = params['weights']
            else:
                model, num_classes, device = self.ftw_model

            mask_geom = None
            if params['mask_path'] and os.path.exists(params['mask_path']):
                self.root.after(0, lambda: self.rs_result_label.config(
                    text="🔄 正在加载 GeoJSON 掩膜…", foreground="blue"))
                mask_geom, _ = load_geojson_mask(params['mask_path'])

            n_total = len(params['filepaths'])
            n_workers = min(params.get('workers', 2), n_total)
            completed = [0]

            def process_one(fpath, idx):
                fname = os.path.basename(fpath)
                result = calculate_cropland_area(
                    image_path=fpath,
                    model=model,
                    device=device,
                    window_size=params['window_size'],
                    overlap=params['overlap'],
                    num_classes=num_classes,
                    band_indices=FTW_BAND_INDICES,
                    mask_geometry=mask_geom,
                    ndvi_threshold=params['ndvi_threshold'],
                )
                return idx, fname, result

            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = {
                    executor.submit(process_one, fp, i): i
                    for i, fp in enumerate(params['filepaths'])
                }
                for future in as_completed(futures):
                    idx, fname, result = future.result()
                    per_file_info.append((idx, fname, result))
                    completed[0] += 1
                    self.root.after(0, lambda c=completed[0], t=n_total, fn=fname:
                        self.rs_result_label.config(
                            text=f"🔄 [{c}/{t}] {fn} 完成…", foreground="blue"))

            per_file_info.sort(key=lambda x: x[0])
            for _, fname, result in per_file_info:
                total_area_km2 += result['area_hectares'] / 100.0
                total_pixels += result['field_pixels']
                pixel_area_sqm = result['pixel_area_sqm']

        except ImportError as e:
            error_msg = f"缺少依赖库: {e}\n请确认已安装: torch, rasterio, segmentation_models_pytorch"
        except Exception as e:
            import traceback
            traceback.print_exc()
            error_msg = f"推理失败: {e}"
        finally:
            self.root.after(0, lambda: self._on_ftw_complete(
                total_area_km2, total_pixels, pixel_area_sqm,
                [(n, r) for _, n, r in per_file_info], error_msg))

    def _on_ftw_complete(self, area_km2, pixels, pixel_sqm, per_file, error):
        """FTW 推理完成后的 UI 回调 (主线程)"""
        self.btn_upload_rs.config(state=tk.NORMAL)

        if error:
            self.rs_result_label.config(text=f"❌ {error}", foreground="red")
            messagebox.showerror("FTW 推理失败", error)
            return

        self.rs_extracted_area = round(area_km2, 4)

        detail_lines = "\n".join(
            f"  • {name}: {r['area_hectares']:.2f} ha ({r['field_pixels']:,} px)"
            for name, r in per_file
        )
        summary = (
            f"✅ FTW 提取完成！耕地面积: {area_km2:,.2f} km²\n"
            f"耕地总像素: {pixels:,}  |  单像素面积: {pixel_sqm:.2f} m²\n"
            f"{detail_lines}"
        )
        self.rs_result_label.config(text=summary, foreground="green")
        self.calc_exact(0)

    def fetch_gee_cropland_area(self, r_idx):
        """从 Google Earth Engine 联网获取哈特隆州耕地面积"""
        try:
            self.gee_result_label.config(text="🔄 正在从 GEE 获取数据，请稍候...", foreground="blue")
            self.root.update()

            gee_project = self.gee_project_entry.get().strip()
            if not gee_project:
                messagebox.showwarning("配置错误", "请输入 GEE 项目 ID")
                self.gee_result_label.config(text="请输入 GEE 项目 ID", foreground="red")
                return

            area_kilo = get_cropland_area_km2(gee_project=gee_project)
            self.rs_extracted_area = area_kilo

            self.gee_result_label.config(
                text=f"✅ GEE 数据获取成功！耕地面积: {area_kilo:,.2f} 平方公里",
                foreground="green"
            )

            self.calc_exact(r_idx)

            messagebox.showinfo("GEE 数据获取成功",
                                f"已成功从 Google Earth Engine 获取哈特隆州耕地面积数据！\n"
                                f"耕地面积: {area_kilo:,.2f} 平方公里")

        except Exception as e:
            self.gee_result_label.config(text=f"❌ 获取失败: {str(e)[:50]}...", foreground="red")
            messagebox.showerror("GEE 数据获取失败",
                                 f"从 Google Earth Engine 获取数据时出错:\n{str(e)}")

    # ========================== 气象ET0计算 ==========================
    def open_meteo_config(self, r_idx):
        win = tk.Toplevel(self.root)
        win.title("配置气象参数")
        win.geometry("450x480")
        win.grab_set()

        params = self.meteo_params[r_idx]
        entries = {}

        labels_units = [
            ('Rn', "太阳净辐射 (Rn) [mm/d]:", params['Rn']),
            ('G', "土壤热通量 (G) [MJ/m²]:", params['G']),
            ('T', "地表日平均气温 (T) [°C]:", params['T']),
            ('u2', "地表2m处风速 (u2) [m/s]:", params['u2']),
            ('es', "饱和水汽压 (es) [hPa]:", params['es']),
            ('ea', "实际水汽压 (ea) [hPa]:", params['ea']),
            ('delta', "水汽压变率 (Δ):", params['delta']),
            ('gamma', "湿度计常数 (γ) [hPa/°C]:", params['gamma'])
        ]

        for i, (key, label_text, val) in enumerate(labels_units):
            ttk.Label(win, text=label_text).grid(row=i, column=0, padx=15, pady=8, sticky=tk.W)
            ent = ttk.Entry(win, width=12)
            ent.insert(0, str(val))
            ent.grid(row=i, column=1, padx=10, pady=8)
            entries[key] = ent

        def fetch_weather_data():
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

            def fetch_backup_api():
                url = "https://uapis.cn/api/v1/misc/weather?city=Bokhtar&adcode=&extended=false&forecast=true&hourly=true&minutely=false&indices=false&lang=zh"
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                params = {"province": "Khatlon", "city": "Bokhtar"}
                response = requests.get(url, headers=headers, params=params, timeout=15, verify=False)
                response.raise_for_status()
                data = response.json()

                today = data['forecast'][0]
                t_max = today['temp_max']
                t_min = today['temp_min']
                T_mean = (t_max + t_min) / 2
                wind_kmh = today['wind_speed_day']
                u2_ms = wind_kmh * (1000 / 3600)
                rh_mean = today['humidity']
                uv_index = today.get('uv_index', 5)

                es_kPa = 0.6108 * math.exp((17.27 * T_mean) / (T_mean + 237.3))
                es_hPa = es_kPa * 10
                ea_hPa = es_hPa * (rh_mean / 100)
                delta_kPa = (4098 * es_kPa) / ((T_mean + 237.3) ** 2)
                delta_hPa = delta_kPa * 10
                Rn_est = uv_index * 1.5

                return {
                    'T': T_mean, 'u2': u2_ms, 'es': es_hPa, 'ea': ea_hPa,
                    'delta': delta_hPa, 'gamma': 0.61, 'Rn': Rn_est, 'G': 0.0,
                    'source': 'uapis.cn', 'city': data.get('city', '博赫塔尔')
                }

            try:
                lat, lon = 37.8333, 69.0000
                year = int(self.year_var.get())
                month = int(self.month_var.get())

                start_date = datetime.date(year, month, 1)
                if month == 12:
                    end_date = datetime.date(year, month, 31)
                else:
                    end_date = datetime.date(year, month + 1, 1) - datetime.timedelta(days=1)

                base_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
                params_api = {
                    "parameters": "ALLSKY_SFC_SW_DWN,ALLSKY_SFC_LW_DWN,ALLSKY_SFC_SW_UP,ALLSKY_SFC_LW_UP,T2M,WS2M,RH2M,PS",
                    "community": "AG",
                    "longitude": lon,
                    "latitude": lat,
                    "start": start_date.strftime("%Y%m%d"),
                    "end": end_date.strftime("%Y%m%d"),
                    "format": "JSON"
                }
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }

                response = requests.get(base_url, headers=headers, params=params_api, timeout=30, verify=False)
                response.raise_for_status()
                data = response.json()

                params_data = data.get("properties", {}).get("parameter", {})
                if not params_data:
                    params_data = data.get("parameters", {})

                if not params_data or len(params_data) == 0:
                    raise ValueError("NASA POWER 返回数据为空")

                def monthly_avg(param_name):
                    values = params_data.get(param_name, {})
                    numeric_values = []
                    if isinstance(values, dict):
                        for v in values.values():
                            if isinstance(v, (int, float)) and v != -999:
                                numeric_values.append(v)
                    if not numeric_values:
                        raise ValueError(f"参数 {param_name} 无有效数据")
                    return sum(numeric_values) / len(numeric_values)

                sw_down = monthly_avg("ALLSKY_SFC_SW_DWN")
                sw_up = monthly_avg("ALLSKY_SFC_SW_UP")
                lw_down = monthly_avg("ALLSKY_SFC_LW_DWN")
                lw_up = monthly_avg("ALLSKY_SFC_LW_UP")
                T_mean = monthly_avg("T2M")
                u2_ms = monthly_avg("WS2M")
                RH = monthly_avg("RH2M")
                P_kpa = monthly_avg("PS")

                Rn_MJ = sw_down - sw_up + lw_down - lw_up
                if Rn_MJ < 0:
                    Rn_MJ = 0.0
                Rn_mm = Rn_MJ / 2.45

                G_MJ = 0.0
                es_kPa = 0.6108 * math.exp((17.27 * T_mean) / (T_mean + 237.3))
                es_hPa = es_kPa * 10.0
                ea_hPa = es_hPa * (RH / 100.0)
                delta_kPa = (4098 * es_kPa) / ((T_mean + 237.3) ** 2)
                delta_hPa = delta_kPa * 10.0
                P_hPa = P_kpa * 10.0
                gamma_hPa_per_C = 0.665e-3 * P_hPa

                entries['T'].delete(0, tk.END); entries['T'].insert(0, f"{T_mean:.2f}")
                entries['u2'].delete(0, tk.END); entries['u2'].insert(0, f"{u2_ms:.2f}")
                entries['es'].delete(0, tk.END); entries['es'].insert(0, f"{es_hPa:.2f}")
                entries['ea'].delete(0, tk.END); entries['ea'].insert(0, f"{ea_hPa:.2f}")
                entries['delta'].delete(0, tk.END); entries['delta'].insert(0, f"{delta_hPa:.2f}")
                entries['gamma'].delete(0, tk.END); entries['gamma'].insert(0, f"{gamma_hPa_per_C:.2f}")
                entries['Rn'].delete(0, tk.END); entries['Rn'].insert(0, f"{Rn_mm:.2f}")
                entries['G'].delete(0, tk.END); entries['G'].insert(0, f"{G_MJ:.2f}")

                messagebox.showinfo("联网成功",
                                    f"已成功获取NASA POWER数据\n"
                                    f"位置: 哈特隆州中心 ({lat}, {lon})\n"
                                    f"时间: {start_date.year}年{start_date.month}月")

            except Exception as nasa_error:
                try:
                    result = fetch_backup_api()

                    entries['T'].delete(0, tk.END); entries['T'].insert(0, f"{result['T']:.2f}")
                    entries['u2'].delete(0, tk.END); entries['u2'].insert(0, f"{result['u2']:.2f}")
                    entries['es'].delete(0, tk.END); entries['es'].insert(0, f"{result['es']:.2f}")
                    entries['ea'].delete(0, tk.END); entries['ea'].insert(0, f"{result['ea']:.2f}")
                    entries['delta'].delete(0, tk.END); entries['delta'].insert(0, f"{result['delta']:.2f}")
                    entries['gamma'].delete(0, tk.END); entries['gamma'].insert(0, f"{result['gamma']:.2f}")
                    entries['Rn'].delete(0, tk.END); entries['Rn'].insert(0, f"{result['Rn']:.2f}")
                    entries['G'].delete(0, tk.END); entries['G'].insert(0, f"{result['G']:.2f}")

                    messagebox.showinfo("联网成功（备用API）",
                                        f"NASA POWER数据获取失败，已自动切换到备用API\n"
                                        f"已成功获取 {result['city']} 最新气象数据！")

                except Exception as backup_error:
                    messagebox.showerror("联网失败",
                                        f"NASA POWER API: {str(nasa_error)}\n\n"
                                        f"备用API: {str(backup_error)}\n\n"
                                        f"请检查网络或稍后重试。")

        def save_and_calc():
            try:
                for k in entries.keys():
                    self.meteo_params[r_idx][k] = float(entries[k].get())
                self.calculate_et0_from_params(r_idx)
                win.destroy()
            except ValueError:
                messagebox.showerror("错误", "参数输入无效。")

        btn_frame = ttk.Frame(win)
        btn_frame.grid(row=len(labels_units), column=0, columnspan=2, pady=15)

        ttk.Button(btn_frame, text="🌐 联网获取当地气象", command=fetch_weather_data).pack(side=tk.LEFT, padx=10)
        ttk.Button(btn_frame, text="✅ 保存并计算 ET0", command=save_and_calc).pack(side=tk.LEFT, padx=10)

    def calculate_et0_from_params(self, r_idx):
        p = self.meteo_params[r_idx]
        numerator = 0.408 * p['delta'] * (p['Rn'] - p['G']) + p['gamma'] * (900 / (p['T'] + 278)) * p['u2'] * (p['es'] - p['ea'])
        denominator = p['delta'] + p['gamma'] * (1 + 0.34 * p['u2'])
        daily_et0 = numerator / denominator if denominator != 0 else 0

        self.et0_value.set(f"{daily_et0:.2f}")
        self.calc_exact(r_idx)

    def add_crop_row(self, r_idx):
        container = self.vars[r_idx]["crop_container"]
        row_f = ttk.Frame(container)
        row_f.pack(fill=tk.X, pady=2)

        c_type = ttk.Combobox(row_f, values=list(self.fao_kc.keys()), width=8, state="readonly")
        c_type.set("细绒棉")
        c_type.pack(side=tk.LEFT, padx=5)

        c_stage = ttk.Combobox(row_f, values=self.stages, width=8, state="readonly")
        c_stage.set("中期")
        c_stage.pack(side=tk.LEFT, padx=5)

        c_area = ttk.Entry(row_f, width=10)
        c_area.insert(0, "50")
        c_area.pack(side=tk.LEFT, padx=5)

        c_yield = ttk.Entry(row_f, width=10)
        c_yield.insert(0, "300")
        c_yield.pack(side=tk.LEFT, padx=5)

        c_price = ttk.Entry(row_f, width=10)
        c_price.insert(0, "7.5")
        c_price.pack(side=tk.LEFT, padx=5)

        crop_data = {
            "frame": row_f,
            "type": c_type,
            "stage": c_stage,
            "area": c_area,
            "yield": c_yield,
            "price": c_price
        }
        self.crop_vars[r_idx].append(crop_data)

        del_btn = ttk.Button(row_f, text="❌", width=3, style="Small.TButton",
                             command=lambda: self.delete_crop_row(r_idx, crop_data))
        del_btn.pack(side=tk.LEFT, padx=5)
        self.calc_exact(r_idx)

    def delete_crop_row(self, r_idx, crop_data):
        crop_data["frame"].destroy()
        if crop_data in self.crop_vars[r_idx]:
            self.crop_vars[r_idx].remove(crop_data)
        self.calc_exact(r_idx)

    def get_default_crop_by_month(self):
        try:
            month = int(self.month_var.get())
        except:
            month = 6

        if month in [1, 2, 3]:
            return "冬小麦", "发育期", 417300.0, 2.5
        elif month in [4, 5]:
            return "细绒棉", "初期", 350000.0, 7.5
        elif month in [6, 7, 8]:
            return "细绒棉", "中期", 35000000.0, 7.5
        elif month in [9, 10]:
            return "细绒棉", "后期", 3500000.0, 7.5
        else:
            return "冬小麦", "初期", 417300.0, 2.5

    def calc_exact(self, r_idx):
        try:
            month = int(self.month_var.get())
            days_in_month = calendar.monthrange(2026, month)[1]

            v = self.vars[r_idx]
            pop = float(v["pop"].get()) * 10000
            urban_rate = float(v["urban"].get()) / 100
            gdp = float(v["gdp"].get())
            reuse_rate = float(v["reuse"].get()) / 100
            eff = float(v["eff"].get())
            eco = float(v["eco"].get())
            et0_daily = float(self.et0_value.get())

            pop_urban = pop * urban_rate
            pop_rural = pop * (1 - urban_rate)
            try:
                urban_q = float(v["urban_quota"].get())
                rural_q = float(v["rural_quota"].get())
            except (KeyError, ValueError):
                urban_q, rural_q = 145, 80
            live_m3 = (pop_urban * urban_q / 1000 + pop_rural * rural_q / 1000) * days_in_month
            live = live_m3 / 1_000_000
            eco = 0.1 * live + float(v["eco"].get())

            agr = 0
            if self.agr_calc_mode.get() == "rs":
                if self.rs_extracted_area > 0:
                    def_type, def_stage, _, _ = self.get_default_crop_by_month()
                    kc = self.fao_kc[def_type][def_stage]
                    c_area = self.rs_extracted_area * 1_000_000 * 0.85
                    etc_monthly = kc * et0_daily * days_in_month
                    water_m3 = etc_monthly * 0.001 * c_area * 0.05
                    agr = (water_m3 / 1_000_000) / eff if eff > 0 else 0
            else:
                for c_info in self.crop_vars[r_idx]:
                    area_str = c_info["area"].get()
                    if not area_str.strip():
                        continue
                    try:
                        c_type = c_info["type"].get()
                        c_stage = c_info["stage"].get()
                        c_area = float(area_str) * 1_000_000
                        kc = self.fao_kc[c_type][c_stage]

                        etc_monthly = kc * et0_daily * days_in_month
                        water_m3 = etc_monthly * 0.001 * c_area
                        agr += (water_m3 / 1_000_000) / eff if eff > 0 else 0
                    except ValueError:
                        pass

            INDUSTRIAL_WATER_QUOTA = 140
            annual_industrial_water = gdp * 10000 * INDUSTRIAL_WATER_QUOTA * (1 - reuse_rate)

            season_factors = [0.85, 0.80, 0.90, 0.95, 1.05, 1.10, 1.15, 1.15, 1.05, 0.95, 0.85, 0.80]
            season_factor = season_factors[month - 1]

            ind = annual_industrial_water * season_factor / 12 / 1000000
            self.demand_entries[(r_idx, 0)].delete(0, tk.END)
            self.demand_entries[(r_idx, 0)].insert(0, f"{live:.2f}")
            self.demand_entries[(r_idx, 1)].delete(0, tk.END)
            self.demand_entries[(r_idx, 1)].insert(0, f"{eco:.2f}")
            self.demand_entries[(r_idx, 2)].delete(0, tk.END)
            self.demand_entries[(r_idx, 2)].insert(0, f"{agr:.2f}")
            self.demand_entries[(r_idx, 3)].delete(0, tk.END)
            self.demand_entries[(r_idx, 3)].insert(0, f"{ind:.2f}")

            try:
                year = int(self.year_var.get())
                month_val = int(self.month_var.get())
                result = predict_downstream_total(year)
                downstream_monthly = result['downstream_monthly'][month_val - 1] * 1000
                self.demand_entries[(r_idx, 4)].delete(0, tk.END)
                self.demand_entries[(r_idx, 4)].insert(0, f"{downstream_monthly:.2f}")
            except Exception as e:
                print(f"下游国家预测数据获取失败: {e}")

        except Exception:
            pass

    def estimate_economic_params(self):
        p_max, q_max, elec_price = float(self.hydro_pmax.get()), float(self.hydro_qmax.get()), float(self.hydro_price.get())
        a_hydro = ((p_max * 1000) / (q_max * 3600)) * elec_price

        total_revenue_yuan = 0.0

        if self.agr_calc_mode.get() == "rs":
            if self.rs_extracted_area > 0:
                _, _, def_yield, def_price = self.get_default_crop_by_month()
                area_mu = self.rs_extracted_area
                total_revenue_yuan = area_mu * def_yield * def_price
        else:
            for c_info in self.crop_vars[0]:
                if not c_info["area"].get().strip():
                    continue
                try:
                    area_mu = float(c_info["area"].get())
                    c_yield = float(c_info["yield"].get())
                    c_price = float(c_info["price"].get())
                    total_revenue_yuan += area_mu * c_yield * c_price
                except ValueError:
                    pass

        agr_water_demand_m3 = float(self.demand_entries[(0, 2)].get()) * 1_000_000
        alpha = 0.5
        if agr_water_demand_m3 > 0:
            a_agr = (total_revenue_yuan / agr_water_demand_m3) * alpha
        else:
            a_agr = 0.8

        a_dom, a_eco, a_ind = 1.1, 1.0, 9.0
        a_down = 1e-9
        a_surface = [a_dom + a_hydro, a_eco + a_hydro, a_agr + a_hydro, a_ind + a_hydro, a_down + a_hydro]
        b_surface = [0.005, 0.105, 0.005, 1.505, 0.0]
        a_ground = [a_dom, a_eco, a_agr, a_ind, a_down]
        b_ground = [a_dom + 0.4, 0.1, a_agr + 0.3, a_ind + 0.5, 0.0]
        return np.array([a_surface, a_ground]), np.array([b_surface, b_ground]), a_hydro, a_agr

    def _build_date_labels(self, start_year, start_month, end_year, end_month, time_scale):
        import pandas as pd
        labels = []
        if time_scale == "yearly":
            for y in range(start_year, end_year + 1):
                labels.append(str(y))
        elif time_scale == "monthly":
            for y in range(start_year, end_year + 1):
                m_start = start_month if y == start_year else 1
                m_end = end_month if y == end_year else 12
                for m in range(m_start, m_end + 1):
                    labels.append(f"{y}-{m:02d}")
        else:
            start = pd.Timestamp(year=start_year, month=start_month, day=1)
            end = pd.Timestamp(year=end_year, month=end_month, day=1)
            if end_month == 12:
                end = pd.Timestamp(year=end_year, month=12, day=31)
            else:
                end = pd.Timestamp(year=end_year, month=end_month + 1, day=1) - pd.Timedelta(days=1)
            d = start
            while d <= end:
                labels.append(d.strftime("%m-%d"))
                d += pd.Timedelta(days=1)
        return labels

    def run_nsga2_optimization(self):
        self.calc_exact(0)

        try:
            start_year = int(self.start_year_var.get())
            start_month = int(self.start_month_var.get())
            end_year = int(self.end_year_var.get())
            end_month = int(self.end_month_var.get())
            time_scale = self.time_scale_var.get()

            if time_scale == "yearly":
                n_periods = end_year - start_year + 1
                days_per_period = 365.25
            elif time_scale == "monthly":
                n_periods = (end_year - start_year) * 12 + (end_month - start_month + 1)
                days_per_period = 365.25 / 12
            else:
                import pandas as pd
                start = pd.Timestamp(year=start_year, month=start_month, day=1)
                if end_month == 12:
                    end = pd.Timestamp(year=end_year, month=12, day=31)
                else:
                    end = pd.Timestamp(year=end_year, month=end_month + 1, day=1) - pd.Timedelta(days=1)
                n_periods = (end - start).days + 1
                days_per_period = 1.0

            date_labels = self._build_date_labels(start_year, start_month, end_year, end_month, time_scale)

            base_w = float(self.w_surface.get())
            seconds_per_period = days_per_period * 24 * 3600
            base_w_per_period = base_w / max(n_periods, 1)

            if hasattr(self, 'current_monthly_inflow') and self.current_monthly_inflow is not None:
                monthly_inflow = np.asarray(self.current_monthly_inflow)
            else:
                monthly_inflow = np.full(12, 300.0)

            W_supply = np.zeros((n_periods, 2))
            for t in range(n_periods):
                if time_scale == "yearly":
                    month_idx = 0
                    inflow_cms = np.mean(monthly_inflow)
                elif time_scale == "monthly":
                    month_idx = (start_month - 1 + t) % 12
                    inflow_cms = monthly_inflow[month_idx]
                else:
                    cum_days = t
                    m = start_month - 1
                    while cum_days >= calendar.monthrange(start_year + (m // 12), (m % 12) + 1)[1]:
                        cum_days -= calendar.monthrange(start_year + (m // 12), (m % 12) + 1)[1]
                        m += 1
                    month_idx = m % 12
                    inflow_cms = monthly_inflow[month_idx]

                period_inflow = inflow_cms * seconds_per_period / 1_000_000
                W_supply[t, 0] = base_w_per_period + period_inflow
                W_supply[t, 1] = 0

            demand_scale = days_per_period / (365.25 / 12)
            D_demand = np.zeros((n_periods, 5))
            for s in range(5):
                monthly_val = float(self.demand_entries[(0, s)].get())
                D_demand[:, s] = monthly_val * demand_scale

            loss_rates = np.array([float(self.loss_entries[0].get()) / 100])

            a_matrix, b_matrix, a_hydro, a_agr = self.estimate_economic_params()
            T_weights = np.array([float(ent.get()) for ent in self.t_entries])

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
                ok = messagebox.askokcancel(
                    "大规模优化提示",
                    f"当前配置将产生 {n_periods} 个时段、{n_periods * 10} 个优化变量。\n"
                    f"NSGA-II 将使用 pop={pop_size}, gen={n_gen} 进行优化，\n"
                    f"可能需要较长时间。\n\n"
                    f"建议: 对于日粒度超过 1.5 个月的区间，考虑使用月粒度。\n\n"
                    f"是否继续?")
                if not ok:
                    return

            res = run_nsga2_opt(problem_params, pop_size=pop_size, n_gen=n_gen)

            if res is None or res.F is None:
                pop_size2 = pop_size * 2
                n_gen2 = n_gen * 2
                ok2 = messagebox.askokcancel(
                    "优化未收敛",
                    f"首次优化未能找到可行解 (pop={pop_size}, gen={n_gen})。\n"
                    f"是否用更大种群重试 (pop={pop_size2}, gen={n_gen2})?\n\n"
                    f"若仍失败，建议改用月粒度或年度粒度。")
                if not ok2:
                    return
                res = run_nsga2_opt(problem_params, pop_size=pop_size2, n_gen=n_gen2)

            if res is None or res.F is None:
                messagebox.showerror(
                    "优化失败",
                    "NSGA-II 无法找到可行解。\n\n"
                    "可能原因: 时间粒度太细导致约束过多。\n"
                    "建议: 改用月粒度或年度粒度重试。")
                return

            pref_weights = np.array([float(self.w_econ.get()), float(self.w_short.get()), float(self.w_gini.get())])
            F, F_min_norm, F_max_norm = res.F, res.F.min(axis=0), res.F.max(axis=0)
            F_range = np.where(F_max_norm - F_min_norm == 0, 1e-9, F_max_norm - F_min_norm)
            best_idx = np.argmin(np.linalg.norm(((F - F_min_norm) / F_range) * pref_weights, axis=1))

            best_X = res.X[best_idx].reshape((n_periods, 2, 1, 5))
            X_agg = best_X.sum(axis=0)

            self.show_nsga2_results(
                time_scale, -res.F[best_idx, 0], res.F[best_idx, 1], res.F[best_idx, 2],
                X_agg, D_demand.sum(axis=0).reshape(1, -1),
                loss_rates, W_supply.sum(axis=0), a_hydro,
                best_X if n_periods > 1 else None,
                date_labels, start_year, start_month)

        except Exception as e:
            messagebox.showerror("运行错误", str(e))

    def show_nsga2_results(self, time_label, profit, shortage, gini, X_opt, D, loss, W, a_hydro,
                           time_series_X=None, date_labels=None, start_year=None, start_month=None):
        win = tk.Toplevel(self.root)
        win.title(f"NSGA-II 优化配置分析报告 ({time_label})")
        win.geometry("950x700+100+80")

        nb = ttk.Notebook(win)
        nb.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        gini_diag = (
            "(完全公平)" if gini < 0.1 else
            "(满意度较平均)" if gini < 0.2 else
            "(分配偏向高产值部门)" if gini < 0.3 else
            "(偏科严重，存在明显受损部门)"
        )
        n_periods = len(date_labels) if date_labels else 1
        report_lines = [
            f"🎯 哈特隆州水资源配置方案 ({time_label}, {n_periods} 期)",
            "=" * 85,
            f"💰 系统总综合经济效益参考值 : {profit:,.2f} 万元",
            f"📉 系统总缺水量       : {shortage:,.2f} 百万m³",
            f"⚖️ 部门公平性 Gini   : {gini:.4f} {gini_diag}",
            "=" * 85,
            f"\n📍 地区：哈特隆州 (管网传输损耗率: {loss[0] * 100:.1f}%)",
            f"{'部门':<10} | {'需水量':<10} | {'水库放水量':<10} | {'实收水量':<10} | {'满足率'}",
            "-" * 75,
        ]
        received_data = []
        for j, sec in enumerate(self.sectors):
            demand = D[0, j]
            surf_out = X_opt[0, 0, j]
            received = surf_out * (1 - loss[0]) + X_opt[1, 0, j]
            received_data.append(received)
            ratio = (received / demand * 100) if demand > 0 else 100
            report_lines.append(f"{sec:<10} | {demand:<13.2f} | {surf_out:<15.2f} | {received:<16.2f} | {ratio:.1f}%")
        total_surf = X_opt[0, 0, :].sum()
        report_lines.extend([
            "\n" + "=" * 85,
            f"🌊 水库放水总量: {total_surf:.2f} / {W[0]:.2f} 百万m³",
            f"🔌 水力发电贡献参考值: 约 {total_surf * a_hydro:,.2f} 万元",
        ])
        report_text = "\n".join(report_lines)

        tab_txt = ttk.Frame(nb)
        nb.add(tab_txt, text="📄 文本报告")
        txt = tk.Text(tab_txt, font=("Consolas", 11), bg="#1E1E1E", fg="#D4D4D4", padx=15, pady=15)
        txt.pack(fill=tk.BOTH, expand=True)
        txt.insert(tk.END, report_text)
        txt.config(state=tk.DISABLED)

        if time_series_X is not None and date_labels is not None and len(date_labels) > 1:
            tab_ts = ttk.Frame(nb)
            nb.add(tab_ts, text="📈 时序分配")
            fig_ts, ax_ts = plt.subplots(figsize=(9, 4.5))
            colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336']
            for j in range(time_series_X.shape[3]):
                series = time_series_X[:, 0, 0, j] * (1 - loss[0]) + time_series_X[:, 1, 0, j]
                ax_ts.plot(range(len(date_labels)), series, marker='.', label=self.sectors[j],
                          color=colors[j], linewidth=1.5, markersize=3)
            ax_ts.set_ylabel('实收水量 (百万m³)')
            ax_ts.set_title(f'各部门逐{time_label}分配水量')
            step = max(1, len(date_labels) // 12)
            ax_ts.set_xticks(range(0, len(date_labels), step))
            ax_ts.set_xticklabels([date_labels[i] for i in range(0, len(date_labels), step)],
                                  rotation=45, ha='right', fontsize=8)
            ax_ts.legend(fontsize=8)
            ax_ts.grid(True, alpha=0.3)
            fig_ts.tight_layout()
            canvas_ts = FigureCanvasTkAgg(fig_ts, master=tab_ts)
            canvas_ts.draw()
            canvas_ts.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        else:
            tab_bar = ttk.Frame(nb)
            nb.add(tab_bar, text="📊 柱状图")
            fig1, ax1 = plt.subplots(figsize=(7, 4))
            x_pos = np.arange(len(self.sectors))
            w = 0.35
            ax1.bar(x_pos - w / 2, D[0], w, label='需水量', color='#ff9999')
            ax1.bar(x_pos + w / 2, received_data, w, label='实际分配', color='#66b3ff')
            ax1.set_ylabel('水量 (百万m³)')
            ax1.set_title(f'各部门用水需求与实际分配 ({time_label})')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(self.sectors)
            ax1.legend()
            canvas1 = FigureCanvasTkAgg(fig1, master=tab_bar)
            canvas1.draw()
            canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        tab_pie = ttk.Frame(nb)
        nb.add(tab_pie, text="🥧 饼图")
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        colors_pie = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#cc99ff']
        ax2.pie(received_data, labels=self.sectors, autopct='%1.1f%%', colors=colors_pie, startangle=90)
        ax2.set_title('各部门分配水量占比 (合计)')
        canvas2 = FigureCanvasTkAgg(fig2, master=tab_pie)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        btn_frame = ttk.Frame(win)
        btn_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Button(btn_frame, text="💾 导出文本", command=lambda: self._save_text(report_text)).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="📥 导出 CSV",
                   command=lambda: self._export_csv(received_data, D)).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="📗 导出 Excel",
                   command=lambda: self._export_excel(received_data, D, report_lines, time_series_X, date_labels)).pack(side=tk.LEFT, padx=5)
        plt.close('all')

    def _save_text(self, text):
        f = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text", "*.txt")])
        if f:
            with open(f, "w", encoding="utf-8") as fp:
                fp.write(text)
            messagebox.showinfo("成功", "文本已保存")

    def _export_csv(self, received, D):
        f = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if f:
            rows = [["部门", "需水量(百万m³)", "实收水量(百万m³)", "满足率(%)"]]
            for j, sec in enumerate(self.sectors):
                ratio = (received[j] / D[0, j] * 100) if D[0, j] > 0 else 100
                rows.append([sec, f"{D[0, j]:.2f}", f"{received[j]:.2f}", f"{ratio:.1f}"])
            import csv
            with open(f, "w", newline="", encoding="utf-8-sig") as fp:
                csv.writer(fp).writerows(rows)
            messagebox.showinfo("成功", "CSV 已保存")

    def _export_excel(self, received, D, lines, time_series_X=None, date_labels=None):
        f = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel", "*.xlsx")])
        if f:
            import pandas as pd
            data = {"部门": self.sectors,
                    "需水量(百万m³)": [f"{D[0, j]:.2f}" for j in range(len(self.sectors))],
                    "实收水量(百万m³)": [f"{received[j]:.2f}" for j in range(len(self.sectors))]}
            with pd.ExcelWriter(f) as writer:
                pd.DataFrame(data).to_excel(writer, sheet_name="分配结果", index=False)
                pd.DataFrame({"报告": lines}).to_excel(writer, sheet_name="文本报告", index=False)
                if time_series_X is not None and date_labels is not None:
                    ts_data = {"日期": date_labels}
                    for j, sec in enumerate(self.sectors):
                        ts_data[f"{sec}_实收(百万m³)"] = [
                            f"{time_series_X[t, 0, 0, j] * (1 - float(self.loss_entries[0].get()) / 100) + time_series_X[t, 1, 0, j]:.2f}"
                            for t in range(len(date_labels))]
                    pd.DataFrame(ts_data).to_excel(writer, sheet_name="时序分配", index=False)
            messagebox.showinfo("成功", "Excel 已保存")

    # ========================== 水文数据与预测 Tab ==========================
    def build_data_tab(self):
        main_frame = ttk.Frame(self.tab_data)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        ttk.Label(main_frame, text="📡 水文数据管理与径流预测", style="Header.TLabel").pack(anchor=tk.W, pady=(0, 10))

        water_info_frame = ttk.LabelFrame(main_frame, text="📊 实时水情信息", padding=10)
        water_info_frame.pack(fill=tk.X, pady=(0, 10))

        af = ttk.Frame(water_info_frame)
        af.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(af, text="🌊 年平均径流量:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=(0, 10))
        self.annual_avg_var = tk.StringVar(value="--")
        ttk.Label(af, textvariable=self.annual_avg_var, font=("Arial", 10, "bold"),
                  foreground="#2196F3").pack(side=tk.LEFT)
        ttk.Label(af, text="m³/s", font=("Arial", 10)).pack(side=tk.LEFT, padx=5)

        yf = ttk.Frame(water_info_frame)
        yf.pack(fill=tk.X)
        ttk.Label(yf, text="📅 水年份类型:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=(0, 10))
        self.water_year_type_var = tk.StringVar(value="--")
        self.water_year_label = ttk.Label(yf, textvariable=self.water_year_type_var,
                                          font=("Arial", 10, "bold"))
        self.water_year_label.pack(side=tk.LEFT)

        ds_frame = ttk.LabelFrame(main_frame, text="📁 气象/水文数据源", padding=8)
        ds_frame.pack(fill=tk.X, pady=(0, 10))

        df1 = ttk.Frame(ds_frame)
        df1.pack(fill=tk.X, pady=2)
        ttk.Label(df1, text="数据源:").pack(side=tk.LEFT, padx=5)
        self.nc_data_path = tk.StringVar()
        ttk.Entry(df1, textvariable=self.nc_data_path, width=45, state="readonly").pack(side=tk.LEFT, padx=5)
        ttk.Button(df1, text="选择文件", style="Small.TButton",
                   command=self.select_nc_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(df1, text="选择文件夹", style="Small.TButton",
                   command=self.select_nc_folder).pack(side=tk.LEFT, padx=2)
        ttk.Button(df1, text="清空", style="Small.TButton",
                   command=lambda: self.nc_data_path.set("")).pack(side=tk.LEFT, padx=2)

        df2 = ttk.Frame(ds_frame)
        df2.pack(fill=tk.X, pady=2)
        self.target_year_label = ttk.Label(df2, text="目标年份:", font=("Arial", 10))
        self.target_year_label.pack(side=tk.LEFT, padx=5)
        ttk.Button(df2, text="🔄 同步年份", style="Small.TButton",
                   command=self._sync_target_year_display).pack(side=tk.LEFT, padx=5)
        self._sync_target_year_display()

        self.v_init_var = tk.StringVar(value="84.0")
        ttk.Label(df2, text="初蓄(亿m³):").pack(side=tk.LEFT, padx=(15, 5))
        ttk.Entry(df2, textvariable=self.v_init_var, width=8).pack(side=tk.LEFT)

        ttk.Button(df2, text="🔍 预览径流预测", style="Small.TButton",
                   command=lambda: self.preview_water_info(self.nc_data_path.get())).pack(side=tk.LEFT, padx=15)
        self.preview_status = ttk.Label(df2, text="", foreground="gray")
        self.preview_status.pack(side=tk.LEFT, padx=5)

        train_frame = ttk.LabelFrame(main_frame, text="🤖 LSTM 径流预测模型训练", padding=8)
        train_frame.pack(fill=tk.X, pady=(0, 10))

        tf1 = ttk.Frame(train_frame)
        tf1.pack(fill=tk.X, pady=2)
        ttk.Label(tf1, text="训练数据:").pack(side=tk.LEFT, padx=5)
        self.train_data_path = tk.StringVar()
        ttk.Entry(tf1, textvariable=self.train_data_path, width=38).pack(side=tk.LEFT, padx=5)
        ttk.Button(tf1, text="浏览", style="Small.TButton",
                   command=lambda: self.train_data_path.set(
                       filedialog.askopenfilename(title="选择训练数据",
                           filetypes=[("CSV/Excel", "*.csv *.xlsx *.xls"), ("NetCDF", "*.nc"), ("All", "*.*")])
                       or self.train_data_path.get())).pack(side=tk.LEFT, padx=2)

        tf2 = ttk.Frame(train_frame)
        tf2.pack(fill=tk.X, pady=2)

        self.btn_train_lstm = ttk.Button(tf2, text="🚀 开始训练",
                                         command=self.run_lstm_training)
        self.btn_train_lstm.pack(side=tk.LEFT, padx=15)
        self.train_status_label = ttk.Label(tf2, text="等待训练...", foreground="gray")
        self.train_status_label.pack(side=tk.LEFT, padx=5)

    def run_lstm_training(self):
        """后台线程运行 LSTM 训练"""
        data_path = self.train_data_path.get().strip()
        if not data_path or not os.path.exists(data_path):
            messagebox.showwarning("数据错误", "请选择有效的训练数据文件")
            return
        self.btn_train_lstm.config(state=tk.DISABLED)
        self.train_status_label.config(text="🔄 训练中...", foreground="blue")

        import threading
        def _train():
            try:
                from .train import train_from_external_data

                hp = {
                    'num_epochs': 100,
                    'learning_rate': 0.001,
                }
                def progress(ep, total, tl, vl, rmse, mae):
                    self.root.after(0, lambda: self.train_status_label.config(
                        text=f"🔄 Epoch {ep}/{total} | Train Loss: {tl:.2f} | Val RMSE: {rmse:.2f}",
                        foreground="blue"))
                result = train_from_external_data(data_path, hyperparams=hp, progress_callback=progress)
                self.root.after(0, lambda: self.train_status_label.config(
                    text=f"✅ 训练完成 | Test RMSE: {result['test_rmse']:.2f} MAE: {result['test_mae']:.2f} R²: {result['test_r2']:.4f}",
                    foreground="green"))
            except Exception as e:
                self.root.after(0, lambda: self.train_status_label.config(
                    text=f"❌ 训练失败: {str(e)[:60]}", foreground="red"))
            finally:
                self.root.after(0, lambda: self.btn_train_lstm.config(state=tk.NORMAL))
        threading.Thread(target=_train, daemon=True).start()

    def classify_water_year(self, avg_discharge):
        THRESHOLD_WET, THRESHOLD_DRY = 730.0, 574.0
        if avg_discharge >= THRESHOLD_WET:
            return "丰水年"
        elif avg_discharge <= THRESHOLD_DRY:
            return "枯水年"
        return "平水年"

    def update_water_info(self, monthly_inflow):
        annual_avg = np.mean(monthly_inflow)
        self.annual_avg_var.set(f"{annual_avg:.2f}")
        water_type = self.classify_water_year(annual_avg)
        self.water_year_type_var.set(water_type)
        colors = {"丰水年": "#4CAF50", "枯水年": "#F44336", "平水年": "#FF9800"}
        self.water_year_label.configure(foreground=colors.get(water_type, "black"))
        return annual_avg, water_type

    def select_nc_file(self):
        fp = filedialog.askopenfilename(title="选择数据文件",
            filetypes=[("NetCDF", "*.nc"), ("CSV", "*.csv"), ("Excel", "*.xlsx *.xls"), ("All", "*.*")])
        if fp:
            self.nc_data_path.set(fp)

    def select_nc_folder(self):
        fp = filedialog.askdirectory(title="选择包含 .nc 文件的文件夹")
        if fp:
            self.nc_data_path.set(fp)

    def _sync_target_year_display(self):
        if not hasattr(self, 'target_year_label'):
            return
        if hasattr(self, 'start_year_var') and hasattr(self, 'end_year_var'):
            sy = self.start_year_var.get()
            sm = getattr(self, 'start_month_var', None)
            sm_val = f"-{sm.get()}" if sm else ""
            ey = self.end_year_var.get()
            em = getattr(self, 'end_month_var', None)
            em_val = f"-{em.get()}" if em else ""
            if sy == ey and sm_val == em_val:
                self.target_year_label.config(text=f"目标年份: {sy}{sm_val}")
            else:
                self.target_year_label.config(text=f"目标时段: {sy}{sm_val} → {ey}{em_val}")
        elif hasattr(self, 'start_year_var'):
            self.target_year_label.config(text=f"目标年份: {self.start_year_var.get()}")

    def preview_water_info(self, filepath):
        try:
            self.preview_status.config(text="🔄 预测中...", foreground="blue")
            target_year = int(self.year_var.get())

            def on_updated(monthly_inflow):
                self.current_monthly_inflow = monthly_inflow
                self.root.after(0, lambda: self.update_water_info(monthly_inflow))

            _ = NurekDamParameters(
                elec_price=float(self.hydro_price.get()),
                unit_water_margin=1.6,
                data_path=filepath,
                v_initial=float(self.v_init_var.get()),
                update_callback=on_updated,
                target_year=target_year,
            )
            self.preview_status.config(text="✅ 预测完成", foreground="green")
            self._sync_target_year_display()
        except Exception as e:
            self.preview_status.config(text=f"预测失败: {e}", foreground="red")
            print(e)


if __name__ == "__main__":
    root = tk.Tk()
    app = WaterAllocationApp(root)
    root.mainloop()
