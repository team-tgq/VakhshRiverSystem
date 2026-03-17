# plugins/routing_plugin/routing_widget.py
import os
import json
import traceback
import numpy as np

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel,
    QTextEdit, QTabWidget, QMessageBox, QHBoxLayout
)

from algorithms.routing.config import (
    FLOOD_FILE, RUNOFF_FILE, FLOOD_OUTPUT_DIR, RUNOFF_OUTPUT_DIR
)
from algorithms.routing.flood_routing import run_flood_routing
from algorithms.routing.runoff_routing import run_runoff_routing
from .charts import LossChart, PredictionChart


class RoutingWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        title = QLabel("洪水演进与汇流模拟模块")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title)

        self.tabs = QTabWidget()
        self.flood_tab = self.build_flood_tab()
        self.runoff_tab = self.build_runoff_tab()

        self.tabs.addTab(self.flood_tab, "Flood Routing")
        self.tabs.addTab(self.runoff_tab, "Runoff Routing")

        layout.addWidget(self.tabs)

    def build_flood_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.flood_info = QLabel(f"数据文件: {FLOOD_FILE}")
        self.flood_info.setWordWrap(True)

        self.flood_run_btn = QPushButton("运行 Flood Routing")
        self.flood_load_btn = QPushButton("加载已有 Flood 结果")
        self.flood_metric = QLabel("指标: 未运行")

        self.flood_loss_chart = LossChart()
        self.flood_pred_chart = PredictionChart()

        self.flood_log = QTextEdit()
        self.flood_log.setReadOnly(True)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self.flood_run_btn)
        btn_row.addWidget(self.flood_load_btn)

        layout.addWidget(self.flood_info)
        layout.addLayout(btn_row)
        layout.addWidget(self.flood_metric)

        chart_row = QHBoxLayout()
        chart_row.addWidget(self.flood_loss_chart)
        chart_row.addWidget(self.flood_pred_chart)
        layout.addLayout(chart_row)

        layout.addWidget(self.flood_log)

        self.flood_run_btn.clicked.connect(self.run_flood_module)
        self.flood_load_btn.clicked.connect(self.load_flood_results)
        return tab

    def build_runoff_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.runoff_info = QLabel(f"数据文件: {RUNOFF_FILE}")
        self.runoff_info.setWordWrap(True)

        self.runoff_run_btn = QPushButton("运行 Runoff Routing")
        self.runoff_load_btn = QPushButton("加载已有 Runoff 结果")
        self.runoff_metric = QLabel("指标: 未运行")

        self.runoff_loss_chart = LossChart()
        self.runoff_pred_chart = PredictionChart()

        self.runoff_log = QTextEdit()
        self.runoff_log.setReadOnly(True)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self.runoff_run_btn)
        btn_row.addWidget(self.runoff_load_btn)

        layout.addWidget(self.runoff_info)
        layout.addLayout(btn_row)
        layout.addWidget(self.runoff_metric)

        chart_row = QHBoxLayout()
        chart_row.addWidget(self.runoff_loss_chart)
        chart_row.addWidget(self.runoff_pred_chart)
        layout.addLayout(chart_row)

        layout.addWidget(self.runoff_log)

        self.runoff_run_btn.clicked.connect(self.run_runoff_module)
        self.runoff_load_btn.clicked.connect(self.load_runoff_results)
        return tab

    # =========================
    # Flood Routing
    # =========================
    def run_flood_module(self):
        try:
            if not os.path.exists(FLOOD_FILE):
                raise FileNotFoundError(
                    "请先将 flood.csv 放入 algorithms/routing/data/ 目录"
                )

            self.flood_log.append("开始运行 Flood Routing...")
            result = run_flood_routing(FLOOD_FILE)

            self.flood_metric.setText(
                f"NSE Downstream: {result['nse_downstream']:.4f} | "
                f"NSE Upstream: {result['nse_upstream']:.4f}"
            )

            self.flood_loss_chart.plot_losses(
                result["train_losses"],
                title="Flood Routing Train Loss"
            )

            self.flood_pred_chart.plot_prediction(
                result["actuals_real"][:, 0],
                result["preds_real"][:, 0],
                title="Downstream Water Level",
                true_label="Actual",
                pred_label="Pred"
            )

            self.save_flood_results(result)
            self.flood_log.append("Flood Routing 运行完成，并已保存结果。")

        except Exception as e:
            self.flood_log.append(str(e))
            self.flood_log.append(traceback.format_exc())
            QMessageBox.critical(self, "错误", str(e))

    def save_flood_results(self, result):
        os.makedirs(FLOOD_OUTPUT_DIR, exist_ok=True)

        metrics = {
            "nse_downstream": float(result["nse_downstream"]),
            "nse_upstream": float(result["nse_upstream"])
        }

        with open(os.path.join(FLOOD_OUTPUT_DIR, "flood_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        np.save(os.path.join(FLOOD_OUTPUT_DIR, "flood_train_losses.npy"), np.array(result["train_losses"]))
        np.save(os.path.join(FLOOD_OUTPUT_DIR, "flood_actuals.npy"), np.array(result["actuals_real"]))
        np.save(os.path.join(FLOOD_OUTPUT_DIR, "flood_preds.npy"), np.array(result["preds_real"]))

    def load_flood_results(self):
        try:
            metrics_path = os.path.join(FLOOD_OUTPUT_DIR, "flood_metrics.json")
            loss_path = os.path.join(FLOOD_OUTPUT_DIR, "flood_train_losses.npy")
            actuals_path = os.path.join(FLOOD_OUTPUT_DIR, "flood_actuals.npy")
            preds_path = os.path.join(FLOOD_OUTPUT_DIR, "flood_preds.npy")

            required_files = [metrics_path, loss_path, actuals_path, preds_path]
            for fp in required_files:
                if not os.path.exists(fp):
                    raise FileNotFoundError(f"未找到结果文件: {fp}")

            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)

            train_losses = np.load(loss_path, allow_pickle=True)
            actuals_real = np.load(actuals_path, allow_pickle=True)
            preds_real = np.load(preds_path, allow_pickle=True)

            self.flood_metric.setText(
                f"NSE Downstream: {metrics['nse_downstream']:.4f} | "
                f"NSE Upstream: {metrics['nse_upstream']:.4f}"
            )

            self.flood_loss_chart.plot_losses(
                train_losses.tolist(),
                title="Flood Routing Train Loss"
            )

            self.flood_pred_chart.plot_prediction(
                actuals_real[:, 0],
                preds_real[:, 0],
                title="Downstream Water Level",
                true_label="Actual",
                pred_label="Pred"
            )

            self.flood_log.append("已加载已有 Flood Routing 结果。")

        except Exception as e:
            self.flood_log.append(str(e))
            self.flood_log.append(traceback.format_exc())
            QMessageBox.critical(self, "错误", str(e))

    # =========================
    # Runoff Routing
    # =========================
    def run_runoff_module(self):
        try:
            if not os.path.exists(RUNOFF_FILE):
                raise FileNotFoundError(
                    "请先将 datashuiwen2005-2017.csv 放入 algorithms/routing/data/ 目录"
                )

            self.runoff_log.append("开始运行 Runoff Routing...")

            model_path = os.path.join(RUNOFF_OUTPUT_DIR, "best_model.pth")
            result = run_runoff_routing(
                file_path=RUNOFF_FILE,
                save_model_path=model_path
            )

            self.runoff_metric.setText(
                f"RMSE: {result['rmse']:.4f} | "
                f"NSE: {result['nse']:.4f} | "
                f"KGE: {result['kge']:.4f}"
            )

            self.runoff_loss_chart.plot_losses(
                result["train_losses"],
                result["val_losses"],
                title="Runoff Routing Loss Curve"
            )

            self.runoff_pred_chart.plot_prediction(
                result["y_true"].flatten(),
                result["y_pred"].flatten(),
                title="Runoff Prediction",
                true_label="Actual",
                pred_label="Pred"
            )

            self.save_runoff_results(result)
            self.runoff_log.append("Runoff Routing 运行完成，并已保存结果。")

        except Exception as e:
            self.runoff_log.append(str(e))
            self.runoff_log.append(traceback.format_exc())
            QMessageBox.critical(self, "错误", str(e))

    def save_runoff_results(self, result):
        os.makedirs(RUNOFF_OUTPUT_DIR, exist_ok=True)

        metrics = {
            "rmse": float(result["rmse"]),
            "nse": float(result["nse"]),
            "kge": float(result["kge"])
        }

        with open(os.path.join(RUNOFF_OUTPUT_DIR, "runoff_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        np.save(os.path.join(RUNOFF_OUTPUT_DIR, "runoff_train_losses.npy"), np.array(result["train_losses"]))
        np.save(os.path.join(RUNOFF_OUTPUT_DIR, "runoff_val_losses.npy"), np.array(result["val_losses"]))
        np.save(os.path.join(RUNOFF_OUTPUT_DIR, "runoff_y_true.npy"), np.array(result["y_true"]))
        np.save(os.path.join(RUNOFF_OUTPUT_DIR, "runoff_y_pred.npy"), np.array(result["y_pred"]))

    def load_runoff_results(self):
        try:
            metrics_path = os.path.join(RUNOFF_OUTPUT_DIR, "runoff_metrics.json")
            train_loss_path = os.path.join(RUNOFF_OUTPUT_DIR, "runoff_train_losses.npy")
            val_loss_path = os.path.join(RUNOFF_OUTPUT_DIR, "runoff_val_losses.npy")
            y_true_path = os.path.join(RUNOFF_OUTPUT_DIR, "runoff_y_true.npy")
            y_pred_path = os.path.join(RUNOFF_OUTPUT_DIR, "runoff_y_pred.npy")

            required_files = [metrics_path, train_loss_path, val_loss_path, y_true_path, y_pred_path]
            for fp in required_files:
                if not os.path.exists(fp):
                    raise FileNotFoundError(f"未找到结果文件: {fp}")

            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)

            train_losses = np.load(train_loss_path, allow_pickle=True)
            val_losses = np.load(val_loss_path, allow_pickle=True)
            y_true = np.load(y_true_path, allow_pickle=True)
            y_pred = np.load(y_pred_path, allow_pickle=True)

            self.runoff_metric.setText(
                f"RMSE: {metrics['rmse']:.4f} | "
                f"NSE: {metrics['nse']:.4f} | "
                f"KGE: {metrics['kge']:.4f}"
            )

            self.runoff_loss_chart.plot_losses(
                train_losses.tolist(),
                val_losses.tolist(),
                title="Runoff Routing Loss Curve"
            )

            self.runoff_pred_chart.plot_prediction(
                y_true.flatten(),
                y_pred.flatten(),
                title="Runoff Prediction",
                true_label="Actual",
                pred_label="Pred"
            )

            self.runoff_log.append("已加载已有 Runoff Routing 结果。")

        except Exception as e:
            self.runoff_log.append(str(e))
            self.runoff_log.append(traceback.format_exc())
            QMessageBox.critical(self, "错误", str(e))