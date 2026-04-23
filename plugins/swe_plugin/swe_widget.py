from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import rasterio
import rasterio.features
import shapefile
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import colormaps, colors
from rasterio.enums import Resampling
from rasterio.transform import from_bounds

from PyQt5.QtCore import QObject, QThread, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from algorithms.swe import swe_assessment


DISPLAY_LAYER_KEY = "swe_raster"
DISPLAY_LAYER_LABEL = "SWE"
DISPLAY_LONG_EDGE = 320
INPUT_DATA_TEXT = (
    "输入数据说明\n"
    "静态输入：高程、坡度、坡向、分区。\n"
    "动态输入：GFS 日尺度气温与降水、固态降水、前一日 SWE 状态、近几日滚动统计。\n"
    "观测约束：VIIRS 日雪盖分数，用于约束当天是否有雪。\n"
    "说明：界面仅展示 SWE 图层；融雪和 QA 仍在后台计算并供下游模块使用。"
)


class SWEWorker(QObject):
    finished = pyqtSignal()
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, target, kwargs: dict):
        super().__init__()
        self.target = target
        self.kwargs = kwargs

    def run(self) -> None:
        try:
            result = self.target(**self.kwargs)
        except Exception as exc:
            self.failed.emit(str(exc) or repr(exc))
        else:
            self.succeeded.emit(result)
        finally:
            self.finished.emit()


class SWEMapCanvas(FigureCanvas):
    def __init__(self, parent: QWidget | None = None):
        self.figure = Figure(tight_layout=True)
        super().__init__(self.figure)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()

    def _load_boundary_data(self, shapefile_path: str) -> dict[str, object] | None:
        if not os.path.exists(shapefile_path):
            return None
        reader = shapefile.Reader(shapefile_path)
        shapes = reader.shapes()
        if not shapes:
            return None

        segments: list[tuple[list[float], list[float]]] = []
        geometries: list[dict] = []
        for shape_record in shapes:
            geometries.append(shape_record.__geo_interface__)
            points = shape_record.points
            parts = list(shape_record.parts) + [len(points)]
            for start, end in zip(parts[:-1], parts[1:]):
                segment = points[start:end]
                if not segment:
                    continue
                xs = [point[0] for point in segment]
                ys = [point[1] for point in segment]
                segments.append((xs, ys))

        min_lon, min_lat, max_lon, max_lat = reader.bbox
        return {
            "segments": segments,
            "geometries": geometries,
            "bbox": (float(min_lon), float(min_lat), float(max_lon), float(max_lat)),
        }

    def _plot_boundary(self, ax, boundary_data: dict[str, object] | None) -> None:
        if not boundary_data:
            return
        for xs, ys in boundary_data["segments"]:
            ax.plot(xs, ys, color="#d62728", linewidth=1.3, zorder=5)

    def _display_shape(self, width: int, height: int) -> tuple[int, int]:
        longest = max(width, height)
        scale = DISPLAY_LONG_EDGE / float(longest)
        display_height = max(1, int(round(height * scale)))
        display_width = max(1, int(round(width * scale)))
        return display_height, display_width

    def plot_raster(self, raster_path: str, study_area_shp: str, layer_key: str) -> None:
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        boundary_data = self._load_boundary_data(study_area_shp)

        with rasterio.open(raster_path) as dataset:
            display_height, display_width = self._display_shape(dataset.width, dataset.height)
            resampled = dataset.read(
                1,
                masked=True,
                out_shape=(display_height, display_width),
                resampling=Resampling.bilinear,
            )
            array = np.asarray(resampled.filled(np.nan), dtype=np.float32)
            bounds = dataset.bounds
            transform = from_bounds(
                bounds.left,
                bounds.bottom,
                bounds.right,
                bounds.top,
                display_width,
                display_height,
            )
            extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)

        if boundary_data:
            inside_mask = rasterio.features.geometry_mask(
                boundary_data["geometries"],
                out_shape=array.shape,
                transform=transform,
                invert=True,
                all_touched=False,
            )
            array = np.where(inside_mask, array, np.nan).astype(np.float32)

        valid = array[~np.isnan(array)]
        if valid.size:
            vmin, vmax = np.percentile(valid, [5, 95])
            if np.isclose(vmin, vmax):
                vmin, vmax = float(np.nanmin(valid)), float(np.nanmax(valid) + 1.0)
        else:
            vmin, vmax = 0.0, 1.0

        cmap = colormaps["Blues"].copy()
        cmap.set_bad(alpha=0.0)
        image = ax.imshow(
            np.ma.masked_invalid(array),
            extent=extent,
            origin="upper",
            cmap=cmap,
            norm=colors.Normalize(vmin=vmin, vmax=vmax),
            interpolation="bilinear",
        )
        colorbar = self.figure.colorbar(image, ax=ax, fraction=0.035, pad=0.04)
        colorbar.set_label("mm")

        self._plot_boundary(ax, boundary_data)
        if boundary_data:
            min_lon, min_lat, max_lon, max_lat = boundary_data["bbox"]
            ax.set_xlim(min_lon, max_lon)
            ax.set_ylim(min_lat, max_lat)
        ax.set_title(f"{DISPLAY_LAYER_LABEL} - {Path(raster_path).name}")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.3)
        self.draw()


class SWEWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.result = None
        self._worker_thread: QThread | None = None
        self._worker: SWEWorker | None = None
        self._success_message = ""
        self._error_title = "任务失败"
        self.init_ui()
        self.load_existing_results(silent=True)

    def init_ui(self) -> None:
        main_layout = QHBoxLayout(self)

        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        title = QLabel("日更 SWE 估算模块")
        title.setStyleSheet("font-size:18px;font-weight:bold;")

        self.update_btn = QPushButton("更新最新 SWE")
        self.backfill_btn = QPushButton("回算最近")
        self.backfill_days = QSpinBox()
        self.backfill_days.setRange(1, 30)
        self.backfill_days.setValue(7)
        self.load_btn = QPushButton("加载已有结果")
        self.retrain_check = QCheckBox("重新训练模型")

        self.summary_label = QLabel("尚未加载 SWE 结果。")
        self.summary_label.setWordWrap(True)
        self.input_info_label = QLabel(INPUT_DATA_TEXT)
        self.input_info_label.setWordWrap(True)
        self.input_info_label.setStyleSheet(
            "padding:8px;border:1px solid #d9d9d9;background:#fafafa;line-height:1.4;"
        )

        self.list_widget = QListWidget()
        self.log = QTextEdit()
        self.log.setReadOnly(True)

        backfill_row = QHBoxLayout()
        backfill_row.addWidget(self.backfill_btn)
        backfill_row.addWidget(self.backfill_days)

        left_layout.addWidget(title)
        left_layout.addWidget(self.update_btn)
        left_layout.addLayout(backfill_row)
        left_layout.addWidget(self.load_btn)
        left_layout.addWidget(self.retrain_check)
        left_layout.addWidget(self.summary_label)
        left_layout.addWidget(self.input_info_label)
        left_layout.addWidget(QLabel("业务日期列表"))
        left_layout.addWidget(self.list_widget, 1)
        left_layout.addWidget(QLabel("运行日志"))
        left_layout.addWidget(self.log, 2)

        self.map_canvas = SWEMapCanvas()
        right_layout.addWidget(self.map_canvas)

        main_layout.addLayout(left_layout, 1)
        main_layout.addLayout(right_layout, 3)

        self.update_btn.clicked.connect(self.run_update_latest)
        self.backfill_btn.clicked.connect(self.run_backfill)
        self.load_btn.clicked.connect(lambda: self.load_existing_results(silent=False))
        self.list_widget.itemClicked.connect(self.on_item_clicked)

    def _log(self, message: str) -> None:
        self.log.append(message)

    def _set_busy(self, busy: bool) -> None:
        self.update_btn.setEnabled(not busy)
        self.backfill_btn.setEnabled(not busy)
        self.backfill_days.setEnabled(not busy)
        self.load_btn.setEnabled(not busy)
        self.retrain_check.setEnabled(not busy)

    def _start_background_task(
        self,
        *,
        target,
        kwargs: dict,
        start_message: str,
        success_message: str,
        error_title: str,
    ) -> None:
        if self._worker_thread is not None:
            self._log("[WARN] 当前已有任务在运行，请稍候。")
            return

        self._log(start_message)
        self._set_busy(True)
        self._success_message = success_message
        self._error_title = error_title

        self._worker_thread = QThread(self)
        self._worker = SWEWorker(target, kwargs)
        self._worker.moveToThread(self._worker_thread)

        self._worker_thread.started.connect(self._worker.run)
        self._worker.succeeded.connect(self._handle_worker_success)
        self._worker.failed.connect(self._handle_worker_error)
        self._worker.finished.connect(self._worker_thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker_thread.finished.connect(self._handle_worker_finished)
        self._worker_thread.finished.connect(self._worker_thread.deleteLater)
        self._worker_thread.start()

    def _handle_worker_success(self, result: dict) -> None:
        self.result = result
        self.populate_results()
        if self._success_message:
            self._log(self._success_message)

    def _handle_worker_error(self, message: str) -> None:
        self._log(f"[ERROR] {message}")
        QMessageBox.critical(self, self._error_title, message)

    def _handle_worker_finished(self) -> None:
        self._set_busy(False)
        self._worker = None
        self._worker_thread = None
        self._success_message = ""
        self._error_title = "任务失败"

    def _selected_layer_key(self) -> str:
        return DISPLAY_LAYER_KEY

    def _entry_raster_path(self, entry: dict) -> str | None:
        return entry.get(self._selected_layer_key())

    def _latest_entry(self) -> dict:
        if not self.result:
            return {}
        latest = self.result.get("latest_entry")
        if latest:
            return latest
        entries = self.result.get("entries", [])
        return entries[-1] if entries else {}

    def _update_summary(self) -> None:
        if not self.result or not self.result.get("entries"):
            self.summary_label.setText("尚未加载 SWE 结果。")
            return

        latest = self._latest_entry()
        text = (
            f"最新业务日: {latest.get('business_date', '-')}\n"
            f"来源状态: {latest.get('source_status', '-')}\n"
            f"驱动周期: {latest.get('forcing_cycle', '-')}\n"
            f"VIIRS 状态: {latest.get('viirs_status', '-')}\n"
            f"流域 SWE: {latest.get('swe_mm', float('nan')):.2f} mm"
        )
        self.summary_label.setText(text)

    def populate_results(self) -> None:
        self.list_widget.clear()
        if not self.result:
            return

        for entry in self.result.get("entries", []):
            label = (
                f"{entry.get('business_date', '-')} | "
                f"{entry.get('source_status', '-')} | "
                f"VIIRS={entry.get('viirs_status', '-')}"
            )
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, entry)
            self.list_widget.addItem(item)

        if self.list_widget.count():
            preferred_date = self._latest_entry().get("business_date")
            selected_row = self.list_widget.count() - 1
            if preferred_date:
                for row in range(self.list_widget.count()):
                    item = self.list_widget.item(row)
                    entry = item.data(Qt.UserRole)
                    if entry.get("business_date") == preferred_date:
                        selected_row = row
                        break
            self.list_widget.setCurrentRow(selected_row)
            current_item = self.list_widget.currentItem()
            if current_item:
                self.display_entry(current_item.data(Qt.UserRole))

        self._update_summary()

    def display_entry(self, entry: dict | None) -> None:
        if not entry:
            return

        raster_path = self._entry_raster_path(entry)
        if not raster_path or not os.path.exists(raster_path):
            self._log(f"[WARN] 图层结果不存在: {raster_path}")
            return

        self.map_canvas.plot_raster(
            raster_path=raster_path,
            study_area_shp=self.result.get("study_area_shp", ""),
            layer_key=self._selected_layer_key(),
        )

    def refresh_current_item(self) -> None:
        item = self.list_widget.currentItem()
        if item:
            self.display_entry(item.data(Qt.UserRole))

    def on_item_clicked(self, item: QListWidgetItem) -> None:
        self.display_entry(item.data(Qt.UserRole))

    def run_update_latest(self) -> None:
        self._start_background_task(
            target=swe_assessment.run_update_latest_swe,
            kwargs={"force_retrain": self.retrain_check.isChecked()},
            start_message="开始更新最新 SWE 业务日（今天优先，不完整则回退昨天）...",
            success_message="最新 SWE 更新完成。",
            error_title="SWE 更新失败",
        )

    def run_backfill(self) -> None:
        days_back = self.backfill_days.value()
        self._start_background_task(
            target=swe_assessment.run_backfill_swe,
            kwargs={
                "days_back": days_back,
                "force_retrain": self.retrain_check.isChecked(),
            },
            start_message=f"开始回算最近 {days_back} 天的 SWE ...",
            success_message=f"最近 {days_back} 天的 SWE 回算完成。",
            error_title="SWE 回算失败",
        )

    def load_existing_results(self, silent: bool = False) -> None:
        try:
            self.result = swe_assessment.load_existing_results()
            self.populate_results()
            if not silent:
                self._log("已加载已有 SWE 结果。")
        except Exception as exc:
            if not silent:
                self._log(f"[ERROR] {exc}")
                QMessageBox.critical(self, "加载失败", str(exc))
