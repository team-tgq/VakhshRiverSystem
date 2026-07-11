from __future__ import annotations

import csv
import inspect
import os
from pathlib import Path

import numpy as np
import rasterio
import rasterio.features
import rasterio.warp
import shapefile
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import colormaps, colors
from rasterio.enums import Resampling
from rasterio.fill import fillnodata
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
from app.digital_twin_standard import (
    DEFAULT_PERIOD,
    TARGET_CRS,
    baseline_dir,
    default_run_context,
    iter_raw_run_contexts,
    mark_module_complete,
    module_output_path,
    module_processed_dir,
    period_to_date,
    raw_data_dir,
    write_metadata_sidecar,
)
from algorithms.swe.daily_ml_pipeline import MODEL_PATH, run_unified_forcing_offline


DISPLAY_LAYER_KEY = "swe_raster"
DISPLAY_LAYER_LABEL = "SWE"
DISPLAY_LONG_EDGE = 960
INPUT_DATA_TEXT = (
    "输入数据说明\n"
    "静态输入：baseline/DEM.tif 和 baseline/流域边界.shp。\n"
    "统一动态输入：raw/meteorology/daily_forcing/forcing_YYYYMMDD.npz。\n"
    "模块输出：processed/M02_swe/rasters 与 tables。\n"
    "当前模型输出 SWE 与融雪量，不生成没有算法依据的径流量。"
)


def _business_date_to_context(business_date: str | None):
    date_text = str(business_date or "").strip()
    if len(date_text) >= 10 and date_text[4] == "-" and date_text[7] == "-":
        year = int(date_text[:4])
        month = int(date_text[5:7])
        period = f"{year:04d}{month:02d}"
    else:
        period = date_text.replace("-", "") or DEFAULT_PERIOD
        month = int(period[4:6]) if len(period) >= 6 and period[4:6].isdigit() else int(DEFAULT_PERIOD[4:6])

    for context in iter_raw_run_contexts():
        if context.period == period and _swe_raw_input_paths(context):
            return context

    if 3 <= month <= 5:
        period_name = "融雪模拟"
    elif 6 <= month <= 9:
        period_name = "汛期模拟"
    else:
        period_name = "日尺度模拟"
    return default_run_context(period=period, period_name=period_name)


def _entry_to_context(entry: dict):
    standard_period = str(entry.get("standard_period") or "").strip()
    if standard_period:
        period = standard_period.replace("-", "")
        month = int(period[4:6]) if len(period) >= 6 and period[4:6].isdigit() else int(DEFAULT_PERIOD[4:6])
        for context in iter_raw_run_contexts():
            if context.period == period and _swe_raw_input_paths(context):
                return context
        if 3 <= month <= 5:
            period_name = "融雪模拟"
        elif 6 <= month <= 9:
            period_name = "汛期模拟"
        else:
            period_name = "日尺度模拟"
        return default_run_context(period=period, period_name=period_name)
    return _business_date_to_context(entry.get("business_date"))


def _swe_raw_input_paths(context) -> list[Path]:
    forcing_dir = raw_data_dir("meteorology", "daily_forcing")
    pattern = f"forcing_{context.period}*.npz" if len(context.period) == 6 else f"forcing_{context.period}.npz"
    return sorted(path for path in forcing_dir.glob(pattern) if path.is_file())


def _require_swe_raw_input_paths(context) -> list[Path]:
    paths = _swe_raw_input_paths(context)
    if paths:
        return paths
    expected = raw_data_dir("meteorology", "daily_forcing") / f"forcing_{context.period}.npz"
    raise FileNotFoundError(
        "M02 SWE 缺少统一 raw 输入，不能写入 processed 标准成果。\n"
        f"当前时段: {context.period}_{context.period_name}\n"
        f"应提供统一 forcing 文件:\n{expected}"
    )


def _business_date_from_raw_inputs(context, raw_inputs: list[Path]) -> str:
    return period_to_date(context.period)


def _run_unified_swe_inputs(
    *,
    days_back: int = 1,
    force_retrain: bool = False,
    progress_callback=None,
) -> dict:
    if force_retrain and progress_callback:
        progress_callback("统一离线运行复用当前已训练模型，不执行联网重训。")
    forcing_dir = raw_data_dir("meteorology", "daily_forcing")
    dem = baseline_dir() / "DEM.tif"
    with rasterio.open(dem) as dataset:
        dem_shape = (dataset.height, dataset.width)
    compatible: list[Path] = []
    skipped: list[str] = []
    for source in sorted(forcing_dir.glob("forcing_*.npz")):
        with np.load(source, allow_pickle=False) as archive:
            shape = tuple(archive["temp_mean_c"].shape)
        if shape == dem_shape:
            compatible.append(source)
        else:
            skipped.append(f"{source.name}: {shape} != DEM {dem_shape}")
    if not compatible:
        raise RuntimeError("统一 daily_forcing 中没有与 baseline DEM 对齐的输入。")
    selected = compatible[-max(1, int(days_back)):]
    entries = []
    for index, source in enumerate(selected, start=1):
        token = source.stem.rsplit("_", 1)[-1]
        context = default_run_context(period=token, period_name="独立离线运行")
        swe_path = module_output_path("M02", context=context, output_index=0)
        snowmelt_path = module_output_path("M02", context=context, output_index=1)
        statistics_path = module_processed_dir("M02", "tables", create=True) / f"swe_statistics_{token}.csv"
        if progress_callback:
            progress_callback(f"正在重放 {source.name}（{index}/{len(selected)}）。")
        result = run_unified_forcing_offline(
            source,
            dem,
            swe_path,
            snowmelt_path,
            statistics_path,
        )
        entries.append(
            {
                "business_date": result["date"],
                "source_status": "unified_raw_offline_replay",
                "forcing_cycle": "local_cache",
                "viirs_status": "missing" if result["viirs_missing"] else "cached",
                "swe_mm": result["swe_mean_mm"],
                "snowmelt_mm_day": result["snowmelt_mean_mm_day"],
                "swe_raster": str(swe_path),
                "snowmelt_raster": str(snowmelt_path),
                "forcing_cache": str(source),
                "standard_period": token,
                "standard_period_name": "独立离线运行",
                "diagnostics": {},
            }
        )
    return {
        "entries": entries,
        "latest_entry": entries[-1],
        "study_area_shp": str(baseline_dir() / "流域边界.shp"),
        "skipped_inputs": skipped,
    }


def _copy_or_reproject_continuous_raster(src_path: str | Path, dst_path: str | Path) -> Path:
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    if not src_path.exists():
        raise FileNotFoundError(f"源栅格不存在: {src_path}")

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(src_path) as src:
        profile = src.profile.copy()
        src_crs = src.crs
        if src_crs is None:
            raise ValueError(f"源栅格缺少 CRS，不能写入标准成果: {src_path}")

        if src_crs.to_string() == TARGET_CRS:
            data = src.read()
            profile.update(driver="GTiff", compress="lzw")
            with rasterio.open(dst_path, "w", **profile) as dst:
                dst.write(data)
            return dst_path

        transform, width, height = rasterio.warp.calculate_default_transform(
            src_crs,
            TARGET_CRS,
            src.width,
            src.height,
            *src.bounds,
        )
        profile.update(
            driver="GTiff",
            crs=TARGET_CRS,
            transform=transform,
            width=width,
            height=height,
            compress="lzw",
        )
        nodata = src.nodata if src.nodata is not None else np.nan
        destination = np.full((src.count, height, width), nodata, dtype=np.float32)
        for band_index in range(1, src.count + 1):
            rasterio.warp.reproject(
                source=rasterio.band(src, band_index),
                destination=destination[band_index - 1],
                src_transform=src.transform,
                src_crs=src_crs,
                dst_transform=transform,
                dst_crs=TARGET_CRS,
                resampling=Resampling.bilinear,
                src_nodata=src.nodata,
                dst_nodata=src.nodata,
            )
        profile.update(dtype="float32")
        with rasterio.open(dst_path, "w", **profile) as dst:
            dst.write(destination.astype(np.float32))
    return dst_path


class SWEWorker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(str)
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, target, kwargs: dict):
        super().__init__()
        self.target = target
        self.kwargs = kwargs

    def run(self) -> None:
        try:
            call_kwargs = dict(self.kwargs)
            try:
                parameters = inspect.signature(self.target).parameters
            except (TypeError, ValueError):
                parameters = {}
            if "progress_callback" in parameters:
                call_kwargs["progress_callback"] = self.progress.emit
            result = self.target(**call_kwargs)
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
        if max(width, height) >= DISPLAY_LONG_EDGE:
            return height, width
        longest = max(width, height)
        scale = DISPLAY_LONG_EDGE / float(longest)
        display_height = max(1, int(round(height * scale)))
        display_width = max(1, int(round(width * scale)))
        return display_height, display_width

    def _resample_for_display(
        self,
        array: np.ndarray,
        *,
        bounds,
        transform,
        crs,
        width: int,
        height: int,
    ) -> tuple[np.ndarray, object]:
        display_height, display_width = self._display_shape(width, height)
        if display_height == height and display_width == width:
            return array.astype(np.float32), transform

        valid_mask = np.isfinite(array)
        if np.any(valid_mask):
            filled_array = fillnodata(
                array.copy(),
                mask=valid_mask.astype(np.uint8),
                max_search_distance=max(array.shape),
                smoothing_iterations=0,
            ).astype(np.float32)
        else:
            filled_array = array.astype(np.float32)

        display_transform = from_bounds(
            bounds.left,
            bounds.bottom,
            bounds.right,
            bounds.top,
            display_width,
            display_height,
        )
        display_array = np.full((display_height, display_width), np.nan, dtype=np.float32)
        rasterio.warp.reproject(
            source=filled_array,
            destination=display_array,
            src_transform=transform,
            src_crs=crs,
            dst_transform=display_transform,
            dst_crs=crs,
            resampling=Resampling.bilinear,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )
        return display_array, display_transform

    def plot_raster(self, raster_path: str, study_area_shp: str, layer_key: str) -> None:
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        boundary_data = self._load_boundary_data(study_area_shp)

        with rasterio.open(raster_path) as dataset:
            raster = dataset.read(1, masked=True)
            native_array = np.asarray(raster.filled(np.nan), dtype=np.float32)
            bounds = dataset.bounds
            native_transform = dataset.transform
            native_crs = dataset.crs
            extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)

        if boundary_data:
            native_inside_mask = rasterio.features.geometry_mask(
                boundary_data["geometries"],
                out_shape=native_array.shape,
                transform=native_transform,
                invert=True,
                all_touched=False,
            )
            native_array = np.where(native_inside_mask, native_array, np.nan).astype(np.float32)

        array, display_transform = self._resample_for_display(
            native_array,
            bounds=bounds,
            transform=native_transform,
            crs=native_crs,
            width=native_array.shape[1],
            height=native_array.shape[0],
        )

        if boundary_data:
            inside_mask = rasterio.features.geometry_mask(
                boundary_data["geometries"],
                out_shape=array.shape,
                transform=display_transform,
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
        self._log("系统会优先复用已有模型；如果本分支需要首次训练，下面会持续显示训练进度。")
        self._set_busy(True)
        self._success_message = success_message
        self._error_title = error_title

        self._worker_thread = QThread(self)
        self._worker = SWEWorker(target, kwargs)
        self._worker.moveToThread(self._worker_thread)

        self._worker_thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._handle_worker_progress)
        self._worker.succeeded.connect(self._handle_worker_success)
        self._worker.failed.connect(self._handle_worker_error)
        self._worker.finished.connect(self._worker_thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker_thread.finished.connect(self._handle_worker_finished)
        self._worker_thread.finished.connect(self._worker_thread.deleteLater)
        self._worker_thread.start()

    def _handle_worker_success(self, result: dict) -> None:
        exported = self._export_standard_results(result)
        self.result = result
        self.populate_results()
        if exported:
            self._log(f"已同步 {exported} 个业务日的 M02 标准成果。")
        if self._success_message:
            self._log(self._success_message)

    def _handle_worker_progress(self, message: str) -> None:
        if message:
            self._log(message)

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
        diagnostics = latest.get("diagnostics", {}) or {}
        correction_applied = bool(diagnostics.get("temperature_correction_applied", False))
        correction_mean = float(diagnostics.get("temperature_correction_mean_c", 0.0) or 0.0)
        correction_min = float(diagnostics.get("temperature_correction_min_c", 0.0) or 0.0)
        correction_max = float(diagnostics.get("temperature_correction_max_c", 0.0) or 0.0)
        dem_status = diagnostics.get("temperature_dem_status", "missing")
        dem_text = "已启用" if correction_applied else "未启用"
        if correction_applied:
            dem_text = f"{dem_text} ({correction_mean:+.2f} ℃, {correction_min:+.2f} ~ {correction_max:+.2f} ℃)"

        text = (
            f"最新业务日: {latest.get('business_date', '-')}\n"
            f"来源状态: {latest.get('source_status', '-')}\n"
            f"驱动周期: {latest.get('forcing_cycle', '-')}\n"
            f"VIIRS 状态: {latest.get('viirs_status', '-')}\n"
            f"流域 SWE: {latest.get('swe_mm', float('nan')):.2f} mm\n"
            f"DEM 温度订正: {dem_text}\n"
            f"DEM 状态: {dem_status}"
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
            target=_run_unified_swe_inputs,
            kwargs={"force_retrain": self.retrain_check.isChecked()},
            start_message="开始重放统一 raw 中最新的可用 SWE forcing...",
            success_message="最新统一 forcing 的 SWE 离线运行完成。",
            error_title="SWE 更新失败",
        )

    def run_backfill(self) -> None:
        days_back = self.backfill_days.value()
        self._start_background_task(
            target=_run_unified_swe_inputs,
            kwargs={
                "days_back": days_back,
                "force_retrain": self.retrain_check.isChecked(),
            },
            start_message=f"开始重放统一 raw 中最近 {days_back} 个兼容 forcing ...",
            success_message="统一 forcing 批量重放完成。",
            error_title="SWE 回算失败",
        )

    def load_existing_results(self, silent: bool = False) -> None:
        try:
            self.result = self._load_standard_results_from_processed()
            if not self.result.get("entries"):
                if silent:
                    return
                raise FileNotFoundError("标准 processed 目录中没有 M02 SWE 成果，请先运行 SWE 模块生成标准输出。")
            self.populate_results()
            if not silent:
                self._log("已从统一 processed 目录加载已有 SWE 结果。")
        except Exception as exc:
            if not silent:
                self._log(f"[ERROR] {exc}")
                QMessageBox.critical(self, "加载失败", str(exc))

    def _load_standard_results_from_processed(self) -> dict:
        entries = []
        for context in iter_raw_run_contexts():
            raw_inputs = _swe_raw_input_paths(context)
            if not raw_inputs:
                continue
            swe_path = module_output_path("M02", context=context, output_index=0)
            snowmelt_path = module_output_path("M02", context=context, output_index=1)
            if not snowmelt_path.exists():
                continue
            if not swe_path.exists():
                continue
            entries.append(
                {
                    "business_date": _business_date_from_raw_inputs(context, raw_inputs),
                    "source_status": "standard_processed",
                    "forcing_cycle": "",
                    "viirs_status": "",
                    "swe_mm": float("nan"),
                    "snowmelt_mm_day": float("nan"),
                    "swe_raster": str(swe_path),
                    "snowmelt_raster": str(snowmelt_path),
                    "standard_swe_raster": str(swe_path),
                    "standard_snowmelt_raster": str(snowmelt_path),
                    "standard_period": context.period,
                    "standard_period_name": context.period_name,
                    "raw_input_files": [str(path) for path in raw_inputs],
                    "diagnostics": {},
                }
            )
        return {
            "entries": entries,
            "latest_entry": entries[-1] if entries else {},
            "study_area_shp": str(baseline_dir() / "流域边界.shp"),
        }

    def _export_standard_results(self, result: dict | None) -> int:
        if not result:
            return 0
        entries = result.get("entries") or []
        exported = 0
        for entry in entries:
            try:
                self._export_standard_entry(entry)
                exported += 1
            except Exception as exc:
                self._log(f"[WARN] M02 标准成果未导出 {entry.get('business_date', '-')}: {exc}")
        return exported

    def _export_standard_entry(self, entry: dict) -> dict[str, str]:
        context = _entry_to_context(entry)
        raw_inputs = _require_swe_raw_input_paths(context)
        swe_source = entry.get("swe_raster")
        snowmelt_source = entry.get("snowmelt_raster")
        if not swe_source:
            raise ValueError("缺少 SWE 栅格路径。")
        if not snowmelt_source:
            raise ValueError("缺少融雪栅格路径。")

        swe_output = module_output_path("M02", context=context, output_index=0)
        snowmelt_output = module_output_path("M02", context=context, output_index=1)
        if Path(swe_source).resolve() != swe_output.resolve():
            _copy_or_reproject_continuous_raster(swe_source, swe_output)
        if Path(snowmelt_source).resolve() != snowmelt_output.resolve():
            _copy_or_reproject_continuous_raster(snowmelt_source, snowmelt_output)

        source_files = [*raw_inputs, baseline_dir() / "DEM.tif"]
        write_metadata_sidecar(
            swe_output,
            module_code="M02",
            field="swe",
            source_files=source_files,
            extra={
                "scheme": context.scheme,
                "scheme_name": context.scheme_name,
                "period": context.period,
                "period_name": context.period_name,
                "business_date": entry.get("business_date"),
                "swe_mm": entry.get("swe_mm"),
                "source_status": entry.get("source_status"),
                "viirs_status": entry.get("viirs_status"),
                "raw_input_files": [str(path) for path in raw_inputs],
                "model_weight": MODEL_PATH,
                "threshold_or_config": {"mode": "offline_unified_forcing_replay"},
            },
        )
        write_metadata_sidecar(
            snowmelt_output,
            module_code="M02",
            field="snowmelt",
            source_files=source_files,
            extra={
                "scheme": context.scheme,
                "scheme_name": context.scheme_name,
                "period": context.period,
                "period_name": context.period_name,
                "business_date": entry.get("business_date"),
                "snowmelt_mm_day": entry.get("snowmelt_mm_day"),
                "source_status": entry.get("source_status"),
                "raw_input_files": [str(path) for path in raw_inputs],
                "model_weight": MODEL_PATH,
                "threshold_or_config": {"mode": "offline_unified_forcing_replay"},
                "runoff_generated": False,
            },
        )
        mark_module_complete(context, "M02")
        entry["standard_swe_raster"] = str(swe_output)
        entry["standard_snowmelt_raster"] = str(snowmelt_output)
        return {
            "swe": str(swe_output),
            "snowmelt": str(snowmelt_output),
        }
