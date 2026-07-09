import csv
import traceback
from pathlib import Path

from PyQt5.QtWidgets import (
    QFileDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
import numpy as np
import rasterio
import rasterio.warp
from rasterio.enums import Resampling

from algorithms.routing.unity_visualization import launch_unity_visualization
from app.digital_twin_standard import (
    DEFAULT_PERIOD,
    TARGET_CRS,
    default_run_context,
    mark_module_complete,
    module_output_path,
    period_to_date,
    standard_dialog_dir,
    write_metadata_sidecar,
    write_standard_csv,
)


DISCHARGE_COLUMNS = ("discharge_m3_s", "discharge_m3s", "discharge", "flow_m3_s", "flow", "q", "runoff")


def _context_from_output_path(path: str | Path):
    token = Path(path).stem.split("_", 1)[0]
    if not token.isdigit() or len(token) < 6:
        token = DEFAULT_PERIOD
    period = token[:8] if len(token) >= 8 else token[:6]
    month = int(period[4:6]) if len(period) >= 6 else 3
    if 3 <= month <= 5:
        period_name = "融雪模拟"
    elif 6 <= month <= 9:
        period_name = "汛期模拟"
    else:
        period_name = "洪水演进模拟"
    return default_run_context(period=period, period_name=period_name)


def _first_existing_column(fieldnames: list[str], candidates: tuple[str, ...]) -> str | None:
    lookup = {name.lower(): name for name in fieldnames}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def _is_number(value: object) -> bool:
    try:
        float(str(value).strip())
        return True
    except Exception:
        return False


def _standardize_discharge_csv(src_path: str | Path, dst_path: str | Path, context) -> Path:
    src_path = Path(src_path)
    if not src_path.exists():
        raise FileNotFoundError(src_path)

    with src_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        source_fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if not source_fieldnames:
        raise ValueError(f"流量 CSV 缺少表头: {src_path}")
    if not rows:
        raise ValueError(f"流量 CSV 没有数据行: {src_path}")

    date_col = _first_existing_column(source_fieldnames, ("date", "time", "datetime"))
    discharge_col = _first_existing_column(source_fieldnames, DISCHARGE_COLUMNS)
    if discharge_col is None:
        for name in source_fieldnames:
            if name == date_col:
                continue
            if any(_is_number(row.get(name, "")) for row in rows):
                discharge_col = name
                break

    standard_rows = []
    for row in rows:
        standard_rows.append(
            {
                "date": row.get(date_col, "") if date_col else period_to_date(context.period),
                "period": context.period,
                "period_name": context.period_name,
                "scheme": context.scheme,
                "scheme_name": context.scheme_name,
                "module_code": "M03",
                "discharge_m3_s": row.get(discharge_col, "") if discharge_col else "",
                "source_discharge_column": discharge_col or "",
            }
        )

    return write_standard_csv(
        dst_path,
        fieldnames=[
            "date",
            "period",
            "period_name",
            "scheme",
            "scheme_name",
            "module_code",
            "discharge_m3_s",
            "source_discharge_column",
        ],
        rows=standard_rows,
    )


def _copy_or_reproject_continuous_raster(src_path: str | Path, dst_path: str | Path) -> Path:
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    if not src_path.exists():
        raise FileNotFoundError(src_path)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(src_path) as src:
        if src.crs is None:
            raise ValueError(f"源水深栅格缺少 CRS: {src_path}")
        profile = src.profile.copy()
        nodata = src.nodata if src.nodata is not None else -9999.0
        if src.crs.to_string() == TARGET_CRS:
            data = src.read(out_dtype="float32")
            profile.update(driver="GTiff", dtype="float32", compress="lzw", nodata=nodata)
            with rasterio.open(dst_path, "w", **profile) as dst:
                dst.write(data)
            return dst_path

        transform, width, height = rasterio.warp.calculate_default_transform(
            src.crs,
            TARGET_CRS,
            src.width,
            src.height,
            *src.bounds,
        )
        destination = np.full((src.count, height, width), nodata, dtype=np.float32)
        for band_index in range(1, src.count + 1):
            rasterio.warp.reproject(
                source=rasterio.band(src, band_index),
                destination=destination[band_index - 1],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=TARGET_CRS,
                resampling=Resampling.bilinear,
                src_nodata=src.nodata,
                dst_nodata=nodata,
            )
        profile.update(
            driver="GTiff",
            crs=TARGET_CRS,
            transform=transform,
            width=width,
            height=height,
            dtype="float32",
            compress="lzw",
            nodata=nodata,
        )
        with rasterio.open(dst_path, "w", **profile) as dst:
            dst.write(destination)
    return dst_path


def _copy_or_reproject_binary_raster(src_path: str | Path, dst_path: str | Path) -> Path:
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    if not src_path.exists():
        raise FileNotFoundError(src_path)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(src_path) as src:
        if src.crs is None:
            raise ValueError(f"源淹没栅格缺少 CRS: {src_path}")
        if src.crs.to_string() == TARGET_CRS:
            data = src.read(1)
            transform = src.transform
            width = src.width
            height = src.height
        else:
            transform, width, height = rasterio.warp.calculate_default_transform(
                src.crs,
                TARGET_CRS,
                src.width,
                src.height,
                *src.bounds,
            )
            data = np.zeros((height, width), dtype=np.float32)
            rasterio.warp.reproject(
                source=rasterio.band(src, 1),
                destination=data,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=TARGET_CRS,
                resampling=Resampling.nearest,
                src_nodata=src.nodata,
                dst_nodata=0,
            )
    binary = np.where(np.asarray(data) > 0, 1, 0).astype(np.uint8)
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "uint8",
        "crs": TARGET_CRS,
        "transform": transform,
        "nodata": 0,
        "compress": "lzw",
    }
    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(binary, 1)
    return dst_path


class RoutingWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        title = QLabel("洪水演进与汇流三维可视化模块")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")

        description = QLabel(
            "该模块用于启动 Unity 洪水演进三维可视化程序。"
            "Qt 负责入口、状态提示和异常提示，Unity 负责三维场景展示。"
        )
        description.setWordWrap(True)

        self.run_btn = QPushButton("启动三维可视化")
        self.sync_btn = QPushButton("同步洪水演进成果")
        self.status_label = QLabel("状态: 未启动")
        self.log = QTextEdit()
        self.log.setReadOnly(True)

        layout.addWidget(title)
        layout.addWidget(description)
        layout.addWidget(self.run_btn)
        layout.addWidget(self.sync_btn)
        layout.addWidget(self.status_label)
        layout.addWidget(self.log)

        self.run_btn.clicked.connect(self.launch_unity_module)
        self.sync_btn.clicked.connect(self.sync_standard_outputs)

    def launch_unity_module(self):
        try:
            self.log.append("正在启动三维可视化程序...")
            result = launch_unity_visualization()
            self.status_label.setText(f"状态: 已启动，PID={result.pid}")
            self.log.append("三维可视化程序已启动。")
        except Exception as e:
            self.status_label.setText("状态: 启动失败")
            self.log.append(str(e))
            self.log.append(traceback.format_exc())
            QMessageBox.critical(self, "启动失败", str(e))

    def sync_standard_outputs(self):
        discharge_csv, _ = QFileDialog.getOpenFileName(
            self,
            "选择 M03 河道流量 CSV",
            standard_dialog_dir("output", module_code="M03", output_index=0),
            "CSV files (*.csv);;All files (*.*)",
        )
        if not discharge_csv:
            return
        flood_depth_tif, _ = QFileDialog.getOpenFileName(
            self,
            "选择 M03 洪水水深 GeoTIFF",
            standard_dialog_dir("output", module_code="M03", output_index=1),
            "GeoTIFF (*.tif *.tiff);;All files (*.*)",
        )
        if not flood_depth_tif:
            return
        inundation_tif, _ = QFileDialog.getOpenFileName(
            self,
            "选择 M03 模拟淹没范围 GeoTIFF",
            standard_dialog_dir("output", module_code="M03", output_index=2),
            "GeoTIFF (*.tif *.tiff);;All files (*.*)",
        )
        if not inundation_tif:
            return

        try:
            outputs = self._export_standard_outputs(discharge_csv, flood_depth_tif, inundation_tif)
            self.status_label.setText("状态: 标准成果已同步")
            self.log.append("M03 标准成果已同步。")
            self.log.append(f"流量表: {outputs['discharge']}")
            self.log.append(f"洪水水深: {outputs['flood_depth']}")
            self.log.append(f"模拟淹没范围: {outputs['inundation']}")
            QMessageBox.information(self, "同步完成", "已写入 M03 标准成果目录。")
        except Exception as exc:
            self.status_label.setText("状态: 同步失败")
            self.log.append(str(exc))
            self.log.append(traceback.format_exc())
            QMessageBox.critical(self, "同步失败", str(exc))

    def _export_standard_outputs(
        self,
        discharge_csv: str | Path,
        flood_depth_tif: str | Path,
        inundation_tif: str | Path,
    ) -> dict[str, Path]:
        context = _context_from_output_path(discharge_csv)
        standard_discharge = module_output_path("M03", context=context, output_index=0)
        standard_depth = module_output_path("M03", context=context, output_index=1)
        standard_inundation = module_output_path("M03", context=context, output_index=2)

        _standardize_discharge_csv(discharge_csv, standard_discharge, context)
        _copy_or_reproject_continuous_raster(flood_depth_tif, standard_depth)
        _copy_or_reproject_binary_raster(inundation_tif, standard_inundation)

        m02_runoff = module_output_path("M02", context=context, output_index=1)
        source_files = [m02_runoff, discharge_csv, flood_depth_tif, inundation_tif]
        common_extra = {
            "scheme": context.scheme,
            "scheme_name": context.scheme_name,
            "period": context.period,
            "period_name": context.period_name,
            "upstream_runoff": str(m02_runoff),
            "source_note": "由外部洪水演进/汇流模型成果同步到统一 processed 目录。",
        }
        write_metadata_sidecar(
            standard_discharge,
            module_code="M03",
            field="discharge",
            source_files=source_files,
            extra=common_extra,
        )
        write_metadata_sidecar(
            standard_depth,
            module_code="M03",
            field="flood_depth",
            source_files=source_files,
            extra=common_extra,
        )
        write_metadata_sidecar(
            standard_inundation,
            module_code="M03",
            field="inundation",
            source_files=source_files,
            extra=common_extra,
        )
        mark_module_complete(context, "M03")
        return {
            "discharge": standard_discharge,
            "flood_depth": standard_depth,
            "inundation": standard_inundation,
        }
