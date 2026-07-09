from __future__ import annotations

import os
import traceback
from pathlib import Path

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QTextEdit, QMessageBox, QComboBox, QLineEdit, QSizePolicy
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import numpy as np
import rasterio
import rasterio.warp
from rasterio.enums import Resampling

from algorithms.segformer_service.service_config import TASKS
from algorithms.segformer_service.service_runner import run_segformer_service
from app.digital_twin_standard import (
    DEFAULT_PERIOD,
    TARGET_CRS,
    default_run_context,
    module_output_path,
    period_to_date,
    standard_dialog_dir,
    write_metadata_sidecar,
    write_standard_csv,
)
from app.ui_hints import attach_hint, label_with_hint


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
        period_name = "遥感解译"
    return default_run_context(period=period, period_name=period_name)


def _read_snow_cover_as_target(src_path: str | Path):
    path = Path(src_path)
    if not path.exists():
        raise FileNotFoundError(path)

    with rasterio.open(path) as src:
        if src.count < 1:
            raise ValueError(f"GeoTIFF 至少需要 1 个积雪覆盖波段: {path}")
        if src.crs is None:
            raise ValueError(f"积雪覆盖 GeoTIFF 缺少 CRS，不能写入标准成果: {path}")

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

        nodata = src.nodata
        source_crs = src.crs.to_string()
        band_count = src.count

    data = np.asarray(data)
    if nodata is not None and np.isfinite(nodata):
        valid = data != nodata
    else:
        valid = np.isfinite(data)
    snow_cover = np.where(valid & (data > 0), 1, 0).astype(np.uint8)
    return snow_cover, transform, width, height, source_crs, band_count


def _write_snow_cover_tif(snow_cover: np.ndarray, transform, width: int, height: int, dst_path: str | Path) -> Path:
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "uint8",
        "crs": TARGET_CRS,
        "transform": transform,
        "compress": "lzw",
    }
    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(snow_cover.astype(np.uint8), 1)
    return dst_path


def _write_snow_area_table(snow_cover: np.ndarray, transform, context, dst_path: str | Path) -> Path:
    total_pixels = int(snow_cover.size)
    snow_pixels = int(np.count_nonzero(snow_cover > 0))
    pixel_area_km2 = abs(float(transform.a) * float(transform.e)) / 1_000_000.0
    snow_area_km2 = snow_pixels * pixel_area_km2
    snow_cover_ratio = snow_pixels / total_pixels if total_pixels else 0.0
    return write_standard_csv(
        dst_path,
        fieldnames=[
            "date",
            "period",
            "period_name",
            "scheme",
            "scheme_name",
            "module_code",
            "snow_cover_area_km2",
            "snow_cover_ratio",
            "snow_pixels",
            "total_pixels",
            "pixel_area_km2",
        ],
        rows=[
            {
                "date": period_to_date(context.period),
                "period": context.period,
                "period_name": context.period_name,
                "scheme": context.scheme,
                "scheme_name": context.scheme_name,
                "module_code": "M01",
                "snow_cover_area_km2": f"{snow_area_km2:.6f}",
                "snow_cover_ratio": f"{snow_cover_ratio:.6f}",
                "snow_pixels": snow_pixels,
                "total_pixels": total_pixels,
                "pixel_area_km2": f"{pixel_area_km2:.9f}",
            }
        ],
    )


class ImageLabel(QLabel):
    def __init__(self, text="No Image"):
        super().__init__(text)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(260, 200)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background:#f0f0f0;border:1px solid #ccc;")
        self._pix = None

    def set_image(self, path):
        if path and os.path.exists(path):
            pix = QPixmap(path)
            if not pix.isNull():
                self._pix = pix
                self._refresh()
                return
        self._pix = None
        self.clear()
        self.setText("Image Not Found")

    def _refresh(self):
        if self._pix is not None:
            self.setPixmap(self._pix.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        self._refresh()
        super().resizeEvent(event)


class SegFormerWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.current_task = "water"
        self.image_path = ""
        self.result_path = ""
        self.init_ui()
        self.update_defaults()

    def init_ui(self):
        layout = QVBoxLayout(self)

        title = QLabel("SegFormer 专题识别服务")
        title.setStyleSheet("font-size:18px;font-weight:bold;")
        layout.addWidget(title)

        task_row = QHBoxLayout()
        task_hint = "内容：分割任务类型。\n格式：下拉选择，当前支持 water / snow。"
        task_row.addWidget(label_with_hint("任务:", task_hint, stretch=False))

        self.task_combo = QComboBox()
        self.task_combo.addItem("水体识别", "water")
        self.task_combo.addItem("积雪识别", "snow")
        self.task_combo.currentIndexChanged.connect(self.on_task_changed)
        attach_hint(self.task_combo, task_hint)

        self.device_combo = QComboBox()
        self.device_combo.addItems(["cpu", "cuda:0"])
        device_hint = "内容：推理设备。\n格式：下拉选择，cpu 或 cuda:0。"
        attach_hint(self.device_combo, device_hint)

        task_row.addWidget(self.task_combo)
        task_row.addWidget(label_with_hint("设备:", device_hint, stretch=False))
        task_row.addWidget(self.device_combo)
        task_row.addStretch()

        layout.addLayout(task_row)

        path_row = QHBoxLayout()
        self.image_edit = QLineEdit()
        self.image_edit.setReadOnly(True)
        image_hint = "内容：输入影像文件路径。\n格式：png/jpg/jpeg/bmp 文件路径（通过“选择输入图片”填写）。"
        attach_hint(self.image_edit, image_hint)
        self.image_btn = QPushButton("选择输入图片")
        self.image_btn.clicked.connect(self.select_image)
        path_row.addWidget(label_with_hint("图片:", image_hint, stretch=False))
        path_row.addWidget(self.image_edit, 1)
        path_row.addWidget(self.image_btn)
        layout.addLayout(path_row)

        btn_row = QHBoxLayout()
        self.run_btn = QPushButton("运行分割")
        self.load_btn = QPushButton("加载已有结果")
        self.sync_btn = QPushButton("同步积雪标准成果")
        self.run_btn.clicked.connect(self.run_task)
        self.load_btn.clicked.connect(self.load_existing_result)
        self.sync_btn.clicked.connect(self.sync_standard_snow_outputs)
        btn_row.addWidget(self.run_btn)
        btn_row.addWidget(self.load_btn)
        btn_row.addWidget(self.sync_btn)
        layout.addLayout(btn_row)

        img_row = QHBoxLayout()
        self.input_label = ImageLabel("输入图")
        self.result_label = ImageLabel("结果图")
        img_row.addWidget(self.input_label, 1)
        img_row.addWidget(self.result_label, 1)
        layout.addLayout(img_row)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log)

    def on_task_changed(self):
        self.current_task = self.task_combo.currentData()
        self.update_defaults()

    def update_defaults(self):
        input_dir = standard_dialog_dir("raw", module_code="M01")
        if os.path.exists(input_dir):
            files = [
                os.path.join(root, f)
                for root, _, names in os.walk(input_dir)
                for f in names
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
            ]
            if files:
                self.image_path = files[0]
                self.image_edit.setText(self.image_path)
                self.input_label.set_image(self.image_path)

    def select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择输入图片",
            standard_dialog_dir("raw", module_code="M01"),
            "Images/GeoTIFF (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if file_path:
            self.image_path = file_path
            self.image_edit.setText(file_path)
            self.input_label.set_image(file_path)

    def run_task(self):
        try:
            if not self.image_path:
                raise ValueError("请先选择输入图片")

            self.log.append(f"开始运行任务: {TASKS[self.current_task]['name']}")

            result = run_segformer_service(
                task_key=self.current_task,
                image_path=self.image_path,
                device=self.device_combo.currentText()
            )

            self.log.append("执行命令:")
            self.log.append(" ".join(result["command"]))
            self.log.append(f"返回码: {result['returncode']}")

            if result["stdout"]:
                self.log.append("STDOUT:")
                self.log.append(result["stdout"])
            if result["stderr"]:
                self.log.append("STDERR:")
                self.log.append(result["stderr"])

            if result["returncode"] != 0:
                raise RuntimeError("SegFormer 推理服务执行失败，请查看日志。")

            if not os.path.exists(result["overlay_path"]):
                raise FileNotFoundError(f"结果图未生成: {result['overlay_path']}")

            self.result_path = result["overlay_path"]
            self.result_label.set_image(self.result_path)
            self.log.append(f"结果图: {self.result_path}")
            self.log.append(
                "提示: png/jpg 结果仅用于预览。若要进入 M01 标准成果链路，"
                "请点击“同步积雪标准成果”并选择已配准的积雪覆盖 GeoTIFF。"
            )

        except Exception as e:
            self.log.append(f"[ERROR] {str(e)}")
            QMessageBox.critical(self, "错误", str(e))

    def load_existing_result(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "加载已有结果",
            standard_dialog_dir("output", module_code="M01", output_index=1),
            "GeoTIFF (*.tif *.tiff);;Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_path:
            self.result_path = file_path
            if file_path.lower().endswith((".tif", ".tiff")):
                self.result_label.setText("已加载标准 GeoTIFF\n用于统一 processed 链路")
            else:
                self.result_label.set_image(file_path)
            self.log.append(f"已加载标准结果: {file_path}")

    def sync_standard_snow_outputs(self):
        geotiff_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择 M01 积雪覆盖 GeoTIFF",
            standard_dialog_dir("raw", module_code="M01"),
            "GeoTIFF (*.tif *.tiff);;All files (*.*)",
        )
        if not geotiff_path:
            return

        try:
            outputs = self._export_standard_snow_result(geotiff_path)
            self.log.append("M01 积雪标准成果已同步。")
            self.log.append(f"积雪覆盖: {outputs['snow_cover']}")
            self.log.append(f"面积统计: {outputs['area_table']}")
            self.log.append("雪深未写入：当前 SegFormer 结果未提供真实雪深栅格。")
            QMessageBox.information(self, "同步完成", "已写入 M01 积雪覆盖与面积统计；未伪造雪深。")
        except Exception as exc:
            self.log.append(f"[ERROR] {exc}")
            self.log.append(traceback.format_exc())
            QMessageBox.critical(self, "同步失败", str(exc))

    def _export_standard_snow_result(self, geotiff_path: str | Path) -> dict[str, Path]:
        context = _context_from_output_path(geotiff_path)
        snow_cover_tif = module_output_path("M01", context=context, output_index=1)
        area_table = module_output_path("M01", context=context, output_index=2)

        snow_cover, transform, width, height, source_crs, band_count = _read_snow_cover_as_target(geotiff_path)
        _write_snow_cover_tif(snow_cover, transform, width, height, snow_cover_tif)
        _write_snow_area_table(snow_cover, transform, context, area_table)

        source_files = [geotiff_path]
        common_extra = {
            "scheme": context.scheme,
            "scheme_name": context.scheme_name,
            "period": context.period,
            "period_name": context.period_name,
            "source_product": "SegFormer snow-cover GeoTIFF",
            "source_crs": source_crs,
            "source_band_count": band_count,
            "source_note": "普通 png/jpg 推理结果仅作为预览；正式 M01 成果必须来自带 CRS 的已配准积雪覆盖 GeoTIFF。",
        }
        write_metadata_sidecar(
            snow_cover_tif,
            module_code="M01",
            field="snow_cover",
            source_files=source_files,
            extra={
                **common_extra,
                "standard_field": "snow_cover",
                "class_rule": "source_value > 0 -> snow_cover=1, otherwise 0",
            },
        )
        write_metadata_sidecar(
            area_table,
            module_code="M01",
            field="snow_cover",
            source_files=source_files,
            extra={
                **common_extra,
                "standard_field": "snow_cover_area",
                "output_type": "area_statistics_table",
            },
        )
        return {
            "snow_cover": snow_cover_tif,
            "area_table": area_table,
        }
