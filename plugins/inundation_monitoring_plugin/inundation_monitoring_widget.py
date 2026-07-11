import os
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from algorithms.inundation_monitoring.predictor import FloodPredictor
from app.digital_twin_standard import (
    TARGET_CRS,
    ensure_raw_source_path,
    infer_run_context_from_path,
    mark_module_complete,
    module_output_path,
    module_processed_dir,
    period_to_date,
    raw_data_dir,
    standard_dialog_dir,
    write_metadata_sidecar,
)
from app.ui_hints import attach_hint, create_hint_badge, label_with_hint


def _m04_validation_group(path: str | Path) -> str | None:
    source = Path(path).resolve()
    candidates = (
        (
            raw_data_dir("remote_sensing", "sentinel1_sar", "inundation_weak_labeled").resolve(),
            "sentinel1_sar_weak",
        ),
        (
            raw_data_dir("remote_sensing", "sentinel2_multispectral", "inundation_hand_labeled").resolve(),
            "sentinel2_optical_hand",
        ),
    )
    for base, group in candidates:
        try:
            source.relative_to(base)
            return group
        except ValueError:
            continue
    return None


def _write_simple_xlsx(path: str | Path, rows: list[list[object]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    sheet_rows = []
    for row_idx, row in enumerate(rows, start=1):
        cells = []
        for col_idx, value in enumerate(row, start=1):
            col = ""
            index = col_idx
            while index:
                index, rem = divmod(index - 1, 26)
                col = chr(65 + rem) + col
            ref = f"{col}{row_idx}"
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                cells.append(f'<c r="{ref}"><v>{value}</v></c>')
            else:
                text = escape(str(value))
                cells.append(f'<c r="{ref}" t="inlineStr"><is><t>{text}</t></is></c>')
        sheet_rows.append(f'<row r="{row_idx}">{"".join(cells)}</row>')

    sheet_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(sheet_rows)}</sheetData>'
        '</worksheet>'
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            "</Types>",
        )
        zf.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
            "</Relationships>",
        )
        zf.writestr(
            "xl/workbook.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            '<sheets><sheet name="M04统计" sheetId="1" r:id="rId1"/></sheets>'
            "</workbook>",
        )
        zf.writestr(
            "xl/_rels/workbook.xml.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
            "</Relationships>",
        )
        zf.writestr("xl/worksheets/sheet1.xml", sheet_xml)
    return path


class ImageLabel(QLabel):
    def __init__(self, text="No image"):
        super().__init__(text)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(300, 220)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background:#f5f7fa;border:1px solid #cbd5e1;color:#334155;")
        self._pix = None

    def set_qimage(self, qimg: QImage):
        self._pix = QPixmap.fromImage(qimg)
        self._refresh()

    def _refresh(self):
        if self._pix is not None:
            self.setPixmap(self._pix.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        self._refresh()
        super().resizeEvent(event)


class InundationMonitoringWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.predictor = FloodPredictor()
        self.current_image_path = ""
        self.last_result = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        threshold_hint = "SegFormer 淹没概率阈值，范围 0 到 1；值越大，识别结果越保守。"
        threshold_row = QHBoxLayout()
        self.thresh_input = QLineEdit("0.50")
        attach_hint(self.thresh_input, threshold_hint)
        threshold_row.addWidget(label_with_hint("淹没识别阈值", threshold_hint, stretch=False))
        threshold_row.addWidget(self.thresh_input)
        layout.addLayout(threshold_row)

        input_hint = (
            "选择遥感影像文件。优先使用多波段 GeoTIFF；模型会构建 6 个可见/近红外/SWIR "
            "通道加 MNDWI 的 7 通道特征，普通图片会自动退化为 RGB/灰度特征。"
        )
        button_row = QHBoxLayout()
        self.btn_select = QPushButton("选择遥感影像")
        self.btn_select.clicked.connect(self.select_image)
        attach_hint(self.btn_select, input_hint)

        self.btn_open_overlay = QPushButton("打开叠加图")
        self.btn_open_overlay.clicked.connect(self.open_overlay_file)

        self.btn_open_mask = QPushButton("打开掩膜图")
        self.btn_open_mask.clicked.connect(self.open_mask_file)

        button_row.addWidget(self.btn_select)
        button_row.addWidget(create_hint_badge(input_hint))
        button_row.addWidget(self.btn_open_overlay)
        button_row.addWidget(self.btn_open_mask)
        layout.addLayout(button_row)

        image_row = QHBoxLayout()
        self.label_orig = ImageLabel("原始影像")
        self.label_result = ImageLabel("淹没识别结果")
        image_row.addWidget(self.label_orig, 1)
        image_row.addWidget(self.label_result, 1)
        layout.addLayout(image_row, 1)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(130)
        layout.addWidget(self.log)

    def select_image(self):
        validation_dir = raw_data_dir(
            "remote_sensing", "sentinel2_multispectral", "inundation_hand_labeled"
        )
        initial_dir = str(validation_dir if validation_dir.exists() else standard_dialog_dir("raw", module_code="M04"))
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择遥感影像",
            initial_dir,
            "Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp);;All Files (*)",
        )
        if file_path:
            try:
                file_path = str(ensure_raw_source_path(file_path))
                self.current_image_path = file_path
                self.run_prediction(file_path)
            except Exception as exc:
                QMessageBox.warning(self, "输入路径错误", str(exc))

    def run_prediction(self, img_path: str):
        try:
            thresh = float(self.thresh_input.text().strip())
            if not (0.0 <= thresh <= 1.0):
                raise ValueError("阈值必须在 0 到 1 之间。")

            self.log.append(f"开始 SegFormer 7 通道淹没识别: {img_path}")
            self.log.append(f"阈值: {thresh:.2f}")

            group = _m04_validation_group(img_path)
            result_group = group or "business"
            output_root = module_processed_dir("M04", result_group, create=True)
            mask_path = output_root / "masks" / f"{Path(img_path).stem}_mask.png"
            overlay_path = output_root / "overlays" / f"{Path(img_path).stem}_overlay.png"
            result = self.predictor.predict(
                img_path,
                thresh=thresh,
                mask_path=mask_path,
                overlay_path=overlay_path,
            )
            self.last_result = result

            self.show_rgb_image(self.label_orig, result["original"])
            self.show_rgb_image(self.label_result, result["overlay"])

            self.log.append(f"设备: {result['device']}")
            self.log.append(f"淹没占比: {result['water_ratio'] * 100:.2f}%")
            self.log.append(f"掩膜输出: {result['mask_path']}")
            self.log.append(f"叠加图输出: {result['overlay_path']}")
            try:
                if self._is_validation_input(img_path):
                    validation_outputs = self._export_validation_result(result, img_path)
                    self.log.append(f"验证集叠加图: {validation_outputs['overlay']}")
                    self.log.append(f"验证集掩膜图: {validation_outputs['mask']}")
                else:
                    standard_outputs = self._export_standard_result(result, img_path)
                    self.log.append(f"标准淹没范围: {standard_outputs['inundation_tif']}")
                    self.log.append(f"标准统计报表: {standard_outputs['stats_xlsx']}")
            except Exception as export_exc:
                self.log.append(f"[WARN] 标准成果未导出: {export_exc}")
            self.log.append("识别完成\n")
        except Exception as e:
            self.log.append(f"[ERROR] {e}\n")
            QMessageBox.critical(self, "识别失败", str(e))

    def _is_validation_input(self, img_path: str | Path) -> bool:
        return _m04_validation_group(img_path) is not None

    def _export_validation_result(self, result: dict, img_path: str) -> dict[str, str]:
        img_path = str(ensure_raw_source_path(img_path))
        raw_image = Path(img_path)
        group = _m04_validation_group(raw_image)
        if group is None:
            raise ValueError(f"不是 M04 统一验证输入: {raw_image}")
        out_base = module_processed_dir("M04", group, create=True)
        overlay_dst = Path(result["overlay_path"])
        mask_dst = Path(result["mask_path"])

        common_extra = {
            "split": "validation",
            "crs": "not_applicable",
            "crs_name": "not_applicable",
            "data_status": "module_validation_result",
            "threshold": float(result.get("threshold", 0.5)),
            "water_ratio": float(result.get("water_ratio", 0.0)),
            "note": "淹没区识别原生验证集推理结果；样例区域不属于瓦赫什流域业务数据。",
            "validation_group": group,
        }
        write_metadata_sidecar(
            overlay_dst,
            module_code="M04",
            field="inundation_validation_overlay",
            source_files=[raw_image],
            extra={**common_extra, "output_type": "overlay_png"},
        )
        write_metadata_sidecar(
            mask_dst,
            module_code="M04",
            field="inundation_validation_mask",
            source_files=[raw_image],
            extra={**common_extra, "output_type": "mask_png"},
        )

        mark_module_complete(infer_run_context_from_path(raw_image), "M04")
        return {"overlay": str(overlay_dst), "mask": str(mask_dst)}

    def _export_standard_result(self, result: dict, img_path: str) -> dict[str, str]:
        img_path = str(ensure_raw_source_path(img_path))
        context = infer_run_context_from_path(img_path)
        inundation_tif = module_output_path("M04", context=context, output_index=0)
        stats_xlsx = module_output_path("M04", context=context, output_index=1)

        mask = np.asarray(result["mask"], dtype=np.uint8)
        with rasterio.open(img_path) as src:
            if src.crs is None:
                raise ValueError("输入影像缺少 CRS，不能导出为标准 GeoTIFF。")

            if mask.shape != (src.height, src.width):
                nearest = getattr(Image, "Resampling", Image).NEAREST
                mask = np.asarray(
                    Image.fromarray(mask).resize((src.width, src.height), nearest),
                    dtype=np.uint8,
                )

            src_transform = src.transform
            src_crs = src.crs
            dst_array = mask
            dst_transform = src_transform
            dst_crs = src_crs
            dst_width = src.width
            dst_height = src.height

            if src_crs.to_string() != TARGET_CRS:
                dst_transform, dst_width, dst_height = calculate_default_transform(
                    src_crs,
                    TARGET_CRS,
                    src.width,
                    src.height,
                    *src.bounds,
                )
                reprojected = np.zeros((dst_height, dst_width), dtype=np.uint8)
                reproject(
                    source=mask,
                    destination=reprojected,
                    src_transform=src_transform,
                    src_crs=src_crs,
                    dst_transform=dst_transform,
                    dst_crs=TARGET_CRS,
                    resampling=Resampling.nearest,
                    src_nodata=0,
                    dst_nodata=0,
                )
                dst_array = reprojected
                dst_crs = TARGET_CRS

            profile = src.profile.copy()

        inundation_tif.parent.mkdir(parents=True, exist_ok=True)
        profile.update(
            driver="GTiff",
            height=dst_height,
            width=dst_width,
            count=1,
            dtype="uint8",
            crs=dst_crs,
            transform=dst_transform,
            nodata=0,
            compress="lzw",
        )
        with rasterio.open(inundation_tif, "w", **profile) as dst:
            dst.write(dst_array.astype(np.uint8), 1)

        pixel_area_km2 = abs(float(dst_transform.a) * float(dst_transform.e)) / 1_000_000.0
        inundated_pixels = int(np.count_nonzero(dst_array))
        total_pixels = int(dst_array.size)
        inundated_area_km2 = inundated_pixels * pixel_area_km2
        rows = [
            ["date", period_to_date(context.period), ""],
            ["period", context.period, ""],
            ["scheme", context.scheme, ""],
            ["module_code", "M04", ""],
            ["threshold", float(result.get("threshold", 0.0)), ""],
            ["inundated_pixels", inundated_pixels, "pixel"],
            ["total_pixels", total_pixels, "pixel"],
            ["pixel_area_km2", pixel_area_km2, "km2"],
            ["inundated_area_km2", inundated_area_km2, "km2"],
            ["water_ratio_pct", float(result.get("water_ratio", 0.0)) * 100.0, "%"],
            ["source_image", os.path.abspath(img_path), ""],
        ]
        _write_simple_xlsx(stats_xlsx, [["field", "value", "unit"], *rows])

        write_metadata_sidecar(
            inundation_tif,
            module_code="M04",
            field="inundation",
            source_files=[img_path],
            extra={
                "scheme": context.scheme,
                "scheme_name": context.scheme_name,
                "period": context.period,
                "period_name": context.period_name,
                "threshold": float(result.get("threshold", 0.0)),
                "water_ratio": float(result.get("water_ratio", 0.0)),
            },
        )
        write_metadata_sidecar(
            stats_xlsx,
            module_code="M04",
            field="inundated_area",
            source_files=[img_path],
            extra={
                "scheme": context.scheme,
                "scheme_name": context.scheme_name,
                "period": context.period,
                "period_name": context.period_name,
                "intermediate_outputs": [str(inundation_tif)],
            },
        )
        mark_module_complete(context, "M04")
        return {
            "inundation_tif": str(inundation_tif),
            "stats_xlsx": str(stats_xlsx),
        }

    def open_overlay_file(self):
        self.open_result_file("overlay_path")

    def open_mask_file(self):
        self.open_result_file("mask_path")

    def open_result_file(self, key: str):
        if not self.last_result:
            QMessageBox.warning(self, "提示", "请先运行识别。")
            return
        path = self.last_result[key]
        if os.path.exists(path):
            os.startfile(path)
        else:
            QMessageBox.warning(self, "文件不存在", path)

    def show_rgb_image(self, label: ImageLabel, img: np.ndarray):
        arr = np.ascontiguousarray(img.astype(np.uint8))
        qimg = QImage(arr.data, arr.shape[1], arr.shape[0], arr.shape[1] * 3, QImage.Format_RGB888).copy()
        label.set_qimage(qimg)
