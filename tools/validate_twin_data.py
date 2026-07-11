from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.digital_twin_standard import (
    DEFAULT_TWIN_DATA_ROOT,
    MODULE_DIRECTORY_NAMES,
    RAW_DATA_CATEGORIES,
)


REPORT_PATH = PROJECT_ROOT / "reports" / "twin_data_validation_report.json"
EXPECTED_ROOT_DIRS = {"baseline", "raw", "processed"}
ACTIVE_MODULES = ("M01", "M02", "M04", "M05", "M06", "M07", "M08", "M09")
ACTIVE_MODULE_DIRS = {MODULE_DIRECTORY_NAMES[code]: code for code in ACTIVE_MODULES}
META_SUFFIX = ".meta.json"

RAW_REQUIRED_FIELDS = {
    "data_type",
    "dataset_name",
    "source_files",
    "source_origin",
    "consumer_modules",
    "split",
    "sensor",
    "bands",
    "dtype",
    "crs",
    "date",
    "is_module_native",
    "checksum",
    "created_at",
}
PROCESSED_REQUIRED_FIELDS = {
    "module_code",
    "output_type",
    "source_files",
    "model_weight",
    "threshold_or_config",
    "shape",
    "dtype",
    "crs",
    "checksum",
    "created_at",
}

EXPECTED_OUTPUTS = {
    "M01": (
        "snow/val/masks/*.png",
        "snow/val/overlays/*.png",
        "water/val/masks/*.png",
        "water/val/overlays/*.png",
        "validation_metrics.csv",
    ),
    "M02": (
        "rasters/SWE_mm_*.tif",
        "rasters/Snowmelt_mm_day_*.tif",
        "tables/daily_basin_series.csv",
    ),
    "M04": (
        "sentinel1_sar_weak/masks/*.png",
        "sentinel1_sar_weak/overlays/*.png",
        "sentinel2_optical_hand/masks/*.png",
        "sentinel2_optical_hand/overlays/*.png",
    ),
    "M05": ("tables/frame_pair_velocity.csv",),
    "M06": (
        "rasters/*_snow_type.tif",
        "rasters/*_snow_density_gcm3.tif",
        "tables/*_snow_state_statistics.csv",
    ),
    "M07": (
        "rasters/flood_risk_level.tif",
        "rasters/flood_risk_index.tif",
        "tables/landuse_risk_stats.csv",
        "tables/final_weights.txt",
        "tables/landuse_risk_summary.txt",
        "visualizations/flood_risk_map.html",
    ),
    "M08": (
        "tables/allocation_plan.csv",
        "tables/allocation_summary.csv",
    ),
    "M09": (
        "tables/reservoir_storage.csv",
        "tables/estimation_summary.json",
    ),
}


@dataclass
class ValidationReport:
    root: str
    stage: str
    created_at: str = field(
        default_factory=lambda: datetime.now().astimezone().isoformat(timespec="seconds")
    )
    checks: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    rounds: dict = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return not self.errors

    def ok(self, message: str) -> None:
        self.checks.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def error(self, message: str) -> None:
        self.errors.append(message)

    def to_dict(self) -> dict:
        return {**asdict(self), "success": self.success}

    def print(self) -> None:
        print(f"数据根目录: {self.root}")
        print(f"验证阶段: {self.stage}")
        print(f"检查通过: {len(self.checks)}")
        print(f"警告: {len(self.warnings)}")
        print(f"错误: {len(self.errors)}")
        for item in self.warnings:
            print(f"[WARN] {item}")
        for item in self.errors:
            print(f"[ERROR] {item}")
        print("验证结果: " + ("PASS" if self.success else "FAIL"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _payload_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and not path.name.endswith(META_SUFFIX) and path.name != "finish.tag"
    )


def _sidecar(path: Path) -> Path:
    return path.with_suffix(path.suffix + META_SUFFIX)


def _load_json(report: ValidationReport, path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        report.error(f"JSON 无法读取: {path}: {exc}")
        return None
    if not isinstance(payload, dict):
        report.error(f"JSON 顶层必须是对象: {path}")
        return None
    return payload


def _check_sidecar(
    report: ValidationReport,
    path: Path,
    *,
    kind: str,
    expected_module: str | None = None,
    root: Path,
) -> dict | None:
    metadata_path = _sidecar(path)
    if not metadata_path.is_file():
        report.error(f"缺少元数据 sidecar: {path}")
        return None
    metadata = _load_json(report, metadata_path)
    if metadata is None:
        return None

    required = RAW_REQUIRED_FIELDS if kind == "raw" else PROCESSED_REQUIRED_FIELDS
    missing = sorted(required - set(metadata))
    if missing:
        report.error(f"元数据字段缺失 {missing}: {metadata_path}")

    expected_checksum = metadata.get("checksum")
    actual_checksum = _sha256(path)
    if expected_checksum != actual_checksum:
        report.error(
            f"checksum 不一致: {path}; metadata={expected_checksum}, actual={actual_checksum}"
        )

    if kind == "raw":
        consumers = metadata.get("consumer_modules")
        if not isinstance(consumers, list) or not consumers:
            report.error(f"raw consumer_modules 必须为非空列表: {metadata_path}")
        invalid_consumers = sorted(set(consumers or []) - set(ACTIVE_MODULES))
        if invalid_consumers:
            report.error(f"raw 包含非活动模块 consumer {invalid_consumers}: {metadata_path}")
        is_field = bool(metadata.get("is_field_observation"))
        if is_field and not _is_under(path, root / "raw" / "video" / "river_velocity"):
            report.error(f"非河道视频被错误标记为现地实测: {metadata_path}")
    else:
        if metadata.get("module_code") != expected_module:
            report.error(
                f"processed module_code 错误: {metadata_path}; "
                f"expected={expected_module}, actual={metadata.get('module_code')}"
            )
        sources = metadata.get("source_files")
        if not isinstance(sources, list) or not sources:
            report.error(f"processed source_files 必须为非空列表: {metadata_path}")
        for source_text in sources or []:
            if "://" in str(source_text):
                continue
            source = Path(source_text).expanduser()
            if not source.exists():
                report.error(f"processed 来源文件不存在: {metadata_path} -> {source}")
                continue
            if _is_under(source, root / "processed"):
                report.error(f"独立模块不得读取其他 processed 成果: {metadata_path} -> {source}")
            if not (_is_under(source, root / "raw") or _is_under(source, root / "baseline")):
                report.error(f"processed 正式输入未统一到 raw/baseline: {metadata_path} -> {source}")

        model_weight = metadata.get("model_weight")
        if model_weight and not Path(model_weight).is_file():
            report.error(f"模型权重不存在: {metadata_path} -> {model_weight}")
    return metadata


def _check_root_structure(report: ValidationReport, root: Path) -> None:
    if not root.is_dir():
        report.error(f"数据根目录不存在: {root}")
        return
    children = {path.name for path in root.iterdir() if path.is_dir()}
    extra_dirs = sorted(children - EXPECTED_ROOT_DIRS)
    missing_dirs = sorted(EXPECTED_ROOT_DIRS - children)
    if extra_dirs:
        report.error(f"数据根目录存在非标准目录: {extra_dirs}")
    if missing_dirs:
        report.error(f"数据根目录缺少目录: {missing_dirs}")
    root_files = sorted(path.name for path in root.iterdir() if path.is_file())
    if root_files:
        report.error(f"数据根目录不得直接放文件: {root_files}")

    raw_root = root / "raw"
    raw_dirs = {path.name for path in raw_root.iterdir() if path.is_dir()} if raw_root.exists() else set()
    invalid_raw = sorted(raw_dirs - set(RAW_DATA_CATEGORIES))
    if invalid_raw:
        report.error(f"raw 一级目录必须是数据类型，发现: {invalid_raw}")
    for directory in raw_root.rglob("*") if raw_root.exists() else []:
        if directory.is_dir() and directory.name.lower() == "test":
            if not _payload_files(directory):
                report.error(f"不得创建空 test 目录冒充数据: {directory}")

    processed_root = root / "processed"
    processed_dirs = (
        {path.name for path in processed_root.iterdir() if path.is_dir()}
        if processed_root.exists()
        else set()
    )
    invalid_processed = sorted(processed_dirs - set(ACTIVE_MODULE_DIRS))
    if invalid_processed:
        report.error(f"processed 一级目录必须是活动模块，发现: {invalid_processed}")
    forbidden_tokens = ("scheme", "工况", "module_validation", "M03", "discharge")
    for path in processed_root.rglob("*") if processed_root.exists() else []:
        relative = path.relative_to(processed_root).as_posix()
        if any(token.lower() in relative.lower() for token in forbidden_tokens):
            report.error(f"processed 中残留旧方案、验证层或 M03 成果: {relative}")
    report.ok("第一轮：根目录、raw 类型层和 processed 模块层结构检查完成")


def _check_baseline(report: ValidationReport, root: Path) -> None:
    baseline = root / "baseline"
    required = [baseline / "DEM.tif", baseline / "流域边界.shp", baseline / "河网.shp", baseline / "水库边界.shp"]
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            report.error(f"baseline 固定基准缺失或为空: {path}")
            continue
        if path.suffix.lower() == ".shp":
            for suffix in (".dbf", ".shx", ".prj"):
                companion = path.with_suffix(suffix)
                if not companion.is_file() or companion.stat().st_size == 0:
                    report.error(f"Shapefile 组件缺失或为空: {companion}")
        if not _sidecar(path).is_file():
            report.error(f"baseline 主文件缺少来源元数据: {_sidecar(path)}")
    report.ok("第一轮：baseline 固定基准检查完成")


def _check_raw(report: ValidationReport, root: Path) -> None:
    files = _payload_files(root / "raw")
    checksums: dict[str, list[Path]] = defaultdict(list)
    for path in files:
        if path.stat().st_size == 0:
            report.error(f"raw 文件为空: {path}")
            continue
        checksums[_sha256(path)].append(path)
        _check_sidecar(report, path, kind="raw", root=root)
        _check_file_content(report, path, raw=True)

    duplicate_groups = [paths for paths in checksums.values() if len(paths) > 1]
    for paths in duplicate_groups:
        report.warn("raw 存在相同 checksum，请确认是否必须保留多份: " + ", ".join(map(str, paths)))

    for metadata_path in (root / "raw").rglob(f"*{META_SUFFIX}"):
        target = Path(str(metadata_path)[: -len(META_SUFFIX)])
        if not target.is_file():
            report.error(f"raw 存在孤立元数据: {metadata_path}")
    report.stats["raw_payload_files"] = len(files)
    report.stats["raw_duplicate_checksum_groups"] = len(duplicate_groups)
    report.ok("第一轮：raw 非空、sidecar、checksum、真实性标记检查完成")


def _check_csv(report: ValidationReport, path: Path) -> tuple[list[str], list[dict]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            headers = list(reader.fieldnames or [])
            rows = list(reader)
    except Exception as exc:
        report.error(f"CSV 无法读取: {path}: {exc}")
        return [], []
    if not headers or any(not str(item).strip() for item in headers):
        report.error(f"CSV 表头无效: {path}")
    if not rows:
        report.error(f"CSV 没有数据行: {path}")
    return headers, rows


def _check_tif(report: ValidationReport, path: Path) -> tuple[int, int] | None:
    try:
        import numpy as np
        import rasterio

        with rasterio.open(path) as dataset:
            if dataset.width <= 0 or dataset.height <= 0 or dataset.count <= 0:
                report.error(f"GeoTIFF 尺寸或波段无效: {path}")
                return None
            if dataset.crs is None:
                report.error(f"GeoTIFF 缺少 CRS: {path}")
            values = dataset.read(masked=True)
            valid = values.compressed()
            if valid.size == 0:
                report.error(f"GeoTIFF 没有有效像元: {path}")
            elif np.issubdtype(valid.dtype, np.floating) and not np.isfinite(valid).all():
                report.error(f"GeoTIFF 有效像元包含 NaN/Inf: {path}")
            elif float(valid.max()) == float(valid.min()):
                report.warn(f"GeoTIFF 有效像元为单一值，请核查成果质量: {path}")
            return dataset.width, dataset.height
    except Exception as exc:
        report.error(f"GeoTIFF 无法读取: {path}: {exc}")
        return None


def _check_image(report: ValidationReport, path: Path) -> tuple[int, int] | None:
    try:
        import numpy as np
        from PIL import Image

        with Image.open(path) as image:
            image.load()
            if image.width <= 0 or image.height <= 0:
                report.error(f"图片尺寸无效: {path}")
                return None
            values = np.asarray(image)
            if values.size == 0:
                report.error(f"图片没有像素: {path}")
            elif "mask" in path.stem.lower() and np.unique(values).size == 1:
                report.warn(f"mask 只有一个类别，请核查模型效果: {path}")
            return image.width, image.height
    except Exception as exc:
        report.error(f"图片无法读取: {path}: {exc}")
        return None


def _check_video(report: ValidationReport, path: Path) -> dict | None:
    try:
        import cv2

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            report.error(f"视频无法打开: {path}")
            return None
        info = {
            "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": float(capture.get(cv2.CAP_PROP_FPS)),
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
        capture.release()
        if info["frame_count"] < 2 or info["fps"] <= 0 or info["width"] <= 0 or info["height"] <= 0:
            report.error(f"视频帧数、FPS 或尺寸无效: {path}: {info}")
        return info
    except Exception as exc:
        report.error(f"视频检查失败: {path}: {exc}")
        return None


def _check_file_content(report: ValidationReport, path: Path, *, raw: bool = False) -> None:
    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        _check_tif(report, path)
    elif suffix in {".png", ".jpg", ".jpeg", ".bmp"}:
        _check_image(report, path)
    elif suffix == ".csv":
        _check_csv(report, path)
    elif suffix in {".mp4", ".avi", ".mov"}:
        _check_video(report, path)
    elif suffix == ".json":
        _load_json(report, path)
    elif suffix in {".txt", ".html"}:
        try:
            if not path.read_text(encoding="utf-8-sig").strip():
                report.error(f"文本成果为空: {path}")
        except Exception as exc:
            report.error(f"文本成果无法读取: {path}: {exc}")


def _check_output_source_shape(
    report: ValidationReport, path: Path, metadata: dict | None
) -> None:
    if metadata is None or path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        return
    if not ({"masks", "overlays"} & set(path.parts)):
        return
    output_size = _check_image(report, path)
    if output_size is None:
        return
    for source_text in metadata.get("source_files") or []:
        source = Path(source_text)
        if not source.is_file():
            continue
        if source.suffix.lower() in {".tif", ".tiff"}:
            source_size = _check_tif(report, source)
        elif source.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
            source_size = _check_image(report, source)
        else:
            continue
        if source_size != output_size:
            report.error(f"mask/overlay 与输入尺寸不一致: {path}={output_size}, {source}={source_size}")
        return


def _check_finish_tag(report: ValidationReport, module_dir: Path, module_code: str) -> None:
    tag = module_dir / "finish.tag"
    if not tag.is_file() or tag.stat().st_size == 0:
        report.error(f"模块缺少完成标记: {tag}")
        return
    payload = _load_json(report, tag)
    if payload is None:
        return
    if payload.get("module_code") != module_code:
        report.error(f"finish.tag 模块编号错误: {tag}")
    recorded = set(payload.get("completed_outputs") or [])
    actual = {
        path.relative_to(module_dir).as_posix()
        for path in _payload_files(module_dir)
    }
    if recorded != actual:
        report.error(
            f"finish.tag 成果清单与磁盘不一致: {tag}; "
            f"missing={sorted(actual - recorded)}, stale={sorted(recorded - actual)}"
        )


def _check_m01_counts(report: ValidationReport, root: Path) -> None:
    raw_base = root / "raw" / "remote_sensing" / "optical_rgb"
    output_base = root / "processed" / MODULE_DIRECTORY_NAMES["M01"]
    for task in ("snow", "water"):
        task_raw = raw_base / f"segformer_{task}"
        if not task_raw.exists():
            report.error(f"M01 缺少统一 raw: {task_raw}")
            continue
        for split_dir in sorted(path for path in task_raw.iterdir() if path.is_dir()):
            images = _payload_files(split_dir / "images")
            if not images:
                report.error(f"M01 {task}/{split_dir.name} 没有有效影像")
                continue
            masks = list((output_base / task / split_dir.name / "masks").glob("*.png"))
            overlays = list((output_base / task / split_dir.name / "overlays").glob("*.png"))
            if len(masks) != len(images) or len(overlays) != len(images):
                report.error(
                    f"M01 {task}/{split_dir.name} 输入输出数量不一致: "
                    f"images={len(images)}, masks={len(masks)}, overlays={len(overlays)}"
                )


def _check_m04_counts(report: ValidationReport, root: Path) -> None:
    pairs = (
        ("sentinel1_sar/inundation_weak_labeled", "sentinel1_sar_weak"),
        ("sentinel2_multispectral/inundation_hand_labeled", "sentinel2_optical_hand"),
    )
    for raw_suffix, output_suffix in pairs:
        inputs = [
            path
            for path in _payload_files(root / "raw" / "remote_sensing" / raw_suffix)
            if path.suffix.lower() in {".tif", ".tiff"}
        ]
        output_root = root / "processed" / MODULE_DIRECTORY_NAMES["M04"] / output_suffix
        masks = list((output_root / "masks").glob("*.png"))
        overlays = list((output_root / "overlays").glob("*.png"))
        if not inputs or len(masks) != len(inputs) or len(overlays) != len(inputs):
            report.error(
                f"M04 {raw_suffix} 输入输出数量不一致: "
                f"inputs={len(inputs)}, masks={len(masks)}, overlays={len(overlays)}"
            )


def _check_m05_csv(report: ValidationReport, root: Path) -> None:
    path = root / "processed" / MODULE_DIRECTORY_NAMES["M05"] / "tables" / "frame_pair_velocity.csv"
    headers, rows = _check_csv(report, path)
    required = {
        "frame_index",
        "timestamp_s",
        "velocity_px_frame",
        "velocity_m_s",
        "valid_pixel_count",
        "confidence",
        "source_video",
    }
    if not required.issubset(headers):
        report.error(f"M05 CSV 字段缺失: {sorted(required - set(headers))}")
        return
    if any(str(row.get("velocity_m_s", "")).strip() for row in rows):
        report.error("M05 缺少可信空间标定，velocity_m_s 必须全部留空")
    valid_pixel_rows = [row for row in rows if str(row.get("velocity_px_frame", "")).strip()]
    if not valid_pixel_rows:
        report.error("M05 没有任何有效像素速度记录")
    videos = {str(row.get("source_video", "")).strip() for row in rows if row.get("source_video")}
    if len(videos) != 1:
        report.error(f"M05 CSV source_video 不唯一: {sorted(videos)}")
        return
    video = Path(next(iter(videos)))
    info = _check_video(report, video)
    if info and len(rows) != info["frame_count"] - 1:
        report.error(
            f"M05 未处理完整视频: CSV={len(rows)} 帧对, 视频={info['frame_count']} 帧"
        )


def _check_module_specific_rules(report: ValidationReport, root: Path) -> None:
    _check_m01_counts(report, root)
    _check_m04_counts(report, root)
    _check_m05_csv(report, root)

    processed = root / "processed"
    for forbidden in processed.rglob("*"):
        if not forbidden.is_file():
            continue
        name = forbidden.name.lower()
        if "runoff" in name:
            report.error(f"M02 当前没有径流算法，不得生成 runoff: {forbidden}")
        if "outflow" in name and _is_under(forbidden, processed / MODULE_DIRECTORY_NAMES["M09"]):
            report.error(f"M09 当前没有独立出库流量算法，不得生成 outflow: {forbidden}")

    m08_meta_files = (processed / MODULE_DIRECTORY_NAMES["M08"]).rglob(f"*{META_SUFFIX}")
    for metadata_path in m08_meta_files:
        metadata = _load_json(report, metadata_path)
        if metadata and any("M09" in str(source) for source in metadata.get("source_files") or []):
            report.error(f"M08 仍依赖 M09: {metadata_path}")


def _check_processed(report: ValidationReport, root: Path) -> None:
    processed = root / "processed"
    total_outputs = 0
    module_stats = {}
    module_outputs: dict[str, list[Path]] = {}
    contract_error_count = len(report.errors)
    for module_code in ACTIVE_MODULES:
        module_dir = processed / MODULE_DIRECTORY_NAMES[module_code]
        if not module_dir.is_dir():
            report.error(f"缺少模块成果目录: {module_dir}")
            continue
        for pattern in EXPECTED_OUTPUTS[module_code]:
            matches = [path for path in module_dir.glob(pattern) if path.is_file() and path.stat().st_size > 0]
            if not matches:
                report.error(f"{module_code} 缺少预期实际成果: {module_dir / pattern}")

        outputs = _payload_files(module_dir)
        module_outputs[module_code] = outputs
        module_stats[module_code] = len(outputs)
        total_outputs += len(outputs)
        _check_finish_tag(report, module_dir, module_code)
        for path in outputs:
            if path.stat().st_size == 0:
                report.error(f"processed 成果为空: {path}")
                continue
            metadata = _check_sidecar(
                report,
                path,
                kind="processed",
                expected_module=module_code,
                root=root,
            )

        for metadata_path in module_dir.rglob(f"*{META_SUFFIX}"):
            target = Path(str(metadata_path)[: -len(META_SUFFIX)])
            if not target.is_file():
                report.error(f"processed 存在孤立元数据: {metadata_path}")

    report.stats["processed_outputs"] = total_outputs
    report.stats["module_output_counts"] = module_stats
    report.ok("第二轮：八个独立模块的成果、来源链和完成标记检查完成")
    report.rounds["round_2_module_outputs"] = {
        "passed": len(report.errors) == contract_error_count,
        "new_errors": len(report.errors) - contract_error_count,
    }

    quality_error_count = len(report.errors)
    for module_code, outputs in module_outputs.items():
        for path in outputs:
            metadata = _load_json(report, _sidecar(path)) if _sidecar(path).is_file() else None
            _check_file_content(report, path)
            _check_output_source_shape(report, path, metadata)
    _check_module_specific_rules(report, root)
    report.ok("第三轮：栅格、图片、CSV、视频和输入输出数量质量检查完成")
    report.rounds["round_3_output_quality"] = {
        "passed": len(report.errors) == quality_error_count,
        "new_errors": len(report.errors) - quality_error_count,
    }


def validate(root: Path, *, stage: str = "full") -> ValidationReport:
    root = root.expanduser().resolve()
    report = ValidationReport(root=str(root), stage=stage)

    before = len(report.errors)
    _check_root_structure(report, root)
    _check_baseline(report, root)
    _check_raw(report, root)
    report.rounds["round_1_structure_and_raw"] = {
        "passed": len(report.errors) == before,
        "new_errors": len(report.errors) - before,
    }

    if stage == "structure":
        return report
    if stage not in {"baseline-raw", "full"}:
        report.error(f"未知验证阶段: {stage}")
        return report
    if stage == "baseline-raw":
        return report

    _check_processed(report, root)
    return report


def _write_report(path: Path, report: ValidationReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="验证 baseline/raw/processed 独立模块数据架构")
    parser.add_argument(
        "root",
        nargs="?",
        default=str(DEFAULT_TWIN_DATA_ROOT),
        help="瓦赫什流域孪生数据根目录",
    )
    parser.add_argument(
        "--stage",
        choices=("structure", "baseline-raw", "full"),
        default="full",
        help="structure/baseline-raw 仅检查输入；full 检查八个模块实际成果",
    )
    parser.add_argument(
        "--report",
        default=str(REPORT_PATH),
        help="JSON 验证报告路径，必须位于 processed 之外",
    )
    args = parser.parse_args()

    report_path = Path(args.report).expanduser().resolve()
    root = Path(args.root).expanduser().resolve()
    if _is_under(report_path, root / "processed"):
        raise ValueError("验证报告不得写入 processed")
    report = validate(root, stage=args.stage)
    _write_report(report_path, report)
    report.print()
    print(f"验证报告: {report_path}")
    return 0 if report.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
