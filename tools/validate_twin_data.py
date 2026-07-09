from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import rasterio

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.digital_twin_standard import (  # noqa: E402
    DATE_FIELD,
    DEFAULT_TWIN_DATA_ROOT,
    MODULE_SPECS,
    TARGET_CRS,
)


REQUIRED_BASELINE_FILES = (
    "流域边界.shp",
    "流域边界.shx",
    "流域边界.dbf",
    "流域边界.prj",
    "河网.shp",
    "河网.shx",
    "河网.dbf",
    "河网.prj",
    "水库边界.shp",
    "水库边界.shx",
    "水库边界.dbf",
    "水库边界.prj",
    "DEM.tif",
)

REQUIRED_RAW_PERIOD_FILES = {
    "200503_融雪期": (
        "200503_哨兵影像.tif",
        "200503_SAR影像.tif",
        "200503_逐日气象.csv",
        "200503_水库参数.csv",
        "200503_河道视频.mp4",
    ),
    "201707_汛期": (
        "201707_哨兵影像.tif",
        "201707_逐日气象.csv",
        "201707_水库参数.csv",
    ),
}

REQUIRED_PROCESSED_PERIODS = (
    "processed/scheme01_常规调度工况/200503_融雪模拟",
    "processed/scheme01_常规调度工况/201707_汛期模拟",
    "processed/scheme02_优化分水工况/200503_融雪模拟",
)


class ValidationReport:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.checked: list[str] = []

    def ok(self, message: str) -> None:
        self.checked.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def error(self, message: str) -> None:
        self.errors.append(message)

    def print(self) -> None:
        print("=== 瓦赫什流域孪生数据校验 ===")
        for item in self.checked:
            print(f"[OK] {item}")
        for item in self.warnings:
            print(f"[WARN] {item}")
        for item in self.errors:
            print(f"[ERROR] {item}")
        print(f"检查通过项: {len(self.checked)}")
        print(f"警告: {len(self.warnings)}")
        print(f"错误: {len(self.errors)}")

    @property
    def success(self) -> bool:
        return not self.errors


def _require_file(report: ValidationReport, path: Path, label: str) -> None:
    if path.exists() and path.is_file():
        report.ok(f"{label}: {path}")
    else:
        report.error(f"缺少文件 {label}: {path}")


def _require_dir(report: ValidationReport, path: Path, label: str) -> None:
    if path.exists() and path.is_dir():
        report.ok(f"{label}: {path}")
    else:
        report.error(f"缺少目录 {label}: {path}")


def _check_tif_crs(report: ValidationReport, path: Path) -> None:
    try:
        with rasterio.open(path) as ds:
            crs = ds.crs.to_string() if ds.crs else ""
            if crs == TARGET_CRS:
                report.ok(f"GeoTIFF CRS={TARGET_CRS}: {path.name}")
            else:
                report.error(f"GeoTIFF 坐标系不统一: {path}，当前 {crs or 'None'}，应为 {TARGET_CRS}")
    except Exception as exc:
        report.error(f"GeoTIFF 无法读取: {path} ({exc})")


def _check_csv_date(report: ValidationReport, path: Path) -> None:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            fields = reader.fieldnames or []
            if DATE_FIELD not in fields:
                report.error(f"CSV 缺少统一时间字段 {DATE_FIELD}: {path}")
                return
            first_row = next(reader, None)
            if first_row is None:
                report.warn(f"CSV 无数据行: {path}")
            else:
                report.ok(f"CSV 时间字段 {DATE_FIELD}: {path.name}")
    except Exception as exc:
        report.error(f"CSV 无法读取: {path} ({exc})")


def _check_finish_tag(report: ValidationReport, path: Path) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        modules = payload.get("completed_modules", [])
        if not isinstance(modules, list) or not modules:
            report.error(f"finish.tag 缺少 completed_modules: {path}")
            return
        report.ok(f"finish.tag 完成模块 {len(modules)} 个: {path.parent.name}")
    except Exception as exc:
        report.error(f"finish.tag 无法读取: {path} ({exc})")


def _expected_outputs_for_period(period_token: str) -> tuple[str, ...]:
    outputs: list[str] = []
    for spec in MODULE_SPECS:
        for pattern in spec.outputs:
            outputs.append(pattern.format(period=period_token))
    return tuple(outputs)


def validate(root: Path) -> ValidationReport:
    report = ValidationReport()
    _require_dir(report, root, "孪生数据根目录")

    baseline = root / "baseline"
    raw = root / "raw"
    processed = root / "processed"
    _require_dir(report, baseline, "baseline")
    _require_dir(report, raw, "raw")
    _require_dir(report, processed, "processed")

    for name in REQUIRED_BASELINE_FILES:
        _require_file(report, baseline / name, f"baseline/{name}")
    dem = baseline / "DEM.tif"
    if dem.exists():
        _check_tif_crs(report, dem)

    for period_dir_name, files in REQUIRED_RAW_PERIOD_FILES.items():
        period_dir = raw / period_dir_name
        _require_dir(report, period_dir, f"raw/{period_dir_name}")
        for name in files:
            path = period_dir / name
            _require_file(report, path, f"raw/{period_dir_name}/{name}")
            if path.suffix.lower() == ".tif" and path.exists():
                _check_tif_crs(report, path)
            if path.suffix.lower() == ".csv" and path.exists():
                _check_csv_date(report, path)

    for rel in REQUIRED_PROCESSED_PERIODS:
        period_dir = root / rel
        _require_dir(report, period_dir, rel)
        raster_dir = period_dir / "raster"
        table_dir = period_dir / "table"
        _require_dir(report, raster_dir, f"{rel}/raster")
        _require_dir(report, table_dir, f"{rel}/table")
        tag_path = period_dir / "finish.tag"
        _require_file(report, tag_path, f"{rel}/finish.tag")
        if tag_path.exists():
            _check_finish_tag(report, tag_path)

        period_token = period_dir.name.split("_", 1)[0]
        for output in _expected_outputs_for_period(period_token):
            path = period_dir / output
            _require_file(report, path, f"{rel}/{output}")
            if path.suffix.lower() == ".tif" and path.exists():
                _check_tif_crs(report, path)
            if path.suffix.lower() == ".csv" and path.exists():
                _check_csv_date(report, path)

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="校验瓦赫什流域数字孪生标准数据目录")
    parser.add_argument(
        "root",
        nargs="?",
        default=str(DEFAULT_TWIN_DATA_ROOT),
        help="瓦赫什流域孪生数据根目录，默认校验仓库 sample_data",
    )
    args = parser.parse_args()

    report = validate(Path(args.root))
    report.print()
    return 0 if report.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
