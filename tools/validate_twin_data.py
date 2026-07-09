from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import rasterio

try:
    import geopandas as gpd
except Exception:  # pragma: no cover - optional runtime dependency
    gpd = None

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

REQUIRED_METADATA_KEYS = (
    "file",
    "module_code",
    "field",
    "unit",
    "crs",
    "time_step",
    "date_field",
    "date_format",
    "source_files",
)

BOUND_TOLERANCE_M = 1.0


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


def _check_tif_crs(report: ValidationReport, path: Path):
    try:
        with rasterio.open(path) as ds:
            crs = ds.crs.to_string() if ds.crs else ""
            if crs == TARGET_CRS:
                report.ok(f"GeoTIFF CRS={TARGET_CRS}: {path.name}")
                return ds.bounds
            else:
                report.error(f"GeoTIFF 坐标系不统一: {path}，当前 {crs or 'None'}，应为 {TARGET_CRS}")
    except Exception as exc:
        report.error(f"GeoTIFF 无法读取: {path} ({exc})")
    return None


def _check_vector_crs(report: ValidationReport, path: Path):
    if gpd is None:
        report.warn(f"未安装 geopandas，跳过矢量 CRS 校验: {path.name}")
        return None

    try:
        gdf = gpd.read_file(path)
    except Exception as exc:
        report.error(f"矢量文件无法读取: {path} ({exc})")
        return None

    if gdf.empty:
        report.error(f"矢量文件无要素: {path}")
        return None

    epsg = gdf.crs.to_epsg() if gdf.crs is not None else None
    if epsg == 32642:
        report.ok(f"矢量 CRS={TARGET_CRS}: {path.name}")
    else:
        report.error(f"矢量坐标系不统一: {path}，当前 {gdf.crs or 'None'}，应为 {TARGET_CRS}")
        return None

    return tuple(float(value) for value in gdf.total_bounds)


def _bounds_within(inner, outer, tolerance: float = BOUND_TOLERANCE_M) -> bool:
    return (
        inner[0] >= outer[0] - tolerance
        and inner[1] >= outer[1] - tolerance
        and inner[2] <= outer[2] + tolerance
        and inner[3] <= outer[3] + tolerance
    )


def _bounds_tuple(bounds) -> tuple[float, float, float, float]:
    if hasattr(bounds, "left"):
        return (float(bounds.left), float(bounds.bottom), float(bounds.right), float(bounds.top))
    return tuple(float(value) for value in bounds)


def _check_bounds_within_watershed(
    report: ValidationReport,
    path: Path,
    bounds,
    watershed_bounds: tuple[float, float, float, float] | None,
) -> None:
    if bounds is None or watershed_bounds is None:
        return
    inner = _bounds_tuple(bounds)
    if _bounds_within(inner, watershed_bounds):
        report.ok(f"范围位于流域边界内: {path.name}")
    else:
        report.error(
            "数据范围超出流域边界: "
            f"{path}，当前 {inner}，流域边界 {watershed_bounds}"
        )


def _check_metadata_sidecar(report: ValidationReport, path: Path, module_code: str | None = None) -> None:
    metadata_path = path.with_suffix(path.suffix + ".meta.json")
    if not metadata_path.exists():
        report.error(f"缺少元数据说明: {metadata_path}")
        return

    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as exc:
        report.error(f"元数据无法读取: {metadata_path} ({exc})")
        return

    missing = [key for key in REQUIRED_METADATA_KEYS if key not in payload]
    if missing:
        report.error(f"元数据缺少字段 {missing}: {metadata_path}")
        return

    if payload.get("file") != path.name:
        report.error(f"元数据 file 与成果文件名不一致: {metadata_path}")
        return
    if module_code and payload.get("module_code") != module_code:
        report.error(f"元数据 module_code 不一致: {metadata_path}，应为 {module_code}")
        return
    if payload.get("crs") != TARGET_CRS:
        report.error(f"元数据 CRS 不统一: {metadata_path}，当前 {payload.get('crs')}，应为 {TARGET_CRS}")
        return
    if payload.get("time_step") != "daily":
        report.error(f"元数据 time_step 不统一: {metadata_path}，应为 daily")
        return
    if payload.get("date_field") != DATE_FIELD:
        report.error(f"元数据 date_field 不统一: {metadata_path}，应为 {DATE_FIELD}")
        return
    if payload.get("date_format") != "YYYY-MM-DD":
        report.error(f"元数据 date_format 不统一: {metadata_path}")
        return
    if not isinstance(payload.get("source_files"), list) or not payload["source_files"]:
        report.error(f"元数据 source_files 不能为空: {metadata_path}")
        return

    report.ok(f"元数据说明完整: {metadata_path.name}")


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


def _module_code_for_output(relative_output: str) -> str | None:
    normalized = relative_output.replace("\\", "/")
    period_token = Path(normalized).name.split("_", 1)[0]
    for spec in MODULE_SPECS:
        for pattern in spec.outputs:
            if pattern.format(period=period_token) == normalized:
                return spec.code
    return None


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
    watershed_bounds = None
    watershed_shp = baseline / "流域边界.shp"
    if watershed_shp.exists():
        watershed_bounds = _check_vector_crs(report, watershed_shp)
    for name in ("河网.shp", "水库边界.shp"):
        path = baseline / name
        if path.exists():
            bounds = _check_vector_crs(report, path)
            _check_bounds_within_watershed(report, path, bounds, watershed_bounds)
    dem = baseline / "DEM.tif"
    if dem.exists():
        bounds = _check_tif_crs(report, dem)
        _check_bounds_within_watershed(report, dem, bounds, watershed_bounds)
        _check_metadata_sidecar(report, dem)

    for period_dir_name, files in REQUIRED_RAW_PERIOD_FILES.items():
        period_dir = raw / period_dir_name
        _require_dir(report, period_dir, f"raw/{period_dir_name}")
        for name in files:
            path = period_dir / name
            _require_file(report, path, f"raw/{period_dir_name}/{name}")
            if path.suffix.lower() == ".tif" and path.exists():
                bounds = _check_tif_crs(report, path)
                _check_bounds_within_watershed(report, path, bounds, watershed_bounds)
                _check_metadata_sidecar(report, path)
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
            module_code = _module_code_for_output(output)
            if path.suffix.lower() == ".tif" and path.exists():
                bounds = _check_tif_crs(report, path)
                _check_bounds_within_watershed(report, path, bounds, watershed_bounds)
                _check_metadata_sidecar(report, path, module_code=module_code)
            if path.suffix.lower() == ".csv" and path.exists():
                _check_csv_date(report, path)
                _check_metadata_sidecar(report, path, module_code=module_code)
            if path.suffix.lower() == ".xlsx" and path.exists():
                _check_metadata_sidecar(report, path, module_code=module_code)

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
