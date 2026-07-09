from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
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
    DATE_FORMAT,
    DEFAULT_TWIN_DATA_ROOT,
    MODULE_SPECS,
    SAMPLE_TWIN_DATA_ROOT,
    STANDARD_FIELDS,
    STUDY_YEAR_END,
    STUDY_YEAR_START,
    TARGET_CRS,
    TIME_STEP,
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

BASELINE_RAW_METADATA_KEYS = (
    "file",
    "data_role",
    "data_status",
    "source_name",
    "source_files",
    "crs",
    "created_at",
)

BOUND_TOLERANCE_M = 1.0
PY_DATE_FORMAT = "%Y-%m-%d"
SIMULATED_STATUS_VALUES = {"demo", "demo_only", "mock", "sample", "simulated", "simulation"}


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


def _bounds_cover(outer, inner, tolerance: float = BOUND_TOLERANCE_M) -> bool:
    return _bounds_within(inner, outer, tolerance)


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


def _check_bounds_cover_watershed(
    report: ValidationReport,
    path: Path,
    bounds,
    watershed_bounds: tuple[float, float, float, float] | None,
) -> None:
    if bounds is None or watershed_bounds is None:
        return
    outer = _bounds_tuple(bounds)
    if _bounds_cover(outer, watershed_bounds):
        report.ok(f"范围覆盖流域边界: {path.name}")
    else:
        report.error(
            "数据范围未覆盖完整流域边界: "
            f"{path}，当前 {outer}，流域边界 {watershed_bounds}"
        )


def _is_sample_root(root: Path) -> bool:
    try:
        return root.resolve() == SAMPLE_TWIN_DATA_ROOT.resolve()
    except Exception:
        return False


def _check_year_in_study_range(report: ValidationReport, year: int, label: str) -> bool:
    if STUDY_YEAR_START <= year <= STUDY_YEAR_END:
        return True
    report.error(f"时间超出统一研究时段 {STUDY_YEAR_START}-{STUDY_YEAR_END}: {label}")
    return False


def _parse_date(report: ValidationReport, value: str, label: str):
    try:
        parsed = datetime.strptime(str(value).strip(), PY_DATE_FORMAT).date()
    except Exception:
        report.error(f"日期格式不统一，应为 {DATE_FORMAT}: {label}={value!r}")
        return None
    _check_year_in_study_range(report, parsed.year, f"{label}={value}")
    return parsed


def _check_period_token(report: ValidationReport, token: str, label: str) -> None:
    if not token.isdigit() or len(token) not in (6, 8):
        report.error(f"时段命名不规范，应为 YYYYMM 或 YYYYMMDD: {label}")
        return

    year = int(token[:4])
    month = int(token[4:6])
    if not 1 <= month <= 12:
        report.error(f"时段月份不合法: {label}")
        return
    if not _check_year_in_study_range(report, year, label):
        return

    if len(token) == 8:
        _parse_date(report, f"{token[:4]}-{token[4:6]}-{token[6:8]}", label)
    else:
        report.ok(f"时段位于统一研究时段: {label}")


def _period_token_from_folder(report: ValidationReport, folder: Path, label: str) -> str | None:
    token = folder.name.split("_", 1)[0]
    if "_" not in folder.name:
        report.error(f"{label} 文件夹命名不规范，应为 年月_时段名称: {folder}")
        return None
    _check_period_token(report, token, f"{label}/{folder.name}")
    return token


def _check_metadata_sidecar(
    report: ValidationReport,
    path: Path,
    module_code: str | None = None,
    *,
    allow_demo_only: bool = True,
    strict_module: bool = True,
) -> None:
    metadata_path = path.with_suffix(path.suffix + ".meta.json")
    if not metadata_path.exists():
        report.error(f"缺少元数据说明: {metadata_path}")
        return

    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as exc:
        report.error(f"元数据无法读取: {metadata_path} ({exc})")
        return

    required_keys = REQUIRED_METADATA_KEYS if strict_module else BASELINE_RAW_METADATA_KEYS
    missing = [key for key in required_keys if key not in payload]
    if missing:
        if allow_demo_only and not strict_module:
            report.warn(f"样例元数据为旧格式，仅作为模板保留: {metadata_path}，缺少 {missing}")
            return
        report.error(f"元数据缺少字段 {missing}: {metadata_path}")
        return

    if payload.get("file") != path.name:
        report.error(f"元数据 file 与成果文件名不一致: {metadata_path}")
        return
    if payload.get("demo_only") is True and not allow_demo_only:
        report.error(f"正式数据目录不能混入 demo_only=true 的演示数据: {metadata_path}")
        return

    if not strict_module:
        if not isinstance(payload.get("source_files"), list) or not payload["source_files"]:
            report.error(f"元数据 source_files 不能为空: {metadata_path}")
            return
        status = str(payload.get("data_status", "")).strip().lower()
        if not allow_demo_only and status in SIMULATED_STATUS_VALUES:
            report.error(f"正式数据目录不能混入模拟/演示数据状态: {metadata_path}，data_status={status}")
            return
        if payload.get("crs") not in (TARGET_CRS, "not_applicable"):
            report.error(f"元数据 CRS 不统一: {metadata_path}，当前 {payload.get('crs')}，应为 {TARGET_CRS}")
            return
        report.ok(f"元数据说明完整: {metadata_path.name}")
        return

    if module_code and payload.get("module_code") != module_code:
        report.error(f"元数据 module_code 不一致: {metadata_path}，应为 {module_code}")
        return
    field = str(payload.get("field", ""))
    field_info = STANDARD_FIELDS.get(field)
    if field_info is None:
        report.error(f"元数据 field 未在全局字段表中定义: {metadata_path}，field={field}")
        return
    if payload.get("unit") != field_info["unit"]:
        report.error(
            f"元数据 unit 与全局字段表不一致: {metadata_path}，"
            f"当前 {payload.get('unit')}，应为 {field_info['unit']}"
        )
        return
    if payload.get("crs") != TARGET_CRS:
        report.error(f"元数据 CRS 不统一: {metadata_path}，当前 {payload.get('crs')}，应为 {TARGET_CRS}")
        return
    if payload.get("time_step") != TIME_STEP:
        report.error(f"元数据 time_step 不统一: {metadata_path}，应为 {TIME_STEP}")
        return
    if payload.get("date_field") != DATE_FIELD:
        report.error(f"元数据 date_field 不统一: {metadata_path}，应为 {DATE_FIELD}")
        return
    if payload.get("date_format") != DATE_FORMAT:
        report.error(f"元数据 date_format 不统一: {metadata_path}，应为 {DATE_FORMAT}")
        return
    if not isinstance(payload.get("source_files"), list) or not payload["source_files"]:
        report.error(f"元数据 source_files 不能为空: {metadata_path}")
        return
    report.ok(f"元数据说明完整: {metadata_path.name}")


def _check_baseline_raw_metadata(
    report: ValidationReport,
    path: Path,
    *,
    allow_demo_only: bool,
) -> None:
    metadata_path = path.with_suffix(path.suffix + ".meta.json")
    if allow_demo_only and not metadata_path.exists():
        report.warn(f"样例数据缺少旁路元数据，仅作为模板保留: {metadata_path}")
        return
    _check_metadata_sidecar(report, path, allow_demo_only=allow_demo_only, strict_module=False)


def _check_csv_date(report: ValidationReport, path: Path) -> None:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            fields = reader.fieldnames or []
            if DATE_FIELD not in fields:
                report.error(f"CSV 缺少统一时间字段 {DATE_FIELD}: {path}")
                return
            row_count = 0
            for row_index, row in enumerate(reader, start=2):
                row_count += 1
                _parse_date(report, str(row.get(DATE_FIELD, "")), f"{path.name}:第{row_index}行")
            if row_count == 0:
                report.warn(f"CSV 无数据行: {path}")
            else:
                report.ok(f"CSV 日期字段 {DATE_FIELD} 格式和时段有效: {path.name} ({row_count} 行)")
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


def validate(root: Path, *, stage: str = "baseline-raw") -> ValidationReport:
    report = ValidationReport()
    root = root.expanduser().resolve()
    allow_demo_only = _is_sample_root(root)
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
        _check_bounds_cover_watershed(report, dem, bounds, watershed_bounds)
        _check_baseline_raw_metadata(report, dem, allow_demo_only=allow_demo_only)

    raw_period_dirs = sorted(path for path in raw.iterdir() if path.is_dir()) if raw.exists() else []
    if not raw_period_dirs:
        report.error(f"raw 下缺少按 年月_时段名称 组织的原始数据目录: {raw}")
    for period_dir in raw_period_dirs:
        _period_token_from_folder(report, period_dir, "raw")
        period_files = sorted(
            path
            for path in period_dir.iterdir()
            if path.is_file() and not path.name.endswith(".meta.json") and path.name.lower() != "readme.md"
        )
        if not period_files:
            report.error(f"raw 时段目录为空: {period_dir}")
            continue
        report.ok(f"raw 时段目录包含 {len(period_files)} 个文件: {period_dir.name}")
        for path in period_files:
            if path.suffix.lower() in {".meta", ".json"} or path.name.endswith(".meta.json"):
                continue
            if path.suffix.lower() == ".tif" and path.exists():
                bounds = _check_tif_crs(report, path)
                _check_bounds_cover_watershed(report, path, bounds, watershed_bounds)
                _check_baseline_raw_metadata(report, path, allow_demo_only=allow_demo_only)
            if path.suffix.lower() == ".csv" and path.exists():
                _check_csv_date(report, path)
                _check_baseline_raw_metadata(report, path, allow_demo_only=allow_demo_only)
            if path.suffix.lower() not in {".tif", ".csv"} and path.exists():
                _check_baseline_raw_metadata(report, path, allow_demo_only=allow_demo_only)

    if stage == "baseline-raw":
        report.ok("已按 baseline/raw 阶段校验，processed 模型成果等待模块运行后再校验")
        return report

    scheme_dirs = sorted(path for path in processed.iterdir() if path.is_dir()) if processed.exists() else []
    if not scheme_dirs:
        report.error(f"processed 下缺少按 方案_工况 组织的模型成果目录: {processed}")
    for scheme_dir in scheme_dirs:
        if "_" not in scheme_dir.name:
            report.error(f"processed 方案目录命名不规范，应为 方案编号_工况名称: {scheme_dir}")
            continue
        period_dirs = sorted(path for path in scheme_dir.iterdir() if path.is_dir())
        if not period_dirs:
            report.error(f"processed 方案目录下缺少时段成果目录: {scheme_dir}")
            continue
        for period_dir in period_dirs:
            rel = period_dir.relative_to(root).as_posix()
            period_token = _period_token_from_folder(report, period_dir, "processed")
            if period_token is None:
                continue
            raster_dir = period_dir / "raster"
            table_dir = period_dir / "table"
            _require_dir(report, raster_dir, f"{rel}/raster")
            _require_dir(report, table_dir, f"{rel}/table")
            tag_path = period_dir / "finish.tag"
            _require_file(report, tag_path, f"{rel}/finish.tag")
            if tag_path.exists():
                _check_finish_tag(report, tag_path)

            for output in _expected_outputs_for_period(period_token):
                path = period_dir / output
                _require_file(report, path, f"{rel}/{output}")
                module_code = _module_code_for_output(output)
                if path.suffix.lower() == ".tif" and path.exists():
                    bounds = _check_tif_crs(report, path)
                    _check_bounds_cover_watershed(report, path, bounds, watershed_bounds)
                    _check_metadata_sidecar(
                        report,
                        path,
                        module_code=module_code,
                        allow_demo_only=allow_demo_only,
                    )
                if path.suffix.lower() == ".csv" and path.exists():
                    _check_csv_date(report, path)
                    _check_metadata_sidecar(
                        report,
                        path,
                        module_code=module_code,
                        allow_demo_only=allow_demo_only,
                    )
                if path.suffix.lower() == ".xlsx" and path.exists():
                    _check_metadata_sidecar(
                        report,
                        path,
                        module_code=module_code,
                        allow_demo_only=allow_demo_only,
                    )

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="校验瓦赫什流域数字孪生标准数据目录")
    parser.add_argument(
        "root",
        nargs="?",
        default=str(DEFAULT_TWIN_DATA_ROOT),
        help="瓦赫什流域孪生数据根目录，默认校验正式 data 目录",
    )
    parser.add_argument(
        "--stage",
        choices=("baseline-raw", "full"),
        default="baseline-raw",
        help="baseline-raw 只校验真实基础/原始数据；full 额外要求 processed 已生成全部标准成果",
    )
    args = parser.parse_args()

    report = validate(Path(args.root), stage=args.stage)
    report.print()
    return 0 if report.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
