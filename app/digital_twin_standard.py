from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_TWIN_DATA_ROOT = PROJECT_ROOT / "sample_data" / "瓦赫什流域孪生数据"
REAL_TWIN_DATA_ROOT = PROJECT_ROOT / "data" / "瓦赫什流域孪生数据"
TWIN_DATA_ROOT_ENV = "VAKHSH_TWIN_DATA_ROOT"


def _resolve_data_root(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def configured_twin_data_root() -> Path:
    env_value = os.getenv(TWIN_DATA_ROOT_ENV, "").strip()
    if env_value:
        return _resolve_data_root(env_value)

    try:
        from config import TWIN_DATA_ROOT
    except Exception:
        TWIN_DATA_ROOT = ""

    config_value = str(TWIN_DATA_ROOT).strip()
    if config_value:
        return _resolve_data_root(config_value)

    return REAL_TWIN_DATA_ROOT


DEFAULT_TWIN_DATA_ROOT = configured_twin_data_root()

TARGET_CRS = "EPSG:32642"
TARGET_CRS_NAME = "WGS84 UTM 42N"
STUDY_YEAR_START = 2005
STUDY_YEAR_END = 2017
TIME_STEP = "daily"
DATE_FIELD = "date"
DATE_FORMAT = "YYYY-MM-DD"

DEFAULT_SCHEME = "scheme01"
DEFAULT_SCHEME_NAME = "常规调度工况"
DEFAULT_PERIOD = "201707"
DEFAULT_PERIOD_NAME = "汛期模拟"

RAW_TO_PROCESSED_PERIOD_NAMES = {
    "融雪期": "融雪模拟",
    "汛期": "汛期模拟",
}
PROCESSED_TO_RAW_PERIOD_NAMES = {value: key for key, value in RAW_TO_PROCESSED_PERIOD_NAMES.items()}


STANDARD_FIELDS = {
    "snow_depth": {"name": "积雪深度", "unit": "m"},
    "snow_cover": {"name": "积雪覆盖率", "unit": "0-1"},
    "snow_type": {"name": "积雪状态类型", "unit": "class"},
    "snow_density": {"name": "雪密度", "unit": "g/cm3"},
    "swe": {"name": "雪水当量", "unit": "mm"},
    "runoff": {"name": "径流深度", "unit": "mm"},
    "flood_depth": {"name": "洪水水深", "unit": "m"},
    "flood_risk": {"name": "洪涝风险等级", "unit": "class"},
    "discharge": {"name": "河道流量", "unit": "m3/s"},
    "storage": {"name": "库容", "unit": "万m3"},
    "outflow": {"name": "下泄流量", "unit": "m3/s"},
    "allocation": {"name": "分水量", "unit": "百万m3"},
    "demand": {"name": "需水量", "unit": "百万m3"},
    "inundation": {"name": "淹没范围", "unit": "0/1"},
    "inundated_area": {"name": "淹没面积", "unit": "km2"},
    "velocity": {"name": "表面流速", "unit": "m/s"},
    "flow_direction": {"name": "流向角度", "unit": "degree"},
}


@dataclass(frozen=True)
class ModuleSpec:
    code: str
    name: str
    phase: str
    role: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    downstream: tuple[str, ...]


@dataclass(frozen=True)
class TwinRunContext:
    scheme: str = DEFAULT_SCHEME
    scheme_name: str = DEFAULT_SCHEME_NAME
    period: str = DEFAULT_PERIOD
    period_name: str = DEFAULT_PERIOD_NAME
    root: str | Path | None = None

    @property
    def root_path(self) -> Path:
        return twin_root(self.root)

    @property
    def scheme_folder(self) -> str:
        return f"{self.scheme}_{self.scheme_name}"

    @property
    def period_folder(self) -> str:
        return f"{self.period}_{self.period_name}"

    @property
    def period_dir(self) -> Path:
        return processed_period_dir(
            self.scheme,
            self.scheme_name,
            self.period,
            self.period_name,
            root=self.root,
        )


MODULE_SPECS: tuple[ModuleSpec, ...] = (
    ModuleSpec(
        code="M01",
        name="积雪水体识别",
        phase="遥感解译",
        role="主线",
        inputs=("baseline/DEM.tif", "baseline/流域边界.shp", "raw/{period}/{period}_哨兵影像.tif"),
        outputs=(
            "raster/{period}_M01_snow_depth_m.tif",
            "raster/{period}_M01_snow_cover.tif",
            "table/{period}_M01_积雪面积统计表.csv",
        ),
        downstream=("M06", "M02"),
    ),
    ModuleSpec(
        code="M06",
        name="积雪状态分类",
        phase="遥感解译",
        role="主线",
        inputs=(
            "processed/{scheme}/{period}/raster/{period}_M01_snow_depth_m.tif",
            "processed/{scheme}/{period}/raster/{period}_M01_snow_cover.tif",
            "baseline/DEM.tif",
        ),
        outputs=(
            "raster/{period}_M06_snow_type.tif",
            "raster/{period}_M06_snow_density_gcm3.tif",
        ),
        downstream=("M02",),
    ),
    ModuleSpec(
        code="M02",
        name="雪水当量计算",
        phase="水文模拟",
        role="主线",
        inputs=(
            "processed/{scheme}/{period}/raster/{period}_M01_snow_depth_m.tif",
            "processed/{scheme}/{period}/raster/{period}_M01_snow_cover.tif",
            "processed/{scheme}/{period}/raster/{period}_M06_snow_density_gcm3.tif",
            "raw/{period}/{period}_逐日气象.csv",
        ),
        outputs=(
            "raster/{period}_M02_swe_mm.tif",
            "raster/{period}_M02_runoff_mm.tif",
        ),
        downstream=("M03",),
    ),
    ModuleSpec(
        code="M03",
        name="洪水演进汇流",
        phase="水文模拟",
        role="主线",
        inputs=("processed/{scheme}/{period}/raster/{period}_M02_runoff_mm.tif",),
        outputs=(
            "table/{period}_M03_discharge.csv",
            "raster/{period}_M03_flood_depth_m.tif",
            "raster/{period}_M03_inundation.tif",
        ),
        downstream=("M07", "M09"),
    ),
    ModuleSpec(
        code="M09",
        name="水库库容计算",
        phase="水库调度",
        role="主线",
        inputs=("processed/{scheme}/{period}/table/{period}_M03_discharge.csv",),
        outputs=(
            "table/{period}_M09_storage.csv",
            "table/{period}_M09_outflow.csv",
        ),
        downstream=("M08",),
    ),
    ModuleSpec(
        code="M08",
        name="水资源分配",
        phase="水库调度",
        role="主线",
        inputs=(
            "processed/{scheme}/{period}/table/{period}_M09_storage.csv",
            "processed/{scheme}/{period}/table/{period}_M09_outflow.csv",
        ),
        outputs=("table/{period}_M08_分水方案统计表.csv",),
        downstream=(),
    ),
    ModuleSpec(
        code="M07",
        name="洪涝风险评估",
        phase="风险评估",
        role="主线",
        inputs=(
            "processed/{scheme}/{period}/raster/{period}_M03_flood_depth_m.tif",
            "processed/{scheme}/{period}/raster/{period}_M03_inundation.tif",
            "processed/{scheme}/{period}/raster/{period}_实测_淹没范围.tif",
        ),
        outputs=("raster/{period}_M07_洪涝风险分区图.tif",),
        downstream=(),
    ),
    ModuleSpec(
        code="M04",
        name="SAR卫星淹没提取",
        phase="监测对比",
        role="校核",
        inputs=("raw/{period}/{period}_SAR影像.tif",),
        outputs=(
            "raster/{period}_实测_淹没范围.tif",
            "table/{period}_淹没面积统计报表.xlsx",
        ),
        downstream=("M07",),
    ),
    ModuleSpec(
        code="M05",
        name="视频流速监测",
        phase="监测对比",
        role="校核",
        inputs=("raw/{period}/{period}_河道视频.mp4",),
        outputs=("table/{period}_实测_流速数据.csv",),
        downstream=("M03",),
    ),
)


def twin_root(root: str | Path | None = None) -> Path:
    return Path(root).expanduser().resolve() if root else DEFAULT_TWIN_DATA_ROOT


def baseline_dir(root: str | Path | None = None) -> Path:
    return twin_root(root) / "baseline"


def raw_dir(root: str | Path | None = None) -> Path:
    return twin_root(root) / "raw"


def processed_dir(root: str | Path | None = None) -> Path:
    return twin_root(root) / "processed"


def raw_period_name_from_processed(period_name: str) -> str:
    return PROCESSED_TO_RAW_PERIOD_NAMES.get(period_name, period_name)


def raw_period_dir(period: str, period_name: str, root: str | Path | None = None) -> Path:
    return twin_root(root) / "raw" / f"{period}_{period_name}"


def raw_period_dir_for_context(context: TwinRunContext) -> Path:
    raw_name = raw_period_name_from_processed(context.period_name)
    exact = raw_period_dir(context.period, raw_name, root=context.root)
    if exact.exists():
        return exact

    matches = sorted(raw_dir(context.root).glob(f"{context.period}_*"))
    if matches:
        return matches[0]
    return exact


def processed_period_dir(
    scheme: str,
    scheme_name: str,
    period: str,
    period_name: str,
    root: str | Path | None = None,
) -> Path:
    return twin_root(root) / "processed" / f"{scheme}_{scheme_name}" / f"{period}_{period_name}"


def raster_dir(*args, **kwargs) -> Path:
    return processed_period_dir(*args, **kwargs) / "raster"


def table_dir(*args, **kwargs) -> Path:
    return processed_period_dir(*args, **kwargs) / "table"


def default_run_context(
    *,
    root: str | Path | None = None,
    scheme: str = DEFAULT_SCHEME,
    scheme_name: str = DEFAULT_SCHEME_NAME,
    period: str = DEFAULT_PERIOD,
    period_name: str = DEFAULT_PERIOD_NAME,
) -> TwinRunContext:
    return TwinRunContext(
        scheme=scheme,
        scheme_name=scheme_name,
        period=period,
        period_name=period_name,
        root=root,
    )


def period_to_date(period: str) -> str:
    token = str(period).strip()
    if len(token) == 8 and token.isdigit():
        return f"{token[:4]}-{token[4:6]}-{token[6:8]}"
    if len(token) == 6 and token.isdigit():
        return f"{token[:4]}-{token[4:6]}-01"
    return token


def infer_run_context_from_path(
    input_path: str | Path,
    *,
    root: str | Path | None = None,
    scheme: str = DEFAULT_SCHEME,
    scheme_name: str = DEFAULT_SCHEME_NAME,
    fallback_period: str = DEFAULT_PERIOD,
    fallback_period_name: str = DEFAULT_PERIOD_NAME,
) -> TwinRunContext:
    root_path = twin_root(root)
    path = Path(input_path).expanduser().resolve()
    period = fallback_period
    period_name = fallback_period_name

    candidate_folder = ""
    try:
        relative = path.relative_to(root_path)
        if len(relative.parts) >= 2 and relative.parts[0] == "raw":
            candidate_folder = relative.parts[1]
    except ValueError:
        candidate_folder = path.parent.name

    if "_" in candidate_folder:
        maybe_period, maybe_name = candidate_folder.split("_", 1)
        if maybe_period.isdigit():
            period = maybe_period
            period_name = RAW_TO_PROCESSED_PERIOD_NAMES.get(maybe_name, maybe_name or fallback_period_name)
    else:
        stem = path.stem
        maybe_period = stem.split("_", 1)[0]
        if maybe_period.isdigit():
            period = maybe_period

    return TwinRunContext(
        scheme=scheme,
        scheme_name=scheme_name,
        period=period,
        period_name=period_name,
        root=root_path,
    )


def ensure_processed_period(*args, **kwargs) -> Path:
    period_dir = processed_period_dir(*args, **kwargs)
    (period_dir / "raster").mkdir(parents=True, exist_ok=True)
    (period_dir / "table").mkdir(parents=True, exist_ok=True)
    return period_dir


def ensure_run_context(context: TwinRunContext | None = None) -> TwinRunContext:
    context = context or default_run_context()
    ensure_processed_period(
        context.scheme,
        context.scheme_name,
        context.period,
        context.period_name,
        root=context.root,
    )
    return context


def module_output_path(
    module_code: str,
    *,
    context: TwinRunContext | None = None,
    output_index: int = 0,
    relative_output: str | None = None,
) -> Path:
    context = ensure_run_context(context)
    spec = module_spec(module_code)
    pattern = relative_output if relative_output is not None else spec.outputs[output_index]
    relative = pattern.format(
        scheme=context.scheme_folder,
        period=context.period,
    )
    return context.period_dir / relative


def module_output_paths(module_code: str, *, context: TwinRunContext | None = None) -> list[Path]:
    context = ensure_run_context(context)
    spec = module_spec(module_code)
    return [
        module_output_path(module_code, context=context, output_index=index)
        for index, _ in enumerate(spec.outputs)
    ]


def _format_standard_pattern(pattern: str, context: TwinRunContext) -> Path:
    root = twin_root(context.root)
    formatted = pattern.format(
        scheme=context.scheme_folder,
        period=context.period,
    )
    if formatted.startswith("baseline/"):
        return root / formatted
    if formatted.startswith("raw/{period}/"):
        suffix = formatted.split("/", 2)[2]
        return raw_period_dir_for_context(context) / suffix
    if formatted.startswith("raw/"):
        parts = formatted.split("/", 2)
        if len(parts) >= 3 and parts[1] == context.period:
            return raw_period_dir_for_context(context) / parts[2]
        return root / formatted
    if formatted.startswith("processed/{scheme}/{period}/"):
        return context.period_dir / formatted.split("/", 3)[3]
    if formatted.startswith("processed/"):
        prefix = f"processed/{context.scheme_folder}/{context.period}/"
        if formatted.startswith(prefix):
            return context.period_dir / formatted[len(prefix):]
        prefix = f"processed/{context.scheme_folder}/{context.period_folder}/"
        if formatted.startswith(prefix):
            return context.period_dir / formatted[len(prefix):]
        return root / formatted
    return context.period_dir / formatted


def module_input_path(
    module_code: str,
    *,
    context: TwinRunContext | None = None,
    input_index: int = 0,
    required: bool = False,
) -> Path:
    context = context or default_run_context()
    spec = module_spec(module_code)
    path = _format_standard_pattern(spec.inputs[input_index], context)
    if required and not path.exists():
        raise FileNotFoundError(
            f"{spec.code} {spec.name} 缺少统一输入: {path}\n"
            f"请先把真实数据放入 {twin_root(context.root)} 的 baseline/raw/processed 标准目录，"
            "或先运行上游模块生成标准成果。"
        )
    return path


def module_input_paths(
    module_code: str,
    *,
    context: TwinRunContext | None = None,
    required: bool = False,
) -> list[Path]:
    context = context or default_run_context()
    return [
        module_input_path(module_code, context=context, input_index=index, required=required)
        for index, _ in enumerate(module_spec(module_code).inputs)
    ]


def latest_module_output(
    module_code: str,
    *,
    output_index: int = 0,
    root: str | Path | None = None,
) -> Path | None:
    spec = module_spec(module_code)
    pattern = spec.outputs[output_index]
    base = processed_dir(root)
    candidates: list[Path] = []
    if not base.exists():
        return None
    for period_dir in base.glob("*/*"):
        if not period_dir.is_dir() or "_" not in period_dir.name:
            continue
        period = period_dir.name.split("_", 1)[0]
        candidate = period_dir / pattern.format(period=period, scheme=period_dir.parent.name)
        if candidate.exists():
            candidates.append(candidate)
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def standard_dialog_dir(
    kind: str = "raw",
    *,
    module_code: str | None = None,
    context: TwinRunContext | None = None,
    output_index: int = 0,
    root: str | Path | None = None,
) -> str:
    context = context or default_run_context(root=root)
    if module_code:
        try:
            if kind == "output":
                return str(module_output_path(module_code, context=context, output_index=output_index).parent)
            return str(module_input_path(module_code, context=context).parent)
        except Exception:
            pass
    if kind == "baseline":
        return str(baseline_dir(context.root))
    if kind == "processed":
        return str(processed_dir(context.root))
    return str(raw_dir(context.root))


def write_standard_csv(
    output_path: str | Path,
    *,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, object]],
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    return path


def iter_module_specs(role: str | None = None) -> Iterable[ModuleSpec]:
    for spec in MODULE_SPECS:
        if role is None or spec.role == role:
            yield spec


def module_spec(code: str) -> ModuleSpec:
    safe_code = code.upper()
    for spec in MODULE_SPECS:
        if spec.code == safe_code:
            return spec
    raise KeyError(f"未知模块编号: {code}")


def write_metadata_sidecar(
    output_path: str | Path,
    *,
    module_code: str,
    field: str,
    source_files: Iterable[str | Path],
    extra: dict | None = None,
) -> Path:
    path = Path(output_path)
    field_info = STANDARD_FIELDS.get(field, {"name": field, "unit": ""})
    payload = {
        "file": path.name,
        "module_code": module_code,
        "module_name": module_spec(module_code).name,
        "field": field,
        "field_name": field_info["name"],
        "unit": field_info["unit"],
        "crs": TARGET_CRS,
        "crs_name": TARGET_CRS_NAME,
        "time_step": TIME_STEP,
        "date_field": DATE_FIELD,
        "date_format": DATE_FORMAT,
        "source_files": [str(item) for item in source_files],
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    if extra:
        payload.update(extra)
    metadata_path = path.with_suffix(path.suffix + ".meta.json")
    metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata_path


def write_finish_tag(
    period_dir: str | Path,
    *,
    scheme: str,
    period: str,
    completed_modules: Iterable[str],
) -> Path:
    tag_path = Path(period_dir) / "finish.tag"
    payload = {
        "scheme": scheme,
        "period": period,
        "completed_modules": list(completed_modules),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }
    tag_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return tag_path


def mark_module_complete(
    context: TwinRunContext,
    module_codes: str | Iterable[str],
) -> Path:
    context = ensure_run_context(context)
    tag_path = context.period_dir / "finish.tag"
    completed: list[str] = []

    if tag_path.exists():
        try:
            payload = json.loads(tag_path.read_text(encoding="utf-8"))
            completed = [str(item).upper() for item in payload.get("completed_modules", [])]
        except Exception:
            completed = []

    if isinstance(module_codes, str):
        incoming = [module_codes]
    else:
        incoming = list(module_codes)

    for code in incoming:
        normalized = str(code).upper()
        if normalized not in completed:
            completed.append(normalized)

    return write_finish_tag(
        context.period_dir,
        scheme=context.scheme,
        period=context.period,
        completed_modules=completed,
    )


def specs_as_dict() -> list[dict]:
    return [asdict(spec) for spec in MODULE_SPECS]


__all__ = [
    "DATE_FIELD",
    "DATE_FORMAT",
    "DEFAULT_TWIN_DATA_ROOT",
    "DEFAULT_PERIOD",
    "DEFAULT_PERIOD_NAME",
    "DEFAULT_SCHEME",
    "DEFAULT_SCHEME_NAME",
    "MODULE_SPECS",
    "PROCESSED_TO_RAW_PERIOD_NAMES",
    "RAW_TO_PROCESSED_PERIOD_NAMES",
    "SAMPLE_TWIN_DATA_ROOT",
    "REAL_TWIN_DATA_ROOT",
    "STANDARD_FIELDS",
    "STUDY_YEAR_END",
    "STUDY_YEAR_START",
    "TARGET_CRS",
    "TARGET_CRS_NAME",
    "TIME_STEP",
    "TWIN_DATA_ROOT_ENV",
    "TwinRunContext",
    "baseline_dir",
    "configured_twin_data_root",
    "default_run_context",
    "ensure_processed_period",
    "ensure_run_context",
    "infer_run_context_from_path",
    "iter_module_specs",
    "latest_module_output",
    "mark_module_complete",
    "module_input_path",
    "module_input_paths",
    "module_output_path",
    "module_output_paths",
    "module_spec",
    "period_to_date",
    "processed_dir",
    "processed_period_dir",
    "raster_dir",
    "raw_dir",
    "raw_period_dir_for_context",
    "raw_period_name_from_processed",
    "raw_period_dir",
    "specs_as_dict",
    "standard_dialog_dir",
    "table_dir",
    "twin_root",
    "write_finish_tag",
    "write_metadata_sidecar",
    "write_standard_csv",
]
