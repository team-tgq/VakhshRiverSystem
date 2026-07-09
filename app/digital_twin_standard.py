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
TWIN_DATA_ROOT_ENV = "VAKHSH_TWIN_DATA_ROOT"


def configured_twin_data_root() -> Path:
    env_value = os.getenv(TWIN_DATA_ROOT_ENV, "").strip()
    if env_value:
        return Path(env_value).expanduser().resolve()

    try:
        from config import TWIN_DATA_ROOT
    except Exception:
        TWIN_DATA_ROOT = ""

    config_value = str(TWIN_DATA_ROOT).strip()
    if config_value:
        return Path(config_value).expanduser().resolve()

    return SAMPLE_TWIN_DATA_ROOT


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
DEFAULT_PERIOD = "200503"
DEFAULT_PERIOD_NAME = "融雪模拟"

RAW_TO_PROCESSED_PERIOD_NAMES = {
    "融雪期": "融雪模拟",
    "汛期": "汛期模拟",
}


STANDARD_FIELDS = {
    "snow_depth": {"name": "积雪深度", "unit": "m"},
    "snow_cover": {"name": "积雪覆盖率", "unit": "0-1"},
    "snow_density": {"name": "雪密度", "unit": "g/cm3"},
    "swe": {"name": "雪水当量", "unit": "mm"},
    "runoff": {"name": "径流深度", "unit": "mm"},
    "flood_depth": {"name": "洪水水深", "unit": "m"},
    "discharge": {"name": "河道流量", "unit": "m3/s"},
    "storage": {"name": "库容", "unit": "万m3"},
    "outflow": {"name": "下泄流量", "unit": "m3/s"},
    "allocation": {"name": "分水量", "unit": "百万m3"},
    "demand": {"name": "需水量", "unit": "百万m3"},
    "inundation": {"name": "实测淹没范围", "unit": "0/1"},
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


def raw_period_dir(period: str, period_name: str, root: str | Path | None = None) -> Path:
    return twin_root(root) / "raw" / f"{period}_{period_name}"


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
    "RAW_TO_PROCESSED_PERIOD_NAMES",
    "SAMPLE_TWIN_DATA_ROOT",
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
    "mark_module_complete",
    "module_output_path",
    "module_output_paths",
    "module_spec",
    "period_to_date",
    "processed_period_dir",
    "raster_dir",
    "raw_period_dir",
    "specs_as_dict",
    "table_dir",
    "twin_root",
    "write_finish_tag",
    "write_metadata_sidecar",
    "write_standard_csv",
]
