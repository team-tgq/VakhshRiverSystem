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
# 标准数据目录同时承载 2005-2017 历史研究资料和后续现地实测/业务运行资料。
STUDY_YEAR_END = 2026
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
    "flood_risk_index": {"name": "洪涝风险指数", "unit": "0-1"},
    "landuse_risk_stats": {"name": "土地利用风险统计", "unit": "table"},
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
    upstream_inputs: tuple[str, ...] = ()


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
            "baseline/DEM.tif",
            "raw/{period}/{period}_积雪状态GEE.tif",
        ),
        outputs=(
            "raster/{period}_M06_snow_type.tif",
            "raster/{period}_M06_snow_density_gcm3.tif",
        ),
        downstream=("M02",),
        upstream_inputs=(
            "processed/{scheme}/{period}/raster/{period}_M01_snow_depth_m.tif",
            "processed/{scheme}/{period}/raster/{period}_M01_snow_cover.tif",
        ),
    ),
    ModuleSpec(
        code="M02",
        name="雪水当量计算",
        phase="水文模拟",
        role="主线",
        inputs=(
            "baseline/DEM.tif",
            "baseline/流域边界.shp",
            "raw/{period}/{period}_SWE业务日记录.csv",
            "raw/{period}/{period}_逐日气象.csv",
            "raw/{period}/{period}_逐日降水.tif",
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
        inputs=(
            "raw/{period}/{period}_径流历史数据.csv",
            "raw/{period}/{period}_洪水演进历史数据.csv",
        ),
        outputs=(
            "table/{period}_M03_discharge.csv",
            "raster/{period}_M03_flood_depth_m.tif",
            "raster/{period}_M03_inundation.tif",
        ),
        downstream=("M07", "M09"),
        upstream_inputs=("processed/{scheme}/{period}/raster/{period}_M02_runoff_mm.tif",),
    ),
    ModuleSpec(
        code="M09",
        name="水库库容计算",
        phase="水库调度",
        role="主线",
        inputs=(
            "raw/{period}/{period}_水库参数.csv",
            "raw/{period}/{period}_库区遥感估算结果.csv",
        ),
        outputs=(
            "table/{period}_M09_storage.csv",
            "table/{period}_M09_outflow.csv",
        ),
        downstream=("M08",),
        upstream_inputs=("processed/{scheme}/{period}/table/{period}_M03_discharge.csv",),
    ),
    ModuleSpec(
        code="M08",
        name="水资源分配",
        phase="水库调度",
        role="主线",
        inputs=(
            "raw/{period}/{period}_水资源分配配置.csv",
            "raw/{period}/{period}_需水配置.csv",
        ),
        outputs=("table/{period}_M08_分水方案统计表.csv",),
        downstream=(),
        upstream_inputs=(
            "processed/{scheme}/{period}/table/{period}_M09_storage.csv",
            "processed/{scheme}/{period}/table/{period}_M09_outflow.csv",
        ),
    ),
    ModuleSpec(
        code="M07",
        name="洪涝风险评估",
        phase="风险评估",
        role="主线",
        inputs=(
            "baseline/DEM.tif",
            "baseline/河网.shp",
            "raw/{period}/{period}_逐日降水.tif",
            "raw/{period}/{period}_土壤湿度.tif",
            "raw/{period}/{period}_土地覆盖.tif",
        ),
        outputs=(
            "raster/{period}_M07_洪涝风险分区图.tif",
            "raster/{period}_M07_洪涝风险指数.tif",
            "table/{period}_M07_landuse_risk_stats.csv",
            "table/{period}_M07_weights.txt",
            "table/{period}_M07_landuse_risk_summary.txt",
            "table/{period}_M07_flood_risk_map.html",
        ),
        downstream=(),
        upstream_inputs=(
            "processed/{scheme}/{period}/raster/{period}_M03_flood_depth_m.tif",
            "processed/{scheme}/{period}/raster/{period}_M03_inundation.tif",
            "processed/{scheme}/{period}/raster/{period}_实测_淹没范围.tif",
        ),
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


def parse_period_folder_name(folder_name: str) -> tuple[str, str] | None:
    if "_" not in folder_name:
        return None
    period, period_name = folder_name.split("_", 1)
    if not period.isdigit() or not period_name:
        return None
    return period, period_name


def processed_period_name_from_raw(raw_period_name: str) -> str:
    return RAW_TO_PROCESSED_PERIOD_NAMES.get(raw_period_name, raw_period_name)


def iter_raw_run_contexts(
    *,
    root: str | Path | None = None,
    scheme: str = DEFAULT_SCHEME,
    scheme_name: str = DEFAULT_SCHEME_NAME,
) -> Iterable[TwinRunContext]:
    base = raw_dir(root)
    if not base.exists():
        return
    root_path = twin_root(root)
    for period_dir in sorted((path for path in base.iterdir() if path.is_dir()), key=lambda path: path.name):
        parsed = parse_period_folder_name(period_dir.name)
        if parsed is None:
            continue
        period, raw_period_name = parsed
        yield TwinRunContext(
            scheme=scheme,
            scheme_name=scheme_name,
            period=period,
            period_name=processed_period_name_from_raw(raw_period_name),
            root=root_path,
        )


def raw_period_dir_for_context(context: TwinRunContext) -> Path:
    raw_name = raw_period_name_from_processed(context.period_name)
    return raw_period_dir(context.period, raw_name, root=context.root)


def ensure_raw_source_path(path: str | Path, *, root: str | Path | None = None) -> Path:
    raw_base = raw_dir(root).resolve()
    source = Path(path).expanduser().resolve()
    try:
        source.relative_to(raw_base)
    except ValueError as exc:
        raise ValueError(
            f"模块标准输入必须来自统一 raw 目录: {raw_base}\n"
            f"当前选择: {source}\n"
            "请先把原始观测、遥感、气象、视频或业务源文件放入对应 raw/{period}_{时段} 目录。"
        ) from exc
    if not source.exists():
        raise FileNotFoundError(source)
    return source


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


def module_upstream_input_path(
    module_code: str,
    *,
    context: TwinRunContext | None = None,
    input_index: int = 0,
    required: bool = False,
) -> Path:
    context = context or default_run_context()
    spec = module_spec(module_code)
    path = _format_standard_pattern(spec.upstream_inputs[input_index], context)
    if required and not path.exists():
        raise FileNotFoundError(
            f"{spec.code} {spec.name} 缺少上游标准成果: {path}\n"
            "请先运行上游模块生成 processed 标准成果。"
        )
    return path


def module_upstream_input_paths(
    module_code: str,
    *,
    context: TwinRunContext | None = None,
    required: bool = False,
) -> list[Path]:
    context = context or default_run_context()
    return [
        module_upstream_input_path(module_code, context=context, input_index=index, required=required)
        for index, _ in enumerate(module_spec(module_code).upstream_inputs)
    ]


def latest_module_output(
    module_code: str,
    *,
    output_index: int = 0,
    root: str | Path | None = None,
) -> Path | None:
    spec = module_spec(module_code)
    pattern = spec.outputs[output_index]
    candidates: list[Path] = []
    for context in iter_raw_run_contexts(root=root):
        candidate = context.period_dir / pattern.format(period=context.period, scheme=context.scheme_folder)
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
            spec = module_spec(module_code)
            if kind == "raw":
                for pattern in spec.inputs:
                    if pattern.startswith("raw/"):
                        return str(_format_standard_pattern(pattern, context).parent)
            if kind == "baseline":
                for pattern in spec.inputs:
                    if pattern.startswith("baseline/"):
                        return str(_format_standard_pattern(pattern, context).parent)
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


def write_raw_metadata_sidecar(
    input_path: str | Path,
    *,
    data_role: str,
    data_status: str,
    source_name: str,
    source_files: Iterable[str | Path],
    extra: dict | None = None,
) -> Path:
    path = Path(input_path)
    payload = {
        "file": path.name,
        "data_role": data_role,
        "data_status": data_status,
        "source_name": source_name,
        "source_files": [str(item) for item in source_files],
        "crs": TARGET_CRS,
        "crs_name": TARGET_CRS_NAME,
        "time_step": TIME_STEP,
        "date_field": DATE_FIELD,
        "date_format": DATE_FORMAT,
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
    payload: dict = {}
    completed: list[str] = []
    completed_outputs: list[str] = []

    if tag_path.exists():
        try:
            payload = json.loads(tag_path.read_text(encoding="utf-8"))
            completed = [str(item).upper() for item in payload.get("completed_modules", [])]
            completed_outputs = [
                str(item).replace("\\", "/")
                for item in payload.get("completed_outputs", [])
            ]
        except Exception:
            payload = {}
            completed = []
            completed_outputs = []

    if isinstance(module_codes, str):
        incoming = [module_codes]
    else:
        incoming = list(module_codes)

    for code in incoming:
        normalized = str(code).upper()
        if normalized not in completed:
            completed.append(normalized)
        try:
            for output_path in module_output_paths(normalized, context=context):
                if output_path.exists():
                    relative_output = output_path.relative_to(context.period_dir).as_posix()
                    if relative_output not in completed_outputs:
                        completed_outputs.append(relative_output)
        except Exception:
            pass

    payload.update(
        {
            "scheme": context.scheme,
            "period": context.period,
            "completed_modules": completed,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    if completed_outputs:
        payload["completed_outputs"] = completed_outputs
    tag_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return tag_path


RAW_DATA_CATEGORIES = (
    "remote_sensing",
    "meteorology",
    "land_surface",
    "snow_hydrology",
    "reservoir",
    "socioeconomic",
    "configuration",
    "video",
)

MODULE_DIRECTORY_NAMES = {
    "M01": "M01_segformer",
    "M02": "M02_swe",
    "M03": "M03_routing",
    "M04": "M04_inundation",
    "M05": "M05_raft",
    "M06": "M06_snow_state",
    "M07": "M07_flood_risk",
    "M08": "M08_water_allocation",
    "M09": "M09_reservoir_estimation",
}

MODULE_RAW_DEFAULTS = {
    "M01": ("remote_sensing", "optical_rgb"),
    "M02": ("meteorology", "daily_forcing"),
    "M03": (),
    "M04": ("remote_sensing",),
    "M05": ("video", "river_velocity"),
    "M06": ("remote_sensing", "gee", "snow_state"),
    "M07": ("meteorology", "precipitation"),
    "M08": ("configuration", "water_allocation"),
    "M09": ("reservoir",),
}

MODULE_OUTPUT_PATTERNS = {
    "M01": (
        "snow/business/rasters/{period}_snow_depth_m.tif",
        "snow/business/masks/{period}_snow_cover.tif",
        "snow/business/tables/{period}_snow_area_statistics.csv",
    ),
    "M02": (
        "rasters/SWE_mm_{period}.tif",
        "rasters/Snowmelt_mm_day_{period}.tif",
        "tables/daily_basin_series.csv",
    ),
    "M03": (
        "tables/{period}_discharge.csv",
        "rasters/{period}_flood_depth_m.tif",
        "rasters/{period}_inundation.tif",
    ),
    "M04": (
        "business/masks/{period}_inundation.tif",
        "business/tables/{period}_inundation_statistics.xlsx",
    ),
    "M05": ("tables/frame_pair_velocity.csv",),
    "M06": (
        "rasters/{period}_snow_type.tif",
        "rasters/{period}_snow_density_gcm3.tif",
        "tables/{period}_snow_state_statistics.csv",
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


def _safe_parts(parts: Sequence[str | Path]) -> tuple[str, ...]:
    safe: list[str] = []
    for part in parts:
        text = str(part).strip().replace("\\", "/")
        if not text or text in {".", ".."} or text.startswith("/") or ":" in text:
            raise ValueError(f"非法目录片段: {part!r}")
        tokens = [token for token in text.split("/") if token]
        if any(token in {".", ".."} for token in tokens):
            raise ValueError(f"目录片段不能越过统一数据根目录: {part!r}")
        safe.extend(tokens)
    return tuple(safe)


def raw_data_dir(
    category: str,
    *parts: str | Path,
    root: str | Path | None = None,
    create: bool = False,
) -> Path:
    category = str(category).strip()
    if category not in RAW_DATA_CATEGORIES:
        raise ValueError(
            f"未知 raw 数据类型 {category!r}；允许值: {', '.join(RAW_DATA_CATEGORIES)}"
        )
    path = raw_dir(root) / category
    for part in _safe_parts(parts):
        path /= part
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def module_processed_dir(
    module_code: str,
    *parts: str | Path,
    root: str | Path | None = None,
    create: bool = False,
) -> Path:
    code = str(module_code).upper()
    try:
        folder = MODULE_DIRECTORY_NAMES[code]
    except KeyError as exc:
        raise KeyError(f"未知模块编号: {module_code}") from exc
    path = processed_dir(root) / folder
    for part in _safe_parts(parts):
        path /= part
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def file_sha256(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_source_files(source_files: Iterable[str | Path]) -> list[str]:
    values: list[str] = []
    for item in source_files:
        text = str(item)
        if "://" not in text:
            text = str(Path(text).expanduser().resolve())
        values.append(text)
    return values


def _file_description(path: Path) -> dict:
    description: dict = {
        "shape": None,
        "dtype": None,
        "crs": "not_applicable",
    }
    suffix = path.suffix.lower()
    if not path.exists() or path.stat().st_size == 0:
        return description
    if suffix in {".tif", ".tiff"}:
        try:
            import rasterio

            with rasterio.open(path) as dataset:
                description.update(
                    {
                        "shape": [dataset.count, dataset.height, dataset.width],
                        "dtype": list(dataset.dtypes),
                        "crs": str(dataset.crs) if dataset.crs else None,
                        "bands": list(dataset.descriptions),
                        "nodata": dataset.nodata,
                        "bounds": list(dataset.bounds),
                    }
                )
            return description
        except Exception:
            return description
    if suffix in {".png", ".jpg", ".jpeg", ".bmp"}:
        try:
            from PIL import Image

            with Image.open(path) as image:
                description.update(
                    {
                        "shape": [image.height, image.width, len(image.getbands())],
                        "dtype": "uint8",
                        "bands": list(image.getbands()),
                    }
                )
        except Exception:
            pass
    return description


def _write_json_sidecar(path: Path, payload: Mapping) -> Path:
    sidecar = path.with_suffix(path.suffix + ".meta.json")
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return sidecar


def write_raw_metadata(
    input_path: str | Path,
    *,
    data_type: str,
    dataset_name: str,
    source_files: Iterable[str | Path],
    source_origin: str,
    consumer_modules: Iterable[str],
    split: str | None = None,
    sensor: str | None = None,
    bands: Sequence | None = None,
    dtype: str | Sequence | None = None,
    crs: str | None = None,
    date: str | None = None,
    is_module_native: bool,
    extra: Mapping | None = None,
) -> Path:
    path = ensure_raw_source_path(input_path)
    description = _file_description(path)
    payload = {
        "file": path.name,
        "data_type": data_type,
        "dataset_name": dataset_name,
        "source_files": _normalise_source_files(source_files),
        "source_origin": source_origin,
        "consumer_modules": [str(item).upper() for item in consumer_modules],
        "split": split,
        "sensor": sensor,
        "bands": list(bands) if bands is not None else description.get("bands"),
        "dtype": dtype if dtype is not None else description.get("dtype"),
        "crs": crs if crs is not None else description.get("crs"),
        "date": date,
        "is_module_native": bool(is_module_native),
        "checksum": file_sha256(path),
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    if extra:
        payload.update(dict(extra))
    return _write_json_sidecar(path, payload)


def write_processed_metadata(
    output_path: str | Path,
    *,
    module_code: str,
    output_type: str,
    source_files: Iterable[str | Path],
    model_weight: str | Path | None = None,
    threshold_or_config: object = None,
    extra: Mapping | None = None,
) -> Path:
    path = Path(output_path).expanduser().resolve()
    if not path.exists() or path.stat().st_size == 0:
        raise ValueError(f"不能为缺失或空的成果写入元数据: {path}")
    output_root = module_processed_dir(module_code).resolve()
    try:
        path.relative_to(output_root)
    except ValueError as exc:
        raise ValueError(f"{module_code} 成果必须写入 {output_root}: {path}") from exc
    description = _file_description(path)
    payload = {
        "file": path.name,
        "module_code": str(module_code).upper(),
        "module_name": module_spec(module_code).name,
        "output_type": output_type,
        "source_files": _normalise_source_files(source_files),
        "model_weight": str(Path(model_weight).resolve()) if model_weight else None,
        "threshold_or_config": threshold_or_config,
        "shape": description.get("shape"),
        "dtype": description.get("dtype"),
        "crs": description.get("crs"),
        "checksum": file_sha256(path),
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    for key in ("bands", "nodata", "bounds"):
        if key in description:
            payload[key] = description[key]
    if extra:
        payload.update(dict(extra))
    return _write_json_sidecar(path, payload)


# 新架构下模块独立运行。保留 TwinRunContext 只是为了兼容既有插件调用，
# scheme/period_name 不再参与目录构造。
DEFAULT_PERIOD = datetime.now().strftime("%Y%m%d")
DEFAULT_PERIOD_NAME = "独立运行"


MODULE_SPECS = (
    ModuleSpec("M01", "SegFormer积雪/水体识别", "遥感解译", "独立", ("raw/remote_sensing/optical_rgb",), ("snow/{val,test}/{masks,overlays}", "water/{val,test}/{masks,overlays}"), ()),
    ModuleSpec("M06", "积雪状态识别", "遥感解译", "独立", ("raw/remote_sensing/gee/snow_state",), ("rasters/snow_type", "rasters/snow_density", "tables/statistics"), ()),
    ModuleSpec("M02", "雪水当量估算", "水文模拟", "独立", ("raw/meteorology", "raw/snow_hydrology", "baseline/DEM.tif"), ("rasters/SWE", "rasters/Snowmelt", "tables/daily_series"), ()),
    ModuleSpec("M03", "洪水演进汇流", "水文模拟", "暂不处理", (), (), ()),
    ModuleSpec("M09", "库区水量估算", "水库", "独立", ("raw/reservoir",), ("tables/reservoir_storage", "tables/summary"), ()),
    ModuleSpec("M08", "水资源分配", "水资源", "独立", ("raw/configuration/water_allocation", "raw/socioeconomic"), ("tables/allocation_plan", "tables/allocation_summary"), ()),
    ModuleSpec("M07", "洪涝灾害风险等级评估", "风险评估", "独立", ("raw/meteorology/precipitation", "raw/land_surface", "baseline"), ("rasters/risk_level", "rasters/risk_index", "tables", "visualizations"), ()),
    ModuleSpec("M04", "淹没区识别", "遥感解译", "独立验证", ("raw/remote_sensing/sentinel1_sar", "raw/remote_sensing/sentinel2_multispectral"), ("sentinel1_sar_weak/{masks,overlays}", "sentinel2_optical_hand/{masks,overlays}"), ()),
    ModuleSpec("M05", "RAFT光流测速", "视频监测", "独立", ("raw/video/river_velocity",), ("tables/frame_pair_velocity.csv", "visualizations"), ()),
)


def default_run_context(
    *,
    root: str | Path | None = None,
    scheme: str = "independent",
    scheme_name: str = "独立运行",
    period: str = DEFAULT_PERIOD,
    period_name: str = DEFAULT_PERIOD_NAME,
) -> TwinRunContext:
    return TwinRunContext(scheme, scheme_name, period, period_name, root)


def processed_period_dir(*args, root: str | Path | None = None, **kwargs) -> Path:
    return processed_dir(root)


def ensure_processed_period(*args, root: str | Path | None = None, **kwargs) -> Path:
    path = processed_dir(root)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_run_context(context: TwinRunContext | None = None) -> TwinRunContext:
    return context or default_run_context()


def raw_period_dir_for_context(context: TwinRunContext) -> Path:
    return raw_dir(context.root)


def iter_raw_run_contexts(
    *,
    root: str | Path | None = None,
    scheme: str = "independent",
    scheme_name: str = "独立运行",
) -> Iterable[TwinRunContext]:
    tokens: set[str] = set()
    import re

    base = raw_dir(root)
    if base.exists():
        for path in base.rglob("*"):
            if not path.is_file() or path.name.endswith(".meta.json"):
                continue
            tokens.update(re.findall(r"(?<!\d)(20\d{6}|20\d{4})(?!\d)", path.name))
    for token in sorted(tokens):
        yield default_run_context(
            root=root,
            scheme=scheme,
            scheme_name=scheme_name,
            period=token,
        )


def infer_run_context_from_path(
    input_path: str | Path,
    *,
    root: str | Path | None = None,
    scheme: str = "independent",
    scheme_name: str = "独立运行",
    fallback_period: str = DEFAULT_PERIOD,
    fallback_period_name: str = DEFAULT_PERIOD_NAME,
) -> TwinRunContext:
    import re

    match = re.search(r"(?<!\d)(20\d{6}|20\d{4})(?!\d)", str(input_path))
    period = match.group(1) if match else fallback_period
    return default_run_context(
        root=root,
        scheme=scheme,
        scheme_name=scheme_name,
        period=period,
        period_name=fallback_period_name,
    )


def module_output_path(
    module_code: str,
    *,
    context: TwinRunContext | None = None,
    output_index: int = 0,
    relative_output: str | None = None,
) -> Path:
    context = ensure_run_context(context)
    code = str(module_code).upper()
    patterns = MODULE_OUTPUT_PATTERNS[code]
    pattern = relative_output if relative_output is not None else patterns[output_index]
    path = module_processed_dir(code, root=context.root) / pattern.format(period=context.period)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def module_output_paths(module_code: str, *, context: TwinRunContext | None = None) -> list[Path]:
    return [
        module_output_path(module_code, context=context, output_index=index)
        for index in range(len(MODULE_OUTPUT_PATTERNS[str(module_code).upper()]))
    ]


def ensure_raw_source_path(path: str | Path, *, root: str | Path | None = None) -> Path:
    raw_base = raw_dir(root).resolve()
    source = Path(path).expanduser().resolve()
    try:
        source.relative_to(raw_base)
    except ValueError as exc:
        raise ValueError(
            f"模块正式输入必须来自统一 raw 目录: {raw_base}\n当前选择: {source}"
        ) from exc
    if not source.exists():
        raise FileNotFoundError(source)
    return source


def standard_dialog_dir(
    kind: str = "raw",
    *,
    module_code: str | None = None,
    context: TwinRunContext | None = None,
    output_index: int = 0,
    root: str | Path | None = None,
) -> str:
    code = str(module_code).upper() if module_code else None
    if kind.lower() in {"output", "processed"}:
        path = module_processed_dir(code, root=root) if code else processed_dir(root)
    elif code and MODULE_RAW_DEFAULTS.get(code):
        category, *parts = MODULE_RAW_DEFAULTS[code]
        path = raw_data_dir(category, *parts, root=root)
    else:
        path = raw_dir(root)
    return str(path)


def latest_module_output(
    module_code: str,
    *,
    output_index: int = 0,
    root: str | Path | None = None,
) -> Path | None:
    base = module_processed_dir(module_code, root=root)
    if not base.exists():
        return None
    files = [
        path
        for path in base.rglob("*")
        if path.is_file() and not path.name.endswith(".meta.json") and path.name != "finish.tag"
    ]
    return max(files, key=lambda item: item.stat().st_mtime) if files else None


def mark_module_complete(context: TwinRunContext, module_codes: str | Iterable[str]) -> Path:
    codes = [module_codes] if isinstance(module_codes, str) else list(module_codes)
    last_tag: Path | None = None
    for raw_code in codes:
        code = str(raw_code).upper()
        base = module_processed_dir(code, root=context.root, create=True)
        outputs = sorted(
            path.relative_to(base).as_posix()
            for path in base.rglob("*")
            if path.is_file() and not path.name.endswith(".meta.json") and path.name != "finish.tag"
        )
        payload = {
            "module_code": code,
            "completed_outputs": outputs,
            "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        last_tag = base / "finish.tag"
        last_tag.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if last_tag is None:
        raise ValueError("module_codes 不能为空")
    return last_tag


def write_metadata_sidecar(
    output_path: str | Path,
    *,
    module_code: str,
    field: str,
    source_files: Iterable[str | Path],
    extra: dict | None = None,
) -> Path:
    details = dict(extra or {})
    return write_processed_metadata(
        output_path,
        module_code=module_code,
        output_type=str(details.pop("output_type", field)),
        source_files=source_files,
        model_weight=details.pop("model_weight", details.pop("model_checkpoint", None)),
        threshold_or_config=details.pop("threshold_or_config", details.pop("threshold", None)),
        extra=details,
    )


def write_raw_metadata_sidecar(
    input_path: str | Path,
    *,
    data_role: str,
    data_status: str,
    source_name: str,
    source_files: Iterable[str | Path],
    extra: dict | None = None,
) -> Path:
    details = dict(extra or {})
    consumers = details.pop("consumer_modules", details.pop("module_codes", []))
    return write_raw_metadata(
        input_path,
        data_type=str(details.pop("data_type", data_role)),
        dataset_name=str(details.pop("dataset_name", source_name)),
        source_files=source_files,
        source_origin=str(details.pop("source_origin", source_name)),
        consumer_modules=consumers,
        split=details.pop("split", None),
        sensor=details.pop("sensor", None),
        bands=details.pop("bands", None),
        dtype=details.pop("dtype", None),
        crs=details.pop("crs", None),
        date=details.pop("date", details.pop("business_date", None)),
        is_module_native=bool(details.pop("is_module_native", True)),
        extra={"data_status": data_status, **details},
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
    "MODULE_DIRECTORY_NAMES",
    "MODULE_OUTPUT_PATTERNS",
    "MODULE_RAW_DEFAULTS",
    "PROCESSED_TO_RAW_PERIOD_NAMES",
    "RAW_TO_PROCESSED_PERIOD_NAMES",
    "RAW_DATA_CATEGORIES",
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
    "ensure_raw_source_path",
    "ensure_run_context",
    "infer_run_context_from_path",
    "iter_module_specs",
    "iter_raw_run_contexts",
    "latest_module_output",
    "mark_module_complete",
    "module_input_path",
    "module_input_paths",
    "module_output_path",
    "module_output_paths",
    "module_processed_dir",
    "module_spec",
    "module_upstream_input_path",
    "module_upstream_input_paths",
    "period_to_date",
    "parse_period_folder_name",
    "processed_dir",
    "processed_period_name_from_raw",
    "processed_period_dir",
    "raster_dir",
    "raw_dir",
    "raw_data_dir",
    "raw_period_dir_for_context",
    "raw_period_name_from_processed",
    "raw_period_dir",
    "specs_as_dict",
    "standard_dialog_dir",
    "table_dir",
    "twin_root",
    "write_finish_tag",
    "write_metadata_sidecar",
    "write_processed_metadata",
    "write_raw_metadata",
    "write_raw_metadata_sidecar",
    "file_sha256",
    "write_standard_csv",
]
