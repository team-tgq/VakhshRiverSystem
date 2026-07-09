from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TWIN_DATA_ROOT = PROJECT_ROOT / "sample_data" / "瓦赫什流域孪生数据"

TARGET_CRS = "EPSG:32642"
TARGET_CRS_NAME = "WGS84 UTM 42N"
STUDY_YEAR_START = 2005
STUDY_YEAR_END = 2017
TIME_STEP = "daily"
DATE_FIELD = "date"
DATE_FORMAT = "YYYY-MM-DD"


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


def ensure_processed_period(*args, **kwargs) -> Path:
    period_dir = processed_period_dir(*args, **kwargs)
    (period_dir / "raster").mkdir(parents=True, exist_ok=True)
    (period_dir / "table").mkdir(parents=True, exist_ok=True)
    return period_dir


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


def specs_as_dict() -> list[dict]:
    return [asdict(spec) for spec in MODULE_SPECS]


__all__ = [
    "DATE_FIELD",
    "DATE_FORMAT",
    "DEFAULT_TWIN_DATA_ROOT",
    "MODULE_SPECS",
    "STANDARD_FIELDS",
    "STUDY_YEAR_END",
    "STUDY_YEAR_START",
    "TARGET_CRS",
    "TARGET_CRS_NAME",
    "TIME_STEP",
    "baseline_dir",
    "ensure_processed_period",
    "iter_module_specs",
    "module_spec",
    "processed_period_dir",
    "raster_dir",
    "raw_period_dir",
    "specs_as_dict",
    "table_dir",
    "twin_root",
    "write_finish_tag",
    "write_metadata_sidecar",
]
