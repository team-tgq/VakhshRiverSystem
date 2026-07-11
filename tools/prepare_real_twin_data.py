from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import stat
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.digital_twin_standard import (  # noqa: E402
    DEFAULT_TWIN_DATA_ROOT,
    MODULE_DIRECTORY_NAMES,
    default_run_context,
    file_sha256,
    mark_module_complete,
    module_processed_dir,
    raw_data_dir,
    write_processed_metadata,
    write_raw_metadata,
)
from algorithms.segformer_service.service_config import SEGFORMER_PYTHON  # noqa: E402


DEFAULT_REPORT = PROJECT_ROOT / "reports" / "data_architecture_migration_audit.json"
SEGFORMER_SOURCE = PROJECT_ROOT / "algorithms" / "segformer_service" / "data"
INUNDATION_SOURCE = PROJECT_ROOT / "algorithms" / "inundation_monitoring" / "data"
SWE_CACHE = PROJECT_ROOT / "algorithms" / "swe" / "output" / "daily_ml" / "cache"
FLOOD_DATA = PROJECT_ROOT / "algorithms" / "flood" / "data"
RESERVOIR_DIR = PROJECT_ROOT / "algorithms" / "reservoir_estimation"
ALLOCATION_DIR = PROJECT_ROOT / "algorithms" / "water_allocation"
VIDEO_SOURCE = PROJECT_ROOT / "algorithms" / "monitoring" / "data" / "测试数据" / "流速测试视频.mp4"


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    return path


def _path_record(path: Path, root: Path, reason: str) -> dict:
    metadata_path = path.with_suffix(path.suffix + ".meta.json")
    metadata = _read_json(metadata_path) if metadata_path.exists() else {}
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size if path.is_file() else None,
        "checksum": file_sha256(path) if path.is_file() else None,
        "reason": reason,
        "source_name": metadata.get("source_name") or metadata.get("source_origin"),
        "source_files": metadata.get("source_files", []),
    }


def build_audit(root: Path) -> dict:
    raw_root = root / "raw"
    processed_root = root / "processed"
    downloaded: list[dict] = []
    legacy_raw: list[dict] = []
    old_processed: list[dict] = []
    useful_to_migrate: list[dict] = []

    if raw_root.exists():
        for path in sorted(raw_root.rglob("*")):
            if not path.is_file() or path.name.endswith(".meta.json"):
                continue
            relative = path.relative_to(raw_root)
            if relative.parts and relative.parts[0] in {
                "remote_sensing",
                "meteorology",
                "land_surface",
                "snow_hydrology",
                "reservoir",
                "socioeconomic",
                "configuration",
                "video",
            }:
                continue
            if "哨兵影像" in path.name or "SAR影像" in path.name:
                downloaded.append(
                    _path_record(path, root, "此前通过 Planetary Computer 下载的临时 Sentinel 产品")
                )
            elif "积雪状态GEE" in path.name or path.suffix.lower() == ".mp4":
                useful_to_migrate.append(_path_record(path, root, "迁移后再删除旧时段副本"))
            else:
                legacy_raw.append(_path_record(path, root, "旧时段式 raw；不属于最终类型化入口"))

    if processed_root.exists():
        canonical = set(MODULE_DIRECTORY_NAMES.values())
        for child in sorted(processed_root.iterdir()):
            if child.name in canonical:
                continue
            if child.is_file():
                old_processed.append(_path_record(child, root, "processed 根目录旧文件"))
                continue
            for path in sorted(child.rglob("*")):
                if path.is_file():
                    old_processed.append(
                        _path_record(path, root, "旧方案/时段成果；来源链不符合独立模块结构")
                    )

    empty_segformer_masks = [
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in sorted(SEGFORMER_SOURCE.glob("*/masks_png/val/*"))
        if path.is_file() and path.stat().st_size == 0
    ]
    return {
        "created_at": _now(),
        "root": str(root),
        "policy": "audit-before-delete; local-only; no mock; checksum-before-cleanup",
        "delete_downloaded_sentinel": downloaded,
        "delete_legacy_raw_after_migration": legacy_raw,
        "delete_old_processed": old_processed,
        "migrate_before_delete": useful_to_migrate,
        "known_source_quality_issues": {
            "segformer_water_empty_validation_masks": empty_segformer_masks,
            "note": "0字节真值掩膜不迁移、不补造；对应影像仍用于模型推理。",
        },
    }


def _copy_verified(source: Path, target: Path) -> Path:
    if not source.is_file() or source.stat().st_size == 0:
        raise ValueError(f"源文件缺失或为空，拒绝迁移: {source}")
    source_hash = file_sha256(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0 and file_sha256(target) == source_hash:
        return target
    temporary = target.with_suffix(target.suffix + ".migrating")
    if temporary.exists():
        temporary.unlink()
    shutil.copy2(source, temporary)
    if file_sha256(temporary) != source_hash:
        temporary.unlink(missing_ok=True)
        raise IOError(f"迁移 checksum 不一致: {source} -> {target}")
    temporary.replace(target)
    return target


def _raw_meta(
    target: Path,
    *,
    data_type: str,
    dataset_name: str,
    source_files: Iterable[str | Path],
    source_origin: str,
    consumers: Iterable[str],
    split: str | None = None,
    sensor: str | None = None,
    bands: Sequence | None = None,
    dtype: str | Sequence | None = None,
    crs: str | None = None,
    date: str | None = None,
    is_module_native: bool = True,
    extra: Mapping | None = None,
) -> None:
    write_raw_metadata(
        target,
        data_type=data_type,
        dataset_name=dataset_name,
        source_files=source_files,
        source_origin=source_origin,
        consumer_modules=consumers,
        split=split,
        sensor=sensor,
        bands=bands,
        dtype=dtype,
        crs=crs,
        date=date,
        is_module_native=is_module_native,
        extra=extra,
    )


def _migrate_segformer(root: Path, summary: list[str]) -> None:
    copied = 0
    skipped_empty = 0
    for task in ("snow", "water"):
        target_base = raw_data_dir(
            "remote_sensing", "optical_rgb", f"segformer_{task}", "val", root=root, create=True
        )
        for kind, source_folder in (("images", "images_png"), ("masks", "masks_png")):
            for source in sorted((SEGFORMER_SOURCE / task / source_folder / "val").glob("*")):
                if not source.is_file():
                    continue
                if source.stat().st_size == 0:
                    skipped_empty += 1
                    continue
                target = _copy_verified(source, target_base / kind / source.name)
                _raw_meta(
                    target,
                    data_type="optical_rgb_tile" if kind == "images" else "reference_segmentation_mask",
                    dataset_name=f"SegFormer {task} native validation {kind}",
                    source_files=[source],
                    source_origin="algorithms/segformer_service/data module-native validation set",
                    consumers=["M01"],
                    split="val",
                    sensor="optical RGB validation tile" if kind == "images" else None,
                    is_module_native=True,
                    extra={"task": task, "label_available": kind == "masks"},
                )
                copied += 1
    summary.append(f"M01_RAW:copied={copied},skipped_empty_masks={skipped_empty}")


def _migrate_inundation(root: Path, summary: list[str]) -> None:
    copied = 0
    for source in sorted(INUNDATION_SOURCE.glob("*.tif")):
        if "_S1Weak" in source.name:
            parts = ("sentinel1_sar", "inundation_weak_labeled")
            sensor = "Sentinel-1 SAR"
            dataset_name = "Sen1Floods11 Bolivia weak-labeled sample"
            expected_bands = 2
        elif "_S2Hand" in source.name:
            parts = ("sentinel2_multispectral", "inundation_hand_labeled")
            sensor = "Sentinel-2 multispectral"
            dataset_name = "Sen1Floods11 India hand-labeled sample"
            expected_bands = 13
        else:
            continue
        target = _copy_verified(source, raw_data_dir("remote_sensing", *parts, root=root, create=True) / source.name)
        _raw_meta(
            target,
            data_type="sar_multiband_validation" if expected_bands == 2 else "multispectral_validation",
            dataset_name=dataset_name,
            source_files=[source],
            source_origin="algorithms/inundation_monitoring/data module-native validation sample",
            consumers=["M04"],
            split="validation",
            sensor=sensor,
            is_module_native=True,
            extra={
                "expected_band_count": expected_bands,
                "geographic_role": "external module validation; not Vakhsh basin business data",
            },
        )
        copied += 1
    summary.append(f"M04_RAW:copied={copied}")


def _migrate_snow_state(root: Path, summary: list[str]) -> None:
    copied = 0
    target_dir = raw_data_dir("remote_sensing", "gee", "snow_state", root=root, create=True)
    candidates = sorted((root / "raw").glob("*_*/??????_积雪状态GEE.tif"))
    if not candidates:
        candidates = sorted(target_dir.glob("*_积雪状态GEE.tif"))
    for source in candidates:
        target = source if source.parent == target_dir else _copy_verified(source, target_dir / source.name)
        token = source.name.split("_", 1)[0]
        date_text = f"{token[:4]}-{token[4:6]}-01" if len(token) == 6 else token
        _raw_meta(
            target,
            data_type="gee_snow_state_product",
            dataset_name="GEE Snow_State and Runoff_Probability product",
            source_files=[source],
            source_origin="existing module GEE export retained from local project data",
            consumers=["M06"],
            sensor="Google Earth Engine composite",
            bands=["Snow_State", "Runoff_Probability"],
            date=date_text,
            is_module_native=True,
            extra={"is_field_observation": False, "class_codes": {"1": "no_snow", "2": "dry_snow", "3": "wet_snow"}},
        )
        copied += 1
    summary.append(f"M06_RAW:copied={copied}")


def _local_swe_grid_reference(token: str, expected_shape: tuple[int, ...]) -> tuple[dict, Path]:
    import numpy as np
    import xarray as xr

    manifest = _read_json(SWE_CACHE.parent / "manifest.json")
    cycle_token = None
    target_date = f"{token[:4]}-{token[4:6]}-{token[6:8]}"
    for entry in manifest.get("entries", []):
        if entry.get("business_date") == target_date:
            cycle = str(entry.get("forcing_cycle") or "")
            cycle_token = cycle.replace("T", "").replace("Z", "")
            break

    candidates: list[Path] = []
    if cycle_token:
        candidates.append(SWE_CACHE / "gfs" / cycle_token / "terrain_f000.grib2")
    candidates.extend(sorted((SWE_CACHE / "gfs").glob("*/terrain_f000.grib2"), reverse=True))
    visited: set[Path] = set()
    for terrain_path in candidates:
        terrain_path = terrain_path.resolve()
        if terrain_path in visited or not terrain_path.is_file():
            continue
        visited.add(terrain_path)
        with xr.open_dataset(terrain_path, engine="cfgrib") as dataset:
            if "latitude" not in dataset.coords or "longitude" not in dataset.coords or "orog" not in dataset:
                continue
            latitudes = dataset["latitude"].values.astype(np.float32)
            longitudes = dataset["longitude"].values.astype(np.float32)
            orography = dataset["orog"].values.astype(np.float32)
        if orography.shape != expected_shape:
            continue
        if latitudes[0] < latitudes[-1]:
            latitudes = latitudes[::-1].copy()
            orography = orography[::-1, :].copy()
        return {
            "latitudes": latitudes,
            "longitudes": longitudes,
            "orography_m": orography,
        }, terrain_path
    raise FileNotFoundError(
        f"找不到与 forcing_{token}.npz 网格 {expected_shape} 对应的本地 GFS terrain_f000.grib2"
    )


def _migrate_swe_forcing(root: Path, summary: list[str]) -> None:
    import numpy as np

    copied = 0
    target_dir = raw_data_dir("meteorology", "daily_forcing", root=root, create=True)
    all_sources = sorted((SWE_CACHE / "forcing").glob("forcing_*.npz"))
    if not all_sources:
        summary.append("M02_RAW:daily_forcing=0")
        return
    latest_year = max(int(source.stem.rsplit("_", 1)[-1][:4]) for source in all_sources)
    sources = [
        source
        for source in all_sources
        if int(source.stem.rsplit("_", 1)[-1][:4]) == latest_year
    ]
    for stale in target_dir.glob("forcing_*.npz"):
        token = stale.stem.rsplit("_", 1)[-1]
        if int(token[:4]) == latest_year:
            continue
        stale.unlink()
        stale.with_suffix(stale.suffix + ".meta.json").unlink(missing_ok=True)

    for source in sources:
        token = source.stem.rsplit("_", 1)[-1]
        date_text = f"{token[:4]}-{token[4:6]}-{token[6:8]}"
        with np.load(source, allow_pickle=False) as archive:
            payload = {name: archive[name] for name in archive.files}
        source_files: list[Path] = [source]
        if not {"latitudes", "longitudes", "orography_m"}.issubset(payload):
            grid_payload, terrain_path = _local_swe_grid_reference(
                token,
                tuple(payload["temp_mean_c"].shape),
            )
            payload.update(grid_payload)
            source_files.append(terrain_path)
        target = target_dir / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, **payload)
        _raw_meta(
            target,
            data_type="daily_swe_forcing_archive",
            dataset_name="SWE daily forcing cache",
            source_files=source_files,
            source_origin="existing SWE pipeline cache derived from official GFS and VIIRS inputs with local GFS grid reference",
            consumers=["M02"],
            sensor="GFS + VIIRS",
            bands=[
                "temp_mean_c", "temp_min_c", "temp_max_c", "precipitation_mm",
                "solid_precip_mm", "snow_cover_fraction", "viirs_available", "viirs_missing",
                "latitudes", "longitudes", "orography_m",
            ],
            dtype="NPZ mixed arrays",
            crs="EPSG:4326",
            date=date_text,
            is_module_native=True,
            extra={
                "contains_model_output": False,
                "storage_format": "numpy_npz",
                "grid_shape": list(payload["temp_mean_c"].shape),
                "grid_reference_embedded": True,
            },
        )
        copied += 1
    summary.append(f"M02_RAW:daily_forcing={copied},selected_year={latest_year}")


def _migrate_flood_factors(root: Path, summary: list[str]) -> None:
    copied = 0
    processed = FLOOD_DATA / "processed"
    for source in sorted((processed / "daily").glob("rain_mm_demgrid_*.tif")):
        date_text = source.stem.replace("rain_mm_demgrid_", "")
        target = _copy_verified(source, raw_data_dir("meteorology", "precipitation", root=root, create=True) / source.name)
        grib = FLOOD_DATA / "raw" / "daily_gfs" / f"gfs_apcp24_{date_text}.grib2"
        _raw_meta(
            target,
            data_type="daily_precipitation_dem_aligned",
            dataset_name="GFS 24-hour accumulated precipitation",
            source_files=[source, grib],
            source_origin="NOAA/NCEP GFS public product, locally reprojected by the original M07 pipeline",
            consumers=["M07"],
            sensor="GFS",
            bands=["precipitation_mm"],
            date=date_text,
            is_module_native=True,
        )
        copied += 1
    for source in sorted((processed / "daily").glob("soil_moist_demgrid_*.tif")):
        date_text = source.stem.replace("soil_moist_demgrid_", "")
        target = _copy_verified(source, raw_data_dir("land_surface", "soil_moisture", root=root, create=True) / source.name)
        grib = FLOOD_DATA / "raw" / "daily_gfs" / f"gfs_soilw_0_0.1m_{date_text}.grib2"
        _raw_meta(
            target,
            data_type="surface_soil_moisture_dem_aligned",
            dataset_name="GFS 0-0.1m soil moisture",
            source_files=[source, grib],
            source_origin="NOAA/NCEP GFS public product, locally reprojected by the original M07 pipeline",
            consumers=["M07"],
            sensor="GFS",
            bands=["volumetric_soil_moisture"],
            date=date_text,
            is_module_native=True,
        )
        copied += 1
    landcover = processed / "landcover_demgrid.tif"
    if landcover.exists():
        target = _copy_verified(landcover, raw_data_dir("land_surface", "land_cover", root=root, create=True) / landcover.name)
        _raw_meta(
            target,
            data_type="categorical_land_cover_dem_aligned",
            dataset_name="ESA WorldCover land-cover classes",
            source_files=[landcover, FLOOD_DATA / "raw" / "worldcover_map.tif"],
            source_origin="ESA WorldCover public product, locally reprojected by the original M07 pipeline",
            consumers=["M07"],
            sensor="ESA WorldCover",
            bands=["land_cover_class"],
            is_module_native=True,
        )
        copied += 1
    summary.append(f"M07_RAW:factors={copied}")


def _migrate_video(root: Path, summary: list[str]) -> None:
    target_dir = raw_data_dir("video", "river_velocity", root=root, create=True)
    source = VIDEO_SOURCE
    if not source.exists():
        legacy = root / "raw" / "202605_综合业务评估" / "202605_河道视频.mp4"
        source = legacy if legacy.exists() else source
    if not source.exists():
        summary.append("M05_RAW:missing_video")
        return
    target = _copy_verified(source, target_dir / "202605_清水河道流速测试视频.mp4")
    _raw_meta(
        target,
        data_type="river_surface_velocity_video",
        dataset_name="2026-05 清水河道流速测试视频",
        source_files=[source],
        source_origin="module-provided clear-water channel test video; not a field observation",
        consumers=["M05"],
        sensor="test video camera",
        date="2026-05",
        is_module_native=True,
        extra={"is_field_observation": False, "scene_type": "clear_water_channel"},
    )
    summary.append("M05_RAW:video=1")


def _normalise_hypsometry(root: Path) -> Path:
    source = RESERVOIR_DIR / "output" / "Nurek" / "reservoir_hypsometry.csv"
    rows: list[dict] = []
    with source.open("r", encoding="utf-8-sig", newline="") as file:
        for item in csv.DictReader(file):
            rows.append(
                {
                    "reservoir_id": "nurek",
                    "elevation_m": item["Elevation (m)"],
                    "area_km2": item["Area (sq.km)"],
                    "storage_mcm": item["Storage (mcm)"],
                }
            )
    target = _write_csv(
        raw_data_dir("reservoir", "hypsometry", root=root, create=True) / "reservoir_hypsometry.csv",
        ["reservoir_id", "elevation_m", "area_km2", "storage_mcm"],
        rows,
    )
    _raw_meta(
        target,
        data_type="reservoir_hypsometry_curve",
        dataset_name="Nurek elevation-area-storage curve",
        source_files=[source, RESERVOIR_DIR / "reservoir_core.py"],
        source_origin="original reservoir module curve file",
        consumers=["M09"],
        is_module_native=True,
    )
    return target


def _prepare_reservoir_csv(root: Path, summary: list[str]) -> None:
    curve = _normalise_hypsometry(root)
    parameters = _write_csv(
        raw_data_dir("reservoir", "parameters", root=root, create=True) / "reservoir_parameters.csv",
        ["reservoir_id", "reservoir_name", "parameter", "value", "unit", "source"],
        [
            {"reservoir_id": "nurek", "reservoir_name": "Nurek", "parameter": "total_capacity", "value": 10.5, "unit": "km3", "source": "reservoir_core.NurekReservoirEstimator"},
            {"reservoir_id": "nurek", "reservoir_name": "Nurek", "parameter": "active_storage", "value": 4.2, "unit": "km3", "source": "reservoir_core.NurekReservoirEstimator"},
            {"reservoir_id": "nurek", "reservoir_name": "Nurek", "parameter": "curve_file", "value": curve.name, "unit": "path", "source": "reservoir_hypsometry.csv"},
        ],
    )
    _raw_meta(
        parameters,
        data_type="reservoir_algorithm_parameters",
        dataset_name="Nurek estimator parameters actually used by code",
        source_files=[RESERVOIR_DIR / "reservoir_core.py", curve],
        source_origin="current M09 implementation constants",
        consumers=["M09"],
        is_module_native=True,
        extra={"omitted_fields": "dead/flood/normal levels are not used by the current algorithm"},
    )

    native_observations = RESERVOIR_DIR / "output" / "Nurek" / "estimation_area.csv"
    observation_rows: list[dict] = []
    if native_observations.exists():
        with native_observations.open("r", encoding="utf-8-sig", newline="") as file:
            for item in csv.DictReader(file):
                observation_rows.append(
                    {
                        "date": item.get("date", ""),
                        "reservoir_id": "nurek",
                        "sensor": item.get("sensor", ""),
                        "water_level_m": "",
                        "surface_area_km2": item.get("post_outlier_removal_area_km2", ""),
                        "source": str(native_observations),
                        "quality_status": "module_native_estimate; source imagery unavailable for independent verification",
                    }
                )
    observations = _write_csv(
        raw_data_dir("reservoir", "observations", root=root, create=True) / "reservoir_observations.csv",
        ["date", "reservoir_id", "sensor", "water_level_m", "surface_area_km2", "source", "quality_status"],
        observation_rows,
    )
    _raw_meta(
        observations,
        data_type="reservoir_area_observation_record",
        dataset_name="M09 native reservoir area estimates",
        source_files=[native_observations],
        source_origin="existing original module estimation_area.csv",
        consumers=["M09"],
        date="2022-06-04",
        is_module_native=True,
        extra={"authenticity_status": "underlying source imagery not present; do not label as field measurement"},
    )
    summary.append(f"M09_RAW:parameters=1,observations={len(observation_rows)},curve=1")


def _latest_complete_discharge_year(source: Path) -> tuple[int, dict[int, float]]:
    monthly: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    with source.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            try:
                date_value = datetime.strptime(row["date"], "%Y-%m-%d")
                monthly[date_value.year][date_value.month].append(float(row["discharge"]))
            except (KeyError, TypeError, ValueError):
                continue
    years = [year for year, months in monthly.items() if len(months) == 12]
    if not years:
        raise ValueError(f"未找到完整年度流量记录: {source}")
    year = max(years)
    averages = {month: sum(values) / len(values) for month, values in monthly[year].items()}
    return year, averages


def _copy_public_allocation_data(root: Path) -> list[Path]:
    source_dir = ALLOCATION_DIR / "resources" / "data"
    targets: list[Path] = []
    for name in (
        "Afghanistan_water_discharger.csv",
        "Turkmenistan_water_data.csv",
        "Uzbekistan_water_data.csv",
        "AQUASTAT Dissemination System.csv",
    ):
        source = source_dir / name
        target = _copy_verified(source, raw_data_dir("socioeconomic", "water_demand", root=root, create=True) / name)
        _raw_meta(
            target,
            data_type="national_water_demand_statistics",
            dataset_name=name,
            source_files=[source],
            source_origin="FAO AQUASTAT public official statistics bundled with the original M08 module",
            consumers=["M08"],
            is_module_native=True,
        )
        targets.append(target)
    return targets


def _prepare_allocation_csv(root: Path, summary: list[str]) -> None:
    source = ALLOCATION_DIR / "resources" / "data" / "ERA5_daily_with_discharge.csv"
    source_year, monthly = _latest_complete_discharge_year(source)
    supply_dir = raw_data_dir("configuration", "water_allocation", "supply", root=root, create=True)
    global_config = _write_csv(
        supply_dir / "global_supply_config.csv",
        ["start_year", "start_month", "end_year", "end_month", "time_scale", "initial_surface_supply_mcm", "inflow_source_year"],
        [{"start_year": source_year, "start_month": 1, "end_year": source_year, "end_month": 12, "time_scale": "monthly", "initial_surface_supply_mcm": 850, "inflow_source_year": source_year}],
    )
    inflow = _write_csv(
        supply_dir / "monthly_inflow.csv",
        ["year", "month", "inflow_m3_s", "source", "quality_status"],
        [
            {"year": source_year, "month": month, "inflow_m3_s": f"{monthly[month]:.6f}", "source": str(source), "quality_status": "historical module input"}
            for month in range(1, 13)
        ],
    )
    demand = _write_csv(
        raw_data_dir("configuration", "water_allocation", "demand", root=root, create=True) / "demand_parameters.csv",
        ["parameter", "value", "unit", "ui_key"],
        [
            {"parameter": "population", "value": 387, "unit": "10k_people", "ui_key": "pop"},
            {"parameter": "urbanization_rate", "value": 23, "unit": "percent", "ui_key": "urban"},
            {"parameter": "population_growth_rate", "value": 1.8, "unit": "percent", "ui_key": "pop_growth"},
            {"parameter": "industrial_reuse_rate", "value": 25, "unit": "percent", "ui_key": "reuse"},
            {"parameter": "gdp", "value": 82, "unit": "100m_cny", "ui_key": "gdp"},
            {"parameter": "domestic_reuse_rate", "value": 15, "unit": "percent", "ui_key": "dom_reuse"},
            {"parameter": "irrigation_efficiency", "value": 0.85, "unit": "ratio", "ui_key": "eff"},
            {"parameter": "transmission_loss_rate", "value": 12, "unit": "percent", "ui_key": "loss"},
            {"parameter": "ecological_base_water", "value": 50, "unit": "million_m3", "ui_key": "eco"},
            {"parameter": "Rn", "value": 10.0, "unit": "mm_day", "ui_key": "meteo.Rn"},
            {"parameter": "G", "value": 0.0, "unit": "MJ_m2", "ui_key": "meteo.G"},
            {"parameter": "T", "value": 20.0, "unit": "degC", "ui_key": "meteo.T"},
            {"parameter": "u2", "value": 2.0, "unit": "m_s", "ui_key": "meteo.u2"},
            {"parameter": "es", "value": 23.4, "unit": "hPa", "ui_key": "meteo.es"},
            {"parameter": "ea", "value": 15.0, "unit": "hPa", "ui_key": "meteo.ea"},
            {"parameter": "delta", "value": 1.45, "unit": "hPa_degC", "ui_key": "meteo.delta"},
            {"parameter": "gamma", "value": 0.66, "unit": "hPa_degC", "ui_key": "meteo.gamma"},
        ],
    )
    crops = _write_csv(
        raw_data_dir("configuration", "water_allocation", "crops", root=root, create=True) / "crops.csv",
        ["crop_type", "growth_stage", "area_km2", "yield_kg_km2", "price_cny_kg", "kc"],
        [{"crop_type": "细绒棉", "growth_stage": "中期", "area_km2": 50, "yield_kg_km2": 300, "price_cny_kg": 7.5, "kc": 1.15}],
    )
    weights = _write_csv(
        raw_data_dir("configuration", "water_allocation", "decision_weights", root=root, create=True) / "decision_weights.csv",
        ["group", "name", "value"],
        [
            {"group": "preference", "name": "economic", "value": 0.33},
            {"group": "preference", "name": "shortage", "value": 0.33},
            {"group": "preference", "name": "gini", "value": 0.34},
        ]
        + [
            {"group": "sector", "name": name, "value": 1.0}
            for name in ("生活", "生态", "农业", "工业", "下游国家")
        ]
        + [
            {"group": "hydropower", "name": "max_power_mw", "value": 335},
            {"group": "hydropower", "name": "max_flow_m3_s", "value": 146},
            {"group": "hydropower", "name": "electricity_price_cny_kwh", "value": 0.4},
        ],
    )
    public_files = _copy_public_allocation_data(root)
    generated = (global_config, inflow, demand, crops, weights)
    for target in generated:
        _raw_meta(
            target,
            data_type="water_allocation_configuration",
            dataset_name=target.stem,
            source_files=[
                PROJECT_ROOT / "plugins" / "water_allocation_plugin" / "water_allocation_widget.py",
                ALLOCATION_DIR / "resources" / "data" / "ERA5_daily_with_discharge.csv",
            ],
            source_origin="current M08 UI defaults plus bundled historical discharge input",
            consumers=["M08"],
            date=str(source_year),
            is_module_native=True,
            extra={"configuration_not_observation": target != inflow},
        )
    summary.append(f"M08_RAW:config_files={len(generated)},public_statistics={len(public_files)},supply_year={source_year}")


def _remove_old_layout(root: Path, summary: list[str]) -> None:
    def remove_tree(path: Path) -> None:
        def clear_readonly(function, target, _error) -> None:
            os.chmod(target, stat.S_IREAD | stat.S_IWRITE)
            function(target)

        shutil.rmtree(path, onexc=clear_readonly)

    raw_root = (root / "raw").resolve()
    allowed_raw = {
        "remote_sensing", "meteorology", "land_surface", "snow_hydrology",
        "reservoir", "socioeconomic", "configuration", "video",
    }
    removed_raw: list[str] = []
    if raw_root.exists():
        for child in list(raw_root.iterdir()):
            if child.name in allowed_raw:
                continue
            child_resolved = child.resolve()
            child_resolved.relative_to(raw_root)
            if child.is_dir():
                remove_tree(child)
            else:
                child.unlink()
            removed_raw.append(child.name)

    processed_root = (root / "processed").resolve()
    allowed_processed = set(MODULE_DIRECTORY_NAMES.values())
    removed_processed: list[str] = []
    processed_root.mkdir(parents=True, exist_ok=True)
    for child in list(processed_root.iterdir()):
        if child.name in allowed_processed:
            continue
        child.resolve().relative_to(processed_root)
        if child.is_dir():
            remove_tree(child)
        else:
            child.unlink()
        removed_processed.append(child.name)

    removed_root_files: list[str] = []
    for name in ("manifest.json", "current_module_outputs_manifest.json"):
        path = root / name
        if path.exists():
            path.unlink()
            removed_root_files.append(name)
    summary.append(f"CLEANUP:raw={removed_raw},processed={removed_processed},root_files={removed_root_files}")


def _validate_migrated_raw(root: Path) -> dict:
    raw_root = root / "raw"
    errors: list[str] = []
    files = [
        path for path in raw_root.rglob("*")
        if path.is_file() and not path.name.endswith(".meta.json")
    ]
    for path in files:
        if path.stat().st_size == 0:
            errors.append(f"empty:{path.relative_to(root).as_posix()}")
            continue
        sidecar = path.with_suffix(path.suffix + ".meta.json")
        if not sidecar.exists():
            errors.append(f"missing_meta:{path.relative_to(root).as_posix()}")
            continue
        payload = _read_json(sidecar)
        required = {
            "data_type", "dataset_name", "source_files", "source_origin", "consumer_modules",
            "split", "sensor", "bands", "dtype", "crs", "date", "is_module_native",
            "checksum", "created_at",
        }
        missing = sorted(required - set(payload))
        if missing:
            errors.append(f"metadata_keys:{path.relative_to(root).as_posix()}:{missing}")
        elif payload.get("checksum") != file_sha256(path):
            errors.append(f"checksum:{path.relative_to(root).as_posix()}")
    top_level = sorted(path.name for path in root.iterdir()) if root.exists() else []
    if top_level != ["baseline", "processed", "raw"]:
        errors.append(f"root_children:{top_level}")
    return {"file_count": len(files), "errors": errors, "passed": not errors}


def migrate(root: Path, report_path: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    for name in ("baseline", "raw", "processed"):
        (root / name).mkdir(exist_ok=True)
    audit = build_audit(root)
    _write_json(report_path, audit)

    summary: list[str] = []
    _migrate_segformer(root, summary)
    _migrate_inundation(root, summary)
    _migrate_snow_state(root, summary)
    _migrate_swe_forcing(root, summary)
    _migrate_flood_factors(root, summary)
    _migrate_video(root, summary)
    _prepare_reservoir_csv(root, summary)
    _prepare_allocation_csv(root, summary)
    _remove_old_layout(root, summary)

    validation = _validate_migrated_raw(root)
    result = {
        "created_at": _now(),
        "root": str(root),
        "audit_report": str(report_path),
        "summary": summary,
        "validation": validation,
    }
    _write_json(report_path.with_name("data_architecture_migration_result.json"), result)
    if not validation["passed"]:
        raise RuntimeError("迁移验证失败:\n" + "\n".join(validation["errors"]))
    return result


def _reset_module_output(root: Path, module_code: str) -> Path:
    target = module_processed_dir(module_code, root=root)
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    return target


def _finish_module(root: Path, module_code: str) -> None:
    mark_module_complete(default_run_context(root=root), module_code)


def _processed_meta(
    path: Path,
    *,
    module_code: str,
    output_type: str,
    source_files: Iterable[str | Path],
    model_weight: str | Path | None = None,
    threshold_or_config=None,
    extra: Mapping | None = None,
) -> None:
    write_processed_metadata(
        path,
        module_code=module_code,
        output_type=output_type,
        source_files=source_files,
        model_weight=model_weight,
        threshold_or_config=threshold_or_config,
        extra=extra,
    )


def run_m01(root: Path, *, device: str = "cuda:0") -> dict:
    import numpy as np
    from PIL import Image

    from algorithms.segformer_service.service_config import TASKS
    from algorithms.segformer_service.service_runner import run_segformer_service

    output_root = _reset_module_output(root, "M01")
    scratch = PROJECT_ROOT / "algorithms" / "segformer_service" / "outputs" / "_unified_meta"
    scratch.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    metric_rows: list[dict] = []
    for task in ("snow", "water"):
        input_root = raw_data_dir(
            "remote_sensing", "optical_rgb", f"segformer_{task}", "val", root=root
        )
        image_dir = input_root / "images"
        reference_dir = input_root / "masks"
        images = sorted(
            path
            for path in image_dir.glob("*")
            if path.is_file() and not path.name.endswith(".meta.json")
        )
        if not images:
            raise FileNotFoundError(f"M01 {task} 缺少统一 raw 验证影像: {image_dir}")
        mask_dir = output_root / task / "val" / "masks"
        overlay_dir = output_root / task / "val" / "overlays"
        mask_dir.mkdir(parents=True, exist_ok=True)
        overlay_dir.mkdir(parents=True, exist_ok=True)
        for index, source in enumerate(images, start=1):
            mask_path = mask_dir / f"{source.stem}_mask.png"
            overlay_path = overlay_dir / f"{source.stem}_overlay.png"
            service_meta = scratch / f"{task}_{source.stem}_service.json"
            result = run_segformer_service(
                task,
                str(source),
                device=device,
                mask_path=str(mask_path),
                overlay_path=str(overlay_path),
                meta_path=str(service_meta),
            )
            if result["returncode"] != 0:
                raise RuntimeError(
                    f"M01 {task} 第 {index}/{len(images)} 张推理失败: {source}\n"
                    f"STDOUT:\n{result['stdout']}\nSTDERR:\n{result['stderr']}"
                )
            for target, output_type in (
                (mask_path, "segmentation_mask"),
                (overlay_path, "segmentation_overlay"),
            ):
                if not target.is_file() or target.stat().st_size == 0:
                    raise RuntimeError(f"SegFormer 未生成有效成果: {target}")
                _processed_meta(
                    target,
                    module_code="M01",
                    output_type=output_type,
                    source_files=[source],
                    model_weight=TASKS[task]["checkpoint"],
                    threshold_or_config={"task": task, "config": TASKS[task]["config"]},
                    extra={"task": task, "split": "val", "source_kind": "module_native_validation"},
                )

            reference_candidates = [
                reference_dir / f"{source.stem}.png",
                reference_dir / f"{source.stem}.jpg",
            ]
            reference = next(
                (path for path in reference_candidates if path.is_file() and path.stat().st_size > 0),
                None,
            )
            if reference is not None:
                predicted = np.asarray(Image.open(mask_path).convert("L")) > 0
                expected_image = Image.open(reference).convert("L")
                if expected_image.size != (predicted.shape[1], predicted.shape[0]):
                    expected_image = expected_image.resize(
                        (predicted.shape[1], predicted.shape[0]), Image.Resampling.NEAREST
                    )
                expected = np.asarray(expected_image) > 0
                intersection = int(np.count_nonzero(predicted & expected))
                union = int(np.count_nonzero(predicted | expected))
                metric_rows.append(
                    {
                        "task": task,
                        "split": "val",
                        "image": source.name,
                        "reference_mask": reference.name,
                        "iou": intersection / union if union else 1.0,
                        "predicted_positive_pixels": int(np.count_nonzero(predicted)),
                        "reference_positive_pixels": int(np.count_nonzero(expected)),
                    }
                )
        counts[task] = len(images)

    if metric_rows:
        metrics = _write_csv(
            output_root / "validation_metrics.csv",
            [
                "task",
                "split",
                "image",
                "reference_mask",
                "iou",
                "predicted_positive_pixels",
                "reference_positive_pixels",
            ],
            metric_rows,
        )
        metric_sources = [
            raw_data_dir(
                "remote_sensing", "optical_rgb", f"segformer_{task}", "val", root=root
            )
            / "masks"
            / row["reference_mask"]
            for task, row in ((item["task"], item) for item in metric_rows)
        ]
        _processed_meta(
            metrics,
            module_code="M01",
            output_type="validation_metrics",
            source_files=metric_sources,
            threshold_or_config={"metric": "binary_iou"},
            extra={"evaluated_pairs": len(metric_rows)},
        )
    _finish_module(root, "M01")
    return {"module": "M01", "counts": counts, "evaluated_pairs": len(metric_rows)}


def run_m04(root: Path, *, threshold: float = 0.5) -> dict:
    from algorithms.inundation_monitoring.inundation_inference import DEFAULT_WEIGHT
    from algorithms.inundation_monitoring.predictor import FloodPredictor

    output_root = _reset_module_output(root, "M04")
    predictor = FloodPredictor()
    jobs = (
        (
            raw_data_dir("remote_sensing", "sentinel1_sar", "inundation_weak_labeled", root=root),
            "sentinel1_sar_weak",
        ),
        (
            raw_data_dir("remote_sensing", "sentinel2_multispectral", "inundation_hand_labeled", root=root),
            "sentinel2_optical_hand",
        ),
    )
    outputs = 0
    water_ratios: dict[str, float] = {}
    for input_dir, group in jobs:
        images = sorted(
            path
            for path in input_dir.glob("*.tif")
            if path.is_file() and not path.name.endswith(".meta.json")
        )
        if not images:
            raise FileNotFoundError(f"M04 缺少统一 raw 输入: {input_dir}")
        for source in images:
            mask_path = output_root / group / "masks" / f"{source.stem}_mask.png"
            overlay_path = output_root / group / "overlays" / f"{source.stem}_overlay.png"
            result = predictor.predict(
                str(source),
                thresh=threshold,
                mask_path=mask_path,
                overlay_path=overlay_path,
            )
            for target, output_type in (
                (mask_path, "inundation_mask"),
                (overlay_path, "inundation_overlay"),
            ):
                _processed_meta(
                    target,
                    module_code="M04",
                    output_type=output_type,
                    source_files=[source],
                    model_weight=DEFAULT_WEIGHT,
                    threshold_or_config={"probability_threshold": threshold},
                    extra={
                        "validation_group": group,
                        "water_ratio": float(result["water_ratio"]),
                        "geographic_role": "external module validation; not Vakhsh basin business data",
                    },
                )
                outputs += 1
            water_ratios[source.name] = float(result["water_ratio"])
    _finish_module(root, "M04")
    return {"module": "M04", "output_count": outputs, "water_ratios": water_ratios}


def run_m06(root: Path) -> dict:
    from algorithms.snow_state.core import (
        SNOW_DENSITY_BY_TYPE_GCM3,
        process_local_gee_product,
    )

    output_root = _reset_module_output(root, "M06")
    input_dir = raw_data_dir("remote_sensing", "gee", "snow_state", root=root)
    sources = sorted(
        path
        for path in input_dir.glob("*.tif")
        if path.is_file() and not path.name.endswith(".meta.json")
    )
    if not sources:
        raise FileNotFoundError(f"M06 缺少统一 raw GEE 产品: {input_dir}")
    results = []
    for source in sources:
        token = source.name.split("_", 1)[0]
        snow_type = output_root / "rasters" / f"{token}_snow_type.tif"
        density = output_root / "rasters" / f"{token}_snow_density_gcm3.tif"
        statistics = output_root / "tables" / f"{token}_snow_state_statistics.csv"
        result = process_local_gee_product(source, snow_type, density, statistics)
        for target, output_type in (
            (snow_type, "snow_type"),
            (density, "snow_density_deterministic_mapping"),
            (statistics, "snow_state_statistics"),
        ):
            _processed_meta(
                target,
                module_code="M06",
                output_type=output_type,
                source_files=[source],
                threshold_or_config={"density_mapping_g_cm3": SNOW_DENSITY_BY_TYPE_GCM3},
                extra={
                    "source_product": "GEE Snow_State + Runoff_Probability",
                    "is_field_observation": False,
                    "density_is_measured": False,
                },
            )
        results.append(result)
    _finish_module(root, "M06")
    return {"module": "M06", "runs": results}


def run_m02(root: Path) -> dict:
    from algorithms.swe.daily_ml_pipeline import MODEL_PATH, run_unified_forcing_offline

    output_root = _reset_module_output(root, "M02")
    forcing_dir = raw_data_dir("meteorology", "daily_forcing", root=root)
    dem = root / "baseline" / "DEM.tif"
    boundary = root / "baseline" / "流域边界.shp"
    for baseline_source in (dem, boundary):
        if not baseline_source.is_file() or baseline_source.stat().st_size == 0:
            raise FileNotFoundError(baseline_source)
    sources = sorted(
        path
        for path in forcing_dir.glob("forcing_*.npz")
        if path.is_file() and not path.name.endswith(".meta.json")
    )
    if not sources:
        raise FileNotFoundError(f"M02 缺少统一 raw forcing: {forcing_dir}")

    results: list[dict] = []
    for source in sources:
        token = source.stem.rsplit("_", 1)[-1]
        swe = output_root / "rasters" / f"SWE_mm_{token}.tif"
        snowmelt = output_root / "rasters" / f"Snowmelt_mm_day_{token}.tif"
        statistics = output_root / "tables" / f"swe_statistics_{token}.csv"
        result = run_unified_forcing_offline(source, dem, swe, snowmelt, statistics)
        for target, output_type in (
            (swe, "swe_mm"),
            (snowmelt, "snowmelt_mm_day"),
            (statistics, "daily_swe_statistics"),
        ):
            _processed_meta(
                target,
                module_code="M02",
                output_type=output_type,
                source_files=[source, dem, boundary],
                model_weight=MODEL_PATH,
                threshold_or_config={
                    "mode": "offline_unified_forcing_replay",
                    "cold_start": True,
                    "viirs_no_snow_threshold": 0.1,
                },
                extra={
                    "business_date": result["date"],
                    "model_version": result["model_version"],
                    "forcing_shape": result["forcing_shape"],
                    "output_shape": result["shape"],
                    "terrain_source": result["terrain_source"],
                    "runoff_output_generated": False,
                },
            )
        results.append(result)

    series = _write_csv(
        output_root / "tables" / "daily_basin_series.csv",
        [
            "date",
            "swe_mean_mm",
            "swe_max_mm",
            "snowmelt_mean_mm_day",
            "snowmelt_max_mm_day",
            "valid_pixel_count",
            "cold_start",
            "viirs_missing",
            "model_version",
        ],
        results,
    )
    _processed_meta(
        series,
        module_code="M02",
        output_type="daily_basin_series",
        source_files=[Path(result["forcing"]) for result in results] + [dem, boundary],
        model_weight=MODEL_PATH,
        threshold_or_config={"mode": "offline_unified_forcing_replay"},
        extra={"run_count": len(results)},
    )
    _finish_module(root, "M02")
    return {"module": "M02", "runs": results, "skipped": [], "runoff_generated": False}


def run_m09(root: Path) -> dict:
    from algorithms.reservoir_estimation.reservoir_core import NurekReservoirEstimator

    output_root = _reset_module_output(root, "M09")
    parameters = raw_data_dir("reservoir", "parameters", root=root) / "reservoir_parameters.csv"
    observations = raw_data_dir("reservoir", "observations", root=root) / "reservoir_observations.csv"
    curve = raw_data_dir("reservoir", "hypsometry", root=root) / "reservoir_hypsometry.csv"
    for source in (parameters, observations, curve):
        if not source.is_file() or source.stat().st_size == 0:
            raise FileNotFoundError(source)

    with parameters.open("r", encoding="utf-8-sig", newline="") as file:
        parameter_rows = list(csv.DictReader(file))
    parameter_values = {row["parameter"]: row["value"] for row in parameter_rows}
    estimator = NurekReservoirEstimator(
        curve_path=curve,
        total_capacity_km3=float(parameter_values["total_capacity"]),
        active_storage_km3=float(parameter_values["active_storage"]),
    )
    with observations.open("r", encoding="utf-8-sig", newline="") as file:
        observation_rows = list(csv.DictReader(file))
    storage_rows: list[dict] = []
    for row in observation_rows:
        level_text = str(row.get("water_level_m", "")).strip()
        area_text = str(row.get("surface_area_km2", "")).strip()
        level = float(level_text) if level_text else None
        area = float(area_text) if area_text else None
        if level is None and area is None:
            continue
        estimate = estimator.estimate_manual(
            date=str(row.get("date", "")), water_level_m=level, area_km2=area
        )
        storage_rows.append(
            {
                "date": estimate.date,
                "reservoir_id": row.get("reservoir_id", "nurek"),
                "sensor": row.get("sensor", ""),
                "input_type": estimate.input_type,
                "water_level_m": estimate.water_level_m,
                "surface_area_km2": estimate.area_km2,
                "estimated_storage_mcm": estimate.estimated_volume_mcm,
                "estimated_storage_km3": estimate.estimated_volume_km3,
                "total_capacity_percent": estimate.total_capacity_percent,
                "active_storage_percent": estimate.active_storage_percent,
                "method": estimate.method,
                "quality_status": row.get("quality_status", ""),
                "warnings": "; ".join(estimate.warnings),
            }
        )
    if not storage_rows:
        raise RuntimeError("M09 observations 中没有可计算的水位或水面面积")

    storage = _write_csv(
        output_root / "tables" / "reservoir_storage.csv",
        [
            "date",
            "reservoir_id",
            "sensor",
            "input_type",
            "water_level_m",
            "surface_area_km2",
            "estimated_storage_mcm",
            "estimated_storage_km3",
            "total_capacity_percent",
            "active_storage_percent",
            "method",
            "quality_status",
            "warnings",
        ],
        storage_rows,
    )
    summary_payload = {
        "reservoir_id": "nurek",
        "observation_count": len(storage_rows),
        "curve_source": str(curve),
        "total_capacity_km3": estimator.total_capacity_km3,
        "active_storage_km3": estimator.active_storage_km3,
        "storage_min_mcm": min(float(row["estimated_storage_mcm"]) for row in storage_rows),
        "storage_max_mcm": max(float(row["estimated_storage_mcm"]) for row in storage_rows),
        "outflow_generated": False,
        "authenticity_note": "source imagery for the bundled 2022 area estimates is unavailable; results are module-native validation, not field observations",
        "created_at": _now(),
    }
    summary = output_root / "tables" / "estimation_summary.json"
    _write_json(summary, summary_payload)
    for target, output_type in (
        (storage, "reservoir_storage_estimate"),
        (summary, "reservoir_estimation_summary"),
    ):
        _processed_meta(
            target,
            module_code="M09",
            output_type=output_type,
            source_files=[parameters, observations, curve],
            threshold_or_config={
                "curve_interpolation": "linear_monotonic",
                "total_capacity_km3": estimator.total_capacity_km3,
                "active_storage_km3": estimator.active_storage_km3,
            },
            extra={"outflow_generated": False, "observation_count": len(storage_rows)},
        )
    _finish_module(root, "M09")
    return {"module": "M09", **summary_payload, "storage_csv": str(storage)}


def run_m08(root: Path, *, pop_size: int = 200, n_gen: int = 400) -> dict:
    from algorithms.water_allocation.core import run_allocation_from_config

    output_root = _reset_module_output(root, "M08")
    config_root = raw_data_dir("configuration", "water_allocation", root=root)
    result = run_allocation_from_config(config_root, pop_size=pop_size, n_gen=n_gen)
    plan = _write_csv(
        output_root / "tables" / "allocation_plan.csv",
        [
            "year",
            "month",
            "sector",
            "demand_million_m3",
            "surface_release_million_m3",
            "groundwater_million_m3",
            "received_million_m3",
            "shortage_million_m3",
            "satisfaction_ratio",
            "loss_rate",
        ],
        result["rows"],
    )
    summary_rows = [
        {"metric": "profit", "value": result["profit"], "unit": "algorithm_objective"},
        {"metric": "shortage", "value": result["shortage_million_m3"], "unit": "million_m3"},
        {"metric": "gini", "value": result["gini"], "unit": "ratio"},
        {"metric": "total_supply", "value": result["total_supply_million_m3"], "unit": "million_m3"},
        {"metric": "total_demand", "value": result["total_demand_million_m3"], "unit": "million_m3"},
    ]
    summary = _write_csv(
        output_root / "tables" / "allocation_summary.csv",
        ["metric", "value", "unit"],
        summary_rows,
    )
    sources = [Path(path) for path in result["config_files"]]
    for target, output_type in (
        (plan, "monthly_sector_allocation_plan"),
        (summary, "allocation_optimization_summary"),
    ):
        _processed_meta(
            target,
            module_code="M08",
            output_type=output_type,
            source_files=sources,
            threshold_or_config=result["optimizer"],
            extra={
                "year": result["year"],
                "inflow_is_explicit": result["inflow_is_explicit"],
                "independent_of_m09": True,
            },
        )
    _finish_module(root, "M08")
    return {
        "module": "M08",
        "year": result["year"],
        "profit": result["profit"],
        "shortage_million_m3": result["shortage_million_m3"],
        "gini": result["gini"],
        "plan_rows": len(result["rows"]),
        "optimizer": result["optimizer"],
    }


def run_m07(root: Path) -> dict:
    from algorithms.flood.risk_assessment_6factors_entropy import run_risk_assessment

    output_root = _reset_module_output(root, "M07")
    rain_dir = raw_data_dir("meteorology", "precipitation", root=root)
    soil_dir = raw_data_dir("land_surface", "soil_moisture", root=root)
    rain_by_date = {
        path.stem.replace("rain_mm_demgrid_", ""): path
        for path in rain_dir.glob("rain_mm_demgrid_*.tif")
    }
    soil_by_date = {
        path.stem.replace("soil_moist_demgrid_", ""): path
        for path in soil_dir.glob("soil_moist_demgrid_*.tif")
    }
    dates = sorted(set(rain_by_date) & set(soil_by_date))
    if not dates:
        raise FileNotFoundError("M07 没有日期匹配的降水与土壤湿度统一 raw")
    target_date = dates[-1]
    rain = rain_by_date[target_date]
    soil = soil_by_date[target_date]
    landcover = raw_data_dir("land_surface", "land_cover", root=root) / "landcover_demgrid.tif"
    dem = root / "baseline" / "DEM.tif"
    rivers = root / "baseline" / "河网.shp"
    study_area = root / "baseline" / "流域边界.shp"
    sources = [rain, soil, landcover, dem, rivers, study_area]
    for source in sources:
        if not source.exists():
            raise FileNotFoundError(source)

    cfg = {
        "study_area_shp": str(study_area),
        "dem_path": str(dem),
        "landcover_path": str(landcover),
        "rivers_path": str(rivers),
        "rain_path": str(rain),
        "soil_path": str(soil),
        "proc_dir": str(root / "raw"),
        "raw_dir": str(root / "raw"),
        "out_dir": str(output_root),
        "out_risk_tif": str(output_root / "rasters" / "flood_risk_index.tif"),
        "out_risk_level_tif": str(output_root / "rasters" / "flood_risk_level.tif"),
        "out_map": str(output_root / "visualizations" / "flood_risk_map.html"),
        "out_weights_txt": str(output_root / "tables" / "final_weights.txt"),
        "out_landuse_stats_csv": str(output_root / "tables" / "landuse_risk_stats.csv"),
        "out_landuse_summary_txt": str(output_root / "tables" / "landuse_risk_summary.txt"),
    }
    for key in (
        "out_risk_tif",
        "out_risk_level_tif",
        "out_map",
        "out_weights_txt",
        "out_landuse_stats_csv",
        "out_landuse_summary_txt",
    ):
        Path(cfg[key]).parent.mkdir(parents=True, exist_ok=True)
    result = run_risk_assessment(
        target_date=target_date,
        auto_prepare_static=False,
        allow_legacy_dynamic=False,
        auto_prepare_dynamic=False,
        cfg=cfg,
    )
    output_types = {
        "risk_tif": "flood_risk_index",
        "risk_level_tif": "flood_risk_level",
        "map_html": "flood_risk_visualization",
        "weights_txt": "factor_weights",
        "landuse_stats_csv": "landuse_risk_statistics",
        "landuse_summary_txt": "landuse_risk_summary",
    }
    for key, output_type in output_types.items():
        target = Path(result[key])
        _processed_meta(
            target,
            module_code="M07",
            output_type=output_type,
            source_files=sources,
            threshold_or_config={
                "weighting": "subjective_entropy_combination",
                "risk_level_method": result["risk_level_method"],
                "risk_level_breaks": result["risk_level_breaks"],
            },
            extra={"target_date": target_date, "final_weights": result["final_weights"]},
        )
    _finish_module(root, "M07")
    return {
        "module": "M07",
        "target_date": target_date,
        "final_weights": result["final_weights"],
        "risk_level_distribution": result["risk_level_distribution"],
    }


def run_m05(
    root: Path,
    *,
    device: str | None = None,
    max_dimension: int = 640,
    iters: int = 12,
) -> dict:
    from algorithms.raft.core import DEFAULT_MODEL_PATH, run_raft_video_full

    output_root = _reset_module_output(root, "M05")
    video_dir = raw_data_dir("video", "river_velocity", root=root)
    videos = sorted(path for path in video_dir.glob("*.mp4") if path.is_file())
    if len(videos) != 1:
        raise ValueError(f"M05 需要且只接受一个统一 raw 视频，当前 {len(videos)} 个: {video_dir}")
    source = videos[0]
    output_csv = output_root / "tables" / "frame_pair_velocity.csv"

    def progress(current: int, total: int) -> None:
        print(f"[M05] frame_pair={current}/{total}", flush=True)

    result = run_raft_video_full(
        source,
        output_csv,
        model_path=DEFAULT_MODEL_PATH,
        device=device,
        max_dimension=max_dimension,
        iters=iters,
        progress_callback=progress,
    )
    _processed_meta(
        output_csv,
        module_code="M05",
        output_type="full_video_frame_pair_velocity",
        source_files=[source],
        model_weight=DEFAULT_MODEL_PATH,
        threshold_or_config={
            "max_dimension": max_dimension,
            "iters": iters,
            "flow_distance_threshold_px": 0.2,
            "direction_filter_deg": 45.0,
        },
        extra={
            "processed_pair_count": result["processed_pair_count"],
            "read_frame_count": result["read_frame_count"],
            "fps": result["fps"],
            "physical_calibration": False,
            "velocity_m_s_status": result["velocity_m_s_status"],
        },
    )
    _finish_module(root, "M05")
    return {"module": "M05", **result}


MODULE_RUNNERS = {
    "M01": run_m01,
    "M02": run_m02,
    "M04": run_m04,
    "M05": run_m05,
    "M06": run_m06,
    "M07": run_m07,
    "M08": run_m08,
    "M09": run_m09,
}


def run_modules(
    root: Path,
    module_codes: Sequence[str],
    *,
    segformer_device: str,
    raft_device: str | None,
    raft_max_dimension: int,
    raft_iters: int,
    allocation_pop_size: int,
    allocation_generations: int,
) -> dict:
    report_path = DEFAULT_REPORT.with_name("independent_module_run_report.json")
    requested_codes = [str(code).strip().upper() for code in module_codes]
    existing = _read_json(report_path)
    same_root = existing.get("root") == str(root)
    results: list[dict] = [
        record
        for record in existing.get("modules", [])
        if same_root and record.get("module") not in requested_codes
    ]
    previous_requested = existing.get("requested_modules", []) if same_root else []
    reported_codes = list(dict.fromkeys([*previous_requested, *requested_codes]))
    payload = {
        "created_at": existing.get("created_at", _now()) if same_root else _now(),
        "root": str(root),
        "interpreter": sys.executable,
        "entrypoint": str(Path(__file__).resolve()),
        "requested_modules": reported_codes,
        "modules": results,
    }
    for raw_code in module_codes:
        code = raw_code.strip().upper()
        if code not in MODULE_RUNNERS:
            raise ValueError(f"不支持独立运行的模块: {raw_code}")
        started_at = _now()
        started_clock = time.perf_counter()
        print(f"[RUN] {code} start", flush=True)
        run_record = {
            "module": code,
            "status": "running",
            "started_at": started_at,
            "runner": f"tools.prepare_real_twin_data.{MODULE_RUNNERS[code].__name__}",
            "interpreter": (
                str(Path(SEGFORMER_PYTHON).resolve())
                if code == "M01"
                else sys.executable
            ),
        }
        results.append(run_record)
        _write_json(report_path, payload)
        try:
            if code == "M01":
                result = run_m01(root, device=segformer_device)
                run_record["runtime_config"] = {"device": segformer_device}
            elif code == "M05":
                result = run_m05(
                    root,
                    device=raft_device,
                    max_dimension=raft_max_dimension,
                    iters=raft_iters,
                )
                run_record["runtime_config"] = {
                    "device": raft_device or "auto",
                    "max_dimension": raft_max_dimension,
                    "iters": raft_iters,
                }
            elif code == "M08":
                result = run_m08(
                    root,
                    pop_size=allocation_pop_size,
                    n_gen=allocation_generations,
                )
                run_record["runtime_config"] = {
                    "population_size": allocation_pop_size,
                    "generations": allocation_generations,
                }
            else:
                result = MODULE_RUNNERS[code](root)
            run_record.update(result)
            run_record["status"] = "completed"
        except Exception as exc:
            run_record["status"] = "failed"
            run_record["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            run_record["finished_at"] = _now()
            run_record["duration_seconds"] = round(time.perf_counter() - started_clock, 3)
            payload["updated_at"] = _now()
            _write_json(report_path, payload)
        print(f"[RUN] {code} complete", flush=True)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="基于仓库本地真实/原生数据重构统一 raw/processed 架构")
    parser.add_argument("--root", default=str(DEFAULT_TWIN_DATA_ROOT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--audit-only", action="store_true", help="只生成删除/迁移候选清单，不改动数据")
    parser.add_argument("--migrate", action="store_true", help="校验复制后清理旧时段/方案目录")
    parser.add_argument("--run-modules", action="store_true", help="从统一 raw 独立运行指定模块")
    parser.add_argument(
        "--modules",
        default="M01,M06,M02,M09,M08,M07,M04,M05",
        help="逗号分隔模块编号；M03 当前不运行",
    )
    parser.add_argument("--segformer-device", default="cuda:0")
    parser.add_argument("--raft-device", default="", help="留空时自动选择 CUDA/CPU")
    parser.add_argument("--raft-max-dimension", type=int, default=640)
    parser.add_argument("--raft-iters", type=int, default=12)
    parser.add_argument("--allocation-pop-size", type=int, default=200)
    parser.add_argument("--allocation-generations", type=int, default=400)
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    report = Path(args.report).expanduser().resolve()
    if sum(bool(value) for value in (args.audit_only, args.migrate, args.run_modules)) != 1:
        parser.error("必须且只能指定 --audit-only、--migrate 或 --run-modules 之一")

    if args.audit_only:
        payload = build_audit(root)
        _write_json(report, payload)
        print(f"[OK] 审计清单: {report}")
        print(f"[AUDIT] downloaded={len(payload['delete_downloaded_sentinel'])}")
        print(f"[AUDIT] legacy_raw={len(payload['delete_legacy_raw_after_migration'])}")
        print(f"[AUDIT] old_processed={len(payload['delete_old_processed'])}")
        print(f"[AUDIT] migrate_first={len(payload['migrate_before_delete'])}")
        return 0

    if args.migrate:
        result = migrate(root, report)
        print("[OK] 类型化 raw 迁移完成")
        for item in result["summary"]:
            print(f"[MIGRATE] {item}")
        print(f"[VERIFY] raw_files={result['validation']['file_count']} passed={result['validation']['passed']}")
        return 0

    module_codes = [item.strip() for item in args.modules.split(",") if item.strip()]
    payload = run_modules(
        root,
        module_codes,
        segformer_device=args.segformer_device,
        raft_device=args.raft_device.strip() or None,
        raft_max_dimension=args.raft_max_dimension,
        raft_iters=args.raft_iters,
        allocation_pop_size=args.allocation_pop_size,
        allocation_generations=args.allocation_generations,
    )
    completed_codes = [
        code.strip().upper()
        for code in module_codes
        if code.strip()
    ]
    print(f"[OK] 独立模块运行完成: {', '.join(completed_codes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
