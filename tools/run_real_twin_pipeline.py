from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.features
import rasterio.warp
from rasterio.enums import Resampling


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.digital_twin_standard import (  # noqa: E402
    DEFAULT_TWIN_DATA_ROOT,
    RAW_TO_PROCESSED_PERIOD_NAMES,
    TARGET_CRS,
    TwinRunContext,
    default_run_context,
    mark_module_complete,
    module_output_path,
    period_to_date,
    write_metadata_sidecar,
    write_standard_csv,
)


SCHEMES = (
    ("scheme01", "常规调度工况"),
    ("scheme02", "优化分水工况"),
)
SECTORS = ("生活", "生态", "农业", "工业", "下游国家")


@dataclass
class GridData:
    profile: dict
    mask: np.ndarray
    dem: np.ndarray
    pixel_area_km2: float


def _period_from_raw_dir(raw_dir: Path) -> tuple[str, str, str]:
    token, raw_name = raw_dir.name.split("_", 1)
    period_name = RAW_TO_PROCESSED_PERIOD_NAMES.get(raw_name, f"{raw_name}模拟")
    return token, raw_name, period_name


def _source_paths(root: Path, raw_dir: Path, period: str) -> dict[str, Path]:
    return {
        "s2": raw_dir / f"{period}_哨兵影像.tif",
        "sar": raw_dir / f"{period}_SAR影像.tif",
        "weather": raw_dir / f"{period}_逐日气象.csv",
        "reservoir": raw_dir / f"{period}_水库参数.csv",
        "video": raw_dir / f"{period}_河道视频.mp4",
        "basin": root / "baseline" / "流域边界.shp",
        "dem": root / "baseline" / "DEM.tif",
    }


def _read_weather(weather_csv: Path) -> dict[str, float]:
    rows = []
    with weather_csv.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            rows.append(row)
    if not rows:
        raise ValueError(f"气象文件无数据行: {weather_csv}")

    def mean_field(name: str) -> float:
        values = [float(row[name]) for row in rows if str(row.get(name, "")).strip() != ""]
        return float(np.mean(values)) if values else 0.0

    return {
        "temp_mean_c": mean_field("temp_mean_c"),
        "temp_min_c": mean_field("temp_min_c"),
        "temp_max_c": mean_field("temp_max_c"),
        "precipitation_mm": mean_field("precipitation_mm"),
        "solid_precip_mm": mean_field("solid_precip_mm"),
    }


def _grid_from_sentinel(s2_path: Path, basin_path: Path, dem_path: Path) -> GridData:
    basin = gpd.read_file(basin_path)
    if basin.crs is None:
        basin = basin.set_crs(TARGET_CRS)

    with rasterio.open(s2_path) as src:
        profile = src.profile.copy()
        if src.crs is None:
            raise ValueError(f"输入哨兵影像缺少 CRS: {s2_path}")
        if src.crs.to_string() != TARGET_CRS:
            raise ValueError(f"输入哨兵影像 CRS 应为 {TARGET_CRS}: {s2_path}")

        basin_utm = basin.to_crs(src.crs)
        mask = rasterio.features.geometry_mask(
            basin_utm.geometry,
            out_shape=(src.height, src.width),
            transform=src.transform,
            invert=True,
        )
        pixel_area_km2 = abs(float(src.transform.a) * float(src.transform.e)) / 1_000_000.0

        dem = np.full((src.height, src.width), np.nan, dtype=np.float32)
        with rasterio.open(dem_path) as dem_src:
            rasterio.warp.reproject(
                source=rasterio.band(dem_src, 1),
                destination=dem,
                src_transform=dem_src.transform,
                src_crs=dem_src.crs,
                src_nodata=dem_src.nodata,
                dst_transform=src.transform,
                dst_crs=src.crs,
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )
        dem = np.where(mask, dem, np.nan).astype(np.float32)

    profile.update(driver="GTiff", crs=TARGET_CRS, compress="deflate", tiled=True, BIGTIFF="IF_SAFER")
    return GridData(profile=profile, mask=mask, dem=dem, pixel_area_km2=pixel_area_km2)


def _write_raster(path: Path, data: np.ndarray, grid: GridData, *, dtype: str, nodata, description: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = grid.profile.copy()
    profile.update(count=1, dtype=dtype, nodata=nodata)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype(dtype), 1)
        dst.set_band_description(1, description)
    return path


def _read_sentinel_features(s2_path: Path, mask: np.ndarray) -> dict[str, np.ndarray]:
    with rasterio.open(s2_path) as src:
        data = src.read().astype(np.float32)
        descriptions = [str(item or "").upper() for item in src.descriptions]

    def by_desc(name: str, fallback_index: int) -> np.ndarray:
        if name in descriptions:
            return data[descriptions.index(name)]
        return data[min(fallback_index, data.shape[0] - 1)]

    if data.shape[0] >= 5:
        blue = by_desc("B02", 0)
        green = by_desc("B03", 1)
        red = by_desc("B04", 2)
        nir = by_desc("B08", 3)
        swir = by_desc("B11", 4)
    else:
        red = data[0]
        green = data[1] if data.shape[0] > 1 else data[0]
        blue = data[2] if data.shape[0] > 2 else data[0]
        nir = green
        swir = red

    valid = mask & (blue > 0) & (green > 0) & (red > 0)
    eps = 1e-6
    ndsi = (green - swir) / (green + swir + eps)
    ndvi = (nir - red) / (nir + red + eps)
    brightness = (blue + green + red) / 3.0
    return {
        "blue": blue,
        "green": green,
        "red": red,
        "nir": nir,
        "swir": swir,
        "ndsi": ndsi,
        "ndvi": ndvi,
        "brightness": brightness,
        "valid": valid,
    }


def run_m01(context: TwinRunContext, sources: dict[str, Path], grid: GridData, weather: dict[str, float]) -> dict[str, Path]:
    features = _read_sentinel_features(sources["s2"], grid.mask)
    valid = features["valid"]
    bright_threshold = np.nanpercentile(features["brightness"][valid], 55) if np.any(valid) else 0.0
    if sources["s2"].exists():
        snow = (
            valid
            & (features["ndsi"] > 0.15)
            & (features["ndvi"] < 0.45)
            & (features["brightness"] >= bright_threshold)
        )
    else:
        snow = np.zeros_like(grid.mask, dtype=bool)

    dem_valid = grid.dem[np.isfinite(grid.dem) & grid.mask]
    if dem_valid.size:
        dem_norm = (grid.dem - float(np.nanpercentile(dem_valid, 35))) / max(
            1.0,
            float(np.nanpercentile(dem_valid, 95) - np.nanpercentile(dem_valid, 35)),
        )
    else:
        dem_norm = np.zeros_like(grid.dem)
    cold_factor = np.clip((2.0 - float(weather["temp_mean_c"])) / 12.0, 0.0, 1.0)
    snow_cover = np.where(grid.mask, snow.astype(np.float32), 0.0)
    snow_depth = np.where(
        snow,
        np.clip(0.05 + 0.65 * np.nan_to_num(dem_norm, nan=0.0) + 0.25 * cold_factor, 0.02, 1.20),
        0.0,
    ).astype(np.float32)

    snow_depth_path = module_output_path("M01", context=context, output_index=0)
    snow_cover_path = module_output_path("M01", context=context, output_index=1)
    stats_path = module_output_path("M01", context=context, output_index=2)
    _write_raster(snow_depth_path, snow_depth, grid, dtype="float32", nodata=-9999.0, description="snow_depth_m")
    _write_raster(snow_cover_path, snow_cover, grid, dtype="float32", nodata=-9999.0, description="snow_cover_fraction")

    snow_area_km2 = float(np.count_nonzero(snow) * grid.pixel_area_km2)
    basin_area_km2 = float(np.count_nonzero(grid.mask) * grid.pixel_area_km2)
    rows = [
        {
            "date": period_to_date(context.period),
            "period": context.period,
            "scheme": context.scheme,
            "module_code": "M01",
            "snow_area_km2": f"{snow_area_km2:.6f}",
            "basin_area_km2": f"{basin_area_km2:.6f}",
            "snow_cover_ratio": f"{(snow_area_km2 / basin_area_km2 if basin_area_km2 else 0.0):.6f}",
            "method": "NDSI/NDVI rule from real Sentinel-2, DEM and weather",
        }
    ]
    write_standard_csv(
        stats_path,
        fieldnames=list(rows[0].keys()),
        rows=rows,
    )

    common_extra = {
        "scheme": context.scheme,
        "scheme_name": context.scheme_name,
        "period": context.period,
        "period_name": context.period_name,
        "data_status": "generated_from_real_raw",
        "method": "rule_based_snow_index_from_sentinel2_dem_weather",
    }
    write_metadata_sidecar(snow_depth_path, module_code="M01", field="snow_depth", source_files=[sources["s2"], sources["dem"], sources["weather"]], extra=common_extra)
    write_metadata_sidecar(snow_cover_path, module_code="M01", field="snow_cover", source_files=[sources["s2"], sources["dem"], sources["weather"]], extra=common_extra)
    write_metadata_sidecar(stats_path, module_code="M01", field="snow_cover", source_files=[snow_cover_path], extra=common_extra)
    return {"snow_depth": snow_depth_path, "snow_cover": snow_cover_path, "stats": stats_path}


def run_m06(context: TwinRunContext, sources: dict[str, Path], grid: GridData, weather: dict[str, float]) -> dict[str, Path]:
    snow_cover_path = module_output_path("M01", context=context, output_index=1)
    with rasterio.open(snow_cover_path) as src:
        snow_cover = src.read(1).astype(np.float32)
    dem_valid = grid.dem[np.isfinite(grid.dem) & grid.mask]
    mean_dem = float(np.nanmean(dem_valid)) if dem_valid.size else 0.0
    temp_grid = float(weather["temp_mean_c"]) - 0.0065 * (np.nan_to_num(grid.dem, nan=mean_dem) - mean_dem)
    snow = (snow_cover > 0.5) & grid.mask
    snow_type = np.full(snow_cover.shape, 1, dtype=np.uint8)
    snow_type[snow & (temp_grid <= 0.0)] = 2
    snow_type[snow & (temp_grid > 0.0)] = 3
    density = np.zeros(snow_cover.shape, dtype=np.float32)
    density[snow_type == 2] = 0.25
    density[snow_type == 3] = 0.40

    type_path = module_output_path("M06", context=context, output_index=0)
    density_path = module_output_path("M06", context=context, output_index=1)
    _write_raster(type_path, snow_type, grid, dtype="uint8", nodata=255, description="snow_type")
    _write_raster(density_path, density, grid, dtype="float32", nodata=-9999.0, description="snow_density_gcm3")
    extra = {
        "scheme": context.scheme,
        "scheme_name": context.scheme_name,
        "period": context.period,
        "period_name": context.period_name,
        "data_status": "generated_from_real_raw",
        "method": "temperature_dem_phase_partition_from_m01_snow_cover",
    }
    write_metadata_sidecar(type_path, module_code="M06", field="snow_type", source_files=[snow_cover_path, sources["dem"], sources["weather"]], extra=extra)
    write_metadata_sidecar(density_path, module_code="M06", field="snow_density", source_files=[type_path], extra=extra)
    return {"snow_type": type_path, "snow_density": density_path}


def run_m02(context: TwinRunContext, sources: dict[str, Path], grid: GridData, weather: dict[str, float]) -> dict[str, Path]:
    with rasterio.open(module_output_path("M01", context=context, output_index=0)) as src:
        snow_depth = src.read(1).astype(np.float32)
    with rasterio.open(module_output_path("M06", context=context, output_index=1)) as src:
        density = src.read(1).astype(np.float32)
    with rasterio.open(module_output_path("M01", context=context, output_index=1)) as src:
        snow_cover = src.read(1).astype(np.float32)

    swe = np.where(grid.mask, np.clip(snow_depth * density * 1000.0, 0.0, None), 0.0).astype(np.float32)
    liquid_precip = max(0.0, float(weather["precipitation_mm"]) - float(weather["solid_precip_mm"]))
    melt = max(0.0, float(weather["temp_mean_c"])) * 2.5 * snow_cover
    runoff = np.where(grid.mask, np.clip(liquid_precip + melt, 0.0, None), 0.0).astype(np.float32)

    swe_path = module_output_path("M02", context=context, output_index=0)
    runoff_path = module_output_path("M02", context=context, output_index=1)
    _write_raster(swe_path, swe, grid, dtype="float32", nodata=-9999.0, description="swe_mm")
    _write_raster(runoff_path, runoff, grid, dtype="float32", nodata=-9999.0, description="runoff_mm")
    extra = {
        "scheme": context.scheme,
        "scheme_name": context.scheme_name,
        "period": context.period,
        "period_name": context.period_name,
        "data_status": "generated_from_real_raw",
        "method": "SWE from snow depth and density; runoff from liquid precipitation and degree-day melt",
    }
    write_metadata_sidecar(swe_path, module_code="M02", field="swe", source_files=[module_output_path("M01", context=context, output_index=0), module_output_path("M06", context=context, output_index=1)], extra=extra)
    write_metadata_sidecar(runoff_path, module_code="M02", field="runoff", source_files=[swe_path, sources["weather"]], extra=extra)
    return {"swe": swe_path, "runoff": runoff_path}


def run_m03(context: TwinRunContext, sources: dict[str, Path], grid: GridData) -> dict[str, Path]:
    runoff_path = module_output_path("M02", context=context, output_index=1)
    with rasterio.open(runoff_path) as src:
        runoff = src.read(1).astype(np.float32)
    runoff_valid = runoff[grid.mask]
    mean_runoff = float(np.nanmean(runoff_valid)) if runoff_valid.size else 0.0
    basin_area_m2 = float(np.count_nonzero(grid.mask) * grid.pixel_area_km2 * 1_000_000.0)
    discharge_m3_s = mean_runoff / 1000.0 * basin_area_m2 / 86400.0

    dem_valid = grid.dem[np.isfinite(grid.dem) & grid.mask]
    if dem_valid.size:
        lowland = 1.0 - (grid.dem - float(np.nanmin(dem_valid))) / max(1.0, float(np.nanmax(dem_valid) - np.nanmin(dem_valid)))
    else:
        lowland = np.zeros_like(runoff)
    flood_depth = np.where(
        grid.mask,
        np.clip((runoff / max(1.0, mean_runoff + 1.0)) * (0.05 + 0.60 * np.nan_to_num(lowland, nan=0.0)), 0.0, 2.0),
        0.0,
    ).astype(np.float32)
    inundation = ((flood_depth > 0.10) & grid.mask).astype(np.uint8)

    discharge_path = module_output_path("M03", context=context, output_index=0)
    depth_path = module_output_path("M03", context=context, output_index=1)
    inundation_path = module_output_path("M03", context=context, output_index=2)
    rows = [
        {
            "date": period_to_date(context.period),
            "period": context.period,
            "scheme": context.scheme,
            "module_code": "M03",
            "runoff_mean_mm_day": f"{mean_runoff:.6f}",
            "basin_area_km2": f"{basin_area_m2 / 1_000_000.0:.6f}",
            "discharge_m3_s": f"{discharge_m3_s:.6f}",
            "method": "lumped runoff-to-discharge conversion from M02 runoff raster",
        }
    ]
    write_standard_csv(discharge_path, fieldnames=list(rows[0].keys()), rows=rows)
    _write_raster(depth_path, flood_depth, grid, dtype="float32", nodata=-9999.0, description="flood_depth_m")
    _write_raster(inundation_path, inundation, grid, dtype="uint8", nodata=255, description="inundation")
    extra = {
        "scheme": context.scheme,
        "scheme_name": context.scheme_name,
        "period": context.period,
        "period_name": context.period_name,
        "data_status": "generated_from_real_raw",
        "method": "runoff_dem_lumped_routing",
    }
    write_metadata_sidecar(discharge_path, module_code="M03", field="discharge", source_files=[runoff_path], extra=extra)
    write_metadata_sidecar(depth_path, module_code="M03", field="flood_depth", source_files=[runoff_path, sources["dem"]], extra=extra)
    write_metadata_sidecar(inundation_path, module_code="M03", field="inundation", source_files=[depth_path], extra=extra)
    return {"discharge": discharge_path, "flood_depth": depth_path, "inundation": inundation_path}


def _read_first_numeric(csv_path: Path, field: str) -> float:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            value = str(row.get(field, "")).strip()
            if value:
                return float(value)
    return 0.0


def run_m09(context: TwinRunContext, sources: dict[str, Path]) -> dict[str, Path]:
    discharge_path = module_output_path("M03", context=context, output_index=0)
    discharge = _read_first_numeric(discharge_path, "discharge_m3_s")
    storage_rows_raw = []
    with sources["reservoir"].open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            storage_rows_raw.append(row)
    if not storage_rows_raw:
        raise ValueError(f"水库参数为空: {sources['reservoir']}")

    storages = np.asarray([float(row["storage_mcm"]) for row in storage_rows_raw], dtype=np.float32)
    elevations = np.asarray([float(row["elevation_m"]) for row in storage_rows_raw], dtype=np.float32)
    idx = int(np.clip(round(0.55 * (len(storages) - 1) + min(discharge, 2000.0) / 2000.0 * 0.25 * (len(storages) - 1)), 0, len(storages) - 1))
    storage_mcm = float(storages[idx])
    elevation_m = float(elevations[idx])
    outflow = max(10.0, min(3500.0, discharge * (0.85 if context.scheme == "scheme02" else 1.0)))

    storage_path = module_output_path("M09", context=context, output_index=0)
    outflow_path = module_output_path("M09", context=context, output_index=1)
    date = period_to_date(context.period)
    storage_rows = [
        {
            "date": date,
            "period": context.period,
            "scheme": context.scheme,
            "module_code": "M09",
            "reservoir_name": "Nurek",
            "water_level_m": f"{elevation_m:.3f}",
            "storage_million_m3": f"{storage_mcm:.6f}",
            "storage_10k_m3": f"{storage_mcm * 100.0:.6f}",
            "storage_km3": f"{storage_mcm / 1000.0:.6f}",
            "source_discharge_m3_s": f"{discharge:.6f}",
        }
    ]
    outflow_rows = [
        {
            "date": date,
            "period": context.period,
            "scheme": context.scheme,
            "module_code": "M09",
            "reservoir_name": "Nurek",
            "outflow_m3_s": f"{outflow:.6f}",
            "source_discharge_m3_s": f"{discharge:.6f}",
            "rule": "scheme02 reduces release for optimized allocation" if context.scheme == "scheme02" else "conventional release follows routed discharge",
        }
    ]
    write_standard_csv(storage_path, fieldnames=list(storage_rows[0].keys()), rows=storage_rows)
    write_standard_csv(outflow_path, fieldnames=list(outflow_rows[0].keys()), rows=outflow_rows)
    extra = {
        "scheme": context.scheme,
        "scheme_name": context.scheme_name,
        "period": context.period,
        "period_name": context.period_name,
        "data_status": "generated_from_real_raw",
        "method": "Nurek hypsometry curve with M03 discharge",
    }
    write_metadata_sidecar(storage_path, module_code="M09", field="storage", source_files=[sources["reservoir"], discharge_path], extra=extra)
    write_metadata_sidecar(outflow_path, module_code="M09", field="outflow", source_files=[storage_path, discharge_path], extra=extra)
    return {"storage": storage_path, "outflow": outflow_path}


def run_m08(context: TwinRunContext) -> dict[str, Path]:
    storage_path = module_output_path("M09", context=context, output_index=0)
    outflow_path = module_output_path("M09", context=context, output_index=1)
    storage_mcm = _read_first_numeric(storage_path, "storage_million_m3")
    outflow_m3_s = _read_first_numeric(outflow_path, "outflow_m3_s")
    monthly_supply = max(0.0, outflow_m3_s * 86400.0 * 30.4 / 1_000_000.0)
    reserve_supply = max(0.0, storage_mcm * 0.015)
    total_supply = monthly_supply + reserve_supply
    demand = {
        "生活": 18.0,
        "生态": 25.0,
        "农业": 110.0,
        "工业": 30.0,
        "下游国家": 85.0,
    }
    if context.scheme == "scheme02":
        weights = {"生活": 1.25, "生态": 1.20, "农业": 1.00, "工业": 0.95, "下游国家": 1.05}
    else:
        weights = {"生活": 1.00, "生态": 1.00, "农业": 1.00, "工业": 1.00, "下游国家": 1.00}
    weighted_demand = {sector: demand[sector] * weights[sector] for sector in SECTORS}
    total_weighted = sum(weighted_demand.values())
    rows = []
    for sector in SECTORS:
        allocated = total_supply * weighted_demand[sector] / total_weighted if total_weighted else 0.0
        received = min(allocated, demand[sector])
        shortage = max(0.0, demand[sector] - received)
        rows.append(
            {
                "date": period_to_date(context.period),
                "period": context.period,
                "period_name": context.period_name,
                "scheme": context.scheme,
                "scheme_name": context.scheme_name,
                "module_code": "M08",
                "time_scale": "monthly",
                "sector": sector,
                "demand_million_m3": f"{demand[sector]:.6f}",
                "surface_release_million_m3": f"{allocated:.6f}",
                "groundwater_million_m3": "0.000000",
                "received_million_m3": f"{received:.6f}",
                "shortage_million_m3": f"{shortage:.6f}",
                "satisfaction_ratio_pct": f"{(received / demand[sector] * 100.0 if demand[sector] else 100.0):.3f}",
                "loss_rate_pct": "0.000",
                "gini": "0.000000",
                "profit_10k_yuan": "0.000000",
                "total_supply_million_m3": f"{total_supply:.6f}",
                "n_periods": 1,
            }
        )
    output_path = module_output_path("M08", context=context)
    write_standard_csv(output_path, fieldnames=list(rows[0].keys()), rows=rows)
    write_metadata_sidecar(
        output_path,
        module_code="M08",
        field="allocation",
        source_files=[storage_path, outflow_path],
        extra={
            "scheme": context.scheme,
            "scheme_name": context.scheme_name,
            "period": context.period,
            "period_name": context.period_name,
            "data_status": "generated_from_real_raw",
            "method": "scheme-aware allocation from M09 available supply and sector demands",
            "sectors": list(SECTORS),
        },
    )
    return {"allocation": output_path}


def _write_simple_xlsx(path: Path, rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet_rows = []
    for row_idx, row in enumerate(rows, start=1):
        cells = []
        for col_idx, value in enumerate(row, start=1):
            col = ""
            n = col_idx
            while n:
                n, rem = divmod(n - 1, 26)
                col = chr(65 + rem) + col
            ref = f"{col}{row_idx}"
            text = escape(str(value))
            cells.append(f'<c r="{ref}" t="inlineStr"><is><t>{text}</t></is></c>')
        sheet_rows.append(f'<row r="{row_idx}">{"".join(cells)}</row>')
    sheet_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(sheet_rows)}</sheetData></worksheet>'
    )
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        content_types = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            "</Types>"
        )
        package_rels = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
            "</Relationships>"
        )
        workbook_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            '<sheets><sheet name="stats" sheetId="1" r:id="rId1"/></sheets></workbook>'
        )
        workbook_rels = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
            "</Relationships>"
        )
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", package_rels)
        zf.writestr("xl/workbook.xml", workbook_xml)
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        zf.writestr("xl/worksheets/sheet1.xml", sheet_xml)


def run_m04(context: TwinRunContext, sources: dict[str, Path], grid: GridData) -> dict[str, Path]:
    with rasterio.open(sources["sar"]) as src:
        sar = src.read().astype(np.float32)
    vv = sar[0]
    vh = sar[1] if sar.shape[0] > 1 else sar[0]
    score = vv + vh
    valid = grid.mask & np.isfinite(score) & (score > 0)
    threshold = float(np.nanpercentile(score[valid], 18)) if np.any(valid) else 0.0
    inundation = (valid & (score <= threshold)).astype(np.uint8)
    output_tif = module_output_path("M04", context=context, output_index=0)
    stats_xlsx = module_output_path("M04", context=context, output_index=1)
    _write_raster(output_tif, inundation, grid, dtype="uint8", nodata=255, description="observed_inundation")
    inundated_area = float(np.count_nonzero(inundation) * grid.pixel_area_km2)
    basin_area = float(np.count_nonzero(grid.mask) * grid.pixel_area_km2)
    rows = [
        ["field", "value", "unit"],
        ["date", period_to_date(context.period), ""],
        ["period", context.period, ""],
        ["scheme", context.scheme, ""],
        ["module_code", "M04", ""],
        ["threshold", threshold, "SAR VV+VH"],
        ["inundated_area_km2", inundated_area, "km2"],
        ["basin_area_km2", basin_area, "km2"],
        ["water_ratio_pct", (inundated_area / basin_area * 100.0 if basin_area else 0.0), "%"],
        ["source_sar", str(sources["sar"]), ""],
    ]
    _write_simple_xlsx(stats_xlsx, rows)
    extra = {
        "scheme": context.scheme,
        "scheme_name": context.scheme_name,
        "period": context.period,
        "period_name": context.period_name,
        "data_status": "generated_from_real_raw",
        "method": "low-backscatter threshold from Sentinel-1 RTC VV/VH",
        "threshold": threshold,
    }
    write_metadata_sidecar(output_tif, module_code="M04", field="inundation", source_files=[sources["sar"]], extra=extra)
    write_metadata_sidecar(stats_xlsx, module_code="M04", field="inundated_area", source_files=[sources["sar"], output_tif], extra=extra)
    return {"observed_inundation": output_tif, "stats": stats_xlsx}


def run_m07(context: TwinRunContext, sources: dict[str, Path], grid: GridData) -> dict[str, Path]:
    depth_path = module_output_path("M03", context=context, output_index=1)
    model_inundation_path = module_output_path("M03", context=context, output_index=2)
    observed_path = module_output_path("M04", context=context, output_index=0)
    with rasterio.open(depth_path) as src:
        depth = src.read(1).astype(np.float32)
    with rasterio.open(model_inundation_path) as src:
        model_inundation = src.read(1).astype(np.uint8)
    with rasterio.open(observed_path) as src:
        observed = src.read(1).astype(np.uint8)
    risk = np.zeros(depth.shape, dtype=np.uint8)
    risk[(depth > 0.05) | (model_inundation == 1) | (observed == 1)] = 1
    risk[depth > 0.15] = 2
    risk[depth > 0.30] = 3
    risk[depth > 0.60] = 4
    risk[(observed == 1) & (depth > 0.30)] = 5
    risk = np.where(grid.mask, risk, 0).astype(np.uint8)
    output_path = module_output_path("M07", context=context)
    _write_raster(output_path, risk, grid, dtype="uint8", nodata=255, description="flood_risk_class")
    write_metadata_sidecar(
        output_path,
        module_code="M07",
        field="flood_risk",
        source_files=[depth_path, model_inundation_path, observed_path],
        extra={
            "scheme": context.scheme,
            "scheme_name": context.scheme_name,
            "period": context.period,
            "period_name": context.period_name,
            "data_status": "generated_from_real_raw",
            "method": "risk classes from M03 flood depth, M03 inundation and M04 observed SAR inundation",
        },
    )
    return {"risk": output_path}


def run_m05(context: TwinRunContext, sources: dict[str, Path]) -> dict[str, Path]:
    output_path = module_output_path("M05", context=context)
    date = period_to_date(context.period)
    if sources["video"].exists():
        rows = [
            {
                "date": date,
                "period": context.period,
                "scheme": context.scheme,
                "module_code": "M05",
                "velocity_m_s": "",
                "flow_direction_degree": "",
                "data_status": "video_present_not_run_in_batch",
                "source_video": str(sources["video"]),
                "note": "批处理未运行 RAFT/LK；请在 RAFT 插件中用该视频生成精确流速。",
            }
        ]
        source_files = [sources["video"]]
        data_status = "video_present_not_run_in_batch"
    else:
        rows = [
            {
                "date": date,
                "period": context.period,
                "scheme": context.scheme,
                "module_code": "M05",
                "velocity_m_s": "",
                "flow_direction_degree": "",
                "data_status": "not_available",
                "source_video": "",
                "note": "该时段 raw 中没有真实河道视频，未生成实测流速。",
            }
        ]
        source_files = [f"missing:{sources['video']}"]
        data_status = "not_available"
    write_standard_csv(output_path, fieldnames=list(rows[0].keys()), rows=rows)
    write_metadata_sidecar(
        output_path,
        module_code="M05",
        field="velocity",
        source_files=source_files,
        extra={
            "scheme": context.scheme,
            "scheme_name": context.scheme_name,
            "period": context.period,
            "period_name": context.period_name,
            "data_status": data_status,
            "method": "RAFT/LK optical-flow module requires real raw video",
        },
    )
    return {"velocity": output_path}


def run_context_pipeline(context: TwinRunContext, root: Path, raw_dir: Path) -> None:
    period, _, _ = _period_from_raw_dir(raw_dir)
    sources = _source_paths(root, raw_dir, period)
    for key in ("s2", "sar", "weather", "reservoir", "basin", "dem"):
        if not sources[key].exists():
            raise FileNotFoundError(f"缺少 {key} 输入: {sources[key]}")
    weather = _read_weather(sources["weather"])
    grid = _grid_from_sentinel(sources["s2"], sources["basin"], sources["dem"])
    run_m01(context, sources, grid, weather)
    run_m06(context, sources, grid, weather)
    run_m02(context, sources, grid, weather)
    run_m03(context, sources, grid)
    run_m09(context, sources)
    run_m08(context)
    run_m04(context, sources, grid)
    run_m07(context, sources, grid)
    run_m05(context, sources)
    mark_module_complete(context, ["M01", "M06", "M02", "M03", "M09", "M08", "M04", "M07", "M05"])


def run_pipeline(root: Path) -> None:
    raw_root = root / "raw"
    raw_dirs = sorted(path for path in raw_root.iterdir() if path.is_dir() and "_" in path.name)
    if not raw_dirs:
        raise RuntimeError(f"没有可调度的 raw 时段目录: {raw_root}")
    for raw_dir in raw_dirs:
        period, _, period_name = _period_from_raw_dir(raw_dir)
        for scheme, scheme_name in SCHEMES:
            context = default_run_context(root=root, scheme=scheme, scheme_name=scheme_name, period=period, period_name=period_name)
            print(f"[RUN] {context.scheme_folder}/{context.period_folder}")
            run_context_pipeline(context, root, raw_dir)
            print(f"[OK] {context.scheme_folder}/{context.period_folder}")


def main() -> int:
    parser = argparse.ArgumentParser(description="从真实 raw 统一调度生成 processed 标准成果")
    parser.add_argument("--root", default=str(DEFAULT_TWIN_DATA_ROOT), help="瓦赫什流域孪生数据根目录")
    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()
    run_pipeline(root)
    manifest_path = root / "processed" / "pipeline_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "root": str(root),
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "status": "processed_generated_from_real_raw",
                "note": "M05 requires real raw video; rows are marked not_available when video is absent.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[OK] 统一调度完成: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
