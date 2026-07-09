from __future__ import annotations

import argparse
import calendar
import csv
import json
import math
import shutil
import time
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
import planetary_computer
import rasterio
import requests
from pyproj import Transformer
from pystac_client import Client
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.merge import merge
from rasterio.vrt import WarpedVRT
from rasterio.warp import calculate_default_transform, reproject
from shapely.geometry import LineString
from shapely.ops import polygonize, unary_union


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.digital_twin_standard import (  # noqa: E402
    DATE_FIELD,
    DATE_FORMAT,
    RAW_TO_PROCESSED_PERIOD_NAMES,
    TARGET_CRS,
    TARGET_CRS_NAME,
    TIME_STEP,
)


REAL_TWIN_ROOT = PROJECT_ROOT / "data" / "瓦赫什流域孪生数据"
LOCAL_VAKHSH_SOURCE = PROJECT_ROOT.parent / "Data" / "瓦赫什河"
LOCAL_BASIN_SHP = LOCAL_VAKHSH_SOURCE / "研究区域" / "瓦赫什河流域上游.shp"
LOCAL_DEM = LOCAL_VAKHSH_SOURCE / "瓦赫什河流域上游DEM.tif"
LOCAL_HYDRORIVERS = PROJECT_ROOT / "algorithms" / "flood" / "data" / "raw" / "hydrorivers.gpkg"
LOCAL_RESERVOIR_CURVE = (
    PROJECT_ROOT / "algorithms" / "reservoir_estimation" / "output" / "Nurek" / "reservoir_hypsometry.csv"
)
DEFAULT_LOCAL_RAW_ROOTS = (
    LOCAL_VAKHSH_SOURCE / "raw",
    PROJECT_ROOT / "local_data" / "瓦赫什流域孪生数据" / "raw",
)

OSM_NUREK_RELATION_ID = "2590442"
PLANETARY_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
NASA_POWER_DAILY = "https://power.larc.nasa.gov/api/temporal/daily/point"
USER_AGENT = "VakhshRiverSystem real data preparation"


DEFAULT_FALLBACK_YEAR = 2017
DEFAULT_SENTINEL_DATES = {
    2015: {"03": ("2015-03-31", "2015-03-31"), "07": ("2015-07-30", "2015-07-29")},
    2016: {"03": ("2016-03-24", "2016-03-24"), "07": ("2016-07-22", "2016-07-22")},
    2017: {"03": ("2017-03-13", "2017-03-30"), "07": ("2017-07-31", "2017-07-28")},
}


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def build_periods(fallback_year: int) -> dict[str, dict]:
    if fallback_year not in DEFAULT_SENTINEL_DATES:
        raise ValueError("当前脚本的 Sentinel 自动下载只支持 2015-2017；2005-2014 请提供本地 Landsat/MODIS 等真实 raw 数据")

    year = f"{fallback_year:04d}"
    march_s2, march_s1 = DEFAULT_SENTINEL_DATES[fallback_year]["03"]
    july_s2, july_s1 = DEFAULT_SENTINEL_DATES[fallback_year]["07"]
    return {
        f"{year}03": {
            "raw_name": "融雪期",
            "processed_name": "融雪模拟",
            "month_start": f"{year}-03-01",
            "month_end": f"{year}-03-31",
            "sentinel2_date": march_s2,
            "sentinel1_date": march_s1,
        },
        f"{year}07": {
            "raw_name": "汛期",
            "processed_name": "汛期模拟",
            "month_start": f"{year}-07-01",
            "month_end": f"{year}-07-31",
            "sentinel2_date": july_s2,
            "sentinel1_date": july_s1,
        },
    }


def build_local_periods(local_raw_periods: dict[str, str]) -> dict[str, dict]:
    periods: dict[str, dict] = {}
    for period, raw_name in sorted(local_raw_periods.items()):
        year = int(period[:4])
        month = int(period[4:6])
        last_day = calendar.monthrange(year, month)[1]
        processed_name = RAW_TO_PROCESSED_PERIOD_NAMES.get(raw_name, f"{raw_name}模拟")
        periods[period] = {
            "raw_name": raw_name,
            "processed_name": processed_name,
            "month_start": f"{year:04d}-{month:02d}-01",
            "month_end": f"{year:04d}-{month:02d}-{last_day:02d}",
            "source": "local_raw",
        }
    return periods


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _period_token(folder_name: str) -> str | None:
    token = folder_name.split("_", 1)[0]
    if token.isdigit() and len(token) in (6, 8):
        year = int(token[:4])
        month = int(token[4:6])
        if 2005 <= year <= 2017 and 1 <= month <= 12:
            return token[:6]
    return None


def _copy_local_raw_root(local_raw_root: Path | None, root: Path) -> dict[str, str]:
    if local_raw_root is None:
        return {}

    source_root = local_raw_root.expanduser().resolve()
    if not source_root.exists():
        print(f"[WARN] 本地 raw 根目录不存在，跳过本地优先接入: {source_root}")
        return {}

    copied: dict[str, str] = {}
    target_raw = root / "raw"
    target_raw.mkdir(parents=True, exist_ok=True)
    for source_period_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
        period = _period_token(source_period_dir.name)
        if period is None:
            continue
        target_period_dir = target_raw / source_period_dir.name
        if target_period_dir.exists():
            shutil.rmtree(target_period_dir)
        shutil.copytree(source_period_dir, target_period_dir)
        copied[period] = source_period_dir.name.split("_", 1)[1] if "_" in source_period_dir.name else "业务时段"
        print(f"[OK] 已优先接入本地 raw: {source_period_dir} -> {target_period_dir}")
    return copied


def _resolve_local_raw_root(raw_root_arg: str) -> Path | None:
    if raw_root_arg.strip():
        return Path(raw_root_arg).expanduser().resolve()
    for candidate in DEFAULT_LOCAL_RAW_ROOTS:
        if candidate.exists():
            return candidate.resolve()
    return None


def _sidecar(path: Path, *, role: str, source_name: str, source_files: Iterable[str], **extra) -> None:
    payload = {
        "file": path.name,
        "data_role": role,
        "data_status": "real",
        "source_name": source_name,
        "source_files": list(source_files),
        "crs": TARGET_CRS,
        "crs_name": TARGET_CRS_NAME,
        "time_step": TIME_STEP,
        "date_field": DATE_FIELD,
        "date_format": DATE_FORMAT,
        "created_at": _now(),
    }
    payload.update(extra)
    _write_json(path.with_suffix(path.suffix + ".meta.json"), payload)


def _copy_reproject_basin(target_baseline: Path) -> gpd.GeoDataFrame:
    if not LOCAL_BASIN_SHP.exists():
        raise FileNotFoundError(f"缺少本地真实流域边界: {LOCAL_BASIN_SHP}")

    basin = gpd.read_file(LOCAL_BASIN_SHP)
    if basin.crs is None:
        basin = basin.set_crs("EPSG:4326")
    basin_utm = basin.to_crs(TARGET_CRS)
    out = target_baseline / "流域边界.shp"
    basin_utm.to_file(out, encoding="utf-8")
    _sidecar(
        out,
        role="baseline",
        source_name="本地瓦赫什河上游研究区边界",
        source_files=[str(LOCAL_BASIN_SHP)],
        processing="Reprojected to EPSG:32642; geometry unchanged.",
    )
    return basin_utm


def _reproject_dem(target_baseline: Path, basin_utm: gpd.GeoDataFrame) -> Path:
    if not LOCAL_DEM.exists():
        raise FileNotFoundError(f"缺少本地真实 DEM: {LOCAL_DEM}")

    out = target_baseline / "DEM.tif"
    with rasterio.open(LOCAL_DEM) as src:
        transform, width, height = calculate_default_transform(
            src.crs,
            TARGET_CRS,
            src.width,
            src.height,
            *src.bounds,
        )
        profile = src.profile.copy()
        profile.update(
            crs=TARGET_CRS,
            transform=transform,
            width=width,
            height=height,
            compress="deflate",
            tiled=True,
            BIGTIFF="IF_SAFER",
        )
        with rasterio.open(out, "w", **profile) as dst:
            for band_index in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, band_index),
                    destination=rasterio.band(dst, band_index),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=TARGET_CRS,
                    resampling=Resampling.bilinear,
                    src_nodata=src.nodata,
                    dst_nodata=src.nodata,
                )

    _sidecar(
        out,
        role="baseline",
        source_name="本地瓦赫什河上游 DEM",
        source_files=[str(LOCAL_DEM)],
        processing="Reprojected to EPSG:32642 for unified model access.",
        watershed_bounds=[float(v) for v in basin_utm.total_bounds],
    )
    return out


def _prepare_rivers(target_baseline: Path, basin_utm: gpd.GeoDataFrame) -> Path:
    if not LOCAL_HYDRORIVERS.exists():
        raise FileNotFoundError(f"缺少本地 HydroRIVERS 河网: {LOCAL_HYDRORIVERS}")

    rivers = gpd.read_file(LOCAL_HYDRORIVERS)
    if rivers.crs is None:
        rivers = rivers.set_crs("EPSG:4326")
    rivers_utm = rivers.to_crs(TARGET_CRS)
    clipped = gpd.clip(rivers_utm, basin_utm)
    out = target_baseline / "河网.shp"
    clipped.to_file(out, encoding="utf-8")
    _sidecar(
        out,
        role="baseline",
        source_name="HydroRIVERS clipped to Vakhsh upper basin",
        source_files=[str(LOCAL_HYDRORIVERS), str(LOCAL_BASIN_SHP)],
        processing="Clipped by basin boundary and reprojected to EPSG:32642.",
        feature_count=int(len(clipped)),
    )
    return out


def _download_osm_relation_xml(relation_id: str) -> bytes:
    url = f"https://www.openstreetmap.org/api/0.6/relation/{relation_id}/full"
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=120)
            response.raise_for_status()
            return response.content
        except Exception as exc:
            last_error = exc
            time.sleep(attempt * 2)
    raise RuntimeError(f"无法下载 OSM relation {relation_id}: {last_error}") from last_error


def _osm_relation_polygon(xml_bytes: bytes, relation_id: str):
    root = ET.fromstring(xml_bytes)
    nodes = {
        int(node.attrib["id"]): (float(node.attrib["lon"]), float(node.attrib["lat"]))
        for node in root.findall("node")
    }
    ways: dict[int, list[tuple[float, float]]] = {}
    for way in root.findall("way"):
        refs = [int(nd.attrib["ref"]) for nd in way.findall("nd")]
        coords = [nodes[ref] for ref in refs if ref in nodes]
        if len(coords) >= 2:
            ways[int(way.attrib["id"])] = coords

    relation = root.find(f"relation[@id='{relation_id}']")
    if relation is None:
        raise RuntimeError(f"OSM relation {relation_id} not found in downloaded XML")

    outer_lines = []
    for member in relation.findall("member"):
        if member.attrib.get("type") != "way" or member.attrib.get("role") == "inner":
            continue
        coords = ways.get(int(member.attrib["ref"]))
        if coords:
            outer_lines.append(LineString(coords))
    polygons = list(polygonize(outer_lines))
    if not polygons:
        raise RuntimeError(f"OSM relation {relation_id} has no polygonized outer boundary")
    return unary_union(polygons)


def _prepare_reservoir_boundary(target_baseline: Path) -> Path:
    out = target_baseline / "水库边界.shp"
    if out.exists() and out.with_suffix(out.suffix + ".meta.json").exists():
        return out

    cache_xml = target_baseline / f"osm_relation_{OSM_NUREK_RELATION_ID}.xml"
    if cache_xml.exists():
        xml_bytes = cache_xml.read_bytes()
    else:
        xml_bytes = _download_osm_relation_xml(OSM_NUREK_RELATION_ID)
        cache_xml.write_bytes(xml_bytes)
    polygon = _osm_relation_polygon(xml_bytes, OSM_NUREK_RELATION_ID)
    gdf = gpd.GeoDataFrame(
        [{"name": "Norak Reservoir", "osm_relation": OSM_NUREK_RELATION_ID}],
        geometry=[polygon],
        crs="EPSG:4326",
    ).to_crs(TARGET_CRS)
    gdf.to_file(out, encoding="utf-8")
    _sidecar(
        out,
        role="baseline",
        source_name="OpenStreetMap Norak/Nurek Reservoir relation",
        source_files=[f"https://www.openstreetmap.org/relation/{OSM_NUREK_RELATION_ID}"],
        processing="Downloaded from OpenStreetMap API and reprojected to EPSG:32642.",
        feature_count=int(len(gdf)),
    )
    return out


def prepare_baseline(root: Path) -> gpd.GeoDataFrame:
    target_baseline = root / "baseline"
    target_baseline.mkdir(parents=True, exist_ok=True)
    basin_utm = _copy_reproject_basin(target_baseline)
    _reproject_dem(target_baseline, basin_utm)
    _prepare_rivers(target_baseline, basin_utm)
    _prepare_reservoir_boundary(target_baseline)
    return basin_utm


def _search_items(collection: str, bbox: list[float], date_range: str, acquisition_date: str):
    client = Client.open(PLANETARY_STAC)
    items = list(
        client.search(
            collections=[collection],
            bbox=bbox,
            datetime=date_range,
            limit=300,
        ).items()
    )
    selected = [item for item in items if item.datetime.date().isoformat() == acquisition_date]
    if not selected:
        raise RuntimeError(f"{collection} 未找到 {acquisition_date} 与研究区相交的公开 STAC 条目")
    return selected


def _merge_planetary_assets(
    items,
    asset_names: list[str],
    out_path: Path,
    *,
    target_bounds: tuple[float, float, float, float],
    resolution_m: float,
    basin_geometries,
    resampling: Resampling,
) -> dict:
    arrays = []
    out_transform = None
    signed_ids = []
    for asset_name in asset_names:
        vrts = []
        for item in items:
            signed = planetary_computer.sign(item)
            if asset_name not in signed.assets:
                continue
            src = rasterio.open(signed.assets[asset_name].href)
            vrts.append(WarpedVRT(src, crs=TARGET_CRS, resampling=resampling))
            signed_ids.append(item.id)
        if not vrts:
            raise RuntimeError(f"所选 STAC 条目缺少资产: {asset_name}")
        merged, transform = merge(
            vrts,
            bounds=target_bounds,
            res=(resolution_m, resolution_m),
            nodata=0,
        )
        for vrt in vrts:
            vrt.close()
        arrays.append(merged[0])
        out_transform = transform

    stack = np.stack(arrays)
    mask = geometry_mask(
        basin_geometries,
        out_shape=(stack.shape[1], stack.shape[2]),
        transform=out_transform,
        invert=True,
    )
    stack[:, ~mask] = 0

    profile = {
        "driver": "GTiff",
        "height": stack.shape[1],
        "width": stack.shape[2],
        "count": stack.shape[0],
        "dtype": stack.dtype,
        "crs": TARGET_CRS,
        "transform": out_transform,
        "nodata": 0,
        "compress": "deflate",
        "tiled": True,
        "BIGTIFF": "IF_SAFER",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(stack)
        for index, asset_name in enumerate(asset_names, start=1):
            dst.set_band_description(index, asset_name)

    return {"item_ids": sorted(set(signed_ids)), "asset_names": asset_names, "resolution_m": resolution_m}


def _basin_bbox_wgs84(basin_utm: gpd.GeoDataFrame) -> list[float]:
    bounds = basin_utm.to_crs("EPSG:4326").total_bounds
    return [float(v) for v in bounds]


def _align_bounds_to_resolution(
    bounds: tuple[float, float, float, float],
    resolution_m: float,
) -> tuple[float, float, float, float]:
    minx, miny, maxx, maxy = bounds
    width = math.ceil((maxx - minx) / resolution_m) * resolution_m
    height = math.ceil((maxy - miny) / resolution_m) * resolution_m
    return (minx, maxy - height, minx + width, maxy)


def _download_sentinel(period: str, cfg: dict, root: Path, basin_utm: gpd.GeoDataFrame, resolution_m: float) -> None:
    bbox = _basin_bbox_wgs84(basin_utm)
    target_bounds = _align_bounds_to_resolution(tuple(float(v) for v in basin_utm.total_bounds), resolution_m)
    raw_dir = root / "raw" / f"{period}_{cfg['raw_name']}"
    raw_dir.mkdir(parents=True, exist_ok=True)
    date_range = f"{cfg['month_start']}/{cfg['month_end']}"

    s2_items = _search_items("sentinel-2-l2a", bbox, date_range, cfg["sentinel2_date"])
    s2_path = raw_dir / f"{period}_哨兵影像.tif"
    s2_info = _merge_planetary_assets(
        s2_items,
        ["B02", "B03", "B04", "B08", "B11"],
        s2_path,
        target_bounds=target_bounds,
        resolution_m=resolution_m,
        basin_geometries=list(basin_utm.geometry),
        resampling=Resampling.bilinear,
    )
    _sidecar(
        s2_path,
        role="raw",
        source_name="Copernicus Sentinel-2 L2A via Microsoft Planetary Computer",
        source_files=[f"{PLANETARY_STAC}/collections/sentinel-2-l2a"],
        acquisition_date=cfg["sentinel2_date"],
        stac_item_ids=s2_info["item_ids"],
        bands=s2_info["asset_names"],
        processing=f"Cropped to Vakhsh basin and resampled to {resolution_m:g} m.",
    )

    s1_items = _search_items("sentinel-1-rtc", bbox, date_range, cfg["sentinel1_date"])
    s1_path = raw_dir / f"{period}_SAR影像.tif"
    s1_info = _merge_planetary_assets(
        s1_items,
        ["vv", "vh"],
        s1_path,
        target_bounds=target_bounds,
        resolution_m=resolution_m,
        basin_geometries=list(basin_utm.geometry),
        resampling=Resampling.bilinear,
    )
    _sidecar(
        s1_path,
        role="raw",
        source_name="Copernicus Sentinel-1 RTC via Microsoft Planetary Computer",
        source_files=[f"{PLANETARY_STAC}/collections/sentinel-1-rtc"],
        acquisition_date=cfg["sentinel1_date"],
        stac_item_ids=s1_info["item_ids"],
        bands=s1_info["asset_names"],
        processing=f"Cropped to Vakhsh basin and resampled to {resolution_m:g} m.",
    )


def _download_weather(period: str, cfg: dict, root: Path, basin_utm: gpd.GeoDataFrame) -> None:
    raw_dir = root / "raw" / f"{period}_{cfg['raw_name']}"
    raw_dir.mkdir(parents=True, exist_ok=True)
    centroid = basin_utm.to_crs("EPSG:4326").geometry.union_all().centroid
    params = {
        "parameters": "T2M,T2M_MIN,T2M_MAX,PRECTOTCORR",
        "community": "AG",
        "longitude": f"{centroid.x:.5f}",
        "latitude": f"{centroid.y:.5f}",
        "start": cfg["month_start"].replace("-", ""),
        "end": cfg["month_end"].replace("-", ""),
        "format": "JSON",
    }
    response = requests.get(NASA_POWER_DAILY, params=params, timeout=120)
    response.raise_for_status()
    payload = response.json()
    data = payload["properties"]["parameter"]
    rows = []
    for key in sorted(data["T2M"].keys()):
        date = f"{key[:4]}-{key[4:6]}-{key[6:8]}"
        temp = float(data["T2M"][key])
        precip = float(data["PRECTOTCORR"][key])
        rows.append(
            {
                "date": date,
                "temp_mean_c": temp,
                "temp_min_c": float(data["T2M_MIN"][key]),
                "temp_max_c": float(data["T2M_MAX"][key]),
                "precipitation_mm": precip,
                "solid_precip_mm": precip if temp <= 0 else 0.0,
                "source": "NASA POWER",
                "latitude": round(float(params["latitude"]), 5),
                "longitude": round(float(params["longitude"]), 5),
            }
        )

    out = raw_dir / f"{period}_逐日气象.csv"
    with out.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    _sidecar(
        out,
        role="raw",
        source_name="NASA POWER daily meteorology",
        source_files=[response.url],
        acquisition_date=f"{cfg['month_start']} to {cfg['month_end']}",
        processing="Daily point meteorology extracted at Vakhsh basin centroid; solid_precip_mm is derived by temp_mean_c <= 0.",
    )


def _write_reservoir_parameters(period: str, cfg: dict, root: Path) -> None:
    if not LOCAL_RESERVOIR_CURVE.exists():
        raise FileNotFoundError(f"缺少本地 Nurek 水库水位-面积-库容曲线: {LOCAL_RESERVOIR_CURVE}")

    raw_dir = root / "raw" / f"{period}_{cfg['raw_name']}"
    raw_dir.mkdir(parents=True, exist_ok=True)
    curve = pd.read_csv(LOCAL_RESERVOIR_CURVE)
    out = raw_dir / f"{period}_水库参数.csv"
    renamed = pd.DataFrame(
        {
            "date": cfg["month_start"],
            "reservoir_name": "Nurek",
            "elevation_m": curve["Elevation (m)"],
            "area_sq_km": curve["Area (sq.km)"],
            "storage_mcm": curve["Storage (mcm)"],
            "source": "local Nurek reservoir hypsometry curve",
        }
    )
    renamed.to_csv(out, index=False, encoding="utf-8-sig")
    _sidecar(
        out,
        role="raw",
        source_name="Local Nurek reservoir hypsometry curve",
        source_files=[str(LOCAL_RESERVOIR_CURVE)],
        acquisition_date=cfg["month_start"],
        processing="Static Nurek level-area-storage curve copied to the period raw folder for module access.",
    )


def prepare_raw(
    root: Path,
    basin_utm: gpd.GeoDataFrame,
    resolution_m: float,
    periods: dict[str, dict],
    *,
    skip_periods: set[str] | None = None,
) -> None:
    skip_periods = skip_periods or set()
    for period, cfg in periods.items():
        if period in skip_periods:
            print(f"[OK] {period} 已由本地 raw 提供，跳过公开下载")
            continue
        _download_sentinel(period, cfg, root, basin_utm, resolution_m)
        _download_weather(period, cfg, root, basin_utm)
        _write_reservoir_parameters(period, cfg, root)


def prepare_processed_skeleton(root: Path, periods: dict[str, dict], local_period_names: dict[str, str] | None = None) -> None:
    processed = root / "processed"
    local_period_names = local_period_names or {}
    for scheme in ("scheme01_常规调度工况", "scheme02_优化分水工况"):
        for period, cfg in periods.items():
            period_name = local_period_names.get(period, cfg["processed_name"])
            period_dir = processed / scheme / f"{period}_{period_name}"
            (period_dir / "raster").mkdir(parents=True, exist_ok=True)
            (period_dir / "table").mkdir(parents=True, exist_ok=True)
            readme = period_dir / "README.md"
            readme.write_text(
                "本目录用于保存模块运行后生成的真实模型成果。\n"
                "不要预填模拟 GeoTIFF/CSV；模块完成后再写入成果和 finish.tag。\n",
                encoding="utf-8",
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="构建瓦赫什流域真实数字孪生数据目录")
    parser.add_argument("--root", default=str(REAL_TWIN_ROOT), help="正式孪生数据根目录")
    parser.add_argument("--resolution-m", type=float, default=500.0, help="公开遥感影像重采样分辨率，单位 m")
    parser.add_argument(
        "--local-raw-root",
        default="",
        help="可选：已有真实 raw 根目录，内部应为 raw/{YYYYMM_时段}；存在时优先复制本地 raw",
    )
    parser.add_argument(
        "--fallback-year",
        type=int,
        default=DEFAULT_FALLBACK_YEAR,
        help="本地 raw 缺失时下载的公开数据年份；当前 Sentinel 自动下载支持 2015-2017",
    )
    parser.add_argument("--skip-remote", action="store_true", help="只准备本地 baseline，不下载公开 raw 数据")
    args = parser.parse_args()

    if args.resolution_m <= 0 or not math.isfinite(args.resolution_m):
        raise ValueError("--resolution-m 必须为正数")

    root = Path(args.root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "raw").mkdir(exist_ok=True)
    (root / "processed").mkdir(exist_ok=True)

    basin_utm = prepare_baseline(root)
    local_raw_root = _resolve_local_raw_root(args.local_raw_root)
    local_periods = _copy_local_raw_root(local_raw_root, root)
    if local_periods:
        periods = build_local_periods(local_periods)
        print("[OK] 已发现本地真实 raw，跳过公开 fallback 下载")
    else:
        periods = build_periods(args.fallback_year)
    if not args.skip_remote and not local_periods:
        prepare_raw(root, basin_utm, args.resolution_m, periods, skip_periods=local_periods)
    prepare_processed_skeleton(root, periods)

    manifest = {
        "root": str(root),
        "created_at": _now(),
        "target_crs": TARGET_CRS,
        "local_raw_root": str(local_raw_root.resolve()) if local_raw_root else "",
        "local_raw_periods": local_periods,
        "fallback_year": args.fallback_year,
        "periods": periods,
        "note": "正式目录优先接入本地真实 raw；本地缺失时下载 2005-2017 研究期内可追溯公开数据。processed 初始为空，等待模块运行后生成真实成果。",
    }
    _write_json(root / "manifest.json", manifest)
    print(f"[OK] 真实孪生数据目录已准备: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
