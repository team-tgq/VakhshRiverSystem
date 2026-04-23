from __future__ import annotations

import json
import math
import os
import tempfile
import warnings
import zipfile
import hashlib
from datetime import date, datetime, time, timedelta, timezone
from io import StringIO
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import requests
import shapefile
import xarray as xr
from netCDF4 import Dataset
from rasterio.mask import mask
from rasterio.transform import from_origin
from shapely.geometry import mapping, shape
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error

import rasterio
import rasterio.warp

try:
    import h5py
except ImportError:
    h5py = None

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
warnings.filterwarnings(
    "ignore",
    message="In a future version of xarray decode_timedelta will default to False.*",
    category=FutureWarning,
)


BASE_DIR = Path(__file__).resolve().parent
STUDY_AREA_PATH = BASE_DIR / "study_area.shp"
LEGACY_ERA5_ARCHIVE = BASE_DIR / "output" / "era5_land.nc"

DAILY_OUTPUT_DIR = BASE_DIR / "output" / "daily_ml"
MODEL_DIR = DAILY_OUTPUT_DIR / "models"
RASTER_DIR = DAILY_OUTPUT_DIR / "rasters"
SERIES_DIR = DAILY_OUTPUT_DIR / "series"
CACHE_DIR = DAILY_OUTPUT_DIR / "cache"
GFS_CACHE_DIR = CACHE_DIR / "gfs"
FORCING_CACHE_DIR = CACHE_DIR / "forcing"
VIIRS_CACHE_DIR = CACHE_DIR / "viirs"
VIIRS_DAILY_CACHE_DIR = VIIRS_CACHE_DIR / "daily"
HISTORICAL_VIIRS_CACHE_DIR = VIIRS_CACHE_DIR / "historical_vnp10c1"
HISTORICAL_VIIRS_DAILY_CACHE_DIR = HISTORICAL_VIIRS_CACHE_DIR / "daily"
STATE_CACHE_DIR = CACHE_DIR / "state"
VIIRS_COLLECTION = "5200"
VIIRS_PRODUCT = "VNP10_NRT"
VIIRS_GEOMETA_PLATFORM = "NPP"
VIIRS_REMOTE_ROOT = f"https://nrt3.modaps.eosdis.nasa.gov/archive/allData/{VIIRS_COLLECTION}/{VIIRS_PRODUCT}"
VIIRS_GEOMETA_ROOT = f"https://nrt3.modaps.eosdis.nasa.gov/archive/geoMetaVIIRS/{VIIRS_COLLECTION}/{VIIRS_GEOMETA_PLATFORM}"
VIIRS_TOKEN_ENV_NAMES = ("VIIRS_NRT_TOKEN", "EARTHDATA_TOKEN")
HISTORICAL_VIIRS_PRODUCT = "VNP10C1"
HISTORICAL_VIIRS_VERSION = "2"
HISTORICAL_VIIRS_SOURCE = "nsidc_standard"
HISTORICAL_VIIRS_CMR_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"
HISTORICAL_VIIRS_STATUS = "historical_standard"

MANIFEST_PATH = DAILY_OUTPUT_DIR / "manifest.json"
MODEL_PATH = MODEL_DIR / "daily_swe_gbr.joblib"
TRAINING_METRICS_PATH = MODEL_DIR / "training_metrics.json"
DAILY_SERIES_PATH = SERIES_DIR / "daily_basin_series.csv"
ROUTING_SERIES_PATH = BASE_DIR.parent / "routing" / "data" / "SWE_daily_series.csv"

BUSINESS_TIMEZONE = ZoneInfo("Asia/Dushanbe")
MODEL_VERSION = "daily_swe_gbr_viirs_v3"
TRAINING_START_DATE = date(2024, 2, 1)
TRAINING_END_DATE = date(2025, 1, 31)
NODATA_FLOAT = -9999.0
NODATA_INT = -1

QA_VIIRS_MISSING = 1
QA_VIIRS_CONSTRAINED = 2
QA_COLD_START_EXTERNAL = 4

ELEVATION_BANDS = 5

FEATURE_COLUMNS = [
    "latitude",
    "longitude",
    "elevation_m",
    "slope_deg",
    "aspect_deg",
    "zone_id",
    "doy",
    "temp_mean_c",
    "temp_min_c",
    "temp_max_c",
    "temp_range_c",
    "positive_degree_c",
    "positive_degree_3d",
    "positive_degree_7d",
    "precipitation_mm",
    "solid_precip_mm",
    "solid_precip_3d",
    "solid_precip_7d",
    "solid_precip_14d",
    "solid_precip_30d",
    "temp_mean_3d",
    "temp_mean_7d",
    "temp_mean_14d",
    "temp_mean_30d",
    "prev_swe_mm",
    "prev_swe_3d_mean",
    "prev_swe_7d_mean",
    "snow_cover_fraction",
    "viirs_available",
    "snow_cover_persist_3d",
    "snow_cover_persist_7d",
]


@dataclass
class DailyEntry:
    business_date: str
    forcing_cycle: str
    viirs_status: str
    model_version: str
    is_backfill: bool
    source_status: str
    qa_flag: int
    swe_raster: str
    snowmelt_raster: str
    qa_raster: str
    forcing_cache: str
    swe_mm: float
    snowmelt_mm_day: float


def ensure_directories() -> None:
    for folder in [
        DAILY_OUTPUT_DIR,
        MODEL_DIR,
        RASTER_DIR,
        SERIES_DIR,
        CACHE_DIR,
        GFS_CACHE_DIR,
        FORCING_CACHE_DIR,
        VIIRS_CACHE_DIR,
        VIIRS_DAILY_CACHE_DIR,
        HISTORICAL_VIIRS_CACHE_DIR,
        HISTORICAL_VIIRS_DAILY_CACHE_DIR,
        STATE_CACHE_DIR,
        ROUTING_SERIES_PATH.parent,
    ]:
        folder.mkdir(parents=True, exist_ok=True)


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported JSON value: {type(value)!r}")


def _read_windows_environment_variable(name: str) -> str | None:
    if os.name != "nt":
        return None
    try:
        import winreg
    except ImportError:
        return None

    candidate_keys = [
        (winreg.HKEY_CURRENT_USER, r"Environment"),
        (winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
    ]
    for root, subkey in candidate_keys:
        try:
            with winreg.OpenKey(root, subkey) as handle:
                value, _ = winreg.QueryValueEx(handle, name)
        except OSError:
            continue
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _resolve_token_from_env(names: tuple[str, ...]) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    for name in names:
        value = _read_windows_environment_variable(name)
        if value:
            return value
    return None


def training_window_days() -> int:
    return int((TRAINING_END_DATE - TRAINING_START_DATE).days + 1)


def training_metadata_signature() -> dict[str, Any]:
    return {
        "training_start_date": TRAINING_START_DATE.isoformat(),
        "training_end_date": TRAINING_END_DATE.isoformat(),
        "training_window_days": training_window_days(),
        "historical_viirs_product": HISTORICAL_VIIRS_PRODUCT,
        "historical_viirs_source": HISTORICAL_VIIRS_SOURCE,
        "feature_columns": FEATURE_COLUMNS,
        "model_version": MODEL_VERSION,
    }


def metadata_signature_matches(payload: dict[str, Any]) -> bool:
    expected = training_metadata_signature()
    return all(payload.get(key) == value for key, value in expected.items())


def read_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        return {
            "model_version": MODEL_VERSION,
            "entries": [],
            "daily_series_csv": str(DAILY_SERIES_PATH),
            "routing_series_csv": str(ROUTING_SERIES_PATH),
        }
    with open(MANIFEST_PATH, "r", encoding="utf-8") as file:
        payload = json.load(file)
    payload.setdefault("entries", [])
    payload.setdefault("model_version", MODEL_VERSION)
    payload.setdefault("daily_series_csv", str(DAILY_SERIES_PATH))
    payload.setdefault("routing_series_csv", str(ROUTING_SERIES_PATH))
    return payload


def write_manifest(manifest: dict[str, Any]) -> None:
    ensure_directories()
    with open(MANIFEST_PATH, "w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2, default=_json_default)


def upsert_manifest_entry(entry: DailyEntry) -> dict[str, Any]:
    manifest = read_manifest()
    entries = [item for item in manifest["entries"] if item["business_date"] != entry.business_date]
    entries.append(asdict(entry))
    entries.sort(key=lambda item: item["business_date"])
    manifest["entries"] = entries
    manifest["latest_business_date"] = entries[-1]["business_date"] if entries else None
    manifest["model_version"] = entry.model_version
    write_manifest(manifest)
    return manifest


def _entry_has_available_viirs(entry: dict[str, Any] | None) -> bool:
    if not entry:
        return False
    return str(entry.get("viirs_status") or "").strip().lower() != "missing"


def _resolve_payload_entry(
    manifest: dict[str, Any],
    preferred_business_date: date | None = None,
) -> dict[str, Any]:
    entries = manifest.get("entries", [])
    if not entries:
        return {}

    if preferred_business_date is not None:
        preferred = _find_manifest_entry(manifest, preferred_business_date)
        if preferred is not None:
            return preferred

    for entry in reversed(entries):
        if _manifest_entry_is_reusable(entry) and _entry_has_available_viirs(entry):
            return entry

    return entries[-1]


def build_result_payload(
    manifest: dict[str, Any],
    preferred_business_date: date | None = None,
) -> dict[str, Any]:
    entries = manifest.get("entries", [])
    latest = _resolve_payload_entry(manifest, preferred_business_date=preferred_business_date)
    tif_list = [item["swe_raster"] for item in entries if item.get("swe_raster")]
    return {
        "csv": str(DAILY_SERIES_PATH),
        "tif_list": tif_list,
        "latest_tif": latest.get("swe_raster"),
        "study_area_shp": str(STUDY_AREA_PATH),
        "business_date": latest.get("business_date"),
        "source_used": latest.get("source_status"),
        "status": latest.get("source_status"),
        "latest_confidence_tif": latest.get("qa_raster"),
        "manifest": str(MANIFEST_PATH),
        "entries": entries,
        "latest_entry": latest,
    }


def _find_manifest_entry(manifest: dict[str, Any], business_date_value: date) -> dict[str, Any] | None:
    date_token = business_date_value.isoformat()
    for item in manifest.get("entries", []):
        if item.get("business_date") == date_token:
            return item
    return None


def _manifest_entry_is_reusable(entry: dict[str, Any] | None) -> bool:
    if not entry:
        return False
    required_paths = [
        entry.get("swe_raster"),
        entry.get("forcing_cache"),
    ]
    for path in required_paths:
        if not path or not Path(path).exists():
            return False
    return True


def current_business_date(now_utc: datetime | None = None) -> date:
    now_utc = now_utc or datetime.now(timezone.utc)
    local_now = now_utc.astimezone(BUSINESS_TIMEZONE)
    return local_now.date()


def latest_complete_business_date(now_utc: datetime | None = None) -> date:
    return current_business_date(now_utc=now_utc) - timedelta(days=1)


def business_window(target_date: date) -> tuple[datetime, datetime]:
    start_local = datetime.combine(target_date, time(0, 0), tzinfo=BUSINESS_TIMEZONE)
    end_local = start_local + timedelta(days=1)
    return start_local.astimezone(timezone.utc), end_local.astimezone(timezone.utc)


def cycle_string(cycle_dt: datetime) -> str:
    return cycle_dt.strftime("%Y%m%dT%HZ")


def load_study_area_geometries() -> list[dict[str, Any]]:
    reader = shapefile.Reader(str(STUDY_AREA_PATH))
    geometries = [mapping(shape(record.__geo_interface__)) for record in reader.shapes()]
    if not geometries:
        raise ValueError("Study area shapefile does not contain any geometry.")
    return geometries


def load_study_area_bbox() -> tuple[float, float, float, float]:
    reader = shapefile.Reader(str(STUDY_AREA_PATH))
    if not reader.shapes():
        raise ValueError("Study area shapefile does not contain any shape.")
    min_lon, min_lat, max_lon, max_lat = reader.shapes()[0].bbox
    return float(min_lon), float(min_lat), float(max_lon), float(max_lat)


def _grid_transform(longitudes: np.ndarray, latitudes_desc: np.ndarray) -> Any:
    dx = float(np.median(np.diff(np.sort(longitudes))))
    dy = float(np.median(np.abs(np.diff(np.sort(latitudes_desc)))))
    west = float(np.min(longitudes) - dx / 2.0)
    north = float(np.max(latitudes_desc) + dy / 2.0)
    return from_origin(west, north, dx, dy)


def ensure_lat_desc(
    values: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values)
    lats = np.asarray(latitudes)
    lons = np.asarray(longitudes)
    if lats[0] < lats[-1]:
        return values[::-1, :], lats[::-1], lons
    return values, lats, lons


def build_inside_mask(longitudes: np.ndarray, latitudes_desc: np.ndarray) -> np.ndarray:
    import rasterio.features

    geometries = load_study_area_geometries()
    transform = _grid_transform(longitudes, latitudes_desc)
    outside = rasterio.features.geometry_mask(
        geometries=geometries,
        out_shape=(len(latitudes_desc), len(longitudes)),
        transform=transform,
        invert=False,
        all_touched=False,
    )
    return ~outside


def compute_grid_edges(values: np.ndarray) -> np.ndarray:
    centers = np.asarray(values, dtype=np.float64)
    if centers.ndim != 1 or len(centers) < 2:
        raise ValueError("Grid edge computation requires a 1D array with at least two cells.")

    deltas = np.diff(centers)
    edges = np.empty(len(centers) + 1, dtype=np.float64)
    edges[1:-1] = centers[:-1] + deltas / 2.0
    edges[0] = centers[0] - deltas[0] / 2.0
    edges[-1] = centers[-1] + deltas[-1] / 2.0
    return edges


def composite_swath_to_grid(
    cover_fraction: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    target_longitudes: np.ndarray,
    target_latitudes: np.ndarray,
) -> np.ndarray:
    out = np.full((len(target_latitudes), len(target_longitudes)), np.nan, dtype=np.float32)

    lat = np.asarray(latitudes, dtype=np.float32).ravel()
    lon = np.asarray(longitudes, dtype=np.float32).ravel()
    cover = np.asarray(cover_fraction, dtype=np.float32).ravel()

    valid = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(cover)
    if not np.any(valid):
        return out

    lat = lat[valid]
    lon = lon[valid]
    cover = cover[valid]

    target_lon_edges = compute_grid_edges(np.asarray(target_longitudes, dtype=np.float64))
    target_lat_asc = np.asarray(target_latitudes, dtype=np.float64)[::-1]
    target_lat_edges = compute_grid_edges(target_lat_asc)

    col_idx = np.digitize(lon, target_lon_edges) - 1
    row_idx_asc = np.digitize(lat, target_lat_edges) - 1
    inside = (
        (col_idx >= 0)
        & (col_idx < len(target_longitudes))
        & (row_idx_asc >= 0)
        & (row_idx_asc < len(target_latitudes))
    )
    if not np.any(inside):
        return out

    col_idx = col_idx[inside]
    row_idx = (len(target_latitudes) - 1 - row_idx_asc[inside]).astype(np.int32)
    cover = cover[inside]

    flat_idx = row_idx * len(target_longitudes) + col_idx
    flat_size = len(target_latitudes) * len(target_longitudes)
    cover_sum = np.bincount(flat_idx, weights=cover, minlength=flat_size)
    cover_count = np.bincount(flat_idx, minlength=flat_size)

    flat_out = np.full(flat_size, np.nan, dtype=np.float32)
    nonzero = cover_count > 0
    flat_out[nonzero] = (cover_sum[nonzero] / cover_count[nonzero]).astype(np.float32)
    return flat_out.reshape(out.shape)


def _nanmean_stack(arrays: list[np.ndarray]) -> np.ndarray | None:
    if not arrays:
        return None

    stack = np.stack(arrays, axis=0).astype(np.float32)
    valid = np.isfinite(stack)
    count = valid.sum(axis=0)
    total = np.nansum(stack, axis=0)
    out = np.full(stack.shape[1:], np.nan, dtype=np.float32)
    mask = count > 0
    out[mask] = (total[mask] / count[mask]).astype(np.float32)
    return out


def _grid_signature(longitudes: np.ndarray, latitudes: np.ndarray) -> str:
    digest = hashlib.sha1()
    digest.update(np.asarray(longitudes, dtype=np.float32).tobytes())
    digest.update(np.asarray(latitudes, dtype=np.float32).tobytes())
    return digest.hexdigest()[:12]


def save_raster(
    array: np.ndarray,
    longitudes: np.ndarray,
    latitudes: np.ndarray,
    output_path: Path,
    nodata: float | int,
    dtype: str,
) -> str:
    ensure_directories()
    geometries = load_study_area_geometries()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    oriented, lat_desc, lons = ensure_lat_desc(array, latitudes, longitudes)
    transform = _grid_transform(lons, lat_desc)
    temp_path = output_path.with_suffix(".tmp.tif")

    write_array = oriented.copy()
    if np.issubdtype(write_array.dtype, np.floating):
        write_array = np.where(np.isnan(write_array), nodata, write_array)

    with rasterio.open(
        temp_path,
        "w",
        driver="GTiff",
        height=write_array.shape[0],
        width=write_array.shape[1],
        count=1,
        dtype=dtype,
        crs="EPSG:4326",
        transform=transform,
        nodata=nodata,
        compress="deflate",
    ) as dataset:
        dataset.write(write_array.astype(dtype), 1)

    with rasterio.open(temp_path) as src:
        clipped, clipped_transform = mask(src, geometries, crop=True, nodata=nodata)
        meta = src.meta.copy()
        meta.update(
            {
                "height": clipped.shape[1],
                "width": clipped.shape[2],
                "transform": clipped_transform,
            }
        )

    with rasterio.open(output_path, "w", **meta) as dataset:
        dataset.write(clipped)

    try:
        temp_path.unlink()
    except OSError:
        pass

    return str(output_path)


def load_raster_array(raster_path: str) -> np.ndarray:
    with rasterio.open(raster_path) as dataset:
        array = dataset.read(1).astype(np.float32)
        nodata = dataset.nodata
    if nodata is not None:
        array[array == nodata] = np.nan
    return array


def compute_terrain(orography: np.ndarray, longitudes: np.ndarray, latitudes: np.ndarray) -> dict[str, np.ndarray]:
    orog, lat_desc, lons = ensure_lat_desc(orography, latitudes, longitudes)
    lat_step = float(np.median(np.abs(np.diff(np.sort(lat_desc))))) if len(lat_desc) > 1 else 0.25
    lon_step = float(np.median(np.abs(np.diff(np.sort(lons))))) if len(lons) > 1 else 0.25

    mean_lat_rad = math.radians(float(np.nanmean(lat_desc)))
    dy = max(lat_step * 111_320.0, 1.0)
    dx = max(lon_step * 111_320.0 * max(math.cos(mean_lat_rad), 0.2), 1.0)

    grad_y, grad_x = np.gradient(orog, dy, dx)
    slope_deg = np.degrees(np.arctan(np.sqrt(grad_x ** 2 + grad_y ** 2))).astype(np.float32)
    aspect_deg = (np.degrees(np.arctan2(-grad_x, grad_y)) + 360.0) % 360.0
    aspect_deg = aspect_deg.astype(np.float32)

    mask = build_inside_mask(lons, lat_desc)
    valid_elev = orog[mask]
    if valid_elev.size == 0:
        raise ValueError("Study area mask does not overlap the forcing grid.")

    quantiles = np.linspace(0.0, 1.0, ELEVATION_BANDS + 1)
    edges = np.quantile(valid_elev, quantiles)
    edges[0] -= 1.0
    edges[-1] += 1.0
    elevation_band = np.digitize(orog, edges[1:-1], right=False).astype(np.int16)

    aspect_class = ((aspect_deg + 45.0) // 90.0).astype(np.int16) % 4
    zone_id = (elevation_band * 10 + aspect_class).astype(np.int16)

    return {
        "orography": orog.astype(np.float32),
        "slope_deg": slope_deg,
        "aspect_deg": aspect_deg,
        "elevation_band": elevation_band,
        "zone_id": zone_id,
        "latitudes": lat_desc.astype(np.float32),
        "longitudes": lons.astype(np.float32),
        "inside_mask": mask,
    }


def snow_fraction_from_temperature(temp_min_c: np.ndarray, temp_max_c: np.ndarray) -> np.ndarray:
    full_snow = temp_max_c <= 0.0
    full_rain = temp_min_c >= 2.0
    transition = np.clip((2.0 - ((temp_min_c + temp_max_c) / 2.0)) / 2.0, 0.0, 1.0)
    fraction = np.where(full_snow, 1.0, np.where(full_rain, 0.0, transition))
    return fraction.astype(np.float32)


def safe_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0 or y_pred.size == 0:
        return float("nan")
    if np.allclose(np.std(y_true), 0.0) or np.allclose(np.std(y_pred), 0.0):
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def _rolling_mean(arrays: list[np.ndarray], window: int, fallback: np.ndarray) -> np.ndarray:
    if not arrays:
        return fallback.astype(np.float32)
    window_arrays = arrays[-window:]
    return np.nanmean(np.stack(window_arrays, axis=0), axis=0).astype(np.float32)


def _rolling_sum_positive(arrays: list[np.ndarray], window: int, fallback: np.ndarray) -> np.ndarray:
    if not arrays:
        return np.clip(fallback, 0.0, None).astype(np.float32)
    window_arrays = [np.clip(item, 0.0, None) for item in arrays[-window:]]
    return np.nansum(np.stack(window_arrays, axis=0), axis=0).astype(np.float32)


def _rolling_binary_mean(arrays: list[np.ndarray], window: int, fallback: np.ndarray) -> np.ndarray:
    if not arrays:
        return fallback.astype(np.float32)
    window_arrays = arrays[-window:]
    return np.nanmean(np.stack(window_arrays, axis=0), axis=0).astype(np.float32)


class GFSClient:
    def __init__(self, cache_dir: Path = GFS_CACHE_DIR) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.min_lon, self.min_lat, self.max_lon, self.max_lat = load_study_area_bbox()

    def _build_url(
        self,
        cycle_dt: datetime,
        forecast_hour: int,
        include_height: bool = False,
        include_weasd: bool = True,
    ) -> str:
        file_name = f"gfs.t{cycle_dt:%H}z.pgrb2.0p25.f{forecast_hour:03d}"
        directory = f"/gfs.{cycle_dt:%Y%m%d}/{cycle_dt:%H}/atmos"
        params = {
            "file": file_name,
            "subregion": "",
            "leftlon": f"{math.floor(self.min_lon)}",
            "rightlon": f"{math.ceil(self.max_lon)}",
            "toplat": f"{math.ceil(self.max_lat)}",
            "bottomlat": f"{math.floor(self.min_lat)}",
            "dir": directory,
        }
        if include_height:
            params["lev_surface"] = "on"
            params["var_HGT"] = "on"
        else:
            params["lev_2_m_above_ground"] = "on"
            params["lev_surface"] = "on"
            params["var_TMP"] = "on"
            params["var_APCP"] = "on"
            if include_weasd:
                params["var_WEASD"] = "on"

        query = "&".join(f"{key}={value}" for key, value in params.items())
        return f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?{query}"

    def _download(self, url: str, target_path: Path) -> Path:
        if target_path.exists() and target_path.stat().st_size > 0:
            return target_path

        response = self.session.get(url, timeout=120)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "html" in content_type.lower():
            raise RuntimeError("GFS subset request returned HTML instead of GRIB data.")

        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(response.content)
        return target_path

    def _open_grib(self, path: Path) -> xr.Dataset:
        return xr.open_dataset(
            path,
            engine="cfgrib",
            backend_kwargs={"indexpath": ""},
        )

    def _download_forcing_file(self, cycle_dt: datetime, forecast_hour: int) -> Path:
        cycle_key = cycle_dt.strftime("%Y%m%d%H")
        file_path = self.cache_dir / cycle_key / f"forcing_f{forecast_hour:03d}.grib2"
        return self._download(self._build_url(cycle_dt, forecast_hour), file_path)

    def _download_height_file(self, cycle_dt: datetime) -> Path:
        cycle_key = cycle_dt.strftime("%Y%m%d%H")
        file_path = self.cache_dir / cycle_key / "terrain_f000.grib2"
        return self._download(self._build_url(cycle_dt, 0, include_height=True, include_weasd=False), file_path)

    def resolve_terrain_cycle(self, reference_dt: datetime) -> datetime:
        candidate = reference_dt.replace(minute=0, second=0, microsecond=0)
        cycle_hour = (candidate.hour // 6) * 6
        cycle = candidate.replace(hour=cycle_hour)

        for _ in range(12):
            try:
                self._download_height_file(cycle)
                return cycle
            except Exception:
                cycle -= timedelta(hours=6)

        raise RuntimeError("Unable to resolve a usable GFS terrain cycle.")

    def resolve_cycle(self, start_utc: datetime) -> datetime:
        candidate = start_utc.replace(minute=0, second=0, microsecond=0)
        cycle_hour = (candidate.hour // 6) * 6
        cycle = candidate.replace(hour=cycle_hour)

        for _ in range(8):
            try:
                forecast_hour = int((start_utc - cycle).total_seconds() // 3600)
                if forecast_hour < 0:
                    cycle -= timedelta(hours=6)
                    continue
                self._download_forcing_file(cycle, forecast_hour)
                return cycle
            except Exception:
                cycle -= timedelta(hours=6)

        raise RuntimeError("Unable to resolve a usable GFS cycle for the requested business day.")

    def get_static_terrain(self, cycle_dt: datetime) -> dict[str, np.ndarray]:
        resolved_cycle = self.resolve_terrain_cycle(cycle_dt)
        path = self._download_height_file(resolved_cycle)
        dataset = self._open_grib(path)
        orog = dataset["orog"].values.astype(np.float32)
        latitudes = dataset["latitude"].values.astype(np.float32)
        longitudes = dataset["longitude"].values.astype(np.float32)
        dataset.close()
        return compute_terrain(orog, longitudes, latitudes)

    def get_business_day_forcing(self, business_date_value: date) -> dict[str, Any]:
        start_utc, end_utc = business_window(business_date_value)
        cycle_dt = self.resolve_cycle(start_utc)

        forecast_start = int((start_utc - cycle_dt).total_seconds() // 3600)
        forecast_end = int((end_utc - cycle_dt).total_seconds() // 3600)
        forecast_hours = list(range(forecast_start, forecast_end))
        if not forecast_hours:
            raise ValueError("No forecast hours fall inside the requested business day.")

        temperatures: list[np.ndarray] = []
        cumulative_tp: list[np.ndarray] = []
        cold_start_seed: np.ndarray | None = None
        latitudes: np.ndarray | None = None
        longitudes: np.ndarray | None = None

        for forecast_hour in forecast_hours:
            path = self._download_forcing_file(cycle_dt, forecast_hour)
            dataset = self._open_grib(path)
            temperatures.append(dataset["t2m"].values.astype(np.float32) - 273.15)

            if "tp" in dataset:
                cumulative_tp.append(dataset["tp"].values.astype(np.float32))
            else:
                shape = temperatures[-1].shape
                cumulative_tp.append(np.zeros(shape, dtype=np.float32))

            if cold_start_seed is None and "sdwe" in dataset:
                cold_start_seed = dataset["sdwe"].values.astype(np.float32)

            if latitudes is None:
                latitudes = dataset["latitude"].values.astype(np.float32)
                longitudes = dataset["longitude"].values.astype(np.float32)

            dataset.close()

        if latitudes is None or longitudes is None:
            raise RuntimeError("GFS forcing download did not return any grid metadata.")

        temp_stack = np.stack(temperatures, axis=0)
        precip_steps: list[np.ndarray] = []
        previous = np.zeros_like(cumulative_tp[0], dtype=np.float32)
        for current in cumulative_tp:
            precip_steps.append(np.maximum(current - previous, 0.0).astype(np.float32))
            previous = current
        daily_precip = np.sum(np.stack(precip_steps, axis=0), axis=0).astype(np.float32)

        terrain = self.get_static_terrain(cycle_dt)
        temp_mean = np.mean(temp_stack, axis=0).astype(np.float32)
        temp_min = np.min(temp_stack, axis=0).astype(np.float32)
        temp_max = np.max(temp_stack, axis=0).astype(np.float32)
        solid_fraction = snow_fraction_from_temperature(temp_min, temp_max)
        solid_precip = (daily_precip * solid_fraction).astype(np.float32)

        return {
            "business_date": business_date_value.isoformat(),
            "start_utc": start_utc,
            "end_utc": end_utc,
            "cycle_dt": cycle_dt,
            "forecast_hours": forecast_hours,
            "latitudes": terrain["latitudes"],
            "longitudes": terrain["longitudes"],
            "temp_mean_c": ensure_lat_desc(temp_mean, latitudes, longitudes)[0],
            "temp_min_c": ensure_lat_desc(temp_min, latitudes, longitudes)[0],
            "temp_max_c": ensure_lat_desc(temp_max, latitudes, longitudes)[0],
            "precipitation_mm": ensure_lat_desc(daily_precip, latitudes, longitudes)[0],
            "solid_precip_mm": ensure_lat_desc(solid_precip, latitudes, longitudes)[0],
            "cold_start_seed_mm": ensure_lat_desc(cold_start_seed, latitudes, longitudes)[0]
            if cold_start_seed is not None
            else None,
            "terrain": terrain,
        }

class BaseVIIRSClient:
    def __init__(
        self,
        cache_dir: Path,
        daily_cache_dir: Path,
        token_env_names: tuple[str, ...],
    ) -> None:
        self.cache_dir = cache_dir
        self.daily_cache_dir = daily_cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.daily_cache_dir.mkdir(parents=True, exist_ok=True)
        self.remote_token = _resolve_token_from_env(token_env_names)
        self.session = requests.Session()
        self.session.trust_env = False

    @staticmethod
    def _normalize_target_grid(
        target_longitudes: np.ndarray,
        target_latitudes: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        lons = np.asarray(target_longitudes, dtype=np.float32)
        lats = np.asarray(target_latitudes, dtype=np.float32)
        flip_lat = bool(lats[0] < lats[-1])
        if flip_lat:
            lats = lats[::-1].copy()
        return lons, lats, flip_lat

    def _daily_cache_path(
        self,
        business_date_value: date,
        target_longitudes: np.ndarray,
        target_latitudes_desc: np.ndarray,
    ) -> Path:
        signature = _grid_signature(target_longitudes, target_latitudes_desc)
        return self.daily_cache_dir / f"viirs_cover_{business_date_value:%Y%m%d}_{signature}.npz"

    def _load_daily_cache(
        self,
        business_date_value: date,
        target_longitudes: np.ndarray,
        target_latitudes_desc: np.ndarray,
    ) -> tuple[np.ndarray | None, str | None]:
        cache_path = self._daily_cache_path(business_date_value, target_longitudes, target_latitudes_desc)
        if not cache_path.exists():
            return None, None

        with np.load(cache_path, allow_pickle=False) as payload:
            snow_cover_fraction = payload["snow_cover_fraction"].astype(np.float32)
            viirs_status = str(payload["viirs_status"].tolist())
        return snow_cover_fraction, viirs_status

    def _save_daily_cache(
        self,
        business_date_value: date,
        target_longitudes: np.ndarray,
        target_latitudes_desc: np.ndarray,
        snow_cover_fraction: np.ndarray,
        viirs_status: str,
    ) -> None:
        cache_path = self._daily_cache_path(business_date_value, target_longitudes, target_latitudes_desc)
        np.savez_compressed(
            cache_path,
            snow_cover_fraction=snow_cover_fraction.astype(np.float32),
            viirs_status=np.array(viirs_status),
        )


class RealtimeVIIRSClient(BaseVIIRSClient):
    def __init__(self, cache_dir: Path = VIIRS_CACHE_DIR, daily_cache_dir: Path = VIIRS_DAILY_CACHE_DIR) -> None:
        super().__init__(cache_dir=cache_dir, daily_cache_dir=daily_cache_dir, token_env_names=VIIRS_TOKEN_ENV_NAMES)
        self.external_dir = os.environ.get("VIIRS_NRT_DIR")

    def _candidate_paths(self, business_date_value: date) -> list[Path]:
        candidates: list[Path] = []
        start_utc, end_utc = business_window(business_date_value)
        date_tokens = {
            business_date_value.strftime("%Y%m%d"),
            start_utc.strftime("%Y%m%d"),
            end_utc.strftime("%Y%m%d"),
        }
        for date_token in sorted(date_tokens):
            if self.external_dir:
                candidates.extend(Path(self.external_dir).glob(f"*{date_token}*.tif"))
            candidates.extend(self.cache_dir.glob(f"*{date_token}*.tif"))
        return candidates

    def _geometa_url(self, utc_day: date) -> str:
        return f"{VIIRS_GEOMETA_ROOT}/{utc_day.year}/{VIIRS_PRODUCT.replace('10', '03MOD', 1)}_{utc_day.isoformat()}.txt"

    def _details_url(self, utc_day: date) -> str:
        return f"https://nrt3.modaps.eosdis.nasa.gov/api/v2/content/details/allData/{VIIRS_COLLECTION}/{VIIRS_PRODUCT}/{utc_day.year}/{utc_day.timetuple().tm_yday:03d}"

    def _remote_headers(self) -> dict[str, str]:
        if not self.remote_token:
            return {}
        return {
            "Authorization": f"Bearer {self.remote_token}",
            "User-Agent": "VakhshRiverSystem/1.0",
            "X-Requested-With": "XMLHttpRequest",
        }

    @staticmethod
    def _looks_like_netcdf(first_chunk: bytes) -> bool:
        if not first_chunk:
            return False
        if first_chunk.startswith(b"CDF"):
            return True
        if first_chunk.startswith(b"\x89HDF\r\n\x1a\n"):
            return True
        return False

    def _download_remote_file(self, url: str, target_path: Path) -> Path | None:
        if not self.remote_token:
            return None

        target_path.parent.mkdir(parents=True, exist_ok=True)
        if target_path.exists() and target_path.stat().st_size > 0:
            return target_path

        temp_path = target_path.with_suffix(f"{target_path.suffix}.part")
        if temp_path.exists():
            temp_path.unlink()

        try:
            response = self.session.get(
                url,
                headers=self._remote_headers(),
                stream=True,
                timeout=300,
                allow_redirects=False,
            )
            if response.status_code in (301, 302, 303, 307, 308, 401, 403):
                return None
            response.raise_for_status()

            with open(temp_path, "wb") as handle:
                iterator = response.iter_content(chunk_size=1024 * 1024)
                first_chunk = next(iterator, b"")
                if not self._looks_like_netcdf(first_chunk):
                    return None
                handle.write(first_chunk)
                for chunk in iterator:
                    if chunk:
                        handle.write(chunk)
            temp_path.replace(target_path)
            return target_path
        except (requests.RequestException, OSError):
            return None
        finally:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)

    @staticmethod
    def _bbox_intersects(row: pd.Series, bbox: tuple[float, float, float, float]) -> bool:
        minx, miny, maxx, maxy = bbox
        west = float(row["WestBoundingCoord"])
        east = float(row["EastBoundingCoord"])
        south = float(row["SouthBoundingCoord"])
        north = float(row["NorthBoundingCoord"])
        return not (east < minx or west > maxx or north < miny or south > maxy)

    def _match_remote_files(self, utc_day: date, granule_ids: set[str]) -> dict[str, str]:
        try:
            response = self.session.get(
                self._details_url(utc_day),
                headers=self._remote_headers(),
                timeout=120,
            )
        except requests.RequestException:
            return {}
        if response.status_code != 200:
            return {}

        payload = response.json()
        content = payload.get("content", [])
        available: dict[str, str] = {}
        for item in content:
            name = str(item.get("name", ""))
            downloads_link = str(item.get("downloadsLink", ""))
            if not name or not downloads_link:
                continue
            if name in granule_ids:
                available[name] = downloads_link
        return available

    @staticmethod
    def _parse_geometa_table(raw_text: str) -> pd.DataFrame:
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        header_line = next((line for line in lines if line.startswith("# GranuleID,")), None)
        if header_line is None:
            return pd.DataFrame()

        data_lines = [header_line.lstrip("# ").strip()]
        data_lines.extend(line for line in lines if not line.startswith("#"))
        return pd.read_csv(StringIO("\n".join(data_lines)), skipinitialspace=True)

    @staticmethod
    def _extract_nc_variable(root: Dataset, candidate_paths: list[str], candidate_names: list[str]) -> np.ndarray:
        for candidate_path in candidate_paths:
            cursor: Any = root
            parts = [part for part in candidate_path.split("/") if part]
            found = True
            for part in parts[:-1]:
                if part not in cursor.groups:
                    found = False
                    break
                cursor = cursor.groups[part]
            if found and parts and parts[-1] in cursor.variables:
                return np.asarray(cursor.variables[parts[-1]][:], dtype=np.float32)

        stack: list[Any] = [root]
        names = {name.lower() for name in candidate_names}
        while stack:
            cursor = stack.pop()
            for variable_name, variable in cursor.variables.items():
                if variable_name.lower() in names:
                    return np.asarray(variable[:], dtype=np.float32)
            stack.extend(cursor.groups.values())
        raise KeyError(f"Could not find any of {candidate_names} in VIIRS granule.")

    def _load_remote_granule(
        self,
        granule_path: Path,
        target_longitudes: np.ndarray,
        target_latitudes: np.ndarray,
    ) -> np.ndarray | None:
        with Dataset(granule_path, "r") as dataset:
            cover = self._extract_nc_variable(
                dataset,
                ["SnowData/NDSI_Snow_Cover", "NDSI_Snow_Cover"],
                ["NDSI_Snow_Cover"],
            )
            latitudes = self._extract_nc_variable(
                dataset,
                ["GeolocationData/latitude", "GeolocationData/Latitude", "latitude", "Latitude"],
                ["latitude"],
            )
            longitudes = self._extract_nc_variable(
                dataset,
                ["GeolocationData/longitude", "GeolocationData/Longitude", "longitude", "Longitude"],
                ["longitude"],
            )

        cover = np.where((cover >= 0.0) & (cover <= 100.0), cover, np.nan)
        if np.isfinite(cover).sum() == 0:
            return None
        cover_fraction = np.clip(cover / 100.0, 0.0, 1.0).astype(np.float32)
        return composite_swath_to_grid(cover_fraction, latitudes, longitudes, target_longitudes, target_latitudes)

    def _fetch_remote_cover(
        self,
        business_date_value: date,
        target_longitudes: np.ndarray,
        target_latitudes: np.ndarray,
    ) -> tuple[np.ndarray | None, str]:
        if not self.remote_token:
            return None, "missing"

        start_utc, end_utc = business_window(business_date_value)
        day_count = (end_utc.date() - start_utc.date()).days + 1
        utc_days = [start_utc.date() + timedelta(days=offset) for offset in range(day_count)]
        basin_bbox = load_study_area_bbox()
        composite = np.full((len(target_latitudes), len(target_longitudes)), np.nan, dtype=np.float32)
        matched_granules = 0

        for utc_day in utc_days:
            try:
                geometa_response = self.session.get(self._geometa_url(utc_day), timeout=60)
            except requests.RequestException:
                continue
            if geometa_response.status_code != 200 or not geometa_response.text.strip():
                continue

            rows = self._parse_geometa_table(geometa_response.text)
            if rows.empty:
                continue

            rows["StartDateTime"] = pd.to_datetime(rows["StartDateTime"], utc=True, errors="coerce")
            rows = rows.dropna(subset=["StartDateTime"])
            rows = rows[
                (rows["StartDateTime"] >= pd.Timestamp(start_utc))
                & (rows["StartDateTime"] < pd.Timestamp(end_utc))
            ]
            if rows.empty:
                continue

            rows = rows[rows.apply(lambda record: self._bbox_intersects(record, basin_bbox), axis=1)]
            if rows.empty:
                continue

            granule_ids = set()
            for granule_id in rows["GranuleID"].tolist():
                parts = str(granule_id).split(".")
                if len(parts) < 4:
                    continue
                granule_ids.add(f"{VIIRS_PRODUCT}.{parts[1]}.{parts[2]}.{parts[3]}.nc")
            available = self._match_remote_files(utc_day, granule_ids)

            for granule_id, granule_url in available.items():
                local_path = self.cache_dir / granule_url.rsplit("/", 1)[-1]
                downloaded = self._download_remote_file(granule_url, local_path)
                if downloaded is None:
                    continue
                grid = self._load_remote_granule(downloaded, target_longitudes, target_latitudes)
                if grid is None:
                    continue

                matched_granules += 1
                write_mask = np.isfinite(grid)
                composite = np.where(write_mask, np.fmax(np.nan_to_num(composite, nan=-1.0), grid), composite)
                composite = np.where(composite < 0.0, np.nan, composite).astype(np.float32)

        if matched_granules == 0:
            return None, "missing"
        return composite, "remote_live"

    def get_snow_cover(
        self,
        business_date_value: date,
        target_longitudes: np.ndarray,
        target_latitudes: np.ndarray,
    ) -> tuple[np.ndarray | None, str]:
        working_lons, working_lats, flip_lat = self._normalize_target_grid(target_longitudes, target_latitudes)

        cached_cover, cached_status = self._load_daily_cache(business_date_value, working_lons, working_lats)
        if cached_cover is not None and cached_status is not None:
            if flip_lat:
                cached_cover = cached_cover[::-1, :]
            return cached_cover, cached_status

        remote_cover, remote_status = self._fetch_remote_cover(
            business_date_value,
            working_lons,
            working_lats,
        )
        if remote_cover is not None:
            self._save_daily_cache(business_date_value, working_lons, working_lats, remote_cover, remote_status)
            if flip_lat:
                remote_cover = remote_cover[::-1, :]
            return remote_cover, remote_status

        candidates = self._candidate_paths(business_date_value)
        if not candidates:
            return None, "missing"

        out = np.full((len(working_lats), len(working_lons)), np.nan, dtype=np.float32)
        transform = _grid_transform(working_lons, working_lats)

        with rasterio.open(candidates[0]) as src:
            rasterio.warp.reproject(
                source=rasterio.band(src, 1),
                destination=out,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs="EPSG:4326",
                resampling=rasterio.warp.Resampling.nearest,
            )

        if np.any(np.isfinite(out)) and np.nanmax(out) > 1.0:
            out = np.where(out >= 50.0, 1.0, 0.0)

        out = np.clip(out, 0.0, 1.0).astype(np.float32)
        self._save_daily_cache(business_date_value, working_lons, working_lats, out, "remote_live")
        if flip_lat:
            out = out[::-1, :]
        return out, "remote_live"


class HistoricalVIIRSClient(BaseVIIRSClient):
    def __init__(
        self,
        cache_dir: Path = HISTORICAL_VIIRS_CACHE_DIR,
        daily_cache_dir: Path = HISTORICAL_VIIRS_DAILY_CACHE_DIR,
    ) -> None:
        super().__init__(
            cache_dir=cache_dir,
            daily_cache_dir=daily_cache_dir,
            token_env_names=("EARTHDATA_TOKEN", "VIIRS_NRT_TOKEN"),
        )

    def _cmr_query_params(self, start_date: date, end_date: date) -> dict[str, Any]:
        return {
            "short_name": HISTORICAL_VIIRS_PRODUCT,
            "version": HISTORICAL_VIIRS_VERSION,
            "temporal": f"{start_date.isoformat()}T00:00:00Z,{end_date.isoformat()}T23:59:59Z",
            "page_size": 1000,
        }

    def _cmr_headers(self) -> dict[str, str]:
        return {"User-Agent": "VakhshRiverSystem/1.0"}

    @staticmethod
    def _title_to_business_date(title: str) -> date | None:
        parts = title.split(".")
        if len(parts) < 2 or not parts[1].startswith("A"):
            return None
        stamp = parts[1][1:]
        if len(stamp) != 7:
            return None
        return datetime.strptime(stamp, "%Y%j").date()

    @staticmethod
    def _download_link_from_entry(entry: dict[str, Any]) -> str | None:
        for link in entry.get("links", []):
            href = str(link.get("href", ""))
            if href.endswith(".h5") and "nsidc-cumulus-prod-protected" in href:
                return href
        return None

    @staticmethod
    def _follow_location(base_url: str, location: str | None) -> str | None:
        if not location:
            return None
        if location.startswith("http://") or location.startswith("https://"):
            return location
        if location.startswith("/"):
            return f"{requests.utils.urlparse(base_url).scheme}://{requests.utils.urlparse(base_url).netloc}{location}"
        return None

    def _search_remote_granules(self, start_date: date, end_date: date) -> dict[date, str]:
        try:
            response = self.session.get(
                HISTORICAL_VIIRS_CMR_URL,
                params=self._cmr_query_params(start_date, end_date),
                headers=self._cmr_headers(),
                timeout=120,
            )
            response.raise_for_status()
        except requests.RequestException:
            return {}
        entries = response.json().get("feed", {}).get("entry", [])

        results: dict[date, str] = {}
        for entry in entries:
            business_date_value = self._title_to_business_date(str(entry.get("title", "")))
            download_link = self._download_link_from_entry(entry)
            if business_date_value is None or download_link is None:
                continue
            if start_date <= business_date_value <= end_date:
                results[business_date_value] = download_link
        return results

    def _remote_headers(self) -> dict[str, str]:
        if not self.remote_token:
            return {"User-Agent": "VakhshRiverSystem/1.0"}
        return {
            "Authorization": f"Bearer {self.remote_token}",
            "User-Agent": "VakhshRiverSystem/1.0",
        }

    def _download_remote_file(self, url: str, target_path: Path) -> Path | None:
        if not self.remote_token:
            return None

        target_path.parent.mkdir(parents=True, exist_ok=True)
        if target_path.exists() and target_path.stat().st_size > 0:
            return target_path

        temp_path = target_path.with_suffix(f"{target_path.suffix}.part")
        if temp_path.exists():
            temp_path.unlink()

        try:
            response = self.session.get(
                url,
                headers=self._remote_headers(),
                stream=True,
                timeout=300,
                allow_redirects=False,
            )
            if response.status_code in (301, 302, 303, 307, 308):
                next_url = self._follow_location(url, response.headers.get("location"))
                if not next_url:
                    return None
                response = self.session.get(next_url, stream=True, timeout=300, allow_redirects=True)
            elif response.status_code in (401, 403):
                return None

            response.raise_for_status()
            with open(temp_path, "wb") as handle:
                iterator = response.iter_content(chunk_size=1024 * 1024)
                first_chunk = next(iterator, b"")
                if not RealtimeVIIRSClient._looks_like_netcdf(first_chunk):
                    return None
                handle.write(first_chunk)
                for chunk in iterator:
                    if chunk:
                        handle.write(chunk)
            temp_path.replace(target_path)
            return target_path
        except (requests.RequestException, OSError):
            return None
        finally:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)

    @staticmethod
    def _nearest_indices(source_values: np.ndarray, target_values: np.ndarray) -> np.ndarray:
        source = np.asarray(source_values, dtype=np.float64)
        target = np.asarray(target_values, dtype=np.float64)
        descending = bool(source[0] > source[-1])
        working_source = source[::-1] if descending else source

        insertion = np.searchsorted(working_source, target)
        insertion = np.clip(insertion, 0, len(working_source) - 1)
        left = np.clip(insertion - 1, 0, len(working_source) - 1)
        right = insertion
        choose_right = np.abs(working_source[right] - target) < np.abs(working_source[left] - target)
        indices = np.where(choose_right, right, left)
        if descending:
            indices = (len(source) - 1) - indices
        return indices.astype(np.int32)

    def _load_remote_granule(
        self,
        granule_path: Path,
        target_longitudes: np.ndarray,
        target_latitudes_desc: np.ndarray,
    ) -> np.ndarray | None:
        if h5py is None:
            raise RuntimeError(
                "h5py is required to read historical standard VIIRS (VNP10C1). "
                "Please install h5py in the plugin runtime environment."
            )
        with h5py.File(granule_path, "r") as handle:
            data_group = handle["HDFEOS/GRIDS/VIIRS_Daily_SnowCover_CMG/Data Fields"]
            snow_cover = np.asarray(data_group["Snow_Cover"][:], dtype=np.float32)
            cloud_cover = np.asarray(data_group["Cloud_Cover"][:], dtype=np.float32)
            latitudes = np.asarray(data_group["latitude"][:], dtype=np.float32)
            longitudes = np.asarray(data_group["longitude"][:], dtype=np.float32)

        snow_valid = (snow_cover >= 0.0) & (snow_cover <= 100.0)
        cloud_full = (cloud_cover >= 100.0) & (cloud_cover <= 100.0)
        cover_fraction = np.where(snow_valid, snow_cover / 100.0, np.nan).astype(np.float32)
        cover_fraction = np.where(cloud_full & ~snow_valid, np.nan, cover_fraction).astype(np.float32)
        if np.isfinite(cover_fraction).sum() == 0:
            return None

        lat_indices = self._nearest_indices(latitudes, target_latitudes_desc)
        lon_indices = self._nearest_indices(longitudes, target_longitudes)
        return cover_fraction[np.ix_(lat_indices, lon_indices)].astype(np.float32)

    def _fetch_remote_cover(
        self,
        business_date_value: date,
        target_longitudes: np.ndarray,
        target_latitudes_desc: np.ndarray,
    ) -> tuple[np.ndarray | None, str]:
        if not self.remote_token:
            return None, "missing"

        granules = self._search_remote_granules(business_date_value, business_date_value)
        granule_url = granules.get(business_date_value)
        if granule_url is None:
            return None, "missing"

        local_path = self.cache_dir / granule_url.split("?")[0].rsplit("/", 1)[-1]
        downloaded = self._download_remote_file(granule_url, local_path)
        if downloaded is None:
            return None, "missing"
        cover = self._load_remote_granule(downloaded, target_longitudes, target_latitudes_desc)
        if cover is None:
            return None, "missing"
        return cover, HISTORICAL_VIIRS_STATUS

    def get_snow_cover(
        self,
        business_date_value: date,
        target_longitudes: np.ndarray,
        target_latitudes: np.ndarray,
    ) -> tuple[np.ndarray | None, str]:
        working_lons, working_lats, flip_lat = self._normalize_target_grid(target_longitudes, target_latitudes)

        cached_cover, cached_status = self._load_daily_cache(business_date_value, working_lons, working_lats)
        if cached_cover is not None and cached_status is not None:
            if flip_lat:
                cached_cover = cached_cover[::-1, :]
            return cached_cover, cached_status

        remote_cover, remote_status = self._fetch_remote_cover(business_date_value, working_lons, working_lats)
        if remote_cover is not None:
            self._save_daily_cache(business_date_value, working_lons, working_lats, remote_cover, remote_status)
            if flip_lat:
                remote_cover = remote_cover[::-1, :]
            return remote_cover, remote_status
        return None, "missing"

    def prefetch_snow_cover_range(
        self,
        business_dates: list[date],
        target_longitudes: np.ndarray,
        target_latitudes: np.ndarray,
    ) -> dict[date, tuple[np.ndarray | None, str]]:
        if not business_dates:
            return {}

        working_lons, working_lats, _ = self._normalize_target_grid(target_longitudes, target_latitudes)
        results: dict[date, tuple[np.ndarray | None, str]] = {}
        missing_dates: list[date] = []
        for business_date_value in business_dates:
            cached_cover, cached_status = self._load_daily_cache(business_date_value, working_lons, working_lats)
            if cached_cover is not None and cached_status is not None:
                results[business_date_value] = (cached_cover.astype(np.float32), cached_status)
            else:
                missing_dates.append(business_date_value)

        if missing_dates and self.remote_token:
            granules = self._search_remote_granules(min(missing_dates), max(missing_dates))
            for business_date_value in missing_dates:
                granule_url = granules.get(business_date_value)
                if granule_url is None:
                    results[business_date_value] = (None, "missing")
                    continue

                local_path = self.cache_dir / granule_url.split("?")[0].rsplit("/", 1)[-1]
                downloaded = self._download_remote_file(granule_url, local_path)
                if downloaded is None:
                    results[business_date_value] = (None, "missing")
                    continue

                cover = self._load_remote_granule(downloaded, working_lons, working_lats)
                if cover is None:
                    results[business_date_value] = (None, "missing")
                    continue

                self._save_daily_cache(business_date_value, working_lons, working_lats, cover, HISTORICAL_VIIRS_STATUS)
                results[business_date_value] = (cover.astype(np.float32), HISTORICAL_VIIRS_STATUS)

        for business_date_value in business_dates:
            results.setdefault(business_date_value, (None, "missing"))
        return results


class Era5PseudoLabelTrainer:
    def __init__(self, gfs_client: GFSClient) -> None:
        self.gfs_client = gfs_client

    @staticmethod
    def _label_archive_path(start_date: date, end_date: date) -> Path:
        return CACHE_DIR / "era5_labels" / f"era5_land_{start_date:%Y%m%d}_{end_date:%Y%m%d}.nc"

    @staticmethod
    def _extract_label_dataset(archive_path: Path) -> Path:
        if not archive_path.exists():
            raise FileNotFoundError(f"Pseudo-label archive is missing: {archive_path}")

        if zipfile.is_zipfile(archive_path):
            target_dir = archive_path.parent / f"{archive_path.stem}_unzipped"
            target_dir.mkdir(parents=True, exist_ok=True)
            nc_files = sorted(target_dir.rglob("*.nc"))
            if not nc_files:
                with zipfile.ZipFile(archive_path, "r") as archive:
                    archive.extractall(target_dir)
                nc_files = sorted(target_dir.rglob("*.nc"))
            if not nc_files:
                raise RuntimeError("No NetCDF file was found after extracting ERA5 labels.")
            return nc_files[0]
        return archive_path

    @staticmethod
    def _label_dataset_time_name(dataset: xr.Dataset) -> str:
        return "valid_time" if "valid_time" in dataset.coords else "time"

    def _label_dataset_covers(self, dataset_path: Path, start_date: date, end_date: date) -> bool:
        dataset = xr.open_dataset(dataset_path, engine="netcdf4")
        try:
            time_name = self._label_dataset_time_name(dataset)
            timestamps = pd.to_datetime(dataset[time_name].values)
            if len(timestamps) == 0:
                return False
            return timestamps[0].date() <= start_date and timestamps[-1].date() >= end_date
        finally:
            dataset.close()

    def _download_label_dataset(self, start_date: date, end_date: date, target_path: Path) -> Path:
        try:
            import cdsapi
        except ImportError as exc:
            raise RuntimeError(
                "cdsapi is required to download the full ERA5-Land pseudo-label window. "
                "Install cdsapi or provide a local archive that covers the full training period."
            ) from exc

        min_lon, min_lat, max_lon, max_lat = load_study_area_bbox()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        client = cdsapi.Client()
        chunk_dir = target_path.parent / f"{target_path.stem}_chunks"
        chunk_dir.mkdir(parents=True, exist_ok=True)

        chunk_paths: list[Path] = []
        cursor = start_date.replace(day=1)
        while cursor <= end_date:
            next_month = (cursor.replace(day=28) + timedelta(days=4)).replace(day=1)
            chunk_start = max(start_date, cursor)
            chunk_end = min(end_date, next_month - timedelta(days=1))
            chunk_path = chunk_dir / f"era5_land_{chunk_start:%Y%m%d}_{chunk_end:%Y%m%d}.nc"
            if not chunk_path.exists():
                days = pd.date_range(chunk_start, chunk_end, freq="D")
                request = {
                    "variable": ["snow_depth", "snow_depth_water_equivalent", "2m_temperature"],
                    "year": [chunk_start.strftime("%Y")],
                    "month": [chunk_start.strftime("%m")],
                    "day": sorted({day.strftime("%d") for day in days}),
                    "time": ["00:00", "03:00", "06:00", "09:00", "12:00", "15:00", "18:00", "21:00"],
                    "area": [
                        float(math.ceil(max_lat)),
                        float(math.floor(min_lon)),
                        float(math.floor(min_lat)),
                        float(math.ceil(max_lon)),
                    ],
                    "format": "netcdf",
                }
                client.retrieve("reanalysis-era5-land", request, str(chunk_path))
            chunk_paths.append(chunk_path)
            cursor = next_month

        datasets: list[xr.Dataset] = []
        try:
            for chunk_path in chunk_paths:
                extracted_path = self._extract_label_dataset(chunk_path)
                dataset = xr.open_dataset(extracted_path, engine="netcdf4")
                if "valid_time" in dataset.coords:
                    dataset = dataset.rename({"valid_time": "time"})
                datasets.append(dataset.load())

            combined = xr.concat(datasets, dim="time").sortby("time")
            combined.to_netcdf(target_path)
            combined.close()
        finally:
            for dataset in datasets:
                dataset.close()
        return target_path

    def _ensure_label_dataset(self) -> Path:
        requested_archive = self._label_archive_path(TRAINING_START_DATE, TRAINING_END_DATE)
        candidate_archives = [requested_archive]
        if requested_archive != LEGACY_ERA5_ARCHIVE:
            candidate_archives.append(LEGACY_ERA5_ARCHIVE)

        for archive_path in candidate_archives:
            if not archive_path.exists():
                continue
            dataset_path = self._extract_label_dataset(archive_path)
            if self._label_dataset_covers(dataset_path, TRAINING_START_DATE, TRAINING_END_DATE):
                return dataset_path

        downloaded_archive = self._download_label_dataset(TRAINING_START_DATE, TRAINING_END_DATE, requested_archive)
        dataset_path = self._extract_label_dataset(downloaded_archive)
        if not self._label_dataset_covers(dataset_path, TRAINING_START_DATE, TRAINING_END_DATE):
            raise RuntimeError("Downloaded ERA5-Land pseudo-label archive does not cover the requested training window.")
        return dataset_path

    def _interpolate_static_to_label_grid(
        self,
        static: dict[str, np.ndarray],
        label_lats: np.ndarray,
        label_lons: np.ndarray,
    ) -> dict[str, np.ndarray]:
        source = {
            name: xr.DataArray(
                values,
                dims=("latitude", "longitude"),
                coords={"latitude": static["latitudes"], "longitude": static["longitudes"]},
            )
            for name, values in static.items()
            if name in {"orography", "slope_deg", "aspect_deg", "zone_id"}
        }
        return {
            name: array.interp(latitude=label_lats, longitude=label_lons, method="nearest").values.astype(np.float32)
            for name, array in source.items()
        }

    def _prefetch_training_viirs(
        self,
        dates: list[date],
        label_lons: np.ndarray,
        label_lats: np.ndarray,
    ) -> dict[date, tuple[np.ndarray | None, str]]:
        viirs_client = HistoricalVIIRSClient()
        daily_cover = viirs_client.prefetch_snow_cover_range(dates, label_lons, label_lats)
        return {
            business_date_value: (cover.astype(np.float32) if cover is not None else None, status)
            for business_date_value, (cover, status) in daily_cover.items()
        }

    def prepare_training_frame(self, force_rebuild: bool = False) -> tuple[pd.DataFrame, dict[str, Any]]:
        cache_path = CACHE_DIR / "training_frame.pkl"
        metadata_path = CACHE_DIR / "training_frame_meta.json"
        if cache_path.exists() and metadata_path.exists() and not force_rebuild:
            cached_frame = pd.read_pickle(cache_path)
            cached_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            required_columns = set(FEATURE_COLUMNS + ["date", "swe_mm"])
            if required_columns.issubset(cached_frame.columns) and metadata_signature_matches(cached_metadata):
                return cached_frame, cached_metadata

        label_path = self._ensure_label_dataset()
        dataset = xr.open_dataset(label_path, engine="netcdf4")
        if "valid_time" in dataset.coords:
            dataset = dataset.rename({"valid_time": "time"})

        daily_tmean = (dataset["t2m"] - 273.15).resample(time="1D").mean()
        daily_tmin = (dataset["t2m"] - 273.15).resample(time="1D").min()
        daily_tmax = (dataset["t2m"] - 273.15).resample(time="1D").max()
        daily_swe = dataset["sd"].resample(time="1D").mean() * 1000.0
        daily_depth = dataset["sde"].resample(time="1D").mean()

        label_lats = daily_swe["latitude"].values.astype(np.float32)
        label_lons = daily_swe["longitude"].values.astype(np.float32)
        timestamps = [
            timestamp
            for timestamp in pd.to_datetime(daily_swe["time"].values)
            if TRAINING_START_DATE <= timestamp.date() <= TRAINING_END_DATE
        ]

        terrain = self.gfs_client.get_static_terrain(datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0))
        static_interp = self._interpolate_static_to_label_grid(terrain, label_lats, label_lons)
        viirs_daily = self._prefetch_training_viirs([timestamp.date() for timestamp in timestamps], label_lons, label_lats)

        frames: list[pd.DataFrame] = []
        for timestamp in timestamps:
            swe_mm = daily_swe.sel(time=timestamp).values.astype(np.float32)
            depth_m = daily_depth.sel(time=timestamp).values.astype(np.float32)
            temp_mean = daily_tmean.sel(time=timestamp).values.astype(np.float32)
            temp_min = daily_tmin.sel(time=timestamp).values.astype(np.float32)
            temp_max = daily_tmax.sel(time=timestamp).values.astype(np.float32)
            observed_cover, viirs_status = viirs_daily.get(timestamp.date(), (None, "missing"))
            if observed_cover is None:
                observed_cover = np.full_like(swe_mm, np.nan, dtype=np.float32)

            lon_mesh, lat_mesh = np.meshgrid(label_lons, label_lats)
            frame = pd.DataFrame(
                {
                    "date": timestamp.date(),
                    "latitude": lat_mesh.ravel(),
                    "longitude": lon_mesh.ravel(),
                    "swe_mm": swe_mm.ravel(),
                    "snow_depth_m": depth_m.ravel(),
                    "temp_mean_c": temp_mean.ravel(),
                    "temp_min_c": temp_min.ravel(),
                    "temp_max_c": temp_max.ravel(),
                    "elevation_m": static_interp["orography"].ravel(),
                    "slope_deg": static_interp["slope_deg"].ravel(),
                    "aspect_deg": static_interp["aspect_deg"].ravel(),
                    "zone_id": static_interp["zone_id"].ravel(),
                    "viirs_observed_cover": observed_cover.ravel(),
                    "viirs_status": viirs_status,
                }
            )
            frames.append(frame)

        dataset.close()

        frame = pd.concat(frames, ignore_index=True)
        frame = frame.replace([np.inf, -np.inf], np.nan)
        frame = frame.dropna(
            subset=[
                "swe_mm",
                "snow_depth_m",
                "temp_mean_c",
                "temp_min_c",
                "temp_max_c",
                "elevation_m",
                "slope_deg",
                "aspect_deg",
                "zone_id",
            ]
        ).reset_index(drop=True)
        frame["cell_id"] = (
            frame["latitude"].round(5).astype(str)
            + "_"
            + frame["longitude"].round(5).astype(str)
        )
        frame = frame.sort_values(["cell_id", "date"]).reset_index(drop=True)
        frame["doy"] = pd.to_datetime(frame["date"]).dt.dayofyear.astype(np.int16)
        frame["temp_range_c"] = frame["temp_max_c"] - frame["temp_min_c"]
        frame["positive_degree_c"] = frame["temp_mean_c"].clip(lower=0.0)
        frame["snow_cover_fraction"] = (frame["swe_mm"] > 1.0).astype(np.float32)

        grouped = frame.groupby("cell_id", sort=False)
        frame["prev_swe_mm"] = grouped["swe_mm"].shift(1).fillna(0.0)
        frame["prev_swe_3d_mean"] = grouped["prev_swe_mm"].transform(
            lambda series: series.rolling(3, min_periods=1).mean()
        )
        frame["prev_swe_7d_mean"] = grouped["prev_swe_mm"].transform(
            lambda series: series.rolling(7, min_periods=1).mean()
        )

        snow_depth_delta_mm = grouped["snow_depth_m"].diff().fillna(0.0).clip(lower=0.0) * 150.0
        cold_gate = snow_fraction_from_temperature(
            frame["temp_min_c"].to_numpy(dtype=np.float32),
            frame["temp_max_c"].to_numpy(dtype=np.float32),
        )
        frame["solid_precip_mm"] = snow_depth_delta_mm.to_numpy(dtype=np.float32) * cold_gate
        frame["precipitation_mm"] = frame["solid_precip_mm"]

        frame["temp_mean_3d"] = grouped["temp_mean_c"].transform(
            lambda series: series.rolling(3, min_periods=1).mean()
        )
        frame["temp_mean_7d"] = grouped["temp_mean_c"].transform(
            lambda series: series.rolling(7, min_periods=1).mean()
        )
        frame["temp_mean_14d"] = grouped["temp_mean_c"].transform(
            lambda series: series.rolling(14, min_periods=1).mean()
        )
        frame["temp_mean_30d"] = grouped["temp_mean_c"].transform(
            lambda series: series.rolling(30, min_periods=1).mean()
        )

        frame["solid_precip_3d"] = grouped["solid_precip_mm"].transform(
            lambda series: series.rolling(3, min_periods=1).sum()
        )
        frame["solid_precip_7d"] = grouped["solid_precip_mm"].transform(
            lambda series: series.rolling(7, min_periods=1).sum()
        )
        frame["solid_precip_14d"] = grouped["solid_precip_mm"].transform(
            lambda series: series.rolling(14, min_periods=1).sum()
        )
        frame["solid_precip_30d"] = grouped["solid_precip_mm"].transform(
            lambda series: series.rolling(30, min_periods=1).sum()
        )

        frame["positive_degree_3d"] = grouped["positive_degree_c"].transform(
            lambda series: series.rolling(3, min_periods=1).sum()
        )
        frame["positive_degree_7d"] = grouped["positive_degree_c"].transform(
            lambda series: series.rolling(7, min_periods=1).sum()
        )

        frame["viirs_available"] = frame["viirs_observed_cover"].notna().astype(np.float32)
        frame["snow_cover_recent_3d"] = grouped["viirs_observed_cover"].transform(
            lambda series: series.shift(1).rolling(3, min_periods=1).mean()
        )
        default_cover = (frame["prev_swe_mm"] > 1.0).astype(np.float32)
        frame["snow_cover_fraction"] = frame["viirs_observed_cover"].astype(np.float32)
        frame["snow_cover_fraction"] = frame["snow_cover_fraction"].fillna(frame["snow_cover_recent_3d"])
        frame["snow_cover_fraction"] = frame["snow_cover_fraction"].fillna(default_cover)
        frame["snow_cover_fraction"] = frame["snow_cover_fraction"].clip(0.0, 1.0).astype(np.float32)

        frame["snow_cover_persist_3d"] = grouped["snow_cover_fraction"].transform(
            lambda series: series.rolling(3, min_periods=1).mean()
        )
        frame["snow_cover_persist_7d"] = grouped["snow_cover_fraction"].transform(
            lambda series: series.rolling(7, min_periods=1).mean()
        )

        frame = frame[FEATURE_COLUMNS + ["date", "swe_mm"]].copy()

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_pickle(cache_path)
        viirs_observation_ratio = float(frame["viirs_available"].mean()) if len(frame) else 0.0
        date_viirs_observation = frame.groupby("date")["viirs_available"].mean() if len(frame) else pd.Series(dtype=float)
        metadata = {
            "rows": int(len(frame)),
            "start_date": str(frame["date"].min()),
            "end_date": str(frame["date"].max()),
            **training_metadata_signature(),
            "historical_viirs_coverage_days": int((date_viirs_observation > 0.0).sum()),
            "historical_viirs_fallback_days": int((date_viirs_observation == 0.0).sum()),
            "historical_viirs_observation_ratio": viirs_observation_ratio,
            "historical_viirs_fallback_ratio": float(1.0 - viirs_observation_ratio),
        }
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        return frame, metadata

    def train_or_load_model(self, force_retrain: bool = False) -> dict[str, Any]:
        if MODEL_PATH.exists() and not force_retrain:
            bundle = joblib.load(MODEL_PATH)
            if metadata_signature_matches(bundle):
                return bundle

        training_frame, metadata = self.prepare_training_frame(force_rebuild=force_retrain)
        unique_dates = sorted(pd.to_datetime(training_frame["date"]).dt.date.unique())
        split_index = max(int(len(unique_dates) * 0.8), 1)
        train_dates = set(unique_dates[:split_index])
        valid_dates = set(unique_dates[split_index:]) or train_dates

        train_df = training_frame[training_frame["date"].isin(train_dates)].copy()
        valid_df = training_frame[training_frame["date"].isin(valid_dates)].copy()

        x_train = train_df[FEATURE_COLUMNS]
        y_train = train_df["swe_mm"].to_numpy(dtype=np.float32)
        x_valid = valid_df[FEATURE_COLUMNS]
        y_valid = valid_df["swe_mm"].to_numpy(dtype=np.float32)

        model = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_depth=8,
            max_iter=350,
            min_samples_leaf=80,
            random_state=42,
        )
        model.fit(x_train, y_train)

        valid_pred = model.predict(x_valid).astype(np.float32)
        metrics = {
            "rmse_mm": float(math.sqrt(mean_squared_error(y_valid, valid_pred))),
            "corr": safe_correlation(y_valid, valid_pred),
            "training_rows": int(len(train_df)),
            "validation_rows": int(len(valid_df)),
            "label_window": metadata,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "feature_columns": FEATURE_COLUMNS,
            "training_start_date": TRAINING_START_DATE.isoformat(),
            "training_end_date": TRAINING_END_DATE.isoformat(),
            "training_window_days": training_window_days(),
            "historical_viirs_product": HISTORICAL_VIIRS_PRODUCT,
            "historical_viirs_source": HISTORICAL_VIIRS_SOURCE,
        }

        bundle = {
            "model": model,
            "feature_columns": FEATURE_COLUMNS,
            "metrics": metrics,
            "model_version": MODEL_VERSION,
            "training_start_date": TRAINING_START_DATE.isoformat(),
            "training_end_date": TRAINING_END_DATE.isoformat(),
            "training_window_days": training_window_days(),
            "historical_viirs_product": HISTORICAL_VIIRS_PRODUCT,
            "historical_viirs_source": HISTORICAL_VIIRS_SOURCE,
        }
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, MODEL_PATH)
        TRAINING_METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        return bundle


def _forcing_cache_path(business_date_value: date) -> Path:
    return FORCING_CACHE_DIR / f"forcing_{business_date_value:%Y%m%d}.npz"


def _state_cache_path(business_date_value: date) -> Path:
    return STATE_CACHE_DIR / f"state_{business_date_value:%Y%m%d}.npz"


def save_forcing_cache(
    business_date_value: date,
    forcing: dict[str, Any],
    snow_cover_fraction: np.ndarray,
    viirs_available: np.ndarray,
    viirs_status: str,
) -> str:
    path = _forcing_cache_path(business_date_value)
    np.savez_compressed(
        path,
        temp_mean_c=forcing["temp_mean_c"].astype(np.float32),
        temp_min_c=forcing["temp_min_c"].astype(np.float32),
        temp_max_c=forcing["temp_max_c"].astype(np.float32),
        precipitation_mm=forcing["precipitation_mm"].astype(np.float32),
        solid_precip_mm=forcing["solid_precip_mm"].astype(np.float32),
        snow_cover_fraction=snow_cover_fraction.astype(np.float32),
        viirs_available=viirs_available.astype(np.float32),
        viirs_missing=np.array([1 if viirs_status == "missing" else 0], dtype=np.int16),
    )
    return str(path)


def save_state_cache(
    business_date_value: date,
    swe_mm: np.ndarray,
    snowmelt_mm_day: np.ndarray,
    qa_flag: np.ndarray,
) -> None:
    path = _state_cache_path(business_date_value)
    np.savez_compressed(
        path,
        swe_mm=swe_mm.astype(np.float32),
        snowmelt_mm_day=snowmelt_mm_day.astype(np.float32),
        qa_flag=qa_flag.astype(np.int16),
    )


def load_state_cache(business_date_value: date) -> dict[str, np.ndarray] | None:
    path = _state_cache_path(business_date_value)
    if not path.exists():
        return None
    payload = np.load(path)
    return {
        "swe_mm": payload["swe_mm"].astype(np.float32),
        "snowmelt_mm_day": payload["snowmelt_mm_day"].astype(np.float32),
        "qa_flag": payload["qa_flag"].astype(np.int16),
    }


def load_cached_arrays(
    field_name: str,
    business_dates: list[date],
) -> list[np.ndarray]:
    arrays: list[np.ndarray] = []
    for business_date_value in business_dates:
        path = _forcing_cache_path(business_date_value)
        if not path.exists():
            continue
        payload = np.load(path)
        if field_name in payload:
            arrays.append(payload[field_name].astype(np.float32))
    return arrays


def resolve_snow_cover_features(
    observed_snow_cover: np.ndarray | None,
    snow_history: list[np.ndarray],
    prev_swe_mm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    default_cover = (prev_swe_mm > 1.0).astype(np.float32)
    history_fill = _nanmean_stack(snow_history[-3:])
    if history_fill is None:
        history_fill = np.full_like(default_cover, np.nan, dtype=np.float32)

    if observed_snow_cover is None:
        observed = np.full_like(default_cover, np.nan, dtype=np.float32)
    else:
        observed = np.asarray(observed_snow_cover, dtype=np.float32)

    viirs_available = np.isfinite(observed).astype(np.float32)
    resolved_cover = np.where(np.isfinite(observed), observed, history_fill)
    resolved_cover = np.where(np.isfinite(resolved_cover), resolved_cover, default_cover)
    return np.clip(resolved_cover, 0.0, 1.0).astype(np.float32), viirs_available


def build_feature_frame(
    forcing: dict[str, Any],
    prev_swe_mm: np.ndarray,
    observed_snow_cover: np.ndarray | None,
    history_dates: list[date],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    terrain = forcing["terrain"]
    latitudes = forcing["latitudes"]
    longitudes = forcing["longitudes"]
    lon_mesh, lat_mesh = np.meshgrid(longitudes, latitudes)

    temp_mean_history = load_cached_arrays("temp_mean_c", history_dates)
    solid_history = load_cached_arrays("solid_precip_mm", history_dates)
    snow_history = load_cached_arrays("snow_cover_fraction", history_dates)

    prev_history = []
    for history_date in history_dates:
        cached_state = load_state_cache(history_date)
        if cached_state is not None:
            prev_history.append(cached_state["swe_mm"])

    temp_mean_c = forcing["temp_mean_c"]
    temp_min_c = forcing["temp_min_c"]
    temp_max_c = forcing["temp_max_c"]
    precipitation_mm = forcing["precipitation_mm"]
    solid_precip_mm = forcing["solid_precip_mm"]

    current_positive = np.clip(temp_mean_c, 0.0, None)
    current_cover, viirs_available = resolve_snow_cover_features(observed_snow_cover, snow_history, prev_swe_mm)
    default_cover = (prev_swe_mm > 1.0).astype(np.float32)

    frame = pd.DataFrame(
        {
            "latitude": lat_mesh.ravel(),
            "longitude": lon_mesh.ravel(),
            "elevation_m": terrain["orography"].ravel(),
            "slope_deg": terrain["slope_deg"].ravel(),
            "aspect_deg": terrain["aspect_deg"].ravel(),
            "zone_id": terrain["zone_id"].ravel(),
            "doy": int(pd.Timestamp(forcing["business_date"]).dayofyear),
            "temp_mean_c": temp_mean_c.ravel(),
            "temp_min_c": temp_min_c.ravel(),
            "temp_max_c": temp_max_c.ravel(),
            "temp_range_c": (temp_max_c - temp_min_c).ravel(),
            "positive_degree_c": current_positive.ravel(),
            "positive_degree_3d": _rolling_sum_positive(temp_mean_history + [temp_mean_c], 3, current_positive).ravel(),
            "positive_degree_7d": _rolling_sum_positive(temp_mean_history + [temp_mean_c], 7, current_positive).ravel(),
            "precipitation_mm": precipitation_mm.ravel(),
            "solid_precip_mm": solid_precip_mm.ravel(),
            "solid_precip_3d": _rolling_sum_positive(solid_history + [solid_precip_mm], 3, solid_precip_mm).ravel(),
            "solid_precip_7d": _rolling_sum_positive(solid_history + [solid_precip_mm], 7, solid_precip_mm).ravel(),
            "solid_precip_14d": _rolling_sum_positive(solid_history + [solid_precip_mm], 14, solid_precip_mm).ravel(),
            "solid_precip_30d": _rolling_sum_positive(solid_history + [solid_precip_mm], 30, solid_precip_mm).ravel(),
            "temp_mean_3d": _rolling_mean(temp_mean_history + [temp_mean_c], 3, temp_mean_c).ravel(),
            "temp_mean_7d": _rolling_mean(temp_mean_history + [temp_mean_c], 7, temp_mean_c).ravel(),
            "temp_mean_14d": _rolling_mean(temp_mean_history + [temp_mean_c], 14, temp_mean_c).ravel(),
            "temp_mean_30d": _rolling_mean(temp_mean_history + [temp_mean_c], 30, temp_mean_c).ravel(),
            "prev_swe_mm": prev_swe_mm.ravel(),
            "prev_swe_3d_mean": _rolling_mean(prev_history + [prev_swe_mm], 3, prev_swe_mm).ravel(),
            "prev_swe_7d_mean": _rolling_mean(prev_history + [prev_swe_mm], 7, prev_swe_mm).ravel(),
            "snow_cover_fraction": current_cover.ravel(),
            "viirs_available": viirs_available.ravel(),
            "snow_cover_persist_3d": _rolling_binary_mean(snow_history + [current_cover], 3, default_cover).ravel(),
            "snow_cover_persist_7d": _rolling_binary_mean(snow_history + [current_cover], 7, default_cover).ravel(),
        }
    )
    return frame[FEATURE_COLUMNS], current_cover, viirs_available


def write_daily_series(manifest: dict[str, Any]) -> pd.DataFrame:
    rows = [
        {
            "date": item["business_date"],
            "swe_mm": item["swe_mm"],
            "snowmelt_mm_day": item["snowmelt_mm_day"],
            "source_status": item["source_status"],
            "qa_flag": item["qa_flag"],
            "forcing_cycle": item["forcing_cycle"],
            "viirs_status": item["viirs_status"],
            "model_version": item["model_version"],
            "is_backfill": item["is_backfill"],
        }
        for item in manifest.get("entries", [])
    ]
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values("date").reset_index(drop=True)
    frame.to_csv(DAILY_SERIES_PATH, index=False, encoding="utf-8-sig")
    frame.to_csv(ROUTING_SERIES_PATH, index=False, encoding="utf-8-sig")
    return frame


def _candidate_viirs_cover(
    viirs_client: BaseVIIRSClient,
    business_date_value: date,
    forcing: dict[str, Any],
    fallback_viirs_client: BaseVIIRSClient | None = None,
) -> tuple[np.ndarray | None, str]:
    candidate_clients = [viirs_client]
    if fallback_viirs_client is not None:
        candidate_clients.append(fallback_viirs_client)

    last_status = "missing"
    for client in candidate_clients:
        cover, status = client.get_snow_cover(
            business_date_value,
            forcing["longitudes"],
            forcing["latitudes"],
        )
        last_status = status
        if cover is None:
            continue
        oriented, _, _ = ensure_lat_desc(cover, forcing["latitudes"], forcing["longitudes"])
        return oriented.astype(np.float32), status
    return None, last_status


def _viirs_cover_is_complete(observed_snow_cover: np.ndarray | None, viirs_status: str) -> bool:
    if observed_snow_cover is None:
        return False
    if str(viirs_status or "").strip().lower() == "missing":
        return False
    return bool(np.any(np.isfinite(observed_snow_cover)))


def _load_previous_swe(
    business_date_value: date,
    forcing: dict[str, Any],
) -> tuple[np.ndarray, bool]:
    previous_date = business_date_value - timedelta(days=1)
    cached_state = load_state_cache(previous_date)
    if cached_state is not None:
        return cached_state["swe_mm"], False

    cold_start = forcing.get("cold_start_seed_mm")
    if cold_start is None:
        cold_start = np.zeros_like(forcing["temp_mean_c"], dtype=np.float32)
    return cold_start.astype(np.float32), True


def run_business_day(
    business_date_value: date,
    model_bundle: dict[str, Any],
    gfs_client: GFSClient,
    viirs_client: BaseVIIRSClient,
    is_backfill: bool,
    fallback_viirs_client: BaseVIIRSClient | None = None,
    forcing_override: dict[str, Any] | None = None,
    observed_snow_cover_override: np.ndarray | None = None,
    viirs_status_override: str | None = None,
) -> DailyEntry:
    forcing = forcing_override if forcing_override is not None else gfs_client.get_business_day_forcing(business_date_value)
    prev_swe_mm, used_cold_start = _load_previous_swe(business_date_value, forcing)
    if observed_snow_cover_override is not None or viirs_status_override is not None:
        observed_snow_cover = (
            np.asarray(observed_snow_cover_override, dtype=np.float32)
            if observed_snow_cover_override is not None
            else None
        )
        viirs_status = viirs_status_override or "missing"
    else:
        observed_snow_cover, viirs_status = _candidate_viirs_cover(
            viirs_client,
            business_date_value,
            forcing,
            fallback_viirs_client=fallback_viirs_client,
        )

    history_dates = [business_date_value - timedelta(days=offset) for offset in range(30, 0, -1)]
    feature_frame, feature_cover, viirs_available = build_feature_frame(
        forcing,
        prev_swe_mm,
        observed_snow_cover,
        history_dates,
    )
    prediction = model_bundle["model"].predict(feature_frame).astype(np.float32)

    grid_shape = forcing["temp_mean_c"].shape
    predicted_swe_mm = prediction.reshape(grid_shape)
    base_swe_mm = np.clip(prev_swe_mm + forcing["solid_precip_mm"], 0.0, None)
    swe_mm = np.clip(np.minimum(predicted_swe_mm, base_swe_mm), 0.0, None)

    qa_flag = np.zeros(grid_shape, dtype=np.int16)
    if viirs_status == "missing":
        qa_flag |= QA_VIIRS_MISSING
    else:
        missing_cells = ~np.isfinite(observed_snow_cover)
        if np.any(missing_cells):
            qa_flag = np.where(missing_cells, qa_flag | QA_VIIRS_MISSING, qa_flag)
        no_snow_mask = np.isfinite(observed_snow_cover) & (observed_snow_cover < 0.1)
        swe_mm = np.where(no_snow_mask, np.minimum(swe_mm, base_swe_mm * 0.1), swe_mm)
        qa_flag = np.where(no_snow_mask, qa_flag | QA_VIIRS_CONSTRAINED, qa_flag)

    if used_cold_start:
        qa_flag |= QA_COLD_START_EXTERNAL

    snowmelt_mm = np.clip(base_swe_mm - swe_mm, 0.0, None).astype(np.float32)
    swe_mm = swe_mm.astype(np.float32)

    terrain = forcing["terrain"]
    inside_mask = terrain["inside_mask"]
    swe_masked = np.where(inside_mask, swe_mm, np.nan)
    snowmelt_masked = np.where(inside_mask, snowmelt_mm, np.nan)
    qa_masked = np.where(inside_mask, qa_flag, NODATA_INT)

    day_token = business_date_value.strftime("%Y%m%d")
    swe_path = RASTER_DIR / f"SWE_mm_{day_token}.tif"
    melt_path = RASTER_DIR / f"Snowmelt_mm_day_{day_token}.tif"
    qa_path = RASTER_DIR / f"SWE_QA_{day_token}.tif"

    swe_raster = save_raster(swe_masked, forcing["longitudes"], forcing["latitudes"], swe_path, NODATA_FLOAT, "float32")
    snowmelt_raster = save_raster(
        snowmelt_masked,
        forcing["longitudes"],
        forcing["latitudes"],
        melt_path,
        NODATA_FLOAT,
        "float32",
    )
    qa_raster = save_raster(qa_masked, forcing["longitudes"], forcing["latitudes"], qa_path, NODATA_INT, "int16")
    forcing_cache = save_forcing_cache(business_date_value, forcing, feature_cover, viirs_available, viirs_status)
    save_state_cache(business_date_value, swe_mm, snowmelt_mm, qa_flag)

    source_status = "gfs_ml_cold_start" if used_cold_start else "gfs_ml_recursive"
    basin_swe = float(np.nanmean(swe_masked))
    basin_melt = float(np.nanmean(snowmelt_masked))
    basin_qa = int(np.nanmax(np.where(np.isnan(qa_masked), 0, qa_masked)))

    return DailyEntry(
        business_date=business_date_value.isoformat(),
        forcing_cycle=cycle_string(forcing["cycle_dt"]),
        viirs_status=viirs_status,
        model_version=model_bundle["model_version"],
        is_backfill=bool(is_backfill),
        source_status=source_status,
        qa_flag=basin_qa,
        swe_raster=swe_raster,
        snowmelt_raster=snowmelt_raster,
        qa_raster=qa_raster,
        forcing_cache=forcing_cache,
        swe_mm=basin_swe,
        snowmelt_mm_day=basin_melt,
    )


def load_existing_results() -> dict[str, Any]:
    return build_result_payload(read_manifest())


def ensure_model(force_retrain: bool = False) -> dict[str, Any]:
    ensure_directories()
    trainer = Era5PseudoLabelTrainer(GFSClient())
    return trainer.train_or_load_model(force_retrain=force_retrain)


def _resolve_update_target_inputs(
    manifest: dict[str, Any],
    gfs_client: GFSClient,
    viirs_client: BaseVIIRSClient,
    fallback_viirs_client: BaseVIIRSClient | None = None,
) -> tuple[date, dict[str, Any] | None, np.ndarray | None, str | None]:
    candidate_date = current_business_date()
    existing_entry = _find_manifest_entry(manifest, candidate_date)
    if _manifest_entry_is_reusable(existing_entry) and _entry_has_available_viirs(existing_entry):
        return candidate_date, None, None, None

    try:
        forcing = gfs_client.get_business_day_forcing(candidate_date)
        observed_snow_cover, viirs_status = _candidate_viirs_cover(
            viirs_client,
            candidate_date,
            forcing,
            fallback_viirs_client=fallback_viirs_client,
        )
        if _viirs_cover_is_complete(observed_snow_cover, viirs_status):
            return candidate_date, forcing, observed_snow_cover, viirs_status
    except Exception:
        pass

    return candidate_date - timedelta(days=1), None, None, None


def run_update_latest(force_retrain: bool = False) -> dict[str, Any]:
    ensure_directories()
    manifest = read_manifest()
    gfs_client = GFSClient()
    viirs_client = RealtimeVIIRSClient()
    fallback_viirs_client = HistoricalVIIRSClient()
    (
        target_date,
        forcing_override,
        observed_snow_cover_override,
        viirs_status_override,
    ) = _resolve_update_target_inputs(
        manifest,
        gfs_client,
        viirs_client,
        fallback_viirs_client=fallback_viirs_client,
    )
    if not force_retrain:
        existing_entry = _find_manifest_entry(manifest, target_date)
        if _manifest_entry_is_reusable(existing_entry):
            return build_result_payload(manifest, preferred_business_date=target_date)

    model_bundle = ensure_model(force_retrain=force_retrain)
    entry = run_business_day(
        target_date,
        model_bundle,
        gfs_client,
        viirs_client,
        is_backfill=False,
        fallback_viirs_client=fallback_viirs_client,
        forcing_override=forcing_override,
        observed_snow_cover_override=observed_snow_cover_override,
        viirs_status_override=viirs_status_override,
    )
    manifest = upsert_manifest_entry(entry)
    write_daily_series(manifest)
    return build_result_payload(manifest, preferred_business_date=target_date)


def run_backfill(days_back: int = 7, force_retrain: bool = False) -> dict[str, Any]:
    ensure_directories()
    if days_back <= 0:
        raise ValueError("days_back must be greater than zero.")

    latest_date = latest_complete_business_date()
    start_date = latest_date - timedelta(days=days_back - 1)
    target_dates = [start_date + timedelta(days=offset) for offset in range(days_back)]
    manifest = read_manifest()
    pending_dates = target_dates
    if not force_retrain:
        pending_dates = [
            business_date_value
            for business_date_value in target_dates
            if not _manifest_entry_is_reusable(_find_manifest_entry(manifest, business_date_value))
        ]
        if not pending_dates:
            return build_result_payload(manifest)

    model_bundle = ensure_model(force_retrain=force_retrain)
    gfs_client = GFSClient()
    viirs_client = RealtimeVIIRSClient()
    fallback_viirs_client = HistoricalVIIRSClient()

    for business_date_value in pending_dates:
        entry = run_business_day(
            business_date_value,
            model_bundle,
            gfs_client,
            viirs_client,
            is_backfill=True,
            fallback_viirs_client=fallback_viirs_client,
        )
        manifest = upsert_manifest_entry(entry)

    write_daily_series(manifest)
    return build_result_payload(manifest)


def run_legacy_compatible_assessment() -> dict[str, Any]:
    return run_update_latest(force_retrain=False)
