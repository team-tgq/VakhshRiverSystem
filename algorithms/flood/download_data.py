import warnings
warnings.filterwarnings("ignore")

import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.mask import mask
import numpy as np
import os
import zipfile
import requests
import geopandas as gpd
from datetime import date, datetime, timedelta
from urllib.parse import urlencode


# -------------------------
# Config
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CFG = {
    "study_area_shp": os.path.join(BASE_DIR, "study_area.shp"),
    "dem_tif": os.path.join(BASE_DIR, "dem.tif"),

    "data_dir": os.path.join(BASE_DIR, "data"),
    "raw_dir": os.path.join(BASE_DIR, "data", "raw"),
    "proc_dir": os.path.join(BASE_DIR, "data", "processed"),

    # Your bbox (EPSG:4326)
    "north": 39.861329,
    "south": 38.202345,
    "west": 69.219977,
    "east": 73.704541,
    "gfs_bbox_pad_deg": 0.25,

    # ERA5-Land time selection (example: one month)
    "era5_year": "2018",
    "era5_month": "10",
}

GFS_NOMADS_FILTER_URL = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
GFS_NOMADS_PUB_ROOT = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod"
GFS_DAY_RUN_CYCLE = "00"
GFS_DAY_FORECAST_HOUR = 24
GFS_GRID_DEG = 0.25

# 用 os.makedirs 创建 raw（存放原始下载数据）和 processed（存放对齐后的数据）文件夹。
os.makedirs(CFG["raw_dir"], exist_ok=True)
os.makedirs(CFG["proc_dir"], exist_ok=True)

# 根据study_area.shp 矢量边界，把巨大的原始影像切成仅包含流域部分的形状。
def clip_to_study_area(in_raster, study_gdf, out_raster):
    with rasterio.open(in_raster) as src:
        geom = [study_gdf.to_crs(src.crs).unary_union.__geo_interface__]
        out_img, out_transform = mask(src, geom, crop=True)
        out_meta = src.meta.copy()
        out_meta.update({
            "height": out_img.shape[1],
            "width": out_img.shape[2],
            "transform": out_transform
        })
        with rasterio.open(out_raster, "w", **out_meta) as dst:
            dst.write(out_img)
    return out_raster

# 以 DEM 为基准，强行要求其他数据（如降雨量、土地利用）的行数、列数、坐标系和空间分辨率与其完全一致。
def match_dem_grid(src_tif, dem_tif, out_tif, resampling=Resampling.bilinear):
    """Reproject/resample src_tif to match DEM's crs/transform/shape.
       Force float32 output and NaN nodata to avoid 'all zeros' issues.
    """
    with rasterio.open(dem_tif) as dem:
        dst_crs = dem.crs
        dst_transform = dem.transform
        dst_height = dem.height
        dst_width = dem.width

        # 关键：不要直接用 dem.profile（会带入整型 dtype 和 nodata=32767）
        dst_profile = dem.profile.copy()
        dst_profile.update({
            "driver": "GTiff",
            "crs": dst_crs,
            "transform": dst_transform,
            "height": dst_height,
            "width": dst_width,
            "count": 1,
            "dtype": "float32",     # 关键：强制 float
            "nodata": np.nan,       # 关键：nodata 用 NaN
            "compress": "LZW"
        })

    with rasterio.open(src_tif) as src:
        src_nodata = src.nodata

        dst_data = np.full((dst_height, dst_width), np.nan, dtype=np.float32)

        reproject(
            source=rasterio.band(src, 1),
            destination=dst_data,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src_nodata,     # 关键：告诉 reproject 源nodata
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_nodata=np.nan,         # 关键：目标 nodata
            resampling=resampling,
        )

    with rasterio.open(out_tif, "w", **dst_profile) as dst:
        dst.write(dst_data, 1)

    return out_tif

# 下载 total_precipitation（总降水量）和 volumetric_soil_water_layer_1（第一层土壤水分）
def download_era5_land_nc(out_nc):
    """
    Download ERA5-Land for bbox using CDS API.
    Requires ~/.cdsapirc configured.
    """
    import cdsapi

    c = cdsapi.Client()
    days = [f"{d:02d}" for d in range(1, 32)]
    hours = [f"{h:02d}:00" for h in range(0, 24)]

    # CDS area: [north, west, south, east]
    area = [CFG["north"], CFG["west"], CFG["south"], CFG["east"]]

    req = {
        "variable": [
            "total_precipitation",
            "volumetric_soil_water_layer_1"
        ],
        "year": CFG["era5_year"],
        "month": CFG["era5_month"],
        "day": days,
        "time": hours,
        "area": area,
        "format": "netcdf",
    }

    print("[ERA5-Land] Downloading netCDF...")
    c.retrieve("reanalysis-era5-land", req, out_nc)
    print("[ERA5-Land] Saved:", out_nc)
    return out_nc

# 将下载的 .nc 格式（气象常用）转换为常用的 .tif 格式，并将每小时的数据累加/平均为月度数据。
def era5_nc_to_geotiff(era5_zip_path, out_tp_tif, out_swvl1_tif):
    import os
    import zipfile
    import xarray as xr
    import rioxarray  # noqa

    # 1) unzip
    extract_dir = os.path.splitext(era5_zip_path)[0] + "_unzipped"
    os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(era5_zip_path, "r") as z:
        z.extractall(extract_dir)

    # 2) find netcdf inside
    nc_files = []
    for root, _, files in os.walk(extract_dir):
        for fn in files:
            if fn.lower().endswith(".nc"):
                nc_files.append(os.path.join(root, fn))

    if not nc_files:
        raise RuntimeError(f"No .nc found inside ZIP: {era5_zip_path}")

    nc_path = nc_files[0]
    print("[ERA5] Extracted NetCDF:", nc_path)

    # 3) open netcdf explicitly with netcdf4 engine
    ds = xr.open_dataset(nc_path, engine="netcdf4")
    print("[ERA5] Variables:", list(ds.data_vars.keys()))

    # 4) compute monthly aggregates
    # 自动识别时间维度
    time_dim = None
    for d in ds.dims:
        if "time" in d:
            time_dim = d
            break

    print("Time dimension:", time_dim)

    # tp: meters -> monthly sum -> mm
    tp = ds["tp"].sum(dim=time_dim) * 1000.0

    # soil moisture: monthly mean
    swvl1 = ds["swvl1"].mean(dim=time_dim)

    # 5) write GeoTIFF in EPSG:4326
    def _to_tif(da, out_path):
        if "latitude" in da.dims and "longitude" in da.dims:
            da = da.rename({"latitude": "y", "longitude": "x"})
        elif "lat" in da.dims and "lon" in da.dims:
            da = da.rename({"lat": "y", "lon": "x"})
        if da["y"][0] < da["y"][-1]:
            da = da.sortby("y", ascending=False)
        da = da.astype("float32")
        da.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
        da.rio.write_crs("EPSG:4326", inplace=True)
        da.rio.to_raster(out_path)

    _to_tif(tp, out_tp_tif)
    _to_tif(swvl1, out_swvl1_tif)

    print("[ERA5] GeoTIFF saved:")
    print("  -", out_tp_tif)
    print("  -", out_swvl1_tif)

# 微软的 Planetary Computer 获取 ESA 的 10 米分辨率全球土地覆盖数据
def _resolve_era5_nc_path(era5_source_path):
    import os
    import zipfile

    if not os.path.exists(era5_source_path):
        raise FileNotFoundError(f"ERA5 source not found: {era5_source_path}")

    if era5_source_path.lower().endswith(".nc"):
        return era5_source_path

    if not era5_source_path.lower().endswith(".zip"):
        raise ValueError(f"Unsupported ERA5 file format: {era5_source_path}")

    extract_dir = os.path.splitext(era5_source_path)[0] + "_unzipped"
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(era5_source_path, "r") as z:
        z.extractall(extract_dir)

    nc_files = []
    for root, _, files in os.walk(extract_dir):
        for fn in files:
            if fn.lower().endswith(".nc"):
                nc_files.append(os.path.join(root, fn))

    if not nc_files:
        raise RuntimeError(f"No .nc found in ERA5 archive: {era5_source_path}")

    return nc_files[0]


def _find_time_dim(ds):
    for dim_name in ds.dims:
        if "time" in dim_name.lower():
            return dim_name
    raise RuntimeError("Could not find ERA5 time dimension.")


def _write_dataarray_tif(da, out_path):
    import rioxarray  # noqa

    if "latitude" in da.dims and "longitude" in da.dims:
        da = da.rename({"latitude": "y", "longitude": "x"})
    elif "lat" in da.dims and "lon" in da.dims:
        da = da.rename({"lat": "y", "lon": "x"})

    if da["y"][0] < da["y"][-1]:
        da = da.sortby("y", ascending=False)

    da = da.astype("float32")
    da.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
    da.rio.write_crs("EPSG:4326", inplace=True)
    da.rio.to_raster(out_path)


def era5_nc_to_daily_geotiffs(era5_source_path, out_dir):
    import numpy as np
    import xarray as xr

    os.makedirs(out_dir, exist_ok=True)

    nc_path = _resolve_era5_nc_path(era5_source_path)
    print("[ERA5] Using NetCDF:", nc_path)

    ds = xr.open_dataset(nc_path, engine="netcdf4")
    time_dim = _find_time_dim(ds)

    tp_daily = ds["tp"].resample({time_dim: "1D"}).sum() * 1000.0
    swvl1_daily = ds["swvl1"].resample({time_dim: "1D"}).mean()

    outputs = {
        "rain_raw_files": [],
        "soil_raw_files": [],
    }

    for timestamp in tp_daily[time_dim].values:
        date_str = np.datetime_as_string(timestamp, unit="D")
        compact_date = date_str.replace("-", "")

        rain_out = os.path.join(out_dir, f"era5_tp_mm_{compact_date}.tif")
        soil_out = os.path.join(out_dir, f"era5_swvl1_mean_{compact_date}.tif")

        if not os.path.exists(rain_out):
            _write_dataarray_tif(tp_daily.sel({time_dim: timestamp}), rain_out)
        if not os.path.exists(soil_out):
            _write_dataarray_tif(swvl1_daily.sel({time_dim: timestamp}), soil_out)

        outputs["rain_raw_files"].append(rain_out)
        outputs["soil_raw_files"].append(soil_out)

    print(f"[ERA5] Daily GeoTIFF count: {len(outputs['rain_raw_files'])}")
    return outputs


def _extract_daily_token(path):
    name = os.path.basename(path)
    digits = "".join(ch for ch in name if ch.isdigit())
    for idx in range(len(digits) - 7):
        token = digits[idx:idx + 8]
        if token.startswith("20"):
            return f"{token[:4]}-{token[4:6]}-{token[6:8]}"
    raise ValueError(f"Could not extract date from file name: {path}")


def match_daily_era5_to_dem_grid(rain_raw_files, soil_raw_files, dem_tif, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    outputs = {
        "rain_daily_files": [],
        "soil_daily_files": [],
    }

    for raw_path in rain_raw_files:
        date_token = _extract_daily_token(raw_path)
        out_path = os.path.join(out_dir, f"rain_mm_demgrid_{date_token}.tif")
        if not os.path.exists(out_path):
            match_dem_grid(raw_path, dem_tif, out_path, resampling=Resampling.bilinear)
            print("[ERA5] Matched rainfall to DEM grid:", out_path)
        outputs["rain_daily_files"].append(out_path)

    for raw_path in soil_raw_files:
        date_token = _extract_daily_token(raw_path)
        out_path = os.path.join(out_dir, f"soil_moist_demgrid_{date_token}.tif")
        if not os.path.exists(out_path):
            match_dem_grid(raw_path, dem_tif, out_path, resampling=Resampling.bilinear)
            print("[ERA5] Matched soil moisture to DEM grid:", out_path)
        outputs["soil_daily_files"].append(out_path)

    return outputs


def _parse_target_date(value):
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        text = value.strip()
        for fmt in ("%Y-%m-%d", "%Y%m%d", "%Y_%m_%d"):
            try:
                return datetime.strptime(text, fmt).date()
            except ValueError:
                continue
    raise ValueError(f"Could not parse target date: {value!r}")


def _first_existing_path(paths):
    for path in paths:
        if path and os.path.exists(path):
            return os.path.abspath(path)
    return None


def _resolve_study_area_path(cfg):
    study_area_path = _first_existing_path(
        [
            cfg.get("study_area_shp"),
            CFG["study_area_shp"],
            os.path.join(BASE_DIR, "study_area.shp"),
        ]
    )
    if study_area_path is None:
        raise FileNotFoundError("未找到研究区边界 study_area.shp，无法计算近实时数据下载范围。")
    return study_area_path


def _resolve_dem_clip_path(cfg):
    proc_dir = os.path.abspath(cfg.get("proc_dir", CFG["proc_dir"]))
    dem_path = _first_existing_path(
        [
            cfg.get("dem_path"),
            os.path.join(proc_dir, "dem_clip.tif"),
            os.path.join(CFG["proc_dir"], "dem_clip.tif"),
            os.path.join(BASE_DIR, "dem_clip.tif"),
        ]
    )
    if dem_path is None:
        raise FileNotFoundError("缺少 DEM 裁剪栅格 dem_clip.tif，无法对近实时气象数据进行网格对齐。")
    return dem_path


def _resolve_bbox(cfg):
    dem_path = _first_existing_path(
        [
            cfg.get("dem_path"),
            os.path.join(os.path.abspath(cfg.get("proc_dir", CFG["proc_dir"])), "dem_clip.tif"),
            os.path.join(CFG["proc_dir"], "dem_clip.tif"),
            os.path.join(BASE_DIR, "dem_clip.tif"),
        ]
    )
    if dem_path is not None:
        with rasterio.open(dem_path) as src:
            bounds = src.bounds
            if src.crs is not None and src.crs.to_string() != "EPSG:4326":
                west, south, east, north = rasterio.warp.transform_bounds(
                    src.crs,
                    "EPSG:4326",
                    bounds.left,
                    bounds.bottom,
                    bounds.right,
                    bounds.top,
                    densify_pts=21,
                )
            else:
                west, south, east, north = bounds.left, bounds.bottom, bounds.right, bounds.top
        return {
            "north": float(north),
            "south": float(south),
            "west": float(west),
            "east": float(east),
        }

    keys = ("north", "south", "west", "east")
    if all(key in cfg for key in keys):
        return {key: float(cfg[key]) for key in keys}

    study_area_path = _resolve_study_area_path(cfg)
    study = gpd.read_file(study_area_path).to_crs(4326)
    west, south, east, north = study.total_bounds
    return {
        "north": float(north),
        "south": float(south),
        "west": float(west),
        "east": float(east),
    }


def _expand_bbox(bbox, pad_deg):
    pad = max(float(pad_deg), 0.0)
    return {
        "north": min(90.0, float(bbox["north"]) + pad),
        "south": max(-90.0, float(bbox["south"]) - pad),
        "west": max(-180.0, float(bbox["west"]) - pad),
        "east": min(180.0, float(bbox["east"]) + pad),
    }


def _latest_completed_local_day(today=None):
    if today is None:
        today = datetime.now().date()
    return today - timedelta(days=1)


def _gfs_day_dir(target_day):
    return f"/gfs.{target_day.strftime('%Y%m%d')}/{GFS_DAY_RUN_CYCLE}/atmos"


def _gfs_day_file_name():
    return f"gfs.t{GFS_DAY_RUN_CYCLE}z.pgrb2.0p25.f{GFS_DAY_FORECAST_HOUR:03d}"


def _gfs_public_idx_url(target_day):
    return (
        f"{GFS_NOMADS_PUB_ROOT}/gfs.{target_day.strftime('%Y%m%d')}/"
        f"{GFS_DAY_RUN_CYCLE}/atmos/{_gfs_day_file_name()}.idx"
    )


def _gfs_filter_url(target_day, bbox, level_param, variable_param):
    params = {
        "file": _gfs_day_file_name(),
        level_param: "on",
        variable_param: "on",
        "subregion": "",
        "leftlon": f"{bbox['west']:.6f}",
        "rightlon": f"{bbox['east']:.6f}",
        "toplat": f"{bbox['north']:.6f}",
        "bottomlat": f"{bbox['south']:.6f}",
        "dir": _gfs_day_dir(target_day),
    }
    return f"{GFS_NOMADS_FILTER_URL}?{urlencode(params)}"


def _download_binary(url, out_path, timeout=180, session=None):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp_path = out_path + ".part"
    client = session or requests

    try:
        with client.get(url, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            with open(tmp_path, "wb") as dst:
                for chunk in response.iter_content(1024 * 64):
                    if chunk:
                        dst.write(chunk)
        if os.path.getsize(tmp_path) == 0:
            raise RuntimeError(f"Downloaded empty file from: {url}")
        os.replace(tmp_path, out_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    return out_path


def _ensure_grib_payload(path, label):
    with open(path, "rb") as src:
        prefix = src.read(4)
        if prefix == b"GRIB":
            return
        src.seek(0)
        snippet = src.read(200).decode("utf-8", errors="ignore")
    raise RuntimeError(
        f"{label} 下载失败，数据源返回的不是有效的 GRIB 文件。"
        f"{(' 服务器返回片段: ' + snippet.strip()) if snippet.strip() else ''}"
    )


def _locate_grib_band(path, element_prefix=None, comment_contains=None, short_name_contains=None):
    with rasterio.open(path) as src:
        matches = []
        for band_index in range(1, src.count + 1):
            tags = src.tags(band_index)
            element = tags.get("GRIB_ELEMENT", "")
            comment = tags.get("GRIB_COMMENT", "")
            short_name = tags.get("GRIB_SHORT_NAME", "")

            if element_prefix and not element.startswith(element_prefix):
                continue
            if comment_contains and comment_contains.lower() not in comment.lower():
                continue
            if short_name_contains and short_name_contains.lower() not in short_name.lower():
                continue
            matches.append(band_index)

    if matches:
        return matches[0]
    raise RuntimeError(f"Could not find expected GRIB band in {path}")


def _grib_band_to_tif(src_grib, out_tif, band_index):
    os.makedirs(os.path.dirname(out_tif), exist_ok=True)

    with rasterio.open(src_grib) as src:
        arr = src.read(band_index).astype(np.float32)
        nodata = src.nodata
        if nodata is not None and np.isfinite(nodata):
            arr[arr == nodata] = np.nan

        profile = {
            "driver": "GTiff",
            "height": src.height,
            "width": src.width,
            "count": 1,
            "dtype": "float32",
            "crs": src.crs,
            "transform": src.transform,
            "nodata": np.nan,
            "compress": "LZW",
        }

    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(arr, 1)

    return out_tif


def download_gfs_daily_inputs(target_date=None, cfg=None, force=False, session=None, today=None):
    """Download near-real-time daily rain and soil moisture from NOAA GFS.

    Output file names stay compatible with the flood input resolver:
      - rain_mm_demgrid_YYYY-MM-DD.tif
      - soil_moist_demgrid_YYYY-MM-DD.tif
    """
    cfg = cfg or CFG
    target_day = _parse_target_date(target_date) or _latest_completed_local_day(today=today)
    latest_complete_day = _latest_completed_local_day(today=today)

    if target_day > latest_complete_day:
        raise FileNotFoundError(
            f"近实时数据源当前只支持截至 {latest_complete_day.isoformat()} 的完整逐日数据，"
            f"{target_day.isoformat()} 还没有可用的完整日尺度结果。"
        )

    proc_dir = os.path.abspath(cfg.get("proc_dir", CFG["proc_dir"]))
    raw_dir = os.path.abspath(cfg.get("raw_dir", CFG["raw_dir"]))
    raw_gfs_dir = os.path.join(raw_dir, "daily_gfs")
    raw_tif_dir = os.path.join(raw_dir, "daily_gfs_tif")
    proc_daily_dir = os.path.join(proc_dir, "daily")

    os.makedirs(raw_gfs_dir, exist_ok=True)
    os.makedirs(raw_tif_dir, exist_ok=True)
    os.makedirs(proc_daily_dir, exist_ok=True)

    date_token = target_day.isoformat()
    rain_proc_path = os.path.join(proc_daily_dir, f"rain_mm_demgrid_{date_token}.tif")
    soil_proc_path = os.path.join(proc_daily_dir, f"soil_moist_demgrid_{date_token}.tif")

    if not force and os.path.exists(rain_proc_path) and os.path.exists(soil_proc_path):
        return {
            "target_date": date_token,
            "rain_path": rain_proc_path,
            "soil_path": soil_proc_path,
            "source": "gfs-nomads",
            "actions": [],
        }

    dem_tif = _resolve_dem_clip_path(cfg)
    bbox = _resolve_bbox(cfg)
    bbox = _expand_bbox(bbox, pad_deg=cfg.get("gfs_bbox_pad_deg", GFS_GRID_DEG))
    created_session = session is None
    client = session or requests.Session()

    try:
        idx_url = _gfs_public_idx_url(target_day)
        idx_response = client.get(idx_url, timeout=30)
        if not idx_response.ok:
            raise FileNotFoundError(
                f"近实时数据源暂未提供 {date_token} 的 GFS 日尺度文件，"
                "可能该日期已超出在线保留范围，或服务尚未同步完成。"
            )

        rain_grib_path = os.path.join(raw_gfs_dir, f"gfs_apcp24_{date_token}.grib2")
        soil_grib_path = os.path.join(raw_gfs_dir, f"gfs_soilw_0_0.1m_{date_token}.grib2")
        rain_raw_tif = os.path.join(raw_tif_dir, f"gfs_apcp24_mm_{date_token}.tif")
        soil_raw_tif = os.path.join(raw_tif_dir, f"gfs_soilw_0_0.1m_{date_token}.tif")

        rain_url = _gfs_filter_url(target_day, bbox, "lev_surface", "var_APCP")
        soil_url = _gfs_filter_url(target_day, bbox, "lev_0-0.1_m_below_ground", "var_SOILW")

        actions = []

        if force or not os.path.exists(rain_grib_path):
            _download_binary(rain_url, rain_grib_path, session=client)
            _ensure_grib_payload(rain_grib_path, f"{date_token} 降水")
            actions.append(f"已从 NOAA GFS 获取 {date_token} 的逐日降水数据。")

        if force or not os.path.exists(soil_grib_path):
            _download_binary(soil_url, soil_grib_path, session=client)
            _ensure_grib_payload(soil_grib_path, f"{date_token} 土壤湿度")
            actions.append(f"已从 NOAA GFS 获取 {date_token} 的表层土壤湿度数据。")

        if force or not os.path.exists(rain_raw_tif):
            rain_band = _locate_grib_band(
                rain_grib_path,
                element_prefix="APCP24",
                comment_contains="24 hr Total precipitation",
            )
            _grib_band_to_tif(rain_grib_path, rain_raw_tif, rain_band)

        if force or not os.path.exists(soil_raw_tif):
            soil_band = _locate_grib_band(
                soil_grib_path,
                element_prefix="SOILW",
                short_name_contains="0-0.1",
            )
            _grib_band_to_tif(soil_grib_path, soil_raw_tif, soil_band)

        if force or not os.path.exists(rain_proc_path):
            match_dem_grid(rain_raw_tif, dem_tif, rain_proc_path, resampling=Resampling.bilinear)
            actions.append(f"已生成 {date_token} 的 DEM 网格降水栅格。")

        if force or not os.path.exists(soil_proc_path):
            match_dem_grid(soil_raw_tif, dem_tif, soil_proc_path, resampling=Resampling.bilinear)
            actions.append(f"已生成 {date_token} 的 DEM 网格土壤湿度栅格。")
    finally:
        if created_session:
            client.close()

    return {
        "target_date": date_token,
        "rain_path": rain_proc_path,
        "soil_path": soil_proc_path,
        "source": "gfs-nomads",
        "actions": actions,
    }


def download_worldcover_to_demgrid(dem_tif, out_tif):
    """
    Download ESA WorldCover tiles from Planetary Computer and directly resample/reproject
    into the DEM grid (very low memory, no stackstac).

    Output:
      - GeoTIFF aligned to dem_tif (same CRS/transform/shape)
      - dtype uint8, nodata=0
    """
    import numpy as np
    import rasterio
    from rasterio.warp import reproject, Resampling
    from pystac_client import Client
    import planetary_computer as pc

    # bbox: [minx, miny, maxx, maxy] in EPSG:4326
    bbox = [CFG["west"], CFG["south"], CFG["east"], CFG["north"]]

    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

    # Prefer 2021, fallback all-time
    search = catalog.search(
        collections=["esa-worldcover"],
        bbox=bbox,
        datetime="2021-01-01/2021-12-31",
        max_items=500
    )
    items = list(search.get_items())
    if not items:
        print("[WorldCover] No 2021 items; falling back to all-time search...")
        search = catalog.search(
            collections=["esa-worldcover"],
            bbox=bbox,
            max_items=500
        )
        items = list(search.get_items())

    if not items:
        raise RuntimeError("No ESA WorldCover items found for bbox on Planetary Computer.")

    # Target grid = DEM grid
    with rasterio.open(dem_tif) as dem:
        dst_crs = dem.crs
        dst_transform = dem.transform
        dst_h, dst_w = dem.height, dem.width
        dst_profile = dem.profile.copy()
        dst_profile.update(
            driver="GTiff",
            count=1,
            dtype="uint8",
            nodata=0,
            compress="LZW"
        )

    # Destination landcover (0 = nodata)
    dst = np.zeros((dst_h, dst_w), dtype=np.uint8)

    # Reproject each tile onto DEM grid and mosaic (only fill where dst==0)
    print(f"[WorldCover] Found {len(items)} tile item(s). Reprojecting to DEM grid...")
    for it in items:
        it = pc.sign(it)
        if "map" not in it.assets:
            continue

        href = it.assets["map"].href

        try:
            with rasterio.open(href) as src:
                tmp = np.zeros((dst_h, dst_w), dtype=np.uint8)

                reproject(
                    source=rasterio.band(src, 1),
                    destination=tmp,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    src_nodata=src.nodata if src.nodata is not None else 0,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    dst_nodata=0,
                    resampling=Resampling.nearest,  # categorical
                )

                # Mosaic: write new pixels where dst is nodata and tmp has data
                mask = (dst == 0) & (tmp != 0)
                if np.any(mask):
                    dst[mask] = tmp[mask]

        except Exception as e:
            print(f"[WorldCover] Skip tile {it.id} due to error: {e}")

    # Save
    with rasterio.open(out_tif, "w", **dst_profile) as out:
        out.write(dst, 1)

    print("[WorldCover] Saved DEM-grid landcover:", out_tif)

# 使用 osmnx 直接抓取 OpenStreetMap 里的河流线（Waterway）。离河流的距离通常是洪涝预测的重要特征。
import osmnx as ox

def download_osm_rivers(study_gdf, out_gpkg):
    """
    Faster OSM waterway download:
    - query by polygon (smaller than bbox)
    - restrict tags to river/stream
    """
    poly = study_gdf.to_crs(4326).unary_union
    tags = {"waterway": ["river", "stream"]}

    print("[OSM] Downloading waterways (river/stream) by polygon...")
    try:
        gdf = ox.features_from_polygon(poly, tags=tags)
    except AttributeError:
        # fallback if older osmnx
        gdf = ox.geometries_from_polygon(poly, tags=tags)

    if gdf.empty:
        raise RuntimeError("No OSM waterway features found in study area.")

    gdf = gdf[gdf.geometry.type.isin(["LineString", "MultiLineString"])].copy()
    gdf = gdf.to_crs(4326)

    gdf.to_file(out_gpkg, layer="waterways", driver="GPKG")
    print("[OSM] Saved:", out_gpkg)


def download_hydrorivers(study_gdf, out_gpkg):
    """
    Download HydroRIVERS dataset and clip to study area.
    """

    url = "https://data.hydrosheds.org/file/HydroRIVERS/HydroRIVERS_v10.gdb.zip"
    zip_path = os.path.join(CFG["raw_dir"], "hydrorivers.zip")
    extract_dir = os.path.join(CFG["raw_dir"], "hydrorivers")

    if not os.path.exists(zip_path):
        print("[HydroRIVERS] Downloading...")
        r = requests.get(url, stream=True)
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        print("[HydroRIVERS] Downloaded.")

    if not os.path.exists(extract_dir):
        print("[HydroRIVERS] Extracting...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_dir)

    print("[HydroRIVERS] Loading dataset...")

    # 找到 gdb 文件
    gdb_path = None
    for root, dirs, files in os.walk(extract_dir):
        for d in dirs:
            if d.endswith(".gdb"):
                gdb_path = os.path.join(root, d)

    if gdb_path is None:
        raise RuntimeError("HydroRIVERS gdb not found.")

    rivers = gpd.read_file(gdb_path, layer="HydroRIVERS_v10")

    # 投影统一
    rivers = rivers.to_crs(4326)

    # 裁剪到研究区
    rivers_clip = gpd.clip(rivers, study_gdf)

    rivers_clip.to_file(out_gpkg, driver="GPKG")

    print("[HydroRIVERS] Saved:", out_gpkg)

def raster_stats(path):
    import rasterio
    import numpy as np
    with rasterio.open(path) as src:
        a = src.read(1).astype("float32")
        nd = src.nodata
        if nd is not None and np.isfinite(nd):
            a[a == nd] = np.nan
        return {
            "path": path,
            "dtype": src.dtypes[0],
            "nodata": src.nodata,
            "min": float(np.nanmin(a)),
            "max": float(np.nanmax(a)),
            "nan_pct": float(np.isnan(a).mean() * 100),
        }

def main():
    study = gpd.read_file(CFG["study_area_shp"]).to_crs(4326)

    # Clip DEM to study area (optional but recommended)
    dem_clip = os.path.join(CFG["proc_dir"], "dem_clip.tif")
    if not os.path.exists(dem_clip):
        clip_to_study_area(CFG["dem_tif"], study, dem_clip)
        print("[DEM] Clipped:", dem_clip)

    # ERA5-Land
    era5_nc = os.path.join(CFG["raw_dir"], f"era5_land_{CFG['era5_year']}{CFG['era5_month']}.nc")
    if not os.path.exists(era5_nc):
        download_era5_land_nc(era5_nc)

    raw_daily_dir = os.path.join(CFG["raw_dir"], "daily")
    proc_daily_dir = os.path.join(CFG["proc_dir"], "daily")
    daily_raw_outputs = era5_nc_to_daily_geotiffs(era5_nc, raw_daily_dir)
    daily_proc_outputs = match_daily_era5_to_dem_grid(
        daily_raw_outputs["rain_raw_files"],
        daily_raw_outputs["soil_raw_files"],
        dem_clip,
        proc_daily_dir,
    )

    # WorldCover landuse
    wc_tif = os.path.join(CFG["proc_dir"], "landcover_demgrid.tif")
    if not os.path.exists(wc_tif):
        download_worldcover_to_demgrid(dem_clip, wc_tif)

    # OSM rivers
    # rivers_gpkg = os.path.join(CFG["raw_dir"], "osm_waterways.gpkg")
    # if not os.path.exists(rivers_gpkg):
    #     download_osm_rivers(study, rivers_gpkg)
    rivers_gpkg = os.path.join(CFG["raw_dir"], "hydrorivers.gpkg")
    if not os.path.exists(rivers_gpkg):
        download_hydrorivers(study, rivers_gpkg)

    print("\nAll data prepared in:", CFG["proc_dir"])
    if daily_raw_outputs["soil_raw_files"]:
        print(raster_stats(daily_raw_outputs["soil_raw_files"][0]))
    if daily_proc_outputs["soil_daily_files"]:
        print(raster_stats(daily_proc_outputs["soil_daily_files"][0]))


if __name__ == "__main__":
    main()
