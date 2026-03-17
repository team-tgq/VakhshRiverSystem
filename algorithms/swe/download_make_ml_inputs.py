# download_make_ml_inputs.py
# -*- coding: utf-8 -*-

import os
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

import cdsapi
import xarray as xr
import rasterio
from pyproj import Transformer
import zipfile
import shutil

# =========================
# 你需要修改的参数
# =========================
AOI_SHP = r"study_area.shp"
DEM_TIF = r"dem.tif"

OUT_DIR = r"output"
OUT_CSV = "ml_inputs.csv"
ERA5_NC = "era5_land.nc"

START_DATE = "2024-02-01"
END_DATE = "2024-02-29"

# 网格点采样：0.05° 约 5km（可调小/调大）
GRID_STEP_DEG = 0.03

# snow_class：先固定一个值，保证 ML 可跑
DEFAULT_SNOW_CLASS = "alpine"

# ERA5-Land 变量名（CDS 通常用这俩）
ERA5_T2M_NAME = "t2m"
ERA5_SD_NAME = "sde"



def ensure_real_netcdf(nc_path: Path) -> Path:
    """
    有些下载结果是 zip，但后缀被保存成 .nc。
    这里自动判断：如果是 zip，则解压出真正的 .nc 并返回其路径。
    """
    nc_path = Path(nc_path)
    if not nc_path.exists():
        raise FileNotFoundError(nc_path)

    # 判断是否 zip（不依赖扩展名）
    if zipfile.is_zipfile(nc_path):
        extract_dir = nc_path.parent / (nc_path.stem + "_unzipped")
        extract_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(nc_path, "r") as z:
            z.extractall(extract_dir)

        # 在解压目录里找第一个 .nc 文件
        nc_files = sorted(extract_dir.rglob("*.nc"))
        if not nc_files:
            raise ValueError(f"解压后未找到 .nc 文件，解压目录：{extract_dir}")
        return nc_files[0]

    # 不是 zip，就原样返回
    return nc_path


def open_netcdf_safely(nc_path: Path) -> xr.Dataset:
    """
    自动选择 xarray 可用的 engine 打开 NetCDF。
    """
    nc_path = Path(nc_path)

    # 优先 netcdf4 / h5netcdf（更通用）
    preferred_engines = ["netcdf4", "h5netcdf", "scipy"]
    last_err = None

    for eng in preferred_engines:
        try:
            return xr.open_dataset(nc_path, engine=eng)
        except Exception as e:
            last_err = e

    raise RuntimeError(
        f"无法打开 NetCDF：{nc_path}\n"
        f"已尝试 engines={preferred_engines}\n"
        f"最后错误：{last_err}"
    )

def daterange_days(start: str, end: str):
    s = pd.to_datetime(start).date()
    e = pd.to_datetime(end).date()
    days = pd.date_range(s, e, freq="D")
    return days


def download_era5_land_bbox(north, west, south, east, start_date, end_date, out_nc: Path):
    """用 CDS API 下载 ERA5-Land：2m_temperature + snow_depth（小时数据）"""
    days = daterange_days(start_date, end_date)
    c = cdsapi.Client()

    req = {
        "variable": ["snow_depth", "snow_depth_water_equivalent", "2m_temperature"],
        "year": sorted({d.strftime("%Y") for d in days}),
        "month": sorted({d.strftime("%m") for d in days}),
        "day": sorted({d.strftime("%d") for d in days}),
        "time": ["00:00", "03:00", "06:00", "09:00", "12:00", "15:00", "18:00", "21:00"],
        "area": [float(north), float(west), float(south), float(east)],  # N, W, S, E
        "format": "netcdf",
    }

    print("[INFO] Downloading ERA5-Land to:", out_nc)
    c.retrieve("reanalysis-era5-land", req, str(out_nc))
    print("[INFO] Download complete.")


def build_points_in_aoi(aoi_gdf: gpd.GeoDataFrame, step_deg: float) -> pd.DataFrame:
    """在 AOI bbox 内生成规则经纬网格点，并筛选落入 AOI 多边形内的点"""
    aoi_4326 = aoi_gdf.to_crs(4326)
    minx, miny, maxx, maxy = aoi_4326.total_bounds

    lons = np.arange(minx, maxx + step_deg, step_deg)
    lats = np.arange(miny, maxy + step_deg, step_deg)

    pts = []
    geom_union = aoi_4326.unary_union

    for lat in lats:
        for lon in lons:
            p = Point(float(lon), float(lat))
            if geom_union.contains(p):
                pts.append((lon, lat))

    df = pd.DataFrame(pts, columns=["Longitude", "Latitude"])
    print(f"[INFO] Points inside AOI: {len(df)}")
    return df


def sample_dem_elevation(dem_path: str, pts_df: pd.DataFrame) -> np.ndarray:
    """对每个点抽样 DEM 高程（m）"""
    with rasterio.open(dem_path) as dem:
        to_dem = Transformer.from_crs("EPSG:4326", dem.crs, always_xy=True)
        xs, ys = to_dem.transform(pts_df["Longitude"].values, pts_df["Latitude"].values)

        elev = []
        for x, y in zip(xs, ys):
            val = list(dem.sample([(x, y)]))[0][0]
            if dem.nodata is not None and val == dem.nodata:
                elev.append(np.nan)
            else:
                elev.append(float(val))
        return np.array(elev, dtype=float)


def main():
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_nc = out_dir / ERA5_NC
    out_csv = out_dir / OUT_CSV

    # 1) 读 AOI，取 bbox（EPSG:4326）
    aoi = gpd.read_file(AOI_SHP).to_crs(4326)
    minx, miny, maxx, maxy = aoi.total_bounds
    north, west, south, east = maxy, minx, miny, maxx

    # 2) 下载 ERA5-Land（若已存在可跳过）
    if not out_nc.exists():
        download_era5_land_bbox(north, west, south, east, START_DATE, END_DATE, out_nc)
    else:
        print("[INFO] ERA5 netcdf exists, skip download:", out_nc)

    # 3) 生成 AOI 内采样点 + DEM 高程
    pts_df = build_points_in_aoi(aoi, GRID_STEP_DEG)
    pts_df["Elevation_m"] = sample_dem_elevation(DEM_TIF, pts_df)

    # 4) 读取 ERA5，构造日尺度 TAVG/TMIN/TMAX（°C）与 Snow_Depth_m（m）
    real_nc = ensure_real_netcdf(out_nc)
    print("[INFO] Using NetCDF file:", real_nc)

    ds = open_netcdf_safely(real_nc)
    print(list(ds.data_vars))
    # 修复时间维度名：valid_time -> time
    if "valid_time" in ds.dims or "valid_time" in ds.coords:
        ds = ds.rename({"valid_time": "time"})

    # -----------------------
    # 选择变量（雪深 sde + 雪水当量 sd）
    # -----------------------
    T_VAR = "t2m"
    if T_VAR not in ds.data_vars:
        raise KeyError(f"Cannot find temperature var '{T_VAR}'. Vars: {list(ds.data_vars)}")

    # 1) 雪深：优先用 sde
    if "sde" in ds.data_vars:
        DEPTH_VAR = "sde"
    else:
        raise KeyError(f"Cannot find snow depth var 'sde'. Vars: {list(ds.data_vars)}")

    # 2) 雪水当量：优先用 sd；如果没有，用属性兜底查找
    SWE_VAR = None
    if "sd" in ds.data_vars:
        SWE_VAR = "sd"
    else:
        # 兜底：按 standard_name/long_name 寻找 SWE 变量
        for v in ds.data_vars:
            std = str(ds[v].attrs.get("standard_name", "")).lower()
            ln = str(ds[v].attrs.get("long_name", "")).lower()
            if ("snow" in ln and "water" in ln and "equivalent" in ln) or ("water_equivalent" in std):
                SWE_VAR = v
                break
            # ECMWF SWE 常见标准名：lwe_thickness_of_surface_snow_amount
            if "lwe_thickness_of_surface_snow_amount" in std and "snow depth" not in ln:
                SWE_VAR = v
                break

    if SWE_VAR is None:
        # 如果你确实需要 sd，但这次下载没带上，就给出明确提示
        raise KeyError(
            f"Cannot find SWE var (sd). Vars: {list(ds.data_vars)}\n"
            f"请确认 CDS 下载请求里包含 'snow_depth_water_equivalent'。"
        )

    print("[INFO] Depth var:", DEPTH_VAR, "units:", ds[DEPTH_VAR].attrs.get("units"), "long_name:",
          ds[DEPTH_VAR].attrs.get("long_name"))
    print("[INFO] SWE   var:", SWE_VAR, "units:", ds[SWE_VAR].attrs.get("units"), "long_name:",
          ds[SWE_VAR].attrs.get("long_name"))

    t2m = ds[T_VAR]
    snow_depth = ds[DEPTH_VAR]  # 雪深（m）
    swe = ds[SWE_VAR]  # 雪水当量（m w.e.）

    # K -> °C
    t2m_c = t2m - 273.15

    # 日聚合
    tavg_c = t2m_c.resample(time="1D").mean()
    tmin_c = t2m_c.resample(time="1D").min()
    tmax_c = t2m_c.resample(time="1D").max()

    snow_depth_m = snow_depth.resample(time="1D").mean()  # 雪深 m
    swe_m = swe.resample(time="1D").mean()  # SWE m water equivalent

    # 5) 对每个点抽取每日序列（nearest）
    rows = []
    for lon, lat, elev in pts_df[["Longitude", "Latitude", "Elevation_m"]].itertuples(index=False):
        s_tavg = tavg_c.sel(longitude=lon, latitude=lat, method="nearest").to_series()
        s_tmin = tmin_c.sel(longitude=lon, latitude=lat, method="nearest").to_series()
        s_tmax = tmax_c.sel(longitude=lon, latitude=lat, method="nearest").to_series()

        s_depth = snow_depth_m.sel(longitude=lon, latitude=lat, method="nearest").to_series()
        s_swe = swe_m.sel(longitude=lon, latitude=lat, method="nearest").to_series()

        tmp = pd.DataFrame(
            {
                "datetime": s_tavg.index,
                "Longitude": lon,
                "Latitude": lat,
                "Elevation_m": elev,

                # 雪深（m）
                "Snow_Depth_m": s_depth.values,

                # ERA5 雪水当量（m / cm）
                "SWE_ERA5_m": s_swe.values,
                "SWE_ERA5_cm": np.asarray(s_swe.values, dtype=float) * 100.0,

                # 温度（°C）
                "TAVG_degC": s_tavg.values,
                "TMIN_degC": s_tmin.values,
                "TMAX_degC": s_tmax.values,

                "Snow_Class": DEFAULT_SNOW_CLASS,
            }
        )
        rows.append(tmp)

    ml_inputs = pd.concat(rows, ignore_index=True)

    # 6) 类型清理
    ml_inputs["datetime"] = pd.to_datetime(ml_inputs["datetime"], utc=True, errors="coerce")
    for c in ["Elevation_m", "Snow_Depth_m", "TAVG_degC", "TMIN_degC", "TMAX_degC", "Longitude", "Latitude"]:
        ml_inputs[c] = pd.to_numeric(ml_inputs[c], errors="coerce")
    ml_inputs["Snow_Class"] = ml_inputs["Snow_Class"].astype(str)

    print("[CHECK] depth max (m):", float(ds[DEPTH_VAR].isel(time=0).max()))
    print("[CHECK] swe   max (m):", float(ds[SWE_VAR].isel(time=0).max()))

    # 7) 保存 CSV
    ml_inputs.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("[INFO] Saved:", out_csv)
    print(ml_inputs.head())


if __name__ == "__main__":
    main()