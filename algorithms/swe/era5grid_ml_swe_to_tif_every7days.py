# era5grid_ml_swe_to_tif_every7days.py
# -*- coding: utf-8 -*-

from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr

import rasterio
from rasterio.transform import from_origin
from rasterio.features import geometry_mask
from rasterio.warp import reproject, Resampling

from snowai.swe.machine_learning_model import MachineLearningSWE


# =========================
# 你需要修改的参数
# =========================
ERA5_FILE = r"output\era5_land.nc"  # 你已下载的ERA5-Land文件（可能是伪nc的zip也行）
AOI_SHP  = r"study_area.shp"
OUT_DIR  = r"output\tif"
DEM_TIF = r"dem.tif"
OUT_DIR   = r"output\tif_era5grid"

# 每隔多少天输出一次
STEP_DAYS = 7

# 模型需要的 snow_class：这里先固定 alpine（你研究区高山很合理）
DEFAULT_SNOW_CLASS = "alpine"

# 输出 nodata
NODATA = -9999.0

# 变量名（若不同可改；代码也会自动兜底）
TEMP_VAR_CANDIDATES = ["t2m", "2m_temperature"]
SNOW_VAR_CANDIDATES = ["sd", "sde"]  # 你目前是 sde


def ensure_real_netcdf(path: Path) -> Path:
    """
    兼容：文件后缀是 .nc 但实际是 zip 的情况
    返回真正的 .nc 文件路径
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if zipfile.is_zipfile(path):
        extract_dir = path.parent / (path.stem + "_unzipped")
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, "r") as z:
            z.extractall(extract_dir)
        nc_files = sorted(extract_dir.rglob("*.nc"))
        if not nc_files:
            raise ValueError(f"解压后未找到 .nc：{extract_dir}")
        return nc_files[0]

    return path


def open_netcdf_safely(nc_path: Path) -> xr.Dataset:
    """
    自动选择 engine 打开
    """
    engines = ["netcdf4", "h5netcdf", "scipy"]
    last = None
    for eng in engines:
        try:
            return xr.open_dataset(nc_path, engine=eng)
        except Exception as e:
            last = e
    raise RuntimeError(f"无法打开 {nc_path}，已尝试 {engines}。最后错误：{last}")


def pick_var(ds: xr.Dataset, candidates: list[str], label: str) -> str:
    for v in candidates:
        if v in ds.data_vars:
            return v
    raise KeyError(f"找不到 {label} 变量。候选={candidates}，现有={list(ds.data_vars)}")


def build_aoi_mask(aoi: gpd.GeoDataFrame, lons: np.ndarray, lats: np.ndarray) -> tuple[np.ndarray, rasterio.Affine]:
    """
    基于 ERA5 网格构建 AOI mask
    返回：
      mask (H,W) True表示在AOI内
      transform (用于写tif)
    注意：GeoTIFF 行从北到南，所以需要 lat 降序
    """
    # ERA5 lats 通常是降序，也可能升序，这里统一用降序写栅格
    lons_sorted = np.array(lons)
    lats_sorted = np.array(lats)
    if lats_sorted[0] < lats_sorted[-1]:
        lats_sorted = lats_sorted[::-1]  # 转成降序

    # 计算像元大小（度）
    dx = float(np.median(np.diff(np.sort(lons_sorted))))
    dy = float(np.median(np.diff(np.sort(lats_sorted))))  # 这里diff会是负的，取绝对
    dy = abs(dy)

    west = float(np.min(lons_sorted))
    north = float(np.max(lats_sorted))
    transform = from_origin(west, north, dx, dy)

    H = len(lats_sorted)
    W = len(lons_sorted)

    # geometry_mask 返回 True 表示“被遮罩（在几何外）”
    geoms = [g.__geo_interface__ for g in aoi.to_crs(4326).geometry]
    outside = geometry_mask(
        geometries=geoms,
        out_shape=(H, W),
        transform=transform,
        invert=False,  # False => True表示外部
        all_touched=False
    )
    inside = ~outside
    return inside, transform


def resample_dem_to_era5_grid(dem_path: str, transform_era5, shape_hw: tuple[int, int]) -> np.ndarray:
    """
    将 DEM 重采样到 ERA5 网格（EPSG:4326 + ERA5 transform + H/W）
    使用最近邻（足够用于 elevation 输入）
    """
    H, W = shape_hw
    dst = np.full((H, W), np.nan, dtype=np.float32)

    with rasterio.open(dem_path) as src:
        src_data = src.read(1).astype(np.float32)
        src_nodata = src.nodata

        # DEM nodata -> nan
        if src_nodata is not None:
            src_data = np.where(src_data == src_nodata, np.nan, src_data)

        # reproject 到 ERA5 网格
        reproject(
            source=src_data,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform_era5,
            dst_crs="EPSG:4326",
            resampling=Resampling.nearest,
            src_nodata=np.nan,
            dst_nodata=np.nan
        )

    return dst


def main():
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 读 AOI
    aoi = gpd.read_file(AOI_SHP)

    # 2) 打开 ERA5 数据（兼容伪nc zip）
    real_nc = ensure_real_netcdf(Path(ERA5_FILE))
    print("[INFO] Using NetCDF:", real_nc)

    ds = open_netcdf_safely(real_nc)

    # 兼容 valid_time
    if "valid_time" in ds.dims or "valid_time" in ds.coords:
        ds = ds.rename({"valid_time": "time"})
    if "time" not in ds.coords:
        raise KeyError(f"数据里没有 time 维度/坐标，现有 coords={list(ds.coords)} dims={list(ds.dims)}")

    # 3) 选择变量
    T_VAR = pick_var(ds, TEMP_VAR_CANDIDATES, "temperature(t2m)")
    S_VAR = pick_var(ds, SNOW_VAR_CANDIDATES, "snow depth")

    print("[INFO] Temp var:", T_VAR, "units:", ds[T_VAR].attrs.get("units"))
    print("[INFO] Snow var:", S_VAR, "units:", ds[S_VAR].attrs.get("units"), "long_name:", ds[S_VAR].attrs.get("long_name"))

    # 4) 取经纬度网格
    lons = ds["longitude"].values
    lats = ds["latitude"].values

    # 统一 lat 为降序（用于写栅格）
    lat_desc = lats if lats[0] > lats[-1] else lats[::-1]

    # 5) 构建 AOI mask（H,W）+ transform
    inside_mask, transform = build_aoi_mask(aoi, lons, lat_desc)
    H, W = inside_mask.shape
    print("[INFO] ERA5 grid shape:", (H, W), "inside cells:", int(inside_mask.sum()))

    # 6) DEM 重采样到 ERA5 网格（Elevation_m）
    elev_grid = resample_dem_to_era5_grid(DEM_TIF, transform, (H, W))
    # AOI外设nan
    elev_grid = np.where(inside_mask, elev_grid, np.nan)

    # 7) 构造日尺度变量：TAVG/TMIN/TMAX（°C）和 SnowDepth(m)
    t2m = ds[T_VAR]
    # K -> °C（ERA5 通常是K）
    t2m_units = (t2m.attrs.get("units") or "").lower()
    if "k" in t2m_units and "c" not in t2m_units:
        t2m_c = t2m - 273.15
    else:
        t2m_c = t2m

    # 日聚合（按 UTC）
    tavg = t2m_c.resample(time="1D").mean()
    tmin = t2m_c.resample(time="1D").min()
    tmax = t2m_c.resample(time="1D").max()

    snow = ds[S_VAR].resample(time="1D").mean()

    # 注意：如果 lat 原本是升序，需要对齐到 lat_desc
    def to_lat_desc(arr_day: xr.DataArray) -> np.ndarray:
        arr = arr_day.values  # shape (lat, lon)
        if lats[0] < lats[-1]:
            arr = arr[::-1, :]  # 翻转到降序
        return arr.astype(np.float32)

    # 8) 每隔 7 天输出一次（基于日序列）
    all_days = pd.to_datetime(tavg["time"].values).date
    export_days = all_days[::STEP_DAYS]
    print("[INFO] Total days:", len(all_days), "Export every", STEP_DAYS, "days =>", len(export_days))

    ml = MachineLearningSWE(return_type="numpy")

    for day in export_days:
        # 选这一天
        day_ts = np.datetime64(day)

        tavg_day = to_lat_desc(tavg.sel(time=day_ts))
        tmin_day = to_lat_desc(tmin.sel(time=day_ts))
        tmax_day = to_lat_desc(tmax.sel(time=day_ts))
        snow_day = to_lat_desc(snow.sel(time=day_ts))

        # AOI外设 nan（后续过滤）
        tavg_day = np.where(inside_mask, tavg_day, np.nan)
        tmin_day = np.where(inside_mask, tmin_day, np.nan)
        tmax_day = np.where(inside_mask, tmax_day, np.nan)
        snow_day = np.where(inside_mask, snow_day, np.nan)

        # 9) 组装 ML 输入 DataFrame（只包含 AOI 内有效像元）
        # 展平
        idx = np.where(
            (~np.isnan(elev_grid)) &
            (~np.isnan(snow_day)) &
            (~np.isnan(tavg_day)) &
            (~np.isnan(tmin_day)) &
            (~np.isnan(tmax_day)) &
            (inside_mask)
        )
        if len(idx[0]) == 0:
            print("[WARN] No valid cells for day:", day)
            continue

        # 对应经纬度（注意 lat_desc 是降序）
        rows = idx[0]
        cols = idx[1]
        lon_vals = lons[cols]
        lat_vals = lat_desc[rows]

        df_in = pd.DataFrame({
            "datetime": pd.to_datetime([str(day)] * len(rows), utc=True),
            "Snow_Class": [DEFAULT_SNOW_CLASS] * len(rows),
            "Elevation_m": elev_grid[rows, cols].astype(float),
            "Snow_Depth_m": snow_day[rows, cols].astype(float),
            "TAVG_degC": tavg_day[rows, cols].astype(float),
            "TMIN_degC": tmin_day[rows, cols].astype(float),
            "TMAX_degC": tmax_day[rows, cols].astype(float),
        })

        # 10) 预测 SWE（cm）
        swe_cm = ml.predict(
            data=df_in,
            snow_class="Snow_Class",
            elevation="Elevation_m",
            snow_depth="Snow_Depth_m",
            tavg="TAVG_degC",
            tmin="TMIN_degC",
            tmax="TMAX_degC",
            DOY="datetime",
        ).astype(np.float32)

        # 11) 回填到栅格（AOI外/无效设 nodata）
        out_grid = np.full((H, W), NODATA, dtype=np.float32)
        out_grid[rows, cols] = swe_cm

        # 12) 写 GeoTIFF（EPSG:4326，分辨率=ERA5网格）
        out_tif = out_dir / f"Auto_SWE_ML_cm_{day}.tif"
        meta = {
            "driver": "GTiff",
            "height": H,
            "width": W,
            "count": 1,
            "dtype": "float32",
            "crs": "EPSG:4326",
            "transform": transform,
            "nodata": NODATA,
            "compress": "deflate",
        }

        with rasterio.open(out_tif, "w", **meta) as dst:
            dst.write(out_grid, 1)

        print("[INFO] Saved:", out_tif)

    print("[INFO] Done.")


if __name__ == "__main__":
    main()