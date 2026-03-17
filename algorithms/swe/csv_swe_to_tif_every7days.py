# csv_swe_to_tif_every7days.py
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from rasterio.mask import mask
from shapely.geometry import mapping
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# =========================
# 你需要修改的参数
# =========================
CSV_PATH= os.path.join(BASE_DIR, "output", "swe_ml_distribution.csv")
AOI_SHP= os.path.join(BASE_DIR, "study_area.shp")
OUT_DIR= os.path.join(BASE_DIR, "output", "tif")

# CSV中列名（按你的实际）
DATE_COL = "datetime"
LON_COL  = "Longitude"
LAT_COL  = "Latitude"
VALUE_COL = "SWE_ML_cm"   # 预测SWE列

# nodata
NODATA = -9999.0

# 每隔多少天输出一次
STEP_DAYS = 7

# 输出文件名前缀
OUT_PREFIX = VALUE_COL


def build_grid_from_points(df_day: pd.DataFrame, lon_col: str, lat_col: str, val_col: str):
    """
    将某一天的规则网格点数据重建成2D栅格。
    要求：点是规则经纬网格采样（你之前按 step_deg 生成点的方式满足）。
    返回：grid(2D), transform, crs
    """
    lons = np.sort(df_day[lon_col].unique())
    lats = np.sort(df_day[lat_col].unique())  # 升序

    if len(lons) < 2 or len(lats) < 2:
        raise ValueError("经纬度唯一值不足，无法构建栅格（可能点太少或不是规则网格）。")

    # 分辨率（假设等间距）
    dx = float(np.median(np.diff(lons)))
    dy = float(np.median(np.diff(lats)))

    width = len(lons)
    height = len(lats)

    # GeoTIFF transform 需要左上角（west, north）
    west = float(lons.min())
    north = float(lats.max())
    transform = from_origin(west, north, dx, dy)  # dx, dy 为像元大小（度）

    # 行方向：从北到南（lat 降序）
    lat_desc = lats[::-1]
    lon_to_col = {lon: i for i, lon in enumerate(lons)}
    lat_to_row = {lat: i for i, lat in enumerate(lat_desc)}

    grid = np.full((height, width), np.nan, dtype=np.float32)

    for lon, lat, val in df_day[[lon_col, lat_col, val_col]].itertuples(index=False):
        if pd.isna(val):
            continue
        r = lat_to_row.get(float(lat))
        c = lon_to_col.get(float(lon))
        if r is not None and c is not None:
            grid[r, c] = float(val)

    grid = np.where(np.isnan(grid), NODATA, grid).astype(np.float32)
    return grid, transform, "EPSG:4326"


def write_tif_then_clip(grid, transform, crs, aoi_gdf, out_tif: Path):
    """
    先写未裁剪临时 tif，再用 AOI mask 裁剪输出最终 tif
    """
    out_tif.parent.mkdir(parents=True, exist_ok=True)
    tmp_tif = out_tif.with_suffix(".tmp.tif")

    height, width = grid.shape
    meta = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": grid.dtype,
        "crs": crs,
        "transform": transform,
        "nodata": NODATA,
        "compress": "deflate",
    }

    # 1) 写临时 tif
    with rasterio.open(tmp_tif, "w", **meta) as dst:
        dst.write(grid, 1)

    # 2) 裁剪到 AOI
    aoi_4326 = aoi_gdf.to_crs(4326)
    geoms = [mapping(geom) for geom in aoi_4326.geometry]

    with rasterio.open(tmp_tif) as src:
        clipped, clipped_transform = mask(src, geoms, crop=True, nodata=NODATA)

        clipped_meta = src.meta.copy()
        clipped_meta.update({
            "height": clipped.shape[1],
            "width": clipped.shape[2],
            "transform": clipped_transform,
        })

    with rasterio.open(out_tif, "w", **clipped_meta) as dst:
        dst.write(clipped)

    # 删除临时文件
    try:
        tmp_tif.unlink()
    except Exception:
        pass


def main():
    csv_path = Path(CSV_PATH)
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 读 AOI
    aoi = gpd.read_file(AOI_SHP)

    # 读 CSV
    df = pd.read_csv(csv_path)

    # 检查列
    for c in [DATE_COL, LON_COL, LAT_COL, VALUE_COL]:
        if c not in df.columns:
            raise KeyError(f"CSV 缺少列 {c}，当前列：{list(df.columns)}")

    # datetime 解析
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], utc=True, errors="coerce")
    df = df.dropna(subset=[DATE_COL, LON_COL, LAT_COL, VALUE_COL])

    # 排序并拿日期列表
    df = df.sort_values(DATE_COL)
    all_dates = sorted(df[DATE_COL].dt.date.unique())

    if not all_dates:
        raise ValueError("CSV 中没有有效日期数据。")

    # 每隔 7 天取一次
    export_dates = all_dates[::STEP_DAYS]
    print(f"[INFO] Total dates: {len(all_dates)}")
    print(f"[INFO] Export every {STEP_DAYS} days, dates count: {len(export_dates)}")
    print("[INFO] Export dates:", export_dates)

    # 逐日生成 tif
    for d in export_dates:
        df_day = df[df[DATE_COL].dt.date == d].copy()
        if df_day.empty:
            print("[WARN] No data for date:", d)
            continue

        grid, transform, crs = build_grid_from_points(df_day, LON_COL, LAT_COL, VALUE_COL)

        out_tif = out_dir / f"{OUT_PREFIX}_{d}.tif"
        write_tif_then_clip(grid, transform, crs, aoi, out_tif)
        print("[INFO] Saved:", out_tif)

    print("[INFO] Done.")


if __name__ == "__main__":
    main()