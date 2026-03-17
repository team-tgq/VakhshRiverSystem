# -*- coding: utf-8 -*-
import os
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from rasterio.mask import mask
from shapely.geometry import mapping

from algorithms.swe.snowai.swe.machine_learning_model import MachineLearningSWE


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CFG = {
    "in_csv": os.path.join(BASE_DIR, "output", "ml_inputs.csv"),
    "out_csv": os.path.join(BASE_DIR, "output", "swe_ml_distribution.csv"),
    "aoi_shp": os.path.join(BASE_DIR, "study_area.shp"),
    "out_tif_dir": os.path.join(BASE_DIR, "output", "tif"),

    "date_col": "datetime",
    "lon_col": "Longitude",
    "lat_col": "Latitude",
    "value_col": "SWE_ML_cm",

    "nodata": -9999.0,
    "step_days": 7,
    "out_prefix": "SWE_ML_cm"
}


def clean_for_ml(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "Snow_Class",
        "Elevation_m",
        "Snow_Depth_m",
        "TAVG_degC",
        "TMIN_degC",
        "TMAX_degC",
        "datetime",
    ]

    df = df.copy()

    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")

    df["Snow_Class"] = df.get("Snow_Class", "alpine")
    df["Snow_Class"] = df["Snow_Class"].fillna("alpine").astype(str)

    for c in ["Elevation_m", "Snow_Depth_m", "TAVG_degC", "TMIN_degC", "TMAX_degC"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=required).reset_index(drop=True)
    return df


def predict_swe_to_csv(in_csv: str, out_csv: str) -> str:
    df = pd.read_csv(in_csv)
    df = clean_for_ml(df)

    preds = df.assign(
        SWE_ML_cm=lambda x: MachineLearningSWE(return_type="pandas").predict(
            data=x,
            snow_class="Snow_Class",
            elevation="Elevation_m",
            snow_depth="Snow_Depth_m",
            tavg="TAVG_degC",
            tmin="TMIN_degC",
            tmax="TMAX_degC",
            DOY="datetime",
        )
    )

    if "SWE_ERA5_cm" in preds.columns:
        cols = [
            "datetime",
            "Longitude",
            "Latitude",
            "Elevation_m",
            "Snow_Depth_m",
            "SWE_ERA5_cm",
            "SWE_ML_cm",
            "TAVG_degC",
            "TMIN_degC",
            "TMAX_degC",
        ]
        cols = [c for c in cols if c in preds.columns]
        preds = preds[cols]

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return out_csv


def build_grid_from_points(df_day: pd.DataFrame, lon_col: str, lat_col: str, val_col: str):
    lons = np.sort(df_day[lon_col].unique())
    lats = np.sort(df_day[lat_col].unique())

    if len(lons) < 2 or len(lats) < 2:
        raise ValueError("经纬度唯一值不足，无法构建栅格（可能点太少或不是规则网格）。")

    dx = float(np.median(np.diff(lons)))
    dy = float(np.median(np.diff(lats)))

    width = len(lons)
    height = len(lats)

    west = float(lons.min())
    north = float(lats.max())
    transform = from_origin(west, north, dx, dy)

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

    grid = np.where(np.isnan(grid), CFG["nodata"], grid).astype(np.float32)
    return grid, transform, "EPSG:4326"


def write_tif_then_clip(grid, transform, crs, aoi_gdf, out_tif: Path):
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
        "nodata": CFG["nodata"],
        "compress": "deflate",
    }

    with rasterio.open(tmp_tif, "w", **meta) as dst:
        dst.write(grid, 1)

    aoi_4326 = aoi_gdf.to_crs(4326)
    geoms = [mapping(geom) for geom in aoi_4326.geometry]

    with rasterio.open(tmp_tif) as src:
        clipped, clipped_transform = mask(src, geoms, crop=True, nodata=CFG["nodata"])
        clipped_meta = src.meta.copy()
        clipped_meta.update({
            "height": clipped.shape[1],
            "width": clipped.shape[2],
            "transform": clipped_transform,
        })

    with rasterio.open(out_tif, "w", **clipped_meta) as dst:
        dst.write(clipped)

    try:
        tmp_tif.unlink()
    except Exception:
        pass


def csv_to_tifs(csv_path: str, aoi_shp: str, out_dir: str):
    csv_path = Path(csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    aoi = gpd.read_file(aoi_shp)
    df = pd.read_csv(csv_path)

    for c in [CFG["date_col"], CFG["lon_col"], CFG["lat_col"], CFG["value_col"]]:
        if c not in df.columns:
            raise KeyError(f"CSV 缺少列 {c}，当前列：{list(df.columns)}")

    df[CFG["date_col"]] = pd.to_datetime(df[CFG["date_col"]], utc=True, errors="coerce")
    df = df.dropna(subset=[CFG["date_col"], CFG["lon_col"], CFG["lat_col"], CFG["value_col"]])
    df = df.sort_values(CFG["date_col"])

    all_dates = sorted(df[CFG["date_col"]].dt.date.unique())
    if not all_dates:
        raise ValueError("CSV 中没有有效日期数据。")

    export_dates = all_dates[::CFG["step_days"]]

    tif_list = []

    for d in export_dates:
        df_day = df[df[CFG["date_col"]].dt.date == d].copy()
        if df_day.empty:
            continue

        grid, transform, crs = build_grid_from_points(
            df_day,
            CFG["lon_col"],
            CFG["lat_col"],
            CFG["value_col"]
        )

        out_tif = out_dir / f"{CFG['out_prefix']}_{d}.tif"
        write_tif_then_clip(grid, transform, crs, aoi, out_tif)
        tif_list.append(str(out_tif))

    return tif_list


def run_swe_assessment():
    out_csv = predict_swe_to_csv(CFG["in_csv"], CFG["out_csv"])
    tif_list = csv_to_tifs(out_csv, CFG["aoi_shp"], CFG["out_tif_dir"])

    result = {
        "csv": out_csv,
        "tif_list": tif_list,
        "latest_tif": tif_list[-1] if tif_list else None,
        "study_area_shp": CFG["aoi_shp"]
    }
    return result


def main():
    return run_swe_assessment()


if __name__ == "__main__":
    result = main()
    print("[INFO] CSV:", result["csv"])
    print("[INFO] TIF COUNT:", len(result["tif_list"]))
    print("[INFO] LATEST TIF:", result["latest_tif"])