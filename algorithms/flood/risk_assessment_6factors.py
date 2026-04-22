# algorithms/flood/risk_assessment_6factors.py
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt

import folium
from folium.raster_layers import ImageOverlay
from branca.colormap import linear

try:
    from .input_resolver import resolve_flood_input_paths
except ImportError:
    from input_resolver import resolve_flood_input_paths  # type: ignore


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CFG = {
    "study_area_shp": os.path.join(BASE_DIR, "study_area.shp"),
    "proc_dir": os.path.join(BASE_DIR, "data", "processed"),
    "raw_dir": os.path.join(BASE_DIR, "data", "raw"),
    "out_dir": os.path.join(BASE_DIR, "outputs"),
    "out_risk_tif": os.path.join(BASE_DIR, "outputs", "risk_6factors.tif"),
    "out_map": os.path.join(BASE_DIR, "outputs", "flood_risk_map.html"),
    "weights": {
        "rain": 0.22,
        "soil_moist": 0.18,
        "elev_low": 0.18,
        "slope_low": 0.15,
        "land_imperv": 0.15,
        "river_near": 0.12,
    }
}
os.makedirs(CFG["out_dir"], exist_ok=True)


def read_raster(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        profile = src.profile
        transform = src.transform
        crs = src.crs
        nodata = src.nodata
    if nodata is not None:
        arr[arr == nodata] = np.nan
    return arr, profile, transform, crs


def write_raster(path, arr, profile):
    prof = profile.copy()
    prof.update(dtype="float32", count=1, nodata=np.nan)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr.astype(np.float32), 1)


def minmax_norm(arr):
    a = np.nanmin(arr)
    b = np.nanmax(arr)
    if not np.isfinite(a) or not np.isfinite(b) or (b - a) == 0:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - a) / (b - a)


def slope_from_dem(dem, transform):
    dx = transform.a
    dy = -transform.e
    dzdx = np.gradient(dem, axis=1) / dx
    dzdy = np.gradient(dem, axis=0) / dy
    slope_rad = np.arctan(np.sqrt(dzdx**2 + dzdy**2))
    return np.degrees(slope_rad).astype(np.float32)


def rasterize_rivers_to_mask(rivers_gdf, out_shape, transform):
    shapes = [(geom, 1) for geom in rivers_gdf.geometry if geom is not None]
    mask = rasterize(
        shapes=shapes,
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype=np.uint8
    )
    return mask


def distance_to_river_m(river_mask, transform):
    inv = (river_mask == 0).astype(np.uint8)
    dist_px = distance_transform_edt(inv)
    px = abs(transform.a)
    py = abs(transform.e)
    dist = dist_px * float(np.mean([px, py]))
    return dist.astype(np.float32)


def landcover_to_impervious_factor(lc):
    factor = np.full_like(lc, np.nan, dtype=np.float32)

    mapping = {
        10: 0.25,
        20: 0.35,
        30: 0.45,
        40: 0.60,
        50: 1.00,
        60: 0.70,
        70: 0.30,
        80: 0.10,
        90: 0.55,
        95: 0.40,
        100: 0.35
    }

    for k, v in mapping.items():
        factor[lc == k] = v

    factor[np.isnan(factor) & np.isfinite(lc)] = 0.5
    return factor


def build_folium_map(risk, dem_path, study_area_shp, out_map):
    risk_disp = np.clip(risk, 0, 1)

    cm = linear.YlOrRd_09.scale(0, 1)
    rgba = np.zeros((risk_disp.shape[0], risk_disp.shape[1], 4), dtype=np.uint8)

    flat = risk_disp.flatten()
    rgba_2d = rgba.reshape(-1, 4)

    for i, v in enumerate(flat):
        if np.isnan(v):
            rgba_2d[i] = [0, 0, 0, 0]
        else:
            r, g, b, a = cm.rgba_bytes_tuple(float(v))
            rgba_2d[i] = [r, g, b, 180]

    with rasterio.open(dem_path) as src:
        bounds = src.bounds
        dem_crs = src.crs

    if dem_crs.to_string() != "EPSG:4326":
        from pyproj import Transformer
        transformer = Transformer.from_crs(dem_crs, "EPSG:4326", always_xy=True)
        wlon, slat = transformer.transform(bounds.left, bounds.bottom)
        elon, nlat = transformer.transform(bounds.right, bounds.top)
    else:
        wlon, slat, elon, nlat = bounds.left, bounds.bottom, bounds.right, bounds.top

    center = [(slat + nlat) / 2.0, (wlon + elon) / 2.0]
    m = folium.Map(location=center, zoom_start=8, tiles="cartodbpositron")

    ImageOverlay(
        image=rgba,
        bounds=[[slat, wlon], [nlat, elon]],
        opacity=0.75,
        interactive=True,
        cross_origin=False,
        zindex=1
    ).add_to(m)

    # 加载研究区边界
    aoi = gpd.read_file(study_area_shp)
    if aoi.crs is not None and aoi.crs.to_string() != "EPSG:4326":
        aoi = aoi.to_crs("EPSG:4326")

    folium.GeoJson(
        aoi.to_json(),
        name="Study Area",
        style_function=lambda x: {
            "color": "#0066ff",
            "weight": 3,
            "fillOpacity": 0.0
        },
        tooltip="Study Area"
    ).add_to(m)

    cm.caption = "Flood Risk (6-factor composite, 0~1)"
    cm.add_to(m)

    folium.LayerControl().add_to(m)
    m.save(out_map)


def run_risk_assessment(target_date=None, auto_prepare_static=True, allow_legacy_dynamic=True):
    input_paths = resolve_flood_input_paths(
        CFG,
        target_date=target_date,
        auto_prepare_static=auto_prepare_static,
        allow_legacy_dynamic=allow_legacy_dynamic,
    )

    dem_path = input_paths["dem_path"]
    rain_path = input_paths["rain_path"]
    soil_path = input_paths["soil_path"]
    lc_path = input_paths["landcover_path"]
    rivers_gpkg = input_paths["rivers_path"]

    dem, profile, transform, crs = read_raster(dem_path)
    rain, _, _, _ = read_raster(rain_path)
    soil, _, _, _ = read_raster(soil_path)
    lc, _, _, _ = read_raster(lc_path)

    elev_norm = minmax_norm(dem)
    elev_low = 1.0 - elev_norm

    slope = slope_from_dem(dem, transform)
    slope_norm = minmax_norm(slope)
    slope_low = 1.0 - slope_norm

    rain_norm = minmax_norm(rain)
    soil_norm = minmax_norm(soil)
    land_imperv = landcover_to_impervious_factor(lc)

    rivers = gpd.read_file(rivers_gpkg).to_crs(crs)

    river_mask = rasterize_rivers_to_mask(rivers, dem.shape, transform)
    dist_river = distance_to_river_m(river_mask, transform)
    dist_norm = minmax_norm(dist_river)
    river_near = 1.0 - dist_norm

    w = CFG["weights"]
    risk = (
        rain_norm * w["rain"] +
        soil_norm * w["soil_moist"] +
        elev_low * w["elev_low"] +
        slope_low * w["slope_low"] +
        land_imperv * w["land_imperv"] +
        river_near * w["river_near"]
    )

    risk[np.isnan(dem)] = np.nan

    write_raster(CFG["out_risk_tif"], risk, profile)
    print("[Saved]", CFG["out_risk_tif"])

    build_folium_map(
        risk=risk,
        dem_path=dem_path,
        study_area_shp=input_paths["study_area_shp"],
        out_map=CFG["out_map"]
    )
    print("[Saved]", CFG["out_map"])

    return {
        "risk_tif": CFG["out_risk_tif"],
        "map_html": CFG["out_map"],
        "study_area_shp": input_paths["study_area_shp"],
        "dem_path": dem_path,
        "landcover_path": lc_path,
        "rivers_path": rivers_gpkg,
        "rain_path": rain_path,
        "soil_path": soil_path,
        "requested_target_date": input_paths["requested_target_date"],
        "resolved_target_date": input_paths["resolved_target_date"],
        "dynamic_scale": input_paths["dynamic_scale"],
        "available_dynamic_dates": input_paths["available_dynamic_dates"],
        "static_actions": input_paths.get("static_actions", []),
    }


def main():
    return run_risk_assessment()


if __name__ == "__main__":
    main()
