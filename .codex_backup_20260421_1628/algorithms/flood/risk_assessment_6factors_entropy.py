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


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CFG = {
    "study_area_shp": os.path.join(BASE_DIR, "study_area.shp"),
    "proc_dir": os.path.join(BASE_DIR, "data", "processed"),
    "raw_dir": os.path.join(BASE_DIR, "data", "raw"),
    "out_dir": os.path.join(BASE_DIR, "outputs"),
    "out_risk_tif": os.path.join(BASE_DIR, "outputs", "risk_6factors.tif"),
    "out_map": os.path.join(BASE_DIR, "outputs", "flood_risk_map.html"),
    "out_weights_txt": os.path.join(BASE_DIR, "outputs", "final_weights.txt"),

    "subjective_weights": {
        "rain": 0.22,
        "soil_moist": 0.18,
        "elev_low": 0.18,
        "slope_low": 0.15,
        "land_imperv": 0.15,
        "river_near": 0.12,
    },

    "alpha_subjective": 0.8,
    "entropy_sample_size": 5000,
    "random_seed": 42,
    "slope_clip_max_deg": 10.0,
    "river_decay_distance_m": 1500.0,
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
    arr = arr.astype(np.float32)
    valid = np.isfinite(arr)
    if not np.any(valid):
        return np.full_like(arr, np.nan, dtype=np.float32)

    a = np.nanmin(arr)
    b = np.nanmax(arr)

    if not np.isfinite(a) or not np.isfinite(b) or (b - a) == 0:
        out = np.zeros_like(arr, dtype=np.float32)
        out[~valid] = np.nan
        return out

    out = (arr - a) / (b - a)
    out[~valid] = np.nan
    return out.astype(np.float32)


def slope_from_dem(dem, transform):
    dx = abs(transform.a)
    dy = abs(transform.e)

    dem_safe = dem.copy()
    if np.isnan(dem_safe).any():
        dem_safe[np.isnan(dem_safe)] = np.nanmean(dem_safe)

    dzdx = np.gradient(dem_safe, axis=1) / dx
    dzdy = np.gradient(dem_safe, axis=0) / dy

    slope = np.degrees(np.arctan(np.sqrt(dzdx ** 2 + dzdy ** 2)))
    slope[np.isnan(dem)] = np.nan
    return slope.astype(np.float32)


def rasterize_rivers_to_mask(rivers_gdf, out_shape, transform):
    shapes = [(geom, 1) for geom in rivers_gdf.geometry if geom is not None]
    return rasterize(
        shapes=shapes,
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype=np.uint8
    )


def distance_to_river_m(river_mask, transform):
    inv = (river_mask == 0).astype(np.uint8)
    dist_px = distance_transform_edt(inv)
    px = abs(transform.a)
    py = abs(transform.e)
    return dist_px * float(np.mean([px, py]))


def landcover_to_impervious_factor(lc):
    factor = np.full_like(lc, np.nan, dtype=np.float32)

    mapping = {
        10: 0.25, 20: 0.35, 30: 0.45, 40: 0.60,
        50: 1.00, 60: 0.70, 70: 0.30, 80: 0.10,
        90: 0.55, 95: 0.40, 100: 0.35
    }

    for k, v in mapping.items():
        factor[lc == k] = v

    factor[np.isnan(factor) & np.isfinite(lc)] = 0.5
    return factor.astype(np.float32)


def entropy_weight_sampled(factors, sample_size=5000, random_seed=42, eps=1e-12):
    names = list(factors.keys())
    arrays = [factors[n].astype(np.float64) for n in names]

    valid_mask = np.ones_like(arrays[0], dtype=bool)
    for arr in arrays:
        valid_mask &= np.isfinite(arr)

    idx = np.where(valid_mask.ravel())[0]
    if len(idx) < 2:
        raise ValueError("有效像元过少，无法计算熵权。")

    rng = np.random.default_rng(random_seed)
    sample = rng.choice(idx, size=min(sample_size, len(idx)), replace=False)

    X = np.column_stack([arr.ravel()[sample] for arr in arrays])

    col_sums = X.sum(axis=0)
    zero_cols = col_sums <= eps
    if np.any(zero_cols):
        X[:, zero_cols] = eps
        col_sums = X.sum(axis=0)

    P = X / col_sums
    P = np.clip(P, eps, None)

    k = 1.0 / np.log(X.shape[0])
    E = -k * np.sum(P * np.log(P), axis=0)
    D = 1.0 - E

    if np.all(D <= eps):
        W = np.full(X.shape[1], 1.0 / X.shape[1], dtype=np.float64)
    else:
        W = D / np.sum(D)

    return {n: float(w) for n, w in zip(names, W)}, valid_mask, X.shape[0]


def combine_weights(ws, we, alpha):
    wf = {}
    for k in ws:
        wf[k] = alpha * ws[k] + (1.0 - alpha) * we[k]

    s = sum(wf.values())
    if s > 0:
        wf = {k: v / s for k, v in wf.items()}
    return wf


def save_weights_report(subjective_weights, entropy_weights, final_weights, sample_count, out_path, alpha_subjective):
    lines = []
    lines.append("Flood Risk Weights Report\n")
    lines.append("=" * 60 + "\n")
    lines.append(f"Entropy sample count: {sample_count}\n")
    lines.append(f"Combination alpha (subjective): {alpha_subjective:.3f}\n")
    lines.append(f"Combination beta (entropy): {1.0 - alpha_subjective:.3f}\n")
    lines.append("=" * 60 + "\n\n")

    lines.append("[Subjective Weights]\n")
    for k, v in subjective_weights.items():
        lines.append(f"{k}: {v:.6f}\n")
    lines.append(f"sum: {sum(subjective_weights.values()):.6f}\n\n")

    lines.append("[Entropy Weights]\n")
    for k, v in entropy_weights.items():
        lines.append(f"{k}: {v:.6f}\n")
    lines.append(f"sum: {sum(entropy_weights.values()):.6f}\n\n")

    lines.append("[Final Combined Weights]\n")
    for k, v in final_weights.items():
        lines.append(f"{k}: {v:.6f}\n")
    lines.append(f"sum: {sum(final_weights.values()):.6f}\n")

    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(lines)


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

    cm.caption = "Flood Risk (Combined weighting, 0~1)"
    cm.add_to(m)

    folium.LayerControl().add_to(m)
    m.save(out_map)


def run_risk_assessment(
    study_area_shp=None,
    proc_dir=None,
    raw_dir=None,
    out_dir=None,
    out_risk_tif=None,
    out_map=None,
    out_weights_txt=None,
):
    cfg = dict(CFG)
    if study_area_shp:
        cfg["study_area_shp"] = study_area_shp
    if proc_dir:
        cfg["proc_dir"] = proc_dir
    if raw_dir:
        cfg["raw_dir"] = raw_dir
    if out_dir:
        cfg["out_dir"] = out_dir
    if out_risk_tif:
        cfg["out_risk_tif"] = out_risk_tif
    elif out_dir:
        cfg["out_risk_tif"] = os.path.join(cfg["out_dir"], "risk_6factors.tif")
    if out_map:
        cfg["out_map"] = out_map
    elif out_dir:
        cfg["out_map"] = os.path.join(cfg["out_dir"], "flood_risk_map.html")
    if out_weights_txt:
        cfg["out_weights_txt"] = out_weights_txt
    elif out_dir:
        cfg["out_weights_txt"] = os.path.join(cfg["out_dir"], "final_weights.txt")

    os.makedirs(cfg["out_dir"], exist_ok=True)
    os.makedirs(os.path.dirname(cfg["out_risk_tif"]), exist_ok=True)
    os.makedirs(os.path.dirname(cfg["out_map"]), exist_ok=True)
    os.makedirs(os.path.dirname(cfg["out_weights_txt"]), exist_ok=True)

    dem_path = os.path.join(cfg["proc_dir"], "dem_clip.tif")
    rain_path = os.path.join(cfg["proc_dir"], "rain_mm_demgrid.tif")
    soil_path = os.path.join(cfg["proc_dir"], "soil_moist_demgrid.tif")
    lc_path = os.path.join(cfg["proc_dir"], "landcover_demgrid.tif")

    dem, profile, transform, crs = read_raster(dem_path)
    rain, _, _, _ = read_raster(rain_path)
    soil, _, _, _ = read_raster(soil_path)
    lc, _, _, _ = read_raster(lc_path)

    elev_low = 1.0 - minmax_norm(
        np.clip(dem, np.nanpercentile(dem, 5), np.nanpercentile(dem, 95))
    )

    slope = slope_from_dem(dem, transform)
    slope_low = 1.0 - minmax_norm(
        np.clip(slope, 0, cfg["slope_clip_max_deg"])
    )

    rain_norm = minmax_norm(rain)
    soil_norm = minmax_norm(soil)
    land_imperv = minmax_norm(landcover_to_impervious_factor(lc))

    rivers = gpd.read_file(
        os.path.join(cfg["raw_dir"], "hydrorivers.gpkg")
    ).to_crs(crs)

    river_mask = rasterize_rivers_to_mask(rivers, dem.shape, transform)
    dist = distance_to_river_m(river_mask, transform)
    river_near = minmax_norm(
        np.exp(-dist / cfg["river_decay_distance_m"])
    )

    factors = {
        "rain": rain_norm,
        "soil_moist": soil_norm,
        "elev_low": elev_low,
        "slope_low": slope_low,
        "land_imperv": land_imperv,
        "river_near": river_near,
    }

    entropy_weights, valid_mask, sample_count = entropy_weight_sampled(
        factors=factors,
        sample_size=cfg["entropy_sample_size"],
        random_seed=cfg["random_seed"]
    )

    subjective_weights = cfg["subjective_weights"]
    final_weights = combine_weights(
        subjective_weights,
        entropy_weights,
        cfg["alpha_subjective"]
    )

    print("\n[Final Weights]")
    for k, v in final_weights.items():
        print(f"{k}: {v:.6f}")

    save_weights_report(
        subjective_weights=subjective_weights,
        entropy_weights=entropy_weights,
        final_weights=final_weights,
        sample_count=sample_count,
        out_path=cfg["out_weights_txt"],
        alpha_subjective=cfg["alpha_subjective"]
    )
    print("[Saved]", cfg["out_weights_txt"])

    risk = np.full_like(dem, np.nan, dtype=np.float32)
    risk_val = np.zeros(np.sum(valid_mask), dtype=np.float64)

    for k in factors:
        risk_val += factors[k][valid_mask] * final_weights[k]

    risk[valid_mask] = risk_val.astype(np.float32)
    risk[np.isnan(dem)] = np.nan

    write_raster(cfg["out_risk_tif"], risk, profile)
    print("[Saved]", cfg["out_risk_tif"])

    build_folium_map(
        risk=risk,
        dem_path=dem_path,
        study_area_shp=cfg["study_area_shp"],
        out_map=cfg["out_map"]
    )
    print("[Saved]", cfg["out_map"])

    return {
        "risk_tif": cfg["out_risk_tif"],
        "map_html": cfg["out_map"],
        "weights_txt": cfg["out_weights_txt"],
        "subjective_weights": subjective_weights,
        "entropy_weights": entropy_weights,
        "final_weights": final_weights,
        "study_area_shp": cfg["study_area_shp"],
        "dem_path": dem_path,
    }


def main():
    return run_risk_assessment()


if __name__ == "__main__":
    main()
