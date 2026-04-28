# algorithms/flood/risk_assessment_6factors_entropy.py
import csv
import os
import warnings

warnings.filterwarnings("ignore")

import folium
import geopandas as gpd
import numpy as np
import rasterio
from branca.colormap import linear
from folium.raster_layers import ImageOverlay
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt

try:
    from .input_resolver import resolve_flood_input_paths
except ImportError:
    from input_resolver import resolve_flood_input_paths  # type: ignore


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LANDCOVER_CLASSES = {
    10: {
        "name": "Tree cover / 林地",
        "susceptibility": 0.35,
        "color": "#2E7D32",
        "included_in_ranking": True,
    },
    20: {
        "name": "Shrubland / 灌丛",
        "susceptibility": 0.45,
        "color": "#7CB342",
        "included_in_ranking": True,
    },
    30: {
        "name": "Grassland / 草地",
        "susceptibility": 0.55,
        "color": "#C0CA33",
        "included_in_ranking": True,
    },
    40: {
        "name": "Cropland / 农田",
        "susceptibility": 0.80,
        "color": "#FDD835",
        "included_in_ranking": True,
    },
    50: {
        "name": "Built-up / 建成区",
        "susceptibility": 1.00,
        "color": "#E53935",
        "included_in_ranking": True,
    },
    60: {
        "name": "Bare sparse / 裸地稀疏植被",
        "susceptibility": 0.50,
        "color": "#C49A6C",
        "included_in_ranking": True,
    },
    70: {
        "name": "Snow ice / 冰雪",
        "susceptibility": 0.10,
        "color": "#B3E5FC",
        "included_in_ranking": True,
    },
    80: {
        "name": "Water bodies / 水体",
        "susceptibility": 0.05,
        "color": "#1E88E5",
        "included_in_ranking": False,
    },
    90: {
        "name": "Wetland / 草本湿地",
        "susceptibility": 0.85,
        "color": "#26A69A",
        "included_in_ranking": True,
    },
    95: {
        "name": "Mangroves / 红树林",
        "susceptibility": 0.75,
        "color": "#00897B",
        "included_in_ranking": True,
    },
    100: {
        "name": "Moss lichen / 苔藓地衣",
        "susceptibility": 0.40,
        "color": "#8BC34A",
        "included_in_ranking": True,
    },
}

UNKNOWN_LANDCOVER = {
    "name": "Unknown / 未识别",
    "susceptibility": 0.50,
    "color": "#9E9E9E",
    "included_in_ranking": True,
}

FACTOR_DISPLAY_NAMES = {
    "rain": "逐日降水",
    "soil_moist": "表层土壤湿度",
    "elev_low": "低海拔敏感性",
    "slope_low": "低坡度敏感性",
    "landuse_suscept": "土地利用敏感性",
    "river_near": "近河道敏感性",
}

LANDUSE_STATS_COLUMNS = [
    "landcover_code",
    "landcover_name",
    "included_in_ranking",
    "pixel_count",
    "area_km2",
    "mean_risk",
    "p90_risk",
    "high_risk_ratio",
    "dominant_risk_level",
]

RISK_LEVELS = [
    (0.0, 0.2, "低"),
    (0.2, 0.4, "较低"),
    (0.4, 0.6, "中"),
    (0.6, 0.8, "较高"),
    (0.8, 1.000001, "高"),
]

CFG = {
    "study_area_shp": os.path.join(BASE_DIR, "study_area.shp"),
    "proc_dir": os.path.join(BASE_DIR, "data", "processed"),
    "raw_dir": os.path.join(BASE_DIR, "data", "raw"),
    "out_dir": os.path.join(BASE_DIR, "outputs"),
    "out_risk_tif": os.path.join(BASE_DIR, "outputs", "risk_6factors.tif"),
    "out_map": os.path.join(BASE_DIR, "outputs", "flood_risk_map.html"),
    "out_weights_txt": os.path.join(BASE_DIR, "outputs", "final_weights.txt"),
    "out_landuse_stats_csv": os.path.join(BASE_DIR, "outputs", "landuse_risk_stats.csv"),
    "out_landuse_summary_txt": os.path.join(BASE_DIR, "outputs", "landuse_risk_summary.txt"),
    "subjective_weights": {
        "rain": 0.20,
        "soil_moist": 0.16,
        "elev_low": 0.15,
        "slope_low": 0.12,
        "landuse_suscept": 0.25,
        "river_near": 0.12,
    },
    "alpha_subjective": 0.85,
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
    if np.isnan(dem_safe).all():
        return np.full_like(dem, np.nan, dtype=np.float32)
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
        dtype=np.uint8,
    )


def distance_to_river_m(river_mask, transform):
    inv = (river_mask == 0).astype(np.uint8)
    dist_px = distance_transform_edt(inv)
    px = abs(transform.a)
    py = abs(transform.e)
    return dist_px * float(np.mean([px, py]))


def landcover_class_info(code):
    try:
        normalized = int(round(float(code)))
    except (TypeError, ValueError):
        normalized = None

    if normalized in LANDCOVER_CLASSES:
        return {"code": normalized, **LANDCOVER_CLASSES[normalized]}

    label_suffix = ""
    if normalized is not None:
        label_suffix = f" ({normalized})"
    return {
        "code": normalized,
        "name": f"{UNKNOWN_LANDCOVER['name']}{label_suffix}",
        "susceptibility": UNKNOWN_LANDCOVER["susceptibility"],
        "color": UNKNOWN_LANDCOVER["color"],
        "included_in_ranking": UNKNOWN_LANDCOVER["included_in_ranking"],
    }


def landcover_to_susceptibility_factor(lc):
    factor = np.full_like(lc, np.nan, dtype=np.float32)

    for code, meta in LANDCOVER_CLASSES.items():
        factor[lc == code] = meta["susceptibility"]

    factor[np.isnan(factor) & np.isfinite(lc)] = UNKNOWN_LANDCOVER["susceptibility"]
    return factor.astype(np.float32)


def entropy_weight_sampled(factors, sample_size=5000, random_seed=42, eps=1e-12):
    names = list(factors.keys())
    arrays = [factors[name].astype(np.float64) for name in names]

    valid_mask = np.ones_like(arrays[0], dtype=bool)
    for arr in arrays:
        valid_mask &= np.isfinite(arr)

    idx = np.where(valid_mask.ravel())[0]
    if len(idx) < 2:
        raise ValueError("有效像元过少，无法计算熵权。")

    rng = np.random.default_rng(random_seed)
    sample = rng.choice(idx, size=min(sample_size, len(idx)), replace=False)
    sample_matrix = np.column_stack([arr.ravel()[sample] for arr in arrays])

    col_sums = sample_matrix.sum(axis=0)
    zero_cols = col_sums <= eps
    if np.any(zero_cols):
        sample_matrix[:, zero_cols] = eps
        col_sums = sample_matrix.sum(axis=0)

    proportions = sample_matrix / col_sums
    proportions = np.clip(proportions, eps, None)

    k = 1.0 / np.log(sample_matrix.shape[0])
    entropy = -k * np.sum(proportions * np.log(proportions), axis=0)
    difference = 1.0 - entropy

    if np.all(difference <= eps):
        weights = np.full(sample_matrix.shape[1], 1.0 / sample_matrix.shape[1], dtype=np.float64)
    else:
        weights = difference / np.sum(difference)

    return {name: float(weight) for name, weight in zip(names, weights)}, valid_mask, sample_matrix.shape[0]


def combine_weights(subjective_weights, entropy_weights, alpha):
    final_weights = {}
    for key in subjective_weights:
        final_weights[key] = alpha * subjective_weights[key] + (1.0 - alpha) * entropy_weights[key]

    total = sum(final_weights.values())
    if total > 0:
        final_weights = {key: value / total for key, value in final_weights.items()}
    return final_weights


def build_valid_mask(factors):
    mask = np.ones_like(next(iter(factors.values())), dtype=bool)
    for arr in factors.values():
        mask &= np.isfinite(arr)
    return mask


def compose_weighted_risk(factors, weights):
    valid_mask = build_valid_mask(factors)
    risk = np.full_like(next(iter(factors.values())), np.nan, dtype=np.float32)
    risk_val = np.zeros(np.sum(valid_mask), dtype=np.float64)

    for key, arr in factors.items():
        risk_val += arr[valid_mask] * weights[key]

    risk[valid_mask] = risk_val.astype(np.float32)
    return risk, valid_mask


def save_weights_report(subjective_weights, entropy_weights, final_weights, sample_count, out_path):
    lines = []
    lines.append("Flood Risk Weights Report\n")
    lines.append("=" * 60 + "\n")
    if sample_count is None:
        lines.append("Weighting mode: fixed subjective weights\n")
    else:
        lines.append("Weighting mode: subjective + entropy combination\n")
        lines.append(f"Entropy sample count: {sample_count}\n")
        lines.append(f"Combination alpha (subjective): {CFG['alpha_subjective']:.3f}\n")
        lines.append(f"Combination beta (entropy): {1.0 - CFG['alpha_subjective']:.3f}\n")
    lines.append("=" * 60 + "\n\n")

    lines.append("[Factor Meaning]\n")
    for key, label in FACTOR_DISPLAY_NAMES.items():
        lines.append(f"{key}: {label}\n")
    lines.append("\n")

    lines.append("[Subjective Weights]\n")
    for key, value in subjective_weights.items():
        lines.append(f"{key}: {value:.6f}\n")
    lines.append(f"sum: {sum(subjective_weights.values()):.6f}\n\n")

    if entropy_weights is not None:
        lines.append("[Entropy Weights]\n")
        for key, value in entropy_weights.items():
            lines.append(f"{key}: {value:.6f}\n")
        lines.append(f"sum: {sum(entropy_weights.values()):.6f}\n\n")
    else:
        lines.append("[Entropy Weights]\n")
        lines.append("Not used in fixed-weight mode.\n\n")

    lines.append("[Final Combined Weights]\n")
    for key, value in final_weights.items():
        lines.append(f"{key}: {value:.6f}\n")
    lines.append(f"sum: {sum(final_weights.values()):.6f}\n")

    with open(out_path, "w", encoding="utf-8") as file:
        file.writelines(lines)


def _format_csv_value(value, digits=6):
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "True" if bool(value) else "False"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return ""
        return f"{float(value):.{digits}f}"
    return str(value)


def dominant_risk_level(values):
    if values.size == 0:
        return "无数据"

    clipped = np.clip(values.astype(np.float64), 0.0, 1.0)
    counts = []
    for lower, upper, label in RISK_LEVELS:
        if upper >= 1.0:
            count = int(np.sum((clipped >= lower) & (clipped <= upper)))
        else:
            count = int(np.sum((clipped >= lower) & (clipped < upper)))
        counts.append((count, label))
    return max(counts, key=lambda item: item[0])[1]


def _pixel_area_km2(transform, crs, shape):
    pixel_width = abs(float(transform.a))
    pixel_height = abs(float(transform.e))
    if pixel_width == 0 or pixel_height == 0:
        return np.nan

    if crs is not None and getattr(crs, "is_projected", False):
        return abs(pixel_width * pixel_height) / 1_000_000.0

    if crs is not None and getattr(crs, "is_geographic", False):
        center_lat = float(transform.f + (shape[0] / 2.0) * transform.e)
        meters_per_degree_lat = 110574.0
        meters_per_degree_lon = 111320.0 * np.cos(np.deg2rad(center_lat))
        return abs(pixel_width * meters_per_degree_lon * pixel_height * meters_per_degree_lat) / 1_000_000.0

    return np.nan


def compute_landuse_risk_stats(risk, landcover, transform, crs):
    valid_mask = np.isfinite(risk) & np.isfinite(landcover)
    finite_codes = sorted({int(round(float(code))) for code in np.unique(landcover[valid_mask])})
    ordered_known_codes = [code for code in LANDCOVER_CLASSES if code in finite_codes]
    ordered_unknown_codes = [code for code in finite_codes if code not in LANDCOVER_CLASSES]
    ordered_codes = ordered_known_codes + ordered_unknown_codes

    pixel_area_km2 = _pixel_area_km2(transform, crs, risk.shape)
    rounded_landcover = np.round(landcover)
    rows = []

    for code in ordered_codes:
        meta = landcover_class_info(code)
        code_mask = valid_mask & (rounded_landcover == int(code))
        pixel_count = int(np.sum(code_mask))

        if pixel_count > 0:
            values = risk[code_mask].astype(np.float64)
            mean_risk = float(np.nanmean(values))
            p90_risk = float(np.nanpercentile(values, 90))
            high_risk_ratio = float(np.mean(values >= 0.6))
            dominant_level = dominant_risk_level(values)
        else:
            mean_risk = np.nan
            p90_risk = np.nan
            high_risk_ratio = np.nan
            dominant_level = "无数据"

        rows.append(
            {
                "landcover_code": int(code),
                "landcover_name": meta["name"],
                "included_in_ranking": bool(meta["included_in_ranking"]),
                "pixel_count": pixel_count,
                "area_km2": float(pixel_count * pixel_area_km2) if np.isfinite(pixel_area_km2) else np.nan,
                "mean_risk": mean_risk,
                "p90_risk": p90_risk,
                "high_risk_ratio": high_risk_ratio,
                "dominant_risk_level": dominant_level,
            }
        )

    return rows


def save_landuse_risk_stats(rows, out_path):
    with open(out_path, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=LANDUSE_STATS_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "landcover_code": _format_csv_value(row["landcover_code"], digits=0),
                    "landcover_name": row["landcover_name"],
                    "included_in_ranking": _format_csv_value(row["included_in_ranking"]),
                    "pixel_count": _format_csv_value(row["pixel_count"], digits=0),
                    "area_km2": _format_csv_value(row["area_km2"]),
                    "mean_risk": _format_csv_value(row["mean_risk"]),
                    "p90_risk": _format_csv_value(row["p90_risk"]),
                    "high_risk_ratio": _format_csv_value(row["high_risk_ratio"]),
                    "dominant_risk_level": row["dominant_risk_level"],
                }
            )


def _top_landuse_rows(rows, key):
    ranked = [
        row
        for row in rows
        if row["included_in_ranking"]
        and row["pixel_count"] > 0
        and np.isfinite(row[key])
    ]
    ranked.sort(
        key=lambda row: (
            float(row[key]),
            float(row["area_km2"]) if np.isfinite(row["area_km2"]) else -1.0,
            int(row["pixel_count"]),
        ),
        reverse=True,
    )
    return ranked[:3]


def save_landuse_risk_summary(rows, final_weights, out_path):
    top_high_risk = _top_landuse_rows(rows, "high_risk_ratio")
    top_mean_risk = _top_landuse_rows(rows, "mean_risk")

    lines = []
    lines.append("洪涝风险评估土地利用解释摘要\n")
    lines.append("=" * 60 + "\n\n")

    lines.append("[土地利用敏感性映射]\n")
    for code in sorted(LANDCOVER_CLASSES):
        meta = LANDCOVER_CLASSES[code]
        suffix = "（不参与解释性排名）" if not meta["included_in_ranking"] else ""
        lines.append(f"{code}: {meta['name']} -> {meta['susceptibility']:.2f}{suffix}\n")
    lines.append(
        f"Unknown finite code: {UNKNOWN_LANDCOVER['name']} -> {UNKNOWN_LANDCOVER['susceptibility']:.2f}\n\n"
    )

    lines.append("[最终权重]\n")
    for key, value in final_weights.items():
        lines.append(f"{key} ({FACTOR_DISPLAY_NAMES.get(key, key)}): {value:.6f}\n")
    lines.append("\n")

    lines.append("[高风险占比前 3 类土地利用]\n")
    if top_high_risk:
        for index, row in enumerate(top_high_risk, start=1):
            lines.append(
                f"{index}. {row['landcover_name']} | high_risk_ratio={row['high_risk_ratio']:.2%} | "
                f"mean_risk={row['mean_risk']:.3f}\n"
            )
    else:
        lines.append("暂无可用统计。\n")
    lines.append("\n")

    lines.append("[平均风险前 3 类土地利用]\n")
    if top_mean_risk:
        for index, row in enumerate(top_mean_risk, start=1):
            lines.append(
                f"{index}. {row['landcover_name']} | mean_risk={row['mean_risk']:.3f} | "
                f"high_risk_ratio={row['high_risk_ratio']:.2%}\n"
            )
    else:
        lines.append("暂无可用统计。\n")

    with open(out_path, "w", encoding="utf-8") as file:
        file.writelines(lines)

    return {
        "top_high_risk_landuse": [
            {
                "landcover_code": row["landcover_code"],
                "landcover_name": row["landcover_name"],
                "high_risk_ratio": float(row["high_risk_ratio"]),
                "mean_risk": float(row["mean_risk"]),
            }
            for row in top_high_risk
        ],
        "top_mean_risk_landuse": [
            {
                "landcover_code": row["landcover_code"],
                "landcover_name": row["landcover_name"],
                "mean_risk": float(row["mean_risk"]),
                "high_risk_ratio": float(row["high_risk_ratio"]),
            }
            for row in top_mean_risk
        ],
    }


def save_landuse_explainability_outputs(risk, landcover, transform, crs, final_weights, stats_csv_path, summary_path):
    rows = compute_landuse_risk_stats(risk, landcover, transform, crs)
    save_landuse_risk_stats(rows, stats_csv_path)
    summary_payload = save_landuse_risk_summary(rows, final_weights, summary_path)
    summary_payload["rows"] = rows
    return summary_payload


def _hex_to_rgba(hex_color, alpha):
    color = hex_color.lstrip("#")
    if len(color) != 6:
        return [158, 158, 158, alpha]
    return [int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16), alpha]


def _risk_rgba(risk):
    risk_disp = np.clip(risk, 0, 1)
    cm = linear.YlOrRd_09.scale(0, 1)
    rgba = np.zeros((risk_disp.shape[0], risk_disp.shape[1], 4), dtype=np.uint8)

    flat = risk_disp.flatten()
    rgba_2d = rgba.reshape(-1, 4)
    for index, value in enumerate(flat):
        if np.isnan(value):
            rgba_2d[index] = [0, 0, 0, 0]
        else:
            r, g, b, _ = cm.rgba_bytes_tuple(float(value))
            rgba_2d[index] = [r, g, b, 190]
    return rgba


def _landcover_rgba(landcover):
    rgba = np.zeros((landcover.shape[0], landcover.shape[1], 4), dtype=np.uint8)
    known_codes = set(LANDCOVER_CLASSES)
    rounded_landcover = np.round(landcover)

    for code, meta in LANDCOVER_CLASSES.items():
        rgba[landcover == code] = _hex_to_rgba(meta["color"], 170)

    unknown_mask = np.isfinite(landcover) & ~np.isin(rounded_landcover, list(known_codes))
    rgba[unknown_mask] = _hex_to_rgba(UNKNOWN_LANDCOVER["color"], 170)
    return rgba


def _map_bounds_wgs84(dem_path):
    with rasterio.open(dem_path) as src:
        bounds = src.bounds
        dem_crs = src.crs

    if dem_crs is not None and dem_crs.to_string() != "EPSG:4326":
        from pyproj import Transformer

        transformer = Transformer.from_crs(dem_crs, "EPSG:4326", always_xy=True)
        west_lon, south_lat = transformer.transform(bounds.left, bounds.bottom)
        east_lon, north_lat = transformer.transform(bounds.right, bounds.top)
    else:
        west_lon, south_lat, east_lon, north_lat = bounds.left, bounds.bottom, bounds.right, bounds.top

    return west_lon, south_lat, east_lon, north_lat


def _add_landcover_legend(map_object):
    legend_lines = []
    for code in sorted(LANDCOVER_CLASSES):
        meta = LANDCOVER_CLASSES[code]
        suffix = "（不参与排名）" if not meta["included_in_ranking"] else ""
        legend_lines.append(
            "<div style=\"margin-bottom:4px;\">"
            f"<span style=\"display:inline-block;width:12px;height:12px;background:{meta['color']};"
            "margin-right:6px;border:1px solid #666;\"></span>"
            f"{code} - {meta['name']}{suffix}</div>"
        )
    legend_lines.append(
        "<div style=\"margin-top:6px;font-size:11px;color:#555;\">"
        "Water bodies / 水体会保留展示，但不参与解释性排名。"
        "</div>"
    )

    legend_html = (
        "<div style=\"position: fixed; bottom: 35px; left: 35px; z-index: 9999; "
        "background: rgba(255,255,255,0.92); padding: 12px 14px; border: 1px solid #999; "
        "border-radius: 6px; max-height: 280px; overflow-y: auto; font-size: 12px;\">"
        "<div style=\"font-weight: 600; margin-bottom: 8px;\">土地利用类型</div>"
        + "".join(legend_lines)
        + "</div>"
    )
    map_object.get_root().html.add_child(folium.Element(legend_html))


def build_folium_map(risk, landcover, dem_path, study_area_shp, out_map):
    west_lon, south_lat, east_lon, north_lat = _map_bounds_wgs84(dem_path)
    center = [(south_lat + north_lat) / 2.0, (west_lon + east_lon) / 2.0]
    map_object = folium.Map(location=center, zoom_start=8, tiles="cartodbpositron")

    landcover_overlay = ImageOverlay(
        image=_landcover_rgba(landcover),
        bounds=[[south_lat, west_lon], [north_lat, east_lon]],
        name="土地利用类型",
        opacity=0.78,
        interactive=True,
        cross_origin=False,
        zindex=1,
        show=False,
    )
    landcover_overlay.add_to(map_object)

    risk_overlay = ImageOverlay(
        image=_risk_rgba(risk),
        bounds=[[south_lat, west_lon], [north_lat, east_lon]],
        name="洪涝风险",
        opacity=0.75,
        interactive=True,
        cross_origin=False,
        zindex=2,
        show=True,
    )
    risk_overlay.add_to(map_object)

    aoi = gpd.read_file(study_area_shp)
    if aoi.crs is not None and aoi.crs.to_string() != "EPSG:4326":
        aoi = aoi.to_crs("EPSG:4326")

    folium.GeoJson(
        aoi.to_json(),
        name="研究区边界",
        style_function=lambda _: {"color": "#0066FF", "weight": 3, "fillOpacity": 0.0},
        tooltip="Study Area",
    ).add_to(map_object)

    risk_colormap = linear.YlOrRd_09.scale(0, 1)
    risk_colormap.caption = "洪涝风险（0~1）"
    risk_colormap.add_to(map_object)
    _add_landcover_legend(map_object)

    folium.LayerControl(collapsed=False).add_to(map_object)
    map_object.save(out_map)


def run_risk_assessment(
    target_date=None,
    auto_prepare_static=True,
    allow_legacy_dynamic=True,
    auto_prepare_dynamic=True,
):
    input_paths = resolve_flood_input_paths(
        CFG,
        target_date=target_date,
        auto_prepare_static=auto_prepare_static,
        allow_legacy_dynamic=allow_legacy_dynamic,
        auto_prepare_dynamic=auto_prepare_dynamic,
    )

    dem_path = input_paths["dem_path"]
    rain_path = input_paths["rain_path"]
    soil_path = input_paths["soil_path"]
    landcover_path = input_paths["landcover_path"]
    rivers_path = input_paths["rivers_path"]

    dem, profile, transform, crs = read_raster(dem_path)
    rain, _, _, _ = read_raster(rain_path)
    soil, _, _, _ = read_raster(soil_path)
    landcover, _, _, _ = read_raster(landcover_path)

    elev_low = 1.0 - minmax_norm(
        np.clip(dem, np.nanpercentile(dem, 5), np.nanpercentile(dem, 95))
    )
    slope = slope_from_dem(dem, transform)
    slope_low = 1.0 - minmax_norm(np.clip(slope, 0, CFG["slope_clip_max_deg"]))

    rain_norm = minmax_norm(rain)
    soil_norm = minmax_norm(soil)
    landuse_suscept = landcover_to_susceptibility_factor(landcover)

    rivers = gpd.read_file(rivers_path).to_crs(crs)
    river_mask = rasterize_rivers_to_mask(rivers, dem.shape, transform)
    distance = distance_to_river_m(river_mask, transform)
    river_near = minmax_norm(np.exp(-distance / CFG["river_decay_distance_m"]))

    factors = {
        "rain": rain_norm,
        "soil_moist": soil_norm,
        "elev_low": elev_low,
        "slope_low": slope_low,
        "landuse_suscept": landuse_suscept,
        "river_near": river_near,
    }

    entropy_weights, valid_mask, sample_count = entropy_weight_sampled(
        factors=factors,
        sample_size=CFG["entropy_sample_size"],
        random_seed=CFG["random_seed"],
    )

    subjective_weights = CFG["subjective_weights"]
    final_weights = combine_weights(subjective_weights, entropy_weights, CFG["alpha_subjective"])

    print("\n[Final Weights]")
    for key, value in final_weights.items():
        print(f"{key}: {value:.6f}")

    save_weights_report(
        subjective_weights=subjective_weights,
        entropy_weights=entropy_weights,
        final_weights=final_weights,
        sample_count=sample_count,
        out_path=CFG["out_weights_txt"],
    )
    print("[Saved]", CFG["out_weights_txt"])

    risk, _ = compose_weighted_risk(factors, final_weights)
    risk[~valid_mask] = np.nan
    risk[np.isnan(dem)] = np.nan

    write_raster(CFG["out_risk_tif"], risk, profile)
    print("[Saved]", CFG["out_risk_tif"])

    explainability = save_landuse_explainability_outputs(
        risk=risk,
        landcover=landcover,
        transform=transform,
        crs=crs,
        final_weights=final_weights,
        stats_csv_path=CFG["out_landuse_stats_csv"],
        summary_path=CFG["out_landuse_summary_txt"],
    )
    print("[Saved]", CFG["out_landuse_stats_csv"])
    print("[Saved]", CFG["out_landuse_summary_txt"])

    build_folium_map(
        risk=risk,
        landcover=landcover,
        dem_path=dem_path,
        study_area_shp=input_paths["study_area_shp"],
        out_map=CFG["out_map"],
    )
    print("[Saved]", CFG["out_map"])

    return {
        "risk_tif": CFG["out_risk_tif"],
        "map_html": CFG["out_map"],
        "weights_txt": CFG["out_weights_txt"],
        "landuse_stats_csv": CFG["out_landuse_stats_csv"],
        "landuse_summary_txt": CFG["out_landuse_summary_txt"],
        "landuse_factor_name": "landuse_suscept",
        "subjective_weights": subjective_weights,
        "entropy_weights": entropy_weights,
        "final_weights": final_weights,
        "study_area_shp": input_paths["study_area_shp"],
        "dem_path": dem_path,
        "landcover_path": landcover_path,
        "rivers_path": rivers_path,
        "rain_path": rain_path,
        "soil_path": soil_path,
        "requested_target_date": input_paths["requested_target_date"],
        "resolved_target_date": input_paths["resolved_target_date"],
        "dynamic_scale": input_paths["dynamic_scale"],
        "available_dynamic_dates": input_paths["available_dynamic_dates"],
        "static_actions": input_paths.get("static_actions", []),
        "dynamic_actions": input_paths.get("dynamic_actions", []),
        "top_high_risk_landuse": explainability["top_high_risk_landuse"],
        "top_mean_risk_landuse": explainability["top_mean_risk_landuse"],
    }


def main():
    return run_risk_assessment()


if __name__ == "__main__":
    main()
