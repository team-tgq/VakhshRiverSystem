# algorithms/flood/risk_assessment_6factors.py
import os
import warnings

warnings.filterwarnings("ignore")

import geopandas as gpd
import numpy as np

try:
    from .input_resolver import resolve_flood_input_paths
    from .risk_assessment_6factors_entropy import (
        build_folium_map,
        compose_weighted_risk,
        distance_to_river_m,
        landcover_to_susceptibility_factor,
        minmax_norm,
        rasterize_rivers_to_mask,
        read_raster,
        save_landuse_explainability_outputs,
        save_weights_report,
        slope_from_dem,
        write_raster,
    )
except ImportError:
    from input_resolver import resolve_flood_input_paths  # type: ignore
    from risk_assessment_6factors_entropy import (  # type: ignore
        build_folium_map,
        compose_weighted_risk,
        distance_to_river_m,
        landcover_to_susceptibility_factor,
        minmax_norm,
        rasterize_rivers_to_mask,
        read_raster,
        save_landuse_explainability_outputs,
        save_weights_report,
        slope_from_dem,
        write_raster,
    )


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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
    "weights": {
        "rain": 0.20,
        "soil_moist": 0.16,
        "elev_low": 0.15,
        "slope_low": 0.12,
        "landuse_suscept": 0.25,
        "river_near": 0.12,
    },
    "slope_clip_max_deg": 10.0,
    "river_decay_distance_m": 1500.0,
}

os.makedirs(CFG["out_dir"], exist_ok=True)


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

    final_weights = dict(CFG["weights"])
    risk, valid_mask = compose_weighted_risk(factors, final_weights)
    risk[~valid_mask] = np.nan
    risk[np.isnan(dem)] = np.nan

    save_weights_report(
        subjective_weights=final_weights,
        entropy_weights=None,
        final_weights=final_weights,
        sample_count=None,
        out_path=CFG["out_weights_txt"],
    )
    print("[Saved]", CFG["out_weights_txt"])

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
        "subjective_weights": final_weights,
        "entropy_weights": None,
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
