import os
import ee
import pandas as pd
import geopandas as gpd
from datetime import datetime
from shapely.geometry import Point

from .download_baselayers import download_gsw_extent, download_gsw_frequency, download_dem
from .reservoir_delineation import delineate_reservoir
from .reservoir_curve import generate_curve_post_srtm
from .satellite_composite import get_landsat_composite, get_sentinel_composite
from .satellite_water_area import estimate_water_area
from .area_postprocessing import generate_inferes_products

def run_estimation(
        res_name,
        lon,
        lat,
        boundary,
        start_date,
        end_date,
        max_water_level,
        max_water_area,
        res_year,
        baselayers_dir="algorithms/reservoir_estimation/data",
        output_res_dir="algorithms/reservoir_estimation/output"
):
    reservoir_dir = "algorithms/reservoir_estimation/data"
    ee.Initialize(project='cosmic-descent-489809-a1')

    region_path = os.path.join(reservoir_dir, "region.shp")

    # print("⚠ region.shp not found, generating region automatically...")

    # 大坝中心点
    dam_point = ee.Geometry.Point([lon, lat])

    buffer_km = 5  # 水库长度方向
    width_km = 5  # 水库宽度方向
    point = [lon, lat]

    region = dam_point.buffer(buffer_km * 1000).bounds()

    print("Region 坐标范围:", region.coordinates().getInfo())

    # 保存为 shapefile
    region_geojson = region.getInfo()

    feature = {
        "type": "Feature",
        "geometry": region_geojson,
        "properties": {}
    }

    gdf = gpd.GeoDataFrame.from_features([feature], crs="EPSG:4326")

    gdf.to_file(region_path)

    print("✔ Region saved to:", region_path)

    # -----------------------------
    # Step 2 下载 DEM 与 GSW
    # -----------------------------
    dem_array, reference_image = download_dem(region, f"{reservoir_dir}/DEM.tif")

    max_extent_array = download_gsw_extent(
        region,
        f"{reservoir_dir}/max_extent.tif",
        reference_image
    )

    frequency_array = download_gsw_frequency(
        region,
        f"{reservoir_dir}/frequency.tif",
        reference_image
    )

    # -----------------------------
    # Step 3 水库边界识别
    # -----------------------------
    delineate_reservoir(
        res_name,
        max_water_level,
        point,
        boundary,
        baselayers_dir,
        plot=False
    )

    # -----------------------------
    # Step 4 生成 DAV 曲线
    # -----------------------------
    output_reservoir_dir = os.path.join(output_res_dir, res_name)

    os.makedirs(output_reservoir_dir, exist_ok=True)

    generate_curve_post_srtm(
        res_name,
        max_water_level,
        baselayers_dir,
        output_reservoir_dir,
        plot_curve=False
    )

    curve_path = os.path.join(
        output_reservoir_dir,
        "reservoir_hypsometry.csv"
    )

    # -----------------------------
    # Step 5 计算水面面积
    # -----------------------------
    print("[Step 5] Estimating Water Surface Area...")

    output_csv = os.path.join(
        output_reservoir_dir,
        "estimation_area.csv"
    )

    if os.path.exists(output_csv):
        os.remove(output_csv)

    with open(output_csv, "w") as f:

        f.write(
            "sensor,date,cloud_percentage,quality_flag,"
            "raw_area,clahe_area,water_cluster_area,"
            "zone_filtered_area,local_filtered_area,"
            "pre_filtering_area_km2,post_filtering_area_km2\n"
        )

    print(f"Processing between {start_date} and {end_date}")

    landsat_comp = get_landsat_composite(
        start_date,
        end_date,
        region,
        reference_image=reference_image
    )

    sentinel_comp = get_sentinel_composite(
        start_date,
        end_date,
        region,
        reference_image=reference_image
    )

    for sensor_label, comp in zip(
            ["landsat", "sentinel"],
            [landsat_comp, sentinel_comp]):

        if comp:

            sensor_id = "LS" if sensor_label == "landsat" else "S2"

            result = estimate_water_area(
                comp,
                region,
                baselayers_dir,
                res_name,
                sensor_id
            )

            cloud_pct, raw_area, clahe_area, water_cluster_area, \
            zone_filtered_area, local_filtered_area, quality_flag = result

            if cloud_pct is None:
                print("Skipping due to invalid image")
                continue

            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")

            mid_dt = start_dt + (end_dt - start_dt) / 2

            row_data = {
                "sensor": sensor_label,
                "date": mid_dt.strftime("%Y-%m-%d"),
                "cloud_percentage": round(cloud_pct, 2),
                "quality_flag": quality_flag,
                "raw_area": raw_area,
                "clahe_area": clahe_area,
                "water_cluster_area": water_cluster_area,
                "zone_filtered_area": zone_filtered_area,
                "local_filtered_area": local_filtered_area,
                "pre_filtering_area_km2": water_cluster_area,
                "post_filtering_area_km2": local_filtered_area,
            }

            area_df = pd.DataFrame([row_data])

            area_df.to_csv(
                output_csv,
                mode="a",
                index=False,
                header=False
            )

    # -----------------------------
    # Step 6 后处理
    # -----------------------------
    print("[Step 6] Postprocessing...")

    df_area = pd.read_csv(output_csv)

    products, updated_area_df = generate_inferes_products(
        df_area=df_area,
        curve_path=curve_path,
        year_of_commission=res_year,
        sim_start_year=start_dt.year,
        sim_end_year=end_dt.year,
        res_max_area_km2=max_water_area,
        apply_bias_correction=True,
        rolling_window=1
    )

    updated_area_path = os.path.join(
        output_reservoir_dir,
        "estimation_area.csv"
    )

    updated_area_df.to_csv(updated_area_path, index=False)

    for i in range(5):

        product_df = products[f"level{i}"]

        product_path = os.path.join(
            output_reservoir_dir,
            f"estimation_level{i}_product.csv"
        )

        product_df.to_csv(product_path, index=False)

    print("✔ Estimation completed:", res_name)