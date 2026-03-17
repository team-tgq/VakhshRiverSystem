import rasterio
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from rasterio.mask import mask
from scipy.integrate import cumulative_trapezoid
import pandas as pd
from pyproj import CRS
from rasterio.warp import calculate_default_transform, reproject, Resampling
























# -----------------------------
# 参数配置
# -----------------------------
dem_path = "data/vakhsh_dem.tif"  # 高分辨率 DEM
reservoir_name = "Nurek"

# 努列克坝中心坐标 (lon, lat)
reservoir_center_lon = 69 + 20 / 60 + 53 / 3600
reservoir_center_lat = 38 + 22 / 60 + 18 / 3600
buffer_km = 15  # 缓冲半径（km）

# 输出文件
reservoir_shp = f"{reservoir_name}_polygon.shp"
curve_csv = f"{reservoir_name}_volume_curve.csv"

# -----------------------------
# 读取 DEM
# -----------------------------
with rasterio.open(dem_path) as src:
    dem = src.read(1).astype(float)
    dem_crs = src.crs
    dem_transform = src.transform
    print("Original DEM CRS:", dem_crs)

# -----------------------------
# 选择投影 CRS (UTM)
# -----------------------------
utm_crs = CRS.from_epsg(32642)  # UTM zone 42N
print("Working CRS:", utm_crs)

# -----------------------------
# 构建水库缓冲区 polygon (GeoSeries)
# -----------------------------
point_geo = gpd.GeoSeries([Point(reservoir_center_lon, reservoir_center_lat)], crs="EPSG:4326")
point_proj = point_geo.to_crs(utm_crs)
buffer_geo = point_proj.buffer(buffer_km * 1000)  # 5km 缓冲，保持 GeoSeries

# -----------------------------
# 将 DEM 投影到 UTM
# -----------------------------
with rasterio.open(dem_path) as src:
    if dem_crs != utm_crs:
        transform, width, height = calculate_default_transform(
            src.crs, utm_crs, src.width, src.height, *src.bounds)
        dem_reproj = np.empty((height, width), dtype=np.float32)
        reproject(
            source=dem,
            destination=dem_reproj,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=utm_crs,
            resampling=Resampling.bilinear
        )
        dem = dem_reproj
        dem_transform = transform
        dem_crs = utm_crs
        print("DEM reprojected")
    else:
        print("DEM already in UTM")

# -----------------------------
# 提取缓冲区 DEM
# -----------------------------
with rasterio.open(dem_path) as src:
    # mask 接受 GeoJSON-like geometry dict 或 GeoSeries
    buffer_geo_utm = gpd.GeoSeries([buffer_geo[0]], crs=utm_crs)
    out_image, out_transform = mask(src, buffer_geo_utm.__geo_interface__, crop=True)
    dem_crop = out_image[0].astype(float)
    dem_transform = out_transform

# -----------------------------
# 处理异常值
# -----------------------------
dem_crop[(dem_crop <= 0) | (dem_crop > 5000) | (dem_crop == 32767)] = np.nan
dem_valid = dem_crop[~np.isnan(dem_crop)]
if len(dem_valid) == 0:
    raise ValueError("缓冲区内没有有效 DEM 数据，请检查 DEM 或缓冲区设置")

# -----------------------------
# 水位序列
# -----------------------------
min_elev = int(np.floor(np.min(dem_valid)))
max_elev = int(np.ceil(np.max(dem_valid)))
print(f"Elevation range: {min_elev} - {max_elev}")

elevations = np.arange(min_elev, max_elev + 1, 1)

# -----------------------------
# 面积和体积计算
# -----------------------------
pixel_area = abs(dem_transform[0] * dem_transform[4])
print("Pixel width:", dem_transform[0])
print("Pixel height:", dem_transform[4])
print("Pixel area (m²):", pixel_area)

areas = []
for h in elevations:
    water_mask = dem_crop <= h
    water_mask = np.where(np.isnan(dem_crop), 0, water_mask)
    area_m2 = np.sum(water_mask) * pixel_area
    areas.append(area_m2)

volumes = cumulative_trapezoid(areas, elevations, initial=0)

# -----------------------------
# 保存水库 polygon
# -----------------------------
gdf = gpd.GeoDataFrame({"name": [reservoir_name]}, geometry=buffer_geo, crs=utm_crs)
gdf.to_file(reservoir_shp)
print(f"水库 polygon 已保存: {reservoir_shp}")

# -----------------------------
# 保存库容曲线
# -----------------------------
df_curve = pd.DataFrame({
    "elevation_m": elevations,
    "area_m2": areas,
    "volume_m3": volumes
})
df_curve.to_csv(curve_csv, index=False)
print(f"库容曲线已保存: {curve_csv}")

print(f"Max area (km²): {max(areas)/1e6:.2f}")
print(f"Max volume (km³): {max(volumes)/1e9:.2f}")