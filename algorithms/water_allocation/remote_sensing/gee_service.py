"""GEE 耕地面积查询服务 — 通过 MODIS MCD12Q1 获取哈特隆州耕地面积"""
import ee
import geemap
import geopandas as gpd
from pathlib import Path

# 资源目录
_GEODATA_DIR = Path(__file__).resolve().parent.parent / "resources" / "geodata"
_DEFAULT_GEOJSON = str(_GEODATA_DIR / "khatlon_region.geojson")

REGION_NAME = "Khatlon, Tajikistan"
YEAR = "2025"
SCALE = 500
GEE_PROJECT = "skillful-source-494707-h7"


def get_cropland_area_km2(year=None, scale=500, gee_project=None):
    """
    从 Google Earth Engine 获取哈特隆州耕地面积（平方公里）

    Args:
        year: 年份，默认为当前年份或最近可用年份
        scale: 分辨率（米）
        gee_project: GEE 项目 ID，默认为 None（使用模块默认配置）

    Returns:
        float: 耕地面积（平方公里）
    """
    try:
        project_id = gee_project if gee_project else GEE_PROJECT
        ee.Initialize(project=project_id)

        gdf = gpd.read_file(_DEFAULT_GEOJSON)
        roi = geemap.geopandas_to_ee(gdf)

        target_year = year if year else YEAR

        modis_col = (ee.ImageCollection("MODIS/061/MCD12Q1")
                     .filterDate(f"{target_year}-01-01", f"{target_year}-12-31"))

        count = modis_col.size().getInfo()

        if count == 0:
            all_images = ee.ImageCollection("MODIS/061/MCD12Q1")
            latest = all_images.sort("system:time_start", False).first()
            latest_date = latest.date().format("YYYY-MM-dd").getInfo()
            fallback_year = latest_date[:4]
            target_year = fallback_year

            modis_col = (ee.ImageCollection("MODIS/061/MCD12Q1")
                         .filterDate(f"{target_year}-01-01", f"{target_year}-12-31"))

        modis = modis_col.first()
        land_cover = modis.select("LC_Type1")

        CROPLAND_CODE = 12
        cropland = land_cover.eq(CROPLAND_CODE).clip(roi).rename("cropland")

        area_stats = (cropland
            .multiply(ee.Image.pixelArea())
            .reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=roi.geometry(),
                scale=scale,
                maxPixels=1e13,
                bestEffort=True
            ))

        area_sqm = area_stats.get("cropland").getInfo()
        area_kilo = area_sqm / 1e6

        return area_kilo

    except Exception as e:
        print(f"获取耕地面积时出错: {e}")
        raise e


if __name__ == "__main__":
    ee.Initialize(project=GEE_PROJECT)

    gdf = gpd.read_file(_DEFAULT_GEOJSON)
    roi = geemap.geopandas_to_ee(gdf)

    modis_col = (ee.ImageCollection("MODIS/061/MCD12Q1")
                 .filterDate(f"{YEAR}-01-01", f"{YEAR}-12-31"))

    count = modis_col.size().getInfo()

    if count == 0:
        print(f"⚠ {YEAR} 年没有 MCD12Q1 数据！")
        all_images = ee.ImageCollection("MODIS/061/MCD12Q1")
        latest = all_images.sort("system:time_start", False).first()
        latest_date = latest.date().format("YYYY-MM-dd").getInfo()
        print(f"  最新可用数据日期: {latest_date}")
        fallback_year = latest_date[:4]
        print(f"  自动回退到: {fallback_year} 年\n")
        YEAR = fallback_year

        modis_col = (ee.ImageCollection("MODIS/061/MCD12Q1")
                     .filterDate(f"{YEAR}-01-01", f"{YEAR}-12-31"))

    modis = modis_col.first()
    land_cover = modis.select("LC_Type1")

    CROPLAND_CODE = 12
    cropland = land_cover.eq(CROPLAND_CODE).clip(roi).rename("cropland")

    area_stats = (cropland
        .multiply(ee.Image.pixelArea())
        .reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=roi.geometry(),
            scale=SCALE,
            maxPixels=1e13,
            bestEffort=True
        ))

    area_sqm = area_stats.get("cropland").getInfo()
    area_kilo = area_sqm / 1e6
    print(f"\n{'='*45}")
    print(f"  区域: {REGION_NAME}")
    print(f"  数据: MODIS MCD12Q1 ({YEAR})")
    print(f"  分辨率: {SCALE}m")
    print(f"{'='*45}")
    print(f"  耕地面积: {area_sqm / 1e4:,.6f} 公顷")
    print(f"  耕地面积: {area_kilo:,.6f} 平方公里")
    print(f"{'='*45}")

    total_area = (ee.Image.pixelArea()
        .clip(roi)
        .reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=roi.geometry(),
            scale=SCALE,
            maxPixels=1e13,
            bestEffort=True
        ))

    total_sqm = total_area.get("area").getInfo()
    ratio = area_sqm / total_sqm * 100
    print(f"  耕地占比: {ratio:.2f}%")
    print(f"{'='*45}")

    Map = geemap.Map()

    igbp_palette = [
        "05450a", "086a10", "54a708", "78d203", "009900",
        "c6b044", "dcd159", "dade48", "fbff13", "b6ff05",
        "27ff87", "c24f44", "a5a5a5", "ff6d4c", "69fff8",
        "f9ffa4", "1c0dff"
    ]

    Map.addLayer(land_cover.clip(roi), {
        "min": 1, "max": 17, "palette": igbp_palette
    }, "土地覆盖类型")

    Map.addLayer(cropland.updateMask(cropland), {
        "palette": ["#FFD700"]
    }, "耕地")

    Map.addLayer(roi, {"color": "red"}, "研究区边界")
    Map.centerObject(roi, 7)
    Map
