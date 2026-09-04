"""GEE 耕地面积查询服务 — 通过 ESA WorldCover 10m 获取哈特隆州全年耕地面积"""
import ee
import geemap
import geopandas as gpd
import os
from pathlib import Path

# 资源目录
_GEODATA_DIR = Path(__file__).resolve().parent.parent / "resources" / "geodata"
_DEFAULT_GEOJSON = str(_GEODATA_DIR / "khatlon_region.geojson")

REGION_NAME = "Khatlon, Tajikistan"
# ESA WorldCover: v200=2021, v100=2020 (10m 年度耕地, 含休耕轮作)
WORLDCOVER_VERSION = "v200"
CROPLAND_CODE = 40                     # WorldCover 耕地类
SCALE = 10
GEE_PROJECT = "skillful-source-494707-h7"
# 密钥文件默认路径
_DEFAULT_WEIGHTS = str(Path(__file__).resolve().parent.parent / "resources" / "geedata" / "skillful-source-494707-h7-8f7b2bebcf57.json")

def get_cropland_area_km2(year=None, scale=SCALE, gee_project=None):
    """
    从 Google Earth Engine 获取哈特隆州耕地面积（平方公里）

    数据源: ESA WorldCover 10m (年度产品, 含休耕/轮作的全年耕地, 类码 40)。
    

    Returns:
        float: 耕地面积（平方公里）
    """
    try:
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = _DEFAULT_WEIGHTS
        project_id = gee_project if gee_project else GEE_PROJECT
        ee.Initialize(project=project_id)

        gdf = gpd.read_file(_DEFAULT_GEOJSON)
        roi = geemap.geopandas_to_ee(gdf)

        # 加载 ESA WorldCover 10m
        worldcover = (ee.ImageCollection(f"ESA/WorldCover/{WORLDCOVER_VERSION}")
                      .first().select("Map"))

        # 提取耕地 (类码 40)
        cropland = worldcover.eq(CROPLAND_CODE).clip(roi).rename("cropland")

        # 计算面积 (tileScale 提高以承受 10m 大区域聚合)
        area_stats = (cropland
            .multiply(ee.Image.pixelArea())
            .reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=roi.geometry(),
                scale=scale,
                maxPixels=1e13,
                bestEffort=True,
                tileScale=4,
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

    # 加载 ESA WorldCover 10m
    worldcover = (ee.ImageCollection(f"ESA/WorldCover/{WORLDCOVER_VERSION}")
                  .first().select("Map"))

    cropland = worldcover.eq(CROPLAND_CODE).clip(roi).rename("cropland")

    area_stats = (cropland
        .multiply(ee.Image.pixelArea())
        .reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=roi.geometry(),
            scale=SCALE,
            maxPixels=1e13,
            bestEffort=True,
            tileScale=4,
        ))

    area_sqm = area_stats.get("cropland").getInfo()
    area_kilo = area_sqm / 1e6
    print(f"\n{'='*45}")
    print(f"  区域: {REGION_NAME}")
    print(f"  数据: ESA WorldCover {WORLDCOVER_VERSION} (10m)")
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
            bestEffort=True,
            tileScale=4,
        ))

    total_sqm = total_area.get("area").getInfo()
    ratio = area_sqm / total_sqm * 100
    print(f"  耕地占比: {ratio:.2f}%")
    print(f"{'='*45}")

    Map = geemap.Map()

    Map.addLayer(worldcover.clip(roi), {
        "bands": ["Map"], "min": 10, "max": 100
    }, "WorldCover 土地覆盖")

    Map.addLayer(cropland.updateMask(cropland), {
        "palette": ["#FFD700"]
    }, "耕地")

    Map.addLayer(roi, {"color": "red"}, "研究区边界")
    Map.centerObject(roi, 7)
    Map
