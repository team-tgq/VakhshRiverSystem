from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, List, Tuple


DEFAULT_BBOX = [70.0, 36.0, 76.5, 40.0]
DEFAULT_PROJECT_ID = "northern-window-485210-b3"
DEFAULT_DRIVE_FOLDER = "Pamir_Warning_System_Outputs"
DEFAULT_TASK_PREFIX = "Pamir_Runoff_Warning"

DEFAULT_SOURCES = {
    "dem_source": "USGS/SRTMGL1_003",
    "eco_source": "RESOLVE/ECOREGIONS/2017",
    "opt_s2_source": "COPERNICUS/S2_HARMONIZED",
    "opt_l8_source": "LANDSAT/LC08/C02/T1_TOA",
    "opt_l9_source": "LANDSAT/LC09/C02/T1_TOA",
    "modis_source": "MODIS/061/MOD10A1",
    "sar_source": "COPERNICUS/S1_GRD",
    "river_source": "WWF/HydroSHEDS/v1/FreeFlowingRivers",
}

STATE_LABELS = {
    1: "无雪/裸地",
    2: "干雪/稳定积雪",
    3: "湿雪/融雪活跃区",
}

BAND_DESCRIPTIONS = {
    "Snow_State": "积雪状态分类，1=无雪/裸地，2=干雪，3=湿雪",
    "Runoff_Probability": "湿雪区融雪径流发生概率，0-100",
}


def _load_ee():
    try:
        import ee
    except ImportError as exc:
        raise RuntimeError(
            "未安装 earthengine-api，无法运行积雪状态识别模块。"
            "请先执行: pip install earthengine-api"
        ) from exc
    return ee


def validate_bbox_coords(bbox_coords: Iterable[float]) -> List[float]:
    coords = [float(value) for value in bbox_coords]
    if len(coords) != 4:
        raise ValueError("范围坐标必须包含 4 个值: west,south,east,north")

    west, south, east, north = coords
    if west >= east:
        raise ValueError("范围坐标不合法: west 必须小于 east")
    if south >= north:
        raise ValueError("范围坐标不合法: south 必须小于 north")

    return coords


def parse_bbox_text(bbox_text: str) -> List[float]:
    parts = [part.strip() for part in bbox_text.split(",") if part.strip()]
    if len(parts) != 4:
        raise ValueError("请输入 4 个逗号分隔的范围值，例如: 70.0, 36.0, 76.5, 40.0")
    return validate_bbox_coords(parts)


def ensure_earth_engine(
    authenticate: bool = False,
    project_id: str | None = DEFAULT_PROJECT_ID,
) -> str:
    ee = _load_ee()
    safe_project_id = (project_id or "").strip() or None
    try:
        if safe_project_id:
            ee.Initialize(project=safe_project_id)
        else:
            ee.Initialize()
        return "Google Earth Engine 已初始化。"
    except Exception as exc:
        if not authenticate:
            raise RuntimeError(
                "Google Earth Engine 尚未初始化。"
                "请先点击“初始化 GEE”完成认证，或在命令行中先执行 Earth Engine 登录。"
            ) from exc

    try:
        ee.Authenticate()
        if safe_project_id:
            ee.Initialize(project=safe_project_id)
        else:
            ee.Initialize()
    except Exception as auth_exc:
        raise RuntimeError(
            "Google Earth Engine 认证失败，请检查网络、账号权限和 Project ID。"
        ) from auth_exc

    return "Google Earth Engine 认证并初始化成功。"


def process_toa_img(image, green_band: str, nir_band: str, swir_band: str):
    green = image.select([green_band], ["Green"]).toFloat()
    nir = image.select([nir_band], ["NIR"]).toFloat()
    swir = image.select([swir_band], ["SWIR"]).toFloat()
    ndsi = green.subtract(swir).divide(green.add(swir)).rename("NDSI").toFloat()
    return image.addBands([ndsi, green, nir]).select(["NDSI", "Green", "NIR"])


def generate_runoff_warning(
    target_start: str,
    target_end: str,
    sar_melt_start: str,
    sar_melt_end: str,
    sar_ref_start: str = "2022-07-05",
    sar_ref_end: str = "2022-07-30",
    bbox_coords: List[float] | None = None,
    dem_source: str = DEFAULT_SOURCES["dem_source"],
    eco_source: str = DEFAULT_SOURCES["eco_source"],
    opt_s2_source: str = DEFAULT_SOURCES["opt_s2_source"],
    opt_l8_source: str = DEFAULT_SOURCES["opt_l8_source"],
    opt_l9_source: str = DEFAULT_SOURCES["opt_l9_source"],
    modis_source: str = DEFAULT_SOURCES["modis_source"],
    sar_source: str = DEFAULT_SOURCES["sar_source"],
    river_source: str = DEFAULT_SOURCES["river_source"],
) -> Tuple[Any, Any]:
    """Build the Pamir snow-state and snowmelt-runoff probability product."""
    ee = _load_ee()

    safe_bbox = ee.Geometry.Rectangle(validate_bbox_coords(bbox_coords or DEFAULT_BBOX))
    dem = ee.Image(dem_source)

    ecoregions = ee.FeatureCollection(eco_source)
    eco_boundary = ecoregions.filter(
        ee.Filter.eq("ECO_NAME", "Pamir alpine desert and tundra")
    )
    eco_image = ee.Image.constant(0).paint(eco_boundary, 1)
    high_elevation = dem.gte(3000).clip(safe_bbox)

    combined_mask = eco_image.Or(high_elevation)
    final_pamir_vector = combined_mask.selfMask().reduceToVectors(
        geometry=safe_bbox,
        crs=dem.projection(),
        scale=500,
        geometryType="polygon",
        eightConnected=True,
        maxPixels=1e10,
    )
    roi = final_pamir_vector.geometry()

    slope = ee.Terrain.slope(dem)
    factor_slope = slope.divide(45).clamp(0, 1)

    rivers = ee.FeatureCollection(river_source).filterBounds(roi)
    river_img = ee.Image.constant(1).paint(rivers, 0)
    dist_to_river = river_img.fastDistanceTransform().multiply(
        ee.Image.pixelArea().sqrt()
    )
    factor_dist = ee.Image(1).subtract(dist_to_river.divide(5000)).clamp(0, 1)

    aspect = ee.Terrain.aspect(dem)
    aspect_rad = aspect.subtract(180).multiply(math.pi).divide(180)
    factor_aspect = aspect_rad.cos().add(1).divide(2)

    s2_col = (
        ee.ImageCollection(opt_s2_source)
        .filterBounds(roi)
        .filterDate(target_start, target_end)
        .map(lambda img: process_toa_img(img.divide(10000), "B3", "B8", "B11"))
    )
    l8_col = (
        ee.ImageCollection(opt_l8_source)
        .filterBounds(roi)
        .filterDate(target_start, target_end)
        .map(lambda img: process_toa_img(img, "B3", "B5", "B6"))
    )
    l9_col = (
        ee.ImageCollection(opt_l9_source)
        .filterBounds(roi)
        .filterDate(target_start, target_end)
        .map(lambda img: process_toa_img(img, "B3", "B5", "B6"))
    )

    high_res_img = s2_col.merge(l8_col).merge(l9_col).qualityMosaic("NDSI").clip(roi)
    snomap_mask = (
        high_res_img.select("NDSI")
        .gte(0.40)
        .And(high_res_img.select("NIR").gte(0.11))
        .And(high_res_img.select("Green").gte(0.10))
        .selfMask()
        .rename("Snow_Mask")
    )

    modis_col = (
        ee.ImageCollection(modis_source)
        .filterBounds(roi)
        .filterDate(target_start, target_end)
        .select("NDSI_Snow_Cover")
    )
    modis_mask = (
        modis_col.max()
        .gte(40)
        .And(modis_col.max().lte(100))
        .selfMask()
        .rename("Snow_Mask")
        .toFloat()
        .clip(roi)
    )
    total_snow_area = ee.ImageCollection([modis_mask, snomap_mask]).mosaic().clip(roi)

    def process_sar_wet_snow(orbit_pass):
        s1_col = (
            ee.ImageCollection(sar_source)
            .filterBounds(roi)
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.eq("orbitProperties_pass", orbit_pass))
        )

        ref_img = (
            s1_col.filterDate(sar_ref_start, sar_ref_end)
            .mean()
            .focal_median(radius=30, kernelType="circle", units="meters")
        )
        melt_img = (
            s1_col.filterDate(sar_melt_start, sar_melt_end)
            .mean()
            .focal_median(radius=30, kernelType="circle", units="meters")
        )

        noise_mask = ref_img.select("VV").gt(-20).And(ref_img.select("VH").gt(-24))

        az = 78.0 if orbit_pass == "ASCENDING" else 282.0
        az_rad = ee.Number(az).multiply(math.pi).divide(180)
        slope_rad = ee.Terrain.slope(dem).multiply(math.pi).divide(180)
        asp_rad = ee.Terrain.aspect(dem).multiply(math.pi).divide(180)
        inc_rad = melt_img.select("angle").multiply(math.pi).divide(180)

        cos_lia = slope_rad.cos().multiply(inc_rad.cos()).add(
            slope_rad.sin()
            .multiply(inc_rad.sin())
            .multiply(ee.Image.constant(az_rad).subtract(asp_rad).cos())
        )
        lia = cos_lia.acos().multiply(180).divide(math.pi)
        lia_mask = lia.gte(18).And(lia.lte(78))

        delta_vv = melt_img.select("VV").subtract(ref_img.select("VV"))
        delta_vh = melt_img.select("VH").subtract(ref_img.select("VH"))

        weight_vh = lia.subtract(18).divide(78 - 18).clamp(0, 1)
        weight_vv = ee.Image.constant(1).subtract(weight_vh)
        rc = delta_vh.multiply(weight_vh).add(delta_vv.multiply(weight_vv))

        k_factor = 2.0
        rc_centered = rc.subtract(-2.0)
        wet_prob_sigmoid = ee.Image(1).divide(
            ee.Image(1).add(rc_centered.multiply(k_factor).exp())
        )
        is_wet = wet_prob_sigmoid.gte(0.5)

        return is_wet.updateMask(lia_mask).updateMask(noise_mask).selfMask()

    asc_wet = process_sar_wet_snow("ASCENDING")
    desc_wet = process_sar_wet_snow("DESCENDING")
    sar_wet_signal = asc_wet.unmask(0).Or(desc_wet.unmask(0)).selfMask()

    final_wet_snow = total_snow_area.And(sar_wet_signal).selfMask()
    final_dry_snow = total_snow_area.And(final_wet_snow.unmask(0).Not()).selfMask()

    shady_mask = aspect.gte(315).Or(aspect.lt(45))
    semi_shady_mask = aspect.gte(45).And(aspect.lt(90)).Or(
        aspect.gte(270).And(aspect.lt(315))
    )
    semi_sunny_mask = aspect.gte(90).And(aspect.lt(135)).Or(
        aspect.gte(225).And(aspect.lt(270))
    )
    sunny_mask = aspect.gte(135).And(aspect.lt(225))

    def get_safe_mean_elev(aspect_mask):
        wet_dem = dem.updateMask(final_wet_snow).updateMask(aspect_mask)
        stats = wet_dem.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=roi,
            scale=3000,
            maxPixels=1e13,
            tileScale=16,
            bestEffort=True,
        )
        mean_elev = stats.get("elevation")
        return ee.Number(
            ee.Algorithms.If(ee.Algorithms.IsEqual(mean_elev, None), 8000, mean_elev)
        )

    elevation_threshold_img = (
        ee.Image(8000)
        .where(shady_mask, ee.Image.constant(get_safe_mean_elev(shady_mask)))
        .where(semi_shady_mask, ee.Image.constant(get_safe_mean_elev(semi_shady_mask)))
        .where(semi_sunny_mask, ee.Image.constant(get_safe_mean_elev(semi_sunny_mask)))
        .where(sunny_mask, ee.Image.constant(get_safe_mean_elev(sunny_mask)))
    )

    to_correct_to_wet = final_dry_snow.unmask(0).And(dem.lt(elevation_threshold_img))
    corrected_wet_snow = final_wet_snow.unmask(0).Or(to_correct_to_wet).selfMask()
    corrected_dry_snow = final_dry_snow.unmask(0).And(
        to_correct_to_wet.Not()
    ).selfMask()

    weight_dist, weight_slope, weight_aspect = 0.54, 0.30, 0.16
    runoff_probability = (
        ee.Image(0)
        .add(factor_dist.multiply(weight_dist))
        .add(factor_slope.multiply(weight_slope))
        .add(factor_aspect.multiply(weight_aspect))
        .multiply(100)
    )
    final_runoff_potential = runoff_probability.updateMask(corrected_wet_snow.unmask(0))

    post_snow_state_map = ee.Image.constant(1).clip(roi)
    post_snow_state_map = post_snow_state_map.where(corrected_dry_snow.unmask(0), 2)
    post_snow_state_map = post_snow_state_map.where(corrected_wet_snow.unmask(0), 3)

    final_product = ee.Image(
        [
            post_snow_state_map.rename("Snow_State").toFloat(),
            final_runoff_potential.rename("Runoff_Probability").toFloat(),
        ]
    )

    return final_product, roi


def _build_task_name(task_prefix: str, target_start: str) -> str:
    raw_prefix = task_prefix.strip() or DEFAULT_TASK_PREFIX
    safe_prefix = re.sub(r"[^A-Za-z0-9_-]+", "_", raw_prefix).strip("_")
    safe_prefix = safe_prefix or DEFAULT_TASK_PREFIX
    start_token = target_start.replace("-", "")
    return f"{safe_prefix}_{start_token}"[:100]


def submit_runoff_warning_export(
    target_start: str,
    target_end: str,
    sar_melt_start: str,
    sar_melt_end: str,
    sar_ref_start: str = "2022-07-05",
    sar_ref_end: str = "2022-07-30",
    bbox_coords: List[float] | None = None,
    drive_folder: str = DEFAULT_DRIVE_FOLDER,
    task_prefix: str = DEFAULT_TASK_PREFIX,
    scale: int = 30,
    authenticate: bool = False,
    project_id: str | None = DEFAULT_PROJECT_ID,
    dem_source: str = DEFAULT_SOURCES["dem_source"],
    eco_source: str = DEFAULT_SOURCES["eco_source"],
    opt_s2_source: str = DEFAULT_SOURCES["opt_s2_source"],
    opt_l8_source: str = DEFAULT_SOURCES["opt_l8_source"],
    opt_l9_source: str = DEFAULT_SOURCES["opt_l9_source"],
    modis_source: str = DEFAULT_SOURCES["modis_source"],
    sar_source: str = DEFAULT_SOURCES["sar_source"],
    river_source: str = DEFAULT_SOURCES["river_source"],
) -> Dict[str, Any]:
    ee = _load_ee()
    ensure_earth_engine(authenticate=authenticate, project_id=project_id)

    safe_bbox = validate_bbox_coords(bbox_coords or DEFAULT_BBOX)
    safe_drive_folder = drive_folder.strip() or DEFAULT_DRIVE_FOLDER
    if scale <= 0:
        raise ValueError("导出分辨率必须为正整数")

    final_product, roi = generate_runoff_warning(
        target_start=target_start,
        target_end=target_end,
        sar_melt_start=sar_melt_start,
        sar_melt_end=sar_melt_end,
        sar_ref_start=sar_ref_start,
        sar_ref_end=sar_ref_end,
        bbox_coords=safe_bbox,
        dem_source=dem_source,
        eco_source=eco_source,
        opt_s2_source=opt_s2_source,
        opt_l8_source=opt_l8_source,
        opt_l9_source=opt_l9_source,
        modis_source=modis_source,
        sar_source=sar_source,
        river_source=river_source,
    )

    description = _build_task_name(task_prefix, target_start)
    export_task = ee.batch.Export.image.toDrive(
        image=final_product,
        description=description,
        folder=safe_drive_folder,
        scale=scale,
        region=roi,
        maxPixels=1e13,
        fileFormat="GeoTIFF",
        formatOptions={"cloudOptimized": True},
    )
    export_task.start()

    status = {}
    try:
        status = export_task.status()
    except Exception:
        status = {}

    return {
        "description": description,
        "drive_folder": safe_drive_folder,
        "scale": scale,
        "bbox": safe_bbox,
        "task_id": status.get("id"),
        "task_state": status.get("state", "SUBMITTED"),
        "state_legend": STATE_LABELS.copy(),
        "bands": BAND_DESCRIPTIONS.copy(),
    }

