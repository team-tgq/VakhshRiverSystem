import ee
import math


def process_toa_img(image, green_band, nir_band, swir_band):
    """通用的光学反射率标准化与 NDSI 计算函数"""
    green = image.select([green_band], ['Green']).toFloat()
    nir = image.select([nir_band], ['NIR']).toFloat()
    swir = image.select([swir_band], ['SWIR']).toFloat()
    ndsi = green.subtract(swir).divide(green.add(swir)).rename('NDSI').toFloat()
    return image.addBands([ndsi, green, nir]).select(['NDSI', 'Green', 'NIR'])


def generate_runoff_warning(
        target_start: str,
        target_end: str,
        sar_melt_start: str,
        sar_melt_end: str,
        sar_ref_start: str = '2022-07-05',
        sar_ref_end: str = '2022-07-30',
        bbox_coords: list = [70.0, 36.0, 76.5, 40.0],
        # ==========================================
        # 预留的数据源接口，支持甲方传入自定义数据集 ID
        # ==========================================
        dem_source: str = 'USGS/SRTMGL1_003',
        eco_source: str = 'RESOLVE/ECOREGIONS/2017',
        opt_s2_source: str = 'COPERNICUS/S2_HARMONIZED',
        opt_l8_source: str = 'LANDSAT/LC08/C02/T1_TOA',
        opt_l9_source: str = 'LANDSAT/LC09/C02/T1_TOA',
        modis_source: str = 'MODIS/061/MOD10A1',
        sar_source: str = 'COPERNICUS/S1_GRD',
        river_source: str = 'WWF/HydroSHEDS/v1/FreeFlowingRivers'
) -> tuple:
    """
    满血物理模型版：融雪径流预警系统的核心算法。
    返回: (合并后的多波段预警图层, 区域Geometry)
    """

    # 1. 全局基础空间数据
    safe_bbox = ee.Geometry.Rectangle(bbox_coords)
    dem = ee.Image(dem_source)

    # 多条件融合边界提取
    ecoregions = ee.FeatureCollection(eco_source)
    eco_boundary = ecoregions.filter(ee.Filter.eq('ECO_NAME', 'Pamir alpine desert and tundra'))
    eco_image = ee.Image.constant(0).paint(eco_boundary, 1)
    high_elevation = dem.gte(3000).clip(safe_bbox)

    combined_mask = eco_image.Or(high_elevation)
    final_pamir_vector = combined_mask.selfMask().reduceToVectors(
        geometry=safe_bbox, crs=dem.projection(), scale=500,
        geometryType='polygon', eightConnected=True, maxPixels=1e10
    )
    roi = final_pamir_vector.geometry()

    # 2. 静态 AHP 地形因子构建
    slope = ee.Terrain.slope(dem)
    factor_slope = slope.divide(45).clamp(0, 1)

    rivers = ee.FeatureCollection(river_source).filterBounds(roi)
    river_img = ee.Image.constant(1).paint(rivers, 0)
    dist_to_river = river_img.fastDistanceTransform().multiply(ee.Image.pixelArea().sqrt())
    factor_dist = ee.Image(1).subtract(dist_to_river.divide(5000)).clamp(0, 1)

    aspect = ee.Terrain.aspect(dem)
    aspect_rad = aspect.subtract(180).multiply(math.pi).divide(180)
    factor_aspect = aspect_rad.cos().add(1).divide(2)

    # 3. 光学 SNOMAP 完整约束
    s2_col = (ee.ImageCollection(opt_s2_source)
              .filterBounds(roi).filterDate(target_start, target_end)
              .map(lambda img: process_toa_img(img.divide(10000), 'B3', 'B8', 'B11')))
    l8_col = (ee.ImageCollection(opt_l8_source)
              .filterBounds(roi).filterDate(target_start, target_end)
              .map(lambda img: process_toa_img(img, 'B3', 'B5', 'B6')))
    l9_col = (ee.ImageCollection(opt_l9_source)
              .filterBounds(roi).filterDate(target_start, target_end)
              .map(lambda img: process_toa_img(img, 'B3', 'B5', 'B6')))

    high_res_img = s2_col.merge(l8_col).merge(l9_col).qualityMosaic('NDSI').clip(roi)

    snomap_mask = (high_res_img.select('NDSI').gte(0.40)
                   .And(high_res_img.select('NIR').gte(0.11))
                   .And(high_res_img.select('Green').gte(0.10))
                   .selfMask().rename('Snow_Mask'))

    modis_col = ee.ImageCollection(modis_source).filterBounds(roi).filterDate(target_start, target_end).select(
        'NDSI_Snow_Cover')
    modis_mask = modis_col.max().gte(40).And(modis_col.max().lte(100)).selfMask().rename('Snow_Mask').toFloat().clip(
        roi)
    total_snow_area = ee.ImageCollection([modis_mask, snomap_mask]).mosaic().clip(roi)

    # 4. 微波局部入射角 (LIA) 权重与 Sigmoid 概率软阈值模型
    def process_sar_wet_snow(orbit_pass):
        s1_col = (ee.ImageCollection(sar_source)
                  .filterBounds(roi)
                  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
                  .filter(ee.Filter.eq('instrumentMode', 'IW'))
                  .filter(ee.Filter.eq('orbitProperties_pass', orbit_pass)))

        ref_img = s1_col.filterDate(sar_ref_start, sar_ref_end).mean().focal_median(radius=30, kernelType='circle',
                                                                                    units='meters')
        melt_img = s1_col.filterDate(sar_melt_start, sar_melt_end).mean().focal_median(radius=30, kernelType='circle',
                                                                                       units='meters')

        noise_mask = ref_img.select('VV').gt(-20).And(ref_img.select('VH').gt(-24))

        az = 78.0 if orbit_pass == 'ASCENDING' else 282.0
        az_rad = ee.Number(az).multiply(math.pi).divide(180)
        slope_rad = ee.Terrain.slope(dem).multiply(math.pi).divide(180)
        asp_rad = ee.Terrain.aspect(dem).multiply(math.pi).divide(180)
        inc_rad = melt_img.select('angle').multiply(math.pi).divide(180)

        cos_lia = slope_rad.cos().multiply(inc_rad.cos()).add(
            slope_rad.sin().multiply(inc_rad.sin()).multiply(ee.Image.constant(az_rad).subtract(asp_rad).cos())
        )
        lia = cos_lia.acos().multiply(180).divide(math.pi)
        lia_mask = lia.gte(18).And(lia.lte(78))

        delta_vv = melt_img.select('VV').subtract(ref_img.select('VV'))
        delta_vh = melt_img.select('VH').subtract(ref_img.select('VH'))

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

    asc_wet = process_sar_wet_snow('ASCENDING')
    desc_wet = process_sar_wet_snow('DESCENDING')
    sar_wet_signal = asc_wet.unmask(0).Or(desc_wet.unmask(0)).selfMask()

    final_wet_snow = total_snow_area.And(sar_wet_signal).selfMask()
    final_dry_snow = total_snow_area.And(final_wet_snow.unmask(0).Not()).selfMask()

    # 5. 动态地形热力学校正
    shady_mask = aspect.gte(315).Or(aspect.lt(45))
    semi_shady_mask = (aspect.gte(45).And(aspect.lt(90))).Or(aspect.gte(270).And(aspect.lt(315)))
    semi_sunny_mask = (aspect.gte(90).And(aspect.lt(135))).Or(aspect.gte(225).And(aspect.lt(270)))
    sunny_mask = aspect.gte(135).And(aspect.lt(225))

    def get_safe_mean_elev(aspect_mask):
        wet_dem = dem.updateMask(final_wet_snow).updateMask(aspect_mask)
        stats = wet_dem.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=roi, scale=3000,
            maxPixels=1e13, tileScale=16, bestEffort=True
        )
        mean_elev = stats.get('elevation')
        return ee.Number(ee.Algorithms.If(ee.Algorithms.IsEqual(mean_elev, None), 8000, mean_elev))

    elevation_threshold_img = (ee.Image(8000)
                               .where(shady_mask, ee.Image.constant(get_safe_mean_elev(shady_mask)))
                               .where(semi_shady_mask, ee.Image.constant(get_safe_mean_elev(semi_shady_mask)))
                               .where(semi_sunny_mask, ee.Image.constant(get_safe_mean_elev(semi_sunny_mask)))
                               .where(sunny_mask, ee.Image.constant(get_safe_mean_elev(sunny_mask))))

    to_correct_to_wet = final_dry_snow.unmask(0).And(dem.lt(elevation_threshold_img))
    corrected_wet_snow = final_wet_snow.unmask(0).Or(to_correct_to_wet).selfMask()
    corrected_dry_snow = final_dry_snow.unmask(0).And(to_correct_to_wet.Not()).selfMask()

    # 6. AHP 概率汇总
    weight_dist, weight_slope, weight_aspect = 0.54, 0.30, 0.16
    runoff_probability = (ee.Image(0)
                          .add(factor_dist.multiply(weight_dist))
                          .add(factor_slope.multiply(weight_slope))
                          .add(factor_aspect.multiply(weight_aspect))
                          .multiply(100))

    final_runoff_potential = runoff_probability.updateMask(corrected_wet_snow.unmask(0))

    # ==========================================
    # 7. 渲染与合成（修复了数据类型不一致导致的导出失败报错）
    # ==========================================
    post_snow_state_map = ee.Image.constant(1).clip(roi)
    post_snow_state_map = post_snow_state_map.where(corrected_dry_snow.unmask(0), 2)
    post_snow_state_map = post_snow_state_map.where(corrected_wet_snow.unmask(0), 3)

    # 💡 关键修复点：使用 .toFloat() 强行将两者的波段数据类型对齐为 Float32
    final_product = ee.Image([
        post_snow_state_map.rename('Snow_State').toFloat(),
        final_runoff_potential.rename('Runoff_Probability').toFloat()
    ])

    return final_product, roi