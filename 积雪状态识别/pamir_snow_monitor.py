import ee


def generate_snow_state(
        target_start: str,
        target_end: str,
        ref_start: str = '2022-07-15',
        ref_end: str = '2022-08-15',
        bbox_coords: list = [70.0, 36.0, 76.5, 40.0],
        # 下方为预留的数据源接口，支持甲方传入自定义数据集 ID
        dem_source: str = 'USGS/SRTMGL1_003',
        eco_source: str = 'RESOLVE/ECOREGIONS/2017',
        opt_source: str = 'COPERNICUS/S2_SR_HARMONIZED',
        lst_source: str = 'MODIS/061/MOD11A1',
        sar_source: str = 'COPERNICUS/S1_GRD'
) -> ee.Image:
    """
    根据给定的时间窗口和空间范围，生成积雪物理状态分类图层。

    返回:
        ee.Image: 单波段的分类结果图像（像素值 1-4），可以直接用于导出或发布地图服务。
    """

    # 1. 基础空间边界定义
    safe_bbox = ee.Geometry.Rectangle(bbox_coords)

    # 2. 获取生命生态边界与高程
    ecoregions = ee.FeatureCollection(eco_source)
    eco_boundary = ecoregions.filter(ee.Filter.eq('ECO_NAME', 'Pamir alpine desert and tundra'))
    eco_image = ee.Image.constant(0).paint(eco_boundary, 1)

    dem = ee.Image(dem_source)
    high_elevation = dem.gte(3000).clip(safe_bbox)

    # 3. 核心：空间并集栅格掩膜
    pamir_raster_mask = eco_image.Or(high_elevation).clip(safe_bbox)

    # 4. 局部地形计算
    local_dem = dem.updateMask(pamir_raster_mask).clip(safe_bbox)
    slope = ee.Terrain.slope(local_dem)
    aspect = ee.Terrain.aspect(local_dem)

    is_valid_terrain = slope.lt(45)
    is_sunny_slope = aspect.gt(90).And(aspect.lt(270))
    is_shady_slope = is_sunny_slope.Not()

    # 5. 光学 (NDSI) 计算
    def add_ndsi(image):
        return image.addBands(image.normalizedDifference(['B3', 'B11']).rename('NDSI'))

    target_ndsi = (ee.ImageCollection(opt_source)
                   .filterBounds(safe_bbox)
                   .filterDate(target_start, target_end)
                   .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50))
                   .map(add_ndsi)
                   .select('NDSI')
                   .median()
                   .updateMask(pamir_raster_mask)
                   .clip(safe_bbox))

    is_snow_covered = target_ndsi.gte(0.6)

    # 6. 热力环境 (LST) 计算
    target_lst = (ee.ImageCollection(lst_source)
                  .filterBounds(safe_bbox)
                  .filterDate(target_start, target_end)
                  .select('LST_Day_1km')
                  .mean()
                  .multiply(0.02).subtract(273.15)
                  .updateMask(pamir_raster_mask)
                  .clip(safe_bbox)
                  .resample('bilinear'))

    is_warm = target_lst.gte(0)

    # 7. 微波 (SAR) 计算
    s1 = (ee.ImageCollection(sar_source)
          .filterBounds(safe_bbox)
          .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
          .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
          .filter(ee.Filter.eq('instrumentMode', 'IW')))

    def to_linear(image):
        return ee.Image(10).pow(image.divide(10))

    def to_db(image):
        return ee.Image(10).multiply(image.log10())

    def get_stable_sar(start_date, end_date):
        collection = s1.filterDate(start_date, end_date)
        linear_mean = collection.map(to_linear).mean()
        db_mean = to_db(linear_mean)
        return (db_mean.select('VV').add(db_mean.select('VH')).divide(2)
                .updateMask(pamir_raster_mask).clip(safe_bbox))

    ref_sar = get_stable_sar(ref_start, ref_end)
    target_sar = get_stable_sar(target_start, target_end)

    # 初步湿雪提取
    initial_wet_snow = (target_sar.subtract(ref_sar).lt(-6)
                        .focal_median(radius=1.5, kernelType='circle', units='pixels')
                        .And(is_valid_terrain))

    # 8. 大尺度高程修正 (防崩溃机制)
    # 阳坡平均湿雪海拔
    mean_elev_sunny_dict = local_dem.updateMask(initial_wet_snow.And(is_sunny_slope)).reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=safe_bbox,
        scale=500,
        maxPixels=1e13,
        tileScale=16
    )
    val_sunny = mean_elev_sunny_dict.get('elevation')
    # 捕获 null 值，赋予极高海拔防止误判
    mean_elev_sunny = ee.Number(ee.Algorithms.If(ee.Algorithms.IsEqual(val_sunny, None), 8000, val_sunny))

    # 阴坡平均湿雪海拔
    mean_elev_shady_dict = local_dem.updateMask(initial_wet_snow.And(is_shady_slope)).reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=safe_bbox,
        scale=500,
        maxPixels=1e13,
        tileScale=16
    )
    val_shady = mean_elev_shady_dict.get('elevation')
    mean_elev_shady = ee.Number(ee.Algorithms.If(ee.Algorithms.IsEqual(val_shady, None), 8000, val_shady))

    # 强制修正
    force_wet_sunny = is_snow_covered.And(is_warm).And(is_sunny_slope).And(
        local_dem.lt(ee.Image.constant(mean_elev_sunny)))
    force_wet_shady = is_snow_covered.And(is_warm).And(is_shady_slope).And(
        local_dem.lt(ee.Image.constant(mean_elev_shady)))
    final_wet_snow = initial_wet_snow.Or(force_wet_sunny).Or(force_wet_shady)

    # 9. 状态制图封装
    state_image = ee.Image(0).updateMask(pamir_raster_mask).clip(safe_bbox)

    # 按物理规则渲染像素值
    state_image = state_image.where(is_snow_covered.Not(), 1)
    state_image = state_image.where(is_snow_covered.And(final_wet_snow.Not()).And(is_warm.Not()), 2)
    state_image = state_image.where(is_snow_covered.And(final_wet_snow.Not()).And(is_warm), 3)
    state_image = state_image.where(final_wet_snow, 4)

    # 最终掩膜光学云遮挡区域
    state_image = state_image.updateMask(target_ndsi.mask())

    return state_image