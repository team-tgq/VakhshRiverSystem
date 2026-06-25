"""
遥感处理子模块
- FTW 耕地面积提取 (Sentinel-2 L2A)
- GEE 耕地面积查询
"""
from .ftw_model import (
    create_ftw_model,
    calculate_cropland_area,
    load_geojson_mask,
    FTW_BAND_INDICES,
    FTW_NORM_SCALE,
    LEGACY_BAND_INDICES,
    process_multiple_images,
    find_tiff_files,
)
from .gee_service import get_cropland_area_km2

__all__ = [
    "create_ftw_model",
    "calculate_cropland_area",
    "load_geojson_mask",
    "FTW_BAND_INDICES",
    "FTW_NORM_SCALE",
    "LEGACY_BAND_INDICES",
    "process_multiple_images",
    "find_tiff_files",
    "get_cropland_area_km2",
]
