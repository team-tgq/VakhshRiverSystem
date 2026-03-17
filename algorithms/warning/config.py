WARNING_THRESHOLDS = {
    'blue': 5.0,
    'yellow': 6.0,
    'orange': 7.0,
    'red': 8.0
}

FEATURE_COLS = [
    'area_rainfall', 'station_rainfall', 'temperature', 'humidity',
    'water_level', 'discharge', 'velocity',
    'dem', 'landuse', 'soil_moisture', 'ndvi',
    'catchment_area', 'channel_slope', 'relief'
]

TARGET_COL = 'warning_level'