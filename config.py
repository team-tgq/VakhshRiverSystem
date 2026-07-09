APP_NAME = "水文系统"

PLUGIN_DIR = "plugins"

WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800

# 统一数字孪生数据根目录。
# 正式数据放在 data/瓦赫什流域孪生数据，本目录不进入 Git。
# sample_data/瓦赫什流域孪生数据 只保留为目录模板和演示样例，不能作为真实计算数据。
# 也可通过环境变量 VAKHSH_TWIN_DATA_ROOT 临时覆盖。
TWIN_DATA_ROOT = "data/瓦赫什流域孪生数据"
TWIN_DATA_ENV_VAR = "VAKHSH_TWIN_DATA_ROOT"
