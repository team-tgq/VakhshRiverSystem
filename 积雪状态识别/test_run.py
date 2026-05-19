import ee
from pamir_snow_monitor import generate_snow_state

# 1. 师兄系统的系统初始化 (需自行配置认证凭据)
try:
    ee.Initialize()
except Exception as e:
    ee.Authenticate()
    ee.Initialize()

# 2. 模拟前端传来的业务参数
# 假设用户在网页上选择了 2023年3月中旬，并框选了帕米尔区域
USER_TARGET_START = '2023-03-10'
USER_TARGET_END = '2023-03-15'
USER_BBOX = [70.0, 36.0, 76.5, 40.0]

print("正在构建积雪状态监测物理引擎...")

# 3. 调用核心算法库
# 返回的是一个处于 GEE 云端的 ee.Image 对象
snow_state_img = generate_snow_state(
    target_start=USER_TARGET_START,
    target_end=USER_TARGET_END,
    bbox_coords=USER_BBOX
)

# 4. 后端处理：提交导出任务
task_name = f'Pamir_Snow_State_{USER_TARGET_START.replace("-", "")}'
print(f"正在向 GEE 提交导出任务: {task_name}")

export_task = ee.batch.Export.image.toDrive(
    image=snow_state_img,
    description=task_name,
    folder='Pamir_Snow_System_Output',
    scale=30,
    region=ee.Geometry.Rectangle(USER_BBOX).getInfo()['coordinates'],
    maxPixels=1e13,
    fileFormat='GeoTIFF',
    formatOptions={'cloudOptimized': True}
)

export_task.start()

print("任务已提交！请在 Google Drive 中查看结果。")