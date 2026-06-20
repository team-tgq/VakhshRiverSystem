import ee
from pamir_snow_engine import generate_runoff_warning

# 1. 业务系统初始化 GEE 环境（已为你绑定好 Project ID）
try:
    ee.Initialize(project='northern-window-485210-b3')
    print("GEE 环境初始化成功！")
except Exception as e:
    print("未检测到有效授权或项目未绑定，正在拉取浏览器授权...")
    ee.Authenticate()
    ee.Initialize(project='northern-window-485210-b3')

# 2. 模拟从外部业务系统传来的参数
request_payload = {
    'target_start': '2023-05-10',
    'target_end': '2023-05-15',
    'sar_melt_start': '2023-05-05',
    'sar_melt_end': '2023-05-20',
    'sar_ref_start': '2022-07-05',
    'sar_ref_end': '2022-07-30',
    'bbox_coords': [70.0, 36.0, 76.5, 40.0]
}

print("系统接收到外部请求，正在构建物理模型...")

# 3. 调用核心引擎（参数已完全对齐）
combined_output_img, calculation_roi = generate_runoff_warning(**request_payload)

# 4. 后端处理：提交异步导出任务
task_name = f"Pamir_Runoff_Warning_{request_payload['target_start'].replace('-', '')}"
print(f"核心模型构建完毕，正在提交异步导出任务: {task_name}")

export_task = ee.batch.Export.image.toDrive(
    image=combined_output_img,
    description=task_name,
    folder='Pamir_Warning_System_Outputs',
    scale=30,
    region=calculation_roi,  # 💡 修复点：直接传入 ee.Geometry 对象，拒绝 .getInfo() 导致的深度解析错误
    maxPixels=1e13,
    fileFormat='GeoTIFF',
    formatOptions={'cloudOptimized': True}
)

export_task.start()
print("数据下发指令已成功提交至 GEE 云端后台！请在 Google Drive 或 GEE Task 列表中查看任务进度。")