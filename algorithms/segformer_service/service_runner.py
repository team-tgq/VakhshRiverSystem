# algorithms/segformer_service/service_runner.py
import os
import subprocess
from .service_config import TASKS, RUNTIME_DIR, SEGFORMER_PYTHON


def run_segformer_service(task_key: str, image_path: str, device: str = "cuda:0"):
    if task_key not in TASKS:
        raise ValueError(f"未知任务: {task_key}")

    task = TASKS[task_key]

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"未找到输入图像: {image_path}")
    if not os.path.exists(task["config"]):
        raise FileNotFoundError(f"未找到配置文件: {task['config']}")
    if not os.path.exists(task["checkpoint"]):
        raise FileNotFoundError(f"未找到模型权重: {task['checkpoint']}")

    infer_script = os.path.join(RUNTIME_DIR, "infer_service.py")
    if not os.path.exists(infer_script):
        raise FileNotFoundError(f"未找到推理脚本: {infer_script}")

    image_name = os.path.splitext(os.path.basename(image_path))[0]
    result_image = os.path.join(task["output_dir"], f"{image_name}_{task_key}_overlay.png")
    result_mask = os.path.join(task["output_dir"], f"{image_name}_{task_key}_mask.png")
    result_json = os.path.join(task["output_dir"], f"{image_name}_{task_key}_meta.json")

    cmd = [
        SEGFORMER_PYTHON,
        infer_script,
        "--image", image_path,
        "--config", task["config"],
        "--checkpoint", task["checkpoint"],
        "--overlay-out", result_image,
        "--mask-out", result_mask,
        "--meta-out", result_json,
        "--device", device,
        "--task", task_key,
    ]

    result = subprocess.run(
        cmd,
        cwd=RUNTIME_DIR,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )

    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "overlay_path": result_image,
        "mask_path": result_mask,
        "meta_path": result_json,
        "task_name": task["name"],
        "command": cmd,
    }