import os
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUNTIME_DIR = os.path.join(BASE_DIR, "segformer_runtime")
TASK_FILE = os.path.join(BASE_DIR, "tasks.json")

# 改成你自己的 SegFormer 独立环境解释器
SEGFORMER_PYTHON = r"E:\anaconda\envs\segformer\python.exe"

with open(TASK_FILE, "r", encoding="utf-8") as f:
    _tasks = json.load(f)

TASKS = {}
for key, item in _tasks.items():
    TASKS[key] = {
        "name": item["name"],
        "config": os.path.join(BASE_DIR, item["config"]),
        "checkpoint": os.path.join(BASE_DIR, item["checkpoint"]),
        "input_dir": os.path.join(BASE_DIR, item["input_dir"]),
        "output_dir": os.path.join(BASE_DIR, item["output_dir"]),
    }
    os.makedirs(TASKS[key]["output_dir"], exist_ok=True)