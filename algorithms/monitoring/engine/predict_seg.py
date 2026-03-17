import os
from pathlib import Path

from ultralytics import YOLO

# 1) 模型路径
# MODEL_PATH = "../weights/segment_best.pt"
# 获取当前文件 (predict_seg.py) 所在的目录: .../algorithms/monitoring/engine/
current_dir = Path(__file__).resolve().parent

# 定位到上一级目录 (monitoring): .../algorithms/monitoring/
monitoring_dir = current_dir.parent

# 拼接成绝对路径: .../algorithms/monitoring/weights/segment_best.pt
MODEL_PATH = str(monitoring_dir / "weights" / "segment_best.pt")

def run_segmentation(source_path):
    """
    运行分割模型，供界面调用
    """
    try:
        print(f"正在加载分割模型: {MODEL_PATH}")
        model = YOLO(MODEL_PATH)

        # 运行预测
        results = model.predict(
            source=source_path,
            save=True,  # 保存结果
            save_txt=True,  # 必须保存TXT才能进行下一步计算
            save_conf=True,
            conf=0.25,  # 置信度
            iou=0.50,  # NMS
            show_boxes=False,  # 不显示检测框，只显示分割
            show_labels=True,
            show_conf=True
        )

        # 获取保存结果的路径
        # results[0].save_dir 是本次预测保存的文件夹路径
        save_dir = results[0].save_dir
        file_name = os.path.basename(source_path)
        result_path = os.path.join(save_dir, file_name)

        print(f"分割完成，保存路径: {result_path}")
        return result_path

    except Exception as e:
        print(f"分割过程出错: {e}")
        return None


if __name__ == "__main__":
    # 测试用
    pass