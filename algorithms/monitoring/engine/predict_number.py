import os
from pathlib import Path

from ultralytics import YOLO

# 1) 模型路径
# 获取当前文件 (predict_seg.py) 所在的目录: .../algorithms/monitoring/engine/
current_dir = Path(__file__).resolve().parent

# 定位到上一级目录 (monitoring): .../algorithms/monitoring/
monitoring_dir = current_dir.parent

# 拼接成绝对路径: .../algorithms/monitoring/weights/segment_best.pt
MODEL_PATH = str(monitoring_dir / "weights" / "detect_best.pt")


def run_number_detection(source_path):
    """
    运行数字检测，供界面调用
    返回: (结果图路径, 标签txt路径)
    """
    try:
        print(f"正在加载数字识别模型: {MODEL_PATH}")
        model = YOLO(MODEL_PATH)

        # 运行预测
        results = model.predict(
            source=source_path,
            save=True,
            save_txt=True,  # 必须保存TXT才能进行下一步计算
            save_conf=True,
            conf=0.25,
            iou=0.50,
            show_boxes=True
        )

        # 获取路径
        save_dir = results[0].save_dir
        filename = os.path.basename(source_path)

        # 结果图片路径
        result_img_path = os.path.join(save_dir, filename)

        # 标签文件路径 (YOLO默认在 labels 文件夹下，后缀变为 .txt)
        txt_name = os.path.splitext(filename)[0] + ".txt"
        label_path = os.path.join(save_dir, "labels", txt_name)

        print(f"识别完成，Label路径: {label_path}")
        return result_img_path, label_path

    except Exception as e:
        print(f"数字识别出错: {e}")
        return None, None


if __name__ == "__main__":
    pass