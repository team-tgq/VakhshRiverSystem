import cv2
import os
import numpy as np


def find_non_transparent_bottom_below_point(image_path, point):
    """查找指定点下方的非透明区域底部"""
    img_with_alpha = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img_with_alpha is None:
        return 0
    x, y = point
    height, width = img_with_alpha.shape[:2]
    if img_with_alpha.shape[2] == 4:
        alpha_channel = img_with_alpha[:, :, 3]
        bottom_y = y
        for scan_y in range(y, height):
            if alpha_channel[scan_y, x] > 0:
                bottom_y = scan_y
        return bottom_y
    else:
        return height - 1


def advanced_draw_center(image_path, label_path, output_path=None):
    """在图像上标注中心点坐标并计算水位"""
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        return [], None

    img_height, img_width = img.shape[:2]
    original_img = img.copy()  # 保持原始图像大小

    centers = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            class_id = int(parts[0])
            x_center = int(float(parts[1]) * img_width)
            y_center = int(float(parts[2]) * img_height)
            centers.append({
                'center': (x_center, y_center),
                'class_id': class_id,
                'index': i + 1
            })

    water_level = None

    # 标注所有中心点
    for info in centers:
        x, y = info['center']

        # 绘制中心点
        cv2.circle(original_img, (x, y), 4, (0, 0, 255), -1)  # 红色实心点
        cv2.circle(original_img, (x, y), 6, (255, 0, 0), 1)  # 蓝色外圈

        # 标注坐标
        coord_text = f"({x},{y})"
        class_text = f"ID:{info['class_id']}"

        # 设置字体
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4

        # 计算文本大小
        (coord_w, coord_h), _ = cv2.getTextSize(coord_text, font, font_scale, 1)
        (class_w, class_h), _ = cv2.getTextSize(class_text, font, font_scale, 1)

        # 绘制坐标文本背景
        cv2.rectangle(original_img,
                      (x + 8, y - coord_h - 4),
                      (x + 8 + max(coord_w, class_w), y + 4),
                      (255, 255, 255), -1)
        cv2.rectangle(original_img,
                      (x + 8, y - coord_h - 4),
                      (x + 8 + max(coord_w, class_w), y + 4),
                      (0, 0, 0), 1)

        # 绘制坐标文本
        cv2.putText(original_img, coord_text,
                    (x + 10, y - 2), font, font_scale, (0, 0, 0), 1)
        cv2.putText(original_img, class_text,
                    (x + 10, y - 8), font, font_scale, (0, 0, 0), 1)

    # 如果至少有两个中心点，计算水位并绘制相关线
    if len(centers) >= 2:
        sorted_centers = sorted(centers, key=lambda x: x['class_id'])
        min_pt = sorted_centers[0]
        sec_pt = sorted_centers[1]

        p1 = min_pt['center']
        p2 = sec_pt['center']

        # 绘制连接两个点的线
        cv2.line(original_img, p1, p2, (0, 255, 0), 2)  # 绿色线
        y_distance = abs(p1[1] - p2[1])

        # 查找最低点下方的底部
        min_point_bottom_y = find_non_transparent_bottom_below_point(image_path, p1)
        min_to_bottom = min_point_bottom_y - p1[1]

        if y_distance > 0:
            water_level = (min_pt['class_id'] + 1.25) - (min_to_bottom / y_distance)

            # 绘制从最低点到水底的线（紫色）
            cv2.line(original_img, p1, (p1[0], min_point_bottom_y),
                     (255, 0, 255), 2)

    # 在图像底部添加统计信息
    info_y = img_height - 20
    cv2.putText(original_img, f"Total Points: {len(centers)}",
                (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    if water_level is not None:
        cv2.putText(original_img, f"Water Level: {water_level:.2f} dm",
                    (img_width - 200, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 1)

    # 保存结果
    if output_path:
        cv2.imwrite(output_path, original_img)

    return centers, water_level


def run_water_level_calculation(image_path, label_path, output_dir="final_result"):
    """主程序调用接口"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, f"final_{filename}")

    try:
        centers, water_level = advanced_draw_center(image_path, label_path, output_path)
        return output_path, water_level
    except Exception as e:
        print(f"计算脚本出错: {e}")
        return None, None