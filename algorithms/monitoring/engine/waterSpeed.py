# backend.py
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt


def calculate_velocity_for_ui(video_path, height_m=4.0, fov_degrees=60.0, fps=30.0):
    """
    接收视频路径，执行光流法测速，保存可视化结果，并返回计算状态和流速。
    """
    cap = cv2.VideoCapture(video_path)
    ret1, frame1 = cap.read()
    ret2, frame2 = cap.read()
    cap.release()

    if not ret1 or not ret2:
        return False, "视频读取失败", None

    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    frame_width = frame1.shape[1]
    meters_per_pixel = (2 * height_m * math.tan(math.radians(fov_degrees / 2))) / frame_width

    feature_params = dict(maxCorners=500, qualityLevel=0.05, minDistance=15, blockSize=7)
    p0 = cv2.goodFeaturesToTrack(gray1, mask=None, **feature_params)

    if p0 is None:
        return False, "未检测到有效水面特征", None

    lk_params = dict(winSize=(21, 21), maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    p1, st, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, p0, None, **lk_params)

    good_new = p1[st == 1]
    good_old = p0[st == 1]

    dx = good_new[:, 0] - good_old[:, 0]
    dy = good_new[:, 1] - good_old[:, 1]
    distances = np.sqrt(dx ** 2 + dy ** 2)
    angles = np.mod(np.degrees(np.arctan2(dy, dx)), 360)

    dist_mask = distances > 0.2
    angles_filtered = angles[dist_mask]

    if len(angles_filtered) == 0:
        return False, "过滤后无有效流动特征", None

    median_angle = np.median(angles_filtered)
    angle_diffs = np.abs(angles_filtered - median_angle)
    angle_diffs = np.minimum(angle_diffs, 360 - angle_diffs)
    angle_mask = angle_diffs < 20.0

    final_old = good_old[dist_mask][angle_mask]
    final_new = good_new[dist_mask][angle_mask]
    final_angles = angles_filtered[angle_mask]
    final_distances = distances[dist_mask][angle_mask]

    if len(final_distances) == 0:
        return False, "清洗后特征点不足", None

    # 定义保存路径
    img_paths = [
        "data/waterSpeed/Frames_1-2_1_red_features.jpg",
        "data/waterSpeed/Frames_1-2_2_blue_features.jpg",
        "data/waterSpeed/Frames_1-2_3_black_trajectories.jpg",
        "data/waterSpeed/Frames_1-2_4_direction_rose.png"
    ]

    # 图1：红点
    vis1 = frame1.copy()
    for pt in final_old:
        x, y = pt.ravel()
        cv2.circle(vis1, (int(x), int(y)), 4, (0, 0, 255), -1)
    cv2.imwrite(img_paths[0], vis1)

    # 图2：蓝点
    vis2 = frame2.copy()
    for pt in final_new:
        x, y = pt.ravel()
        cv2.circle(vis2, (int(x), int(y)), 4, (255, 0, 0), -1)
    cv2.imwrite(img_paths[1], vis2)

    # 图3：黑底轨迹
    black_bg = np.zeros_like(frame1)
    for (new, old) in zip(final_new, final_old):
        a, b = new.ravel()
        c, d = old.ravel()
        cv2.line(black_bg, (int(c), int(d)), (int(a), int(b)), (0, 255, 0), 2)
        cv2.circle(black_bg, (int(c), int(d)), 3, (0, 0, 255), -1)
        cv2.circle(black_bg, (int(a), int(b)), 3, (255, 0, 0), -1)
    cv2.imwrite(img_paths[2], black_bg)

    # 图4：玫瑰图
    radians = np.deg2rad(final_angles)
    bins = np.linspace(0.0, 2 * np.pi, 37)
    counts, _ = np.histogram(radians, bins=bins)
    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={'projection': 'polar'})
    ax.bar(bins[:-1], counts, width=np.deg2rad(10), bottom=0.0, color='#4169E1', alpha=0.8, edgecolor='black')
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(-1)
    ax.set_title("Motion Rose Diagram", va='bottom', fontsize=12)
    plt.savefig(img_paths[3], dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 计算流速
    velocity_m_s = (np.mean(final_distances) * meters_per_pixel) / (1.0 / fps)

    return True, velocity_m_s, img_paths