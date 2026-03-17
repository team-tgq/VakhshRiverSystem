import cv2
import numpy as np
import os


def get_dominant_angle(lines):
    if lines is None:
        return 0
    angle_groups = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        if abs(angle) > 45:
            norm_angle = angle - 90 if angle > 0 else angle + 90
        else:
            norm_angle = angle
        angle_groups.append(norm_angle)
    if not angle_groups:
        return 0
    angle_groups.sort()
    clusters = []
    if len(angle_groups) > 0:
        current_cluster = [angle_groups[0]]
        for i in range(1, len(angle_groups)):
            if angle_groups[i] - current_cluster[-1] < 5:
                current_cluster.append(angle_groups[i])
            else:
                clusters.append(current_cluster)
                current_cluster = [angle_groups[i]]
        clusters.append(current_cluster)
    best_cluster = max(clusters, key=len)
    return np.mean(best_cluster)


def straighten_water_gauge(image):
    if image is None:
        return None, 0
    if len(image.shape) == 3 and image.shape[2] == 4:
        bgr = image[:, :, :3]
        alpha = image[:, :, 3]
    else:
        bgr = image
        alpha = np.ones(image.shape[:2], dtype=np.uint8) * 255
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 30, 120)
    edges = cv2.bitwise_and(edges, edges, mask=alpha)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40, minLineLength=30, maxLineGap=15)
    correction_angle = get_dominant_angle(lines)
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, correction_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    def get_crop_and_size(img):
        tmp_alpha = img[:, :, 3] if len(img.shape) == 3 and img.shape[2] == 4 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        coords = cv2.findNonZero(tmp_alpha)
        if coords is None:
            return img, 0, 0
        x, y, cw, ch = cv2.boundingRect(coords)
        return img[y:y + ch, x:x + cw], cw, ch

    final_img, cw, ch = get_crop_and_size(rotated)
    if cw > ch:
        final_img = cv2.rotate(final_img, cv2.ROTATE_90_CLOCKWISE)
        final_img, _, _ = get_crop_and_size(final_img)
    return final_img, correction_angle


def clahe_enhance_with_alpha(image, clip_limit=4.0, grid_size=8):
    if image is None: return None
    if len(image.shape) == 3 and image.shape[2] == 4:
        bgr = image[:, :, :3]
        alpha = image[:, :, 3]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
        enhanced_gray = clahe.apply(gray)
        enhanced_bgr = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
        enhanced_img = np.dstack((enhanced_bgr, alpha))
    elif len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
        enhanced_gray = clahe.apply(gray)
        enhanced_img = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
    else:
        enhanced_img = image.copy()
    return enhanced_img


def denoise_image_with_alpha(image, method='bilateral', kernel_size=3):
    if image is None: return None
    has_alpha = len(image.shape) == 3 and image.shape[2] == 4
    if has_alpha:
        bgr = image[:, :, :3]
        alpha = image[:, :, 3]
        if method == 'bilateral':
            denoised_bgr = cv2.bilateralFilter(bgr, kernel_size, 75, 75)
        else:
            denoised_bgr = cv2.bilateralFilter(bgr, kernel_size, 75, 75)
        denoised_img = np.dstack((denoised_bgr, alpha))
    else:
        denoised_img = cv2.bilateralFilter(image, kernel_size, 75, 75)
    return denoised_img


def save_image_with_alpha(image, output_path):
    if image is None: return False
    try:
        cv2.imwrite(output_path, image)
        return True
    except Exception as e:
        print(f"Save error: {e}")
        return False


# === 供主程序调用的接口函数 ===
def run_preprocessing_pipeline(image_path, output_dir="data/waterleve_temp_processed"):
    """
    供GUI调用：执行旋正、增强、去噪
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"正在读取图片进行预处理: {image_path}")
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print("图片读取失败")
        return None, None, None

    filename = os.path.basename(image_path)

    # 1. 旋正
    straightened_img, angle = straighten_water_gauge(img)
    path_straight = os.path.join(output_dir, f"straight_{filename}")
    save_image_with_alpha(straightened_img, path_straight)

    # 2. 增强
    enhanced_img = clahe_enhance_with_alpha(straightened_img, clip_limit=4.0)
    path_enhanced = os.path.join(output_dir, f"enhanced_{filename}")
    save_image_with_alpha(enhanced_img, path_enhanced)

    # 3. 去噪
    denoised_img = denoise_image_with_alpha(enhanced_img, method='bilateral')
    path_denoised = os.path.join(output_dir, f"denoised_{filename}")
    save_image_with_alpha(denoised_img, path_denoised)

    return path_straight, path_enhanced, path_denoised