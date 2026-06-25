import os
import numpy as np
import torch
import segmentation_models_pytorch as smp
from rasterio.windows import Window
import rasterio

# Sentinel-2 L2A 波段配置 & FTW 预处理参数
FTW_NORM_SCALE = 3000.0
FTW_BAND_INDICES = [1, 2, 3, 4]
LEGACY_BAND_INDICES = [3, 2, 1, 4]

# 权重文件默认路径
from pathlib import Path
_DEFAULT_WEIGHTS = str(Path(__file__).resolve().parent.parent / "resources" / "models" / "3_Class_FULL_FTW_Pretrained_v2.ckpt")


def create_ftw_model(
    num_classes=2,
    encoder_name='efficientnet-b3',
    in_channels=8,
    pretrained_weights_path=None,
    device='cuda'
):
    """创建 FTW 语义分割模型 (U-Net + EfficientNet 骨干)"""
    if pretrained_weights_path is None:
        pretrained_weights_path = _DEFAULT_WEIGHTS

    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=in_channels,
        classes=num_classes,
    )

    if pretrained_weights_path and os.path.exists(pretrained_weights_path):
        state_dict = torch.load(pretrained_weights_path, map_location=device, weights_only=True)

        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        cleaned = {}
        skip_keys = {'criterion.weight', 'train_loss', 'val_loss', 'loss_fn.weight'}
        for k, v in state_dict.items():
            if k in skip_keys:
                continue
            key = k.replace('model.', '') if k.startswith('model.') else k
            cleaned[key] = v

        head_weight = cleaned.get('segmentation_head.0.weight')
        if head_weight is not None:
            ckpt_classes = head_weight.shape[0]
            if ckpt_classes != num_classes:
                print(f"注意: 权重文件为 {ckpt_classes}-class, 传入参数为 {num_classes}-class")
                print(f"      自动按 {ckpt_classes}-class 重新构建模型")
                num_classes = ckpt_classes
                model = smp.Unet(
                    encoder_name=encoder_name,
                    encoder_weights=None,
                    in_channels=in_channels,
                    classes=num_classes,
                )

        model.load_state_dict(cleaned, strict=False)
        print(f"已加载预训练权重: {pretrained_weights_path} ({num_classes}-class)")
    else:
        print("警告: 未提供预训练权重路径或文件不存在, 使用随机初始化参数")

    model.to(device)
    model.eval()
    return model, num_classes


def preprocess_s2_l2a(data):
    """预处理 Sentinel-2 L2A 影像数据, 自动处理 4/8 波段"""
    data = data.astype(np.float32) / FTW_NORM_SCALE
    num_bands = data.shape[0]

    if num_bands == 8:
        return torch.from_numpy(data).float(), True
    elif num_bands == 4:
        data = np.concatenate([data, data], axis=0)
        return torch.from_numpy(data).float(), False
    else:
        raise ValueError(f"不支持的波段数: {num_bands}, 期望 4 或 8")


def compute_ndvi_mask(data_8ch, threshold=0.3):
    """从 8 通道 FTW 输入计算 NDVI 植被掩膜"""
    sr = data_8ch * FTW_NORM_SCALE

    b4_a = sr[0].astype(np.float64)
    b8_a = sr[3].astype(np.float64)
    b4_b = sr[4].astype(np.float64)
    b8_b = sr[7].astype(np.float64)

    eps = 1e-6
    ndvi_a = (b8_a - b4_a) / (b8_a + b4_a + eps)
    ndvi_b = (b8_b - b4_b) / (b8_b + b4_b + eps)

    ndvi_max = np.maximum(ndvi_a, ndvi_b)
    valid = (sr[0] > 0) & (sr[0] != -9999 / FTW_NORM_SCALE)
    vegetation_mask = (ndvi_max >= threshold) & valid
    return vegetation_mask


def _pad_to_multiple(tensor, multiple=32):
    """将 (1, C, H, W) tensor 的 H/W 填充到 multiple 的整数倍"""
    _, _, h, w = tensor.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return tensor, (0, 0, 0, 0)
    padded = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode='reflect')
    return padded, (0, pad_w, 0, pad_h)


def _crop_padding(tensor, padding):
    """移除填充, 恢复到原始 H/W"""
    pad_left, pad_right, pad_top, pad_bottom = padding
    if pad_right == 0 and pad_bottom == 0:
        return tensor
    h, w = tensor.shape[-2], tensor.shape[-1]
    return tensor[..., pad_top:h - pad_bottom, pad_left:w - pad_right]


def _compute_pixel_area_sqm(src):
    """计算单像素面积 (m²), 自动处理地理/投影坐标系"""
    transform = src.transform
    pixel_width = abs(transform[0])
    pixel_height = abs(transform[4])

    if src.crs and src.crs.is_geographic:
        bounds = src.bounds
        center_lat = (bounds.top + bounds.bottom) / 2.0
        lat_rad = np.radians(center_lat)
        meters_per_deg_lat = 111320.0
        meters_per_deg_lon = 111320.0 * np.cos(lat_rad)
        pixel_area_sqm = (pixel_width * meters_per_deg_lon) * (pixel_height * meters_per_deg_lat)
        print(f"检测到地理坐标系 ({src.crs.to_string()}), "
              f"中心纬度 {center_lat:.4f}°, 转换后像素面积: {pixel_area_sqm:.2f} m²")
    else:
        pixel_area_sqm = pixel_width * pixel_height

    return pixel_area_sqm


def load_geojson_mask(geojson_path, target_crs=None):
    """加载 GeoJSON 并返回 shapely 几何对象用于空间过滤"""
    import json
    from shapely.geometry import shape
    from shapely.ops import unary_union

    with open(geojson_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    geojson_crs = None
    if 'crs' in data:
        crs_props = data['crs'].get('properties', {})
        geojson_crs = crs_props.get('name', None)

    polygons = []
    for feature in data.get('features', []):
        geom = feature.get('geometry')
        if geom and geom.get('type') in ('Polygon', 'MultiPolygon'):
            polygons.append(shape(geom))

    if not polygons:
        raise ValueError("GeoJSON 中未找到 Polygon/MultiPolygon 几何")

    geom = unary_union(polygons)
    print(f"已加载 GeoJSON 掩膜: {geojson_path} ({len(polygons)} 个要素, "
          f"CRS: {geojson_crs or '未指定, 假设 EPSG:4326'})")
    return geom, geojson_crs


def _window_bounds(src, win):
    """计算 rasterio 窗口的地理边界"""
    win_transform = src.window_transform(win)
    left = win_transform.c
    top = win_transform.f
    right = left + win.width * win_transform.a
    bottom = top + win.height * win_transform.e
    return left, bottom, right, top


def _rasterize_window_mask(src, win, polygon, all_touched=True):
    """在指定窗口内栅格化多边形, 返回 (H, W) 布尔掩膜"""
    from rasterio.features import rasterize
    from shapely.geometry import box

    win_bounds = _window_bounds(src, win)
    win_box = box(*win_bounds)

    if not polygon.intersects(win_box):
        return None

    mask = rasterize(
        [(polygon, 1)],
        out_shape=(win.height, win.width),
        transform=src.window_transform(win),
        fill=0,
        all_touched=all_touched,
        dtype='uint8',
    )
    return mask.astype(bool)


def calculate_cropland_area(
    image_path,
    model,
    device='cuda',
    window_size=1024,
    overlap=64,
    num_classes=2,
    band_indices=None,
    mask_geometry=None,
    ndvi_threshold=0.3,
):
    """使用滑动窗口在大型遥感影像上进行 FTW 推理, 计算耕地面积"""
    if band_indices is None:
        band_indices = FTW_BAND_INDICES
    if window_size % 32 != 0:
        window_size = ((window_size + 31) // 32) * 32
        print(f"窗口大小已调整为 32 的倍数: {window_size}")

    model.eval()
    total_field_pixels = 0

    with rasterio.open(image_path) as src:
        pixel_area_sqm = _compute_pixel_area_sqm(src)

        width = src.width
        height = src.height
        num_bands = src.count
        is_8band = (num_bands >= 8)

        dual_label = "真实双日期" if is_8band else "伪双时相 (单日期复制)"
        ndvi_label = f"NDVI≥{ndvi_threshold}" if ndvi_threshold > 0 else "禁用"
        print(f"影像尺寸: {width} × {height} 像素")
        print(f"波段数: {num_bands} | 模式: {dual_label} | 单像素面积: {pixel_area_sqm:.2f} m²")
        print(f"NDVI 过滤: {ndvi_label} | 窗口: {window_size} | 重叠: {overlap}px")

        stride = window_size - overlap
        total_windows = ((height - 1) // stride + 1) * ((width - 1) // stride + 1)
        window_count = 0

        for row in range(0, height, stride):
            for col in range(0, width, stride):
                win_w = min(window_size, width - col)
                win_h = min(window_size, height - row)
                win = Window(col, row, win_w, win_h)

                if is_8band:
                    data = src.read([1, 2, 3, 4, 5, 6, 7, 8], window=win)
                else:
                    data = src.read(band_indices, window=win)

                if np.all(data == 0) or np.all(data == -9999):
                    window_count += 1
                    continue

                geo_mask = None
                if mask_geometry is not None:
                    geo_mask = _rasterize_window_mask(src, win, mask_geometry)
                    if geo_mask is None or not geo_mask.any():
                        window_count += 1
                        continue

                tensor_data, _ = preprocess_s2_l2a(data)
                tensor_data = tensor_data.unsqueeze(0).to(device)

                ndvi_mask = None
                if ndvi_threshold > 0:
                    ndvi_mask = compute_ndvi_mask(data, threshold=ndvi_threshold)

                tensor_padded, padding = _pad_to_multiple(tensor_data, multiple=32)

                with torch.no_grad():
                    pred = model(tensor_padded)
                    pred = _crop_padding(pred, padding)

                    if num_classes == 2:
                        probs = torch.sigmoid(pred[:, 1:2, :, :]).squeeze(0).squeeze(0)
                        field_mask = (probs > 0.5)
                    else:
                        field_mask = (torch.argmax(pred, dim=1).squeeze(0) == 1)

                    if geo_mask is not None:
                        field_mask = field_mask & torch.from_numpy(geo_mask).to(field_mask.device)
                    if ndvi_mask is not None:
                        field_mask = field_mask & torch.from_numpy(ndvi_mask).to(field_mask.device)

                    field_pixels = field_mask.sum().item()
                    total_field_pixels += field_pixels

                window_count += 1
                if window_count % 50 == 0:
                    pct = window_count / total_windows * 100
                    print(f"进度: {window_count}/{total_windows} ({pct:.1f}%)  "
                          f"当前耕地像素: {total_field_pixels:,}", end='\r')

    print(f"\n推理完成! 共处理 {window_count} 个窗口")
    print(f"耕地总像素数: {total_field_pixels:,}")

    area_ha = (total_field_pixels * pixel_area_sqm) / 10000.0
    print(f"耕地总面积: {area_ha:.2f} 公顷 ({area_ha / 100:.2f} km²)")

    return {
        'area_hectares': area_ha,
        'field_pixels': total_field_pixels,
        'pixel_area_sqm': pixel_area_sqm,
    }


def process_multiple_images(image_paths, model, device, window_size, overlap, num_classes,
                           band_indices=None, mask_geometry=None, ndvi_threshold=0.3):
    """批量处理多个 TIFF 文件, 累加耕地面积"""
    total_area_ha = 0.0
    total_field_pixels = 0
    per_file_results = []

    print(f"\n{'='*60}")
    print(f"批量处理 {len(image_paths)} 个影像文件")
    print(f"{'='*60}")

    for i, path in enumerate(image_paths, 1):
        print(f"\n[{i}/{len(image_paths)}] 处理: {os.path.basename(path)}")
        try:
            result = calculate_cropland_area(
                image_path=path,
                model=model,
                device=device,
                window_size=window_size,
                overlap=overlap,
                num_classes=num_classes,
                band_indices=band_indices,
                mask_geometry=mask_geometry,
                ndvi_threshold=ndvi_threshold,
            )
            total_area_ha += result['area_hectares']
            total_field_pixels += result['field_pixels']
            per_file_results.append({'file': path, **result})
        except Exception as e:
            print(f"  错误: {e}, 跳过此文件")
            import traceback
            traceback.print_exc()

    return {
        'total_area_hectares': total_area_ha,
        'total_field_pixels': total_field_pixels,
        'per_file': per_file_results,
    }


def find_tiff_files(paths_or_dirs, pattern='*.tif*'):
    """从路径列表/目录中收集所有 TIFF 文件"""
    import glob as glob_mod
    tiff_files = []
    for p in paths_or_dirs:
        if os.path.isfile(p):
            if p.lower().endswith(('.tif', '.tiff')):
                tiff_files.append(p)
        elif os.path.isdir(p):
            matched = glob_mod.glob(os.path.join(p, pattern))
            tiff_files.extend(matched)
        else:
            print(f"警告: 路径不存在, 跳过: {p}")
    return sorted(set(tiff_files))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='FTW 耕地面积提取 — Sentinel-2 L2A 10m 影像 (支持多文件)'
    )
    parser.add_argument('--image', type=str, nargs='+', required=True,
                        help='Sentinel-2 L2A GeoTIFF 路径或目录 (支持多个, 空格分隔)')
    parser.add_argument('--pattern', type=str, default='*.tif*')
    parser.add_argument('--weights', type=str, default=None,
                        help='FTW 预训练权重路径')
    parser.add_argument('--num-classes', type=int, default=3, choices=[2, 3])
    parser.add_argument('--encoder', type=str, default='efficientnet-b3',
                        choices=['efficientnet-b3', 'efficientnet-b5', 'efficientnet-b7'])
    parser.add_argument('--window-size', type=int, default=1024)
    parser.add_argument('--overlap', type=int, default=64)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--legacy-band-order', action='store_true')
    parser.add_argument('--mask', type=str, default=None)
    parser.add_argument('--ndvi-threshold', type=float, default=0.3)

    args = parser.parse_args()

    band_indices = LEGACY_BAND_INDICES if args.legacy_band_order else FTW_BAND_INDICES
    print(f"波段读取索引: {band_indices}")

    mask_geometry = None
    if args.mask:
        mask_geometry, _ = load_geojson_mask(args.mask)
        print(f"将仅统计 GeoJSON 多边形内的耕地像素")

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"计算设备: {device}")

    tiff_files = find_tiff_files(args.image, pattern=args.pattern)
    if not tiff_files:
        print("错误: 未找到任何 TIFF 文件")
        exit(1)

    print(f"找到 {len(tiff_files)} 个 TIFF 文件:")
    for f in tiff_files:
        print(f"  - {f}")

    print(f"\n创建 FTW 模型: U-Net + {args.encoder}, {args.num_classes}-class")
    model, actual_num_classes = create_ftw_model(
        num_classes=args.num_classes,
        encoder_name=args.encoder,
        in_channels=8,
        pretrained_weights_path=args.weights,
        device=device,
    )

    result = process_multiple_images(
        image_paths=tiff_files,
        model=model,
        device=device,
        window_size=args.window_size,
        overlap=args.overlap,
        num_classes=actual_num_classes,
        band_indices=band_indices,
        mask_geometry=mask_geometry,
        ndvi_threshold=args.ndvi_threshold,
    )

    print(f"\n{'='*60}")
    print(f"=== 汇总结果 ===")
    print(f"{'='*60}")
    for r in result['per_file']:
        print(f"  {os.path.basename(r['file'])}: {r['area_hectares']:.2f} 公顷 "
              f"({r['field_pixels']:,} 像素)")
    print(f"  ---")
    print(f"  耕地总面积: {result['total_area_hectares']:.2f} 公顷")
    print(f"            = {result['total_area_hectares'] / 100:.2f} 平方公里")
    print(f"  耕地总像素: {result['total_field_pixels']:,}")
