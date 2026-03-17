import os
import sys
import json
import time
import argparse
import traceback
from pathlib import Path

import cv2
import mmcv
import numpy as np
import torch

from mmseg.apis import init_segmentor, inference_segmentor

CURRENT_DIR = Path(__file__).resolve().parent

# 强制优先使用当前 segformer_runtime 目录下的本地 mmseg
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
from mmseg.apis import init_segmentor, inference_segmentor

def parse_args():
    parser = argparse.ArgumentParser(description='SegFormer old-env inference service')
    parser.add_argument('--image', required=True, help='输入图像路径')
    parser.add_argument('--config', required=True, help='配置文件路径')
    parser.add_argument('--checkpoint', required=True, help='权重文件路径')
    parser.add_argument('--overlay-out', required=True, help='叠加图输出路径')
    parser.add_argument('--mask-out', required=True, help='掩码输出路径')
    parser.add_argument('--meta-out', required=True, help='JSON元数据输出路径')
    parser.add_argument('--device', default='cuda:0', help='推理设备，例如 cuda:0 / cpu')
    parser.add_argument('--task', default='water', help='任务名，如 water / snow')
    return parser.parse_args()


def ensure_parent_dir(path_str: str):
    Path(path_str).parent.mkdir(parents=True, exist_ok=True)


def check_exists(path_str: str, name: str):
    if not Path(path_str).exists():
        raise FileNotFoundError(f'{name}不存在: {path_str}')


def normalize_device(device_str: str) -> str:
    device_str = str(device_str).strip().lower()
    if device_str == 'cpu':
        return 'cpu'
    if device_str.startswith('cuda'):
        if torch.cuda.is_available():
            return device_str
        print('[WARN] CUDA 不可用，自动切换到 CPU')
        return 'cpu'
    return 'cpu'


def get_palette_and_classes(task: str):
    task = str(task).lower()
    if task == 'snow':
        classes = ('background', 'snow')
        palette = [[0, 0, 0], [0, 255, 255]]
    else:
        classes = ('background', 'water')
        palette = [[0, 0, 0], [0, 0, 255]]
    return classes, palette


def save_mask(seg_map: np.ndarray, mask_out: str):
    ensure_parent_dir(mask_out)

    unique_vals = np.unique(seg_map)
    if set(unique_vals.tolist()).issubset({0, 1}):
        mask = (seg_map * 255).astype(np.uint8)
    else:
        mask = seg_map.astype(np.uint8)

    ok = cv2.imwrite(mask_out, mask)
    if not ok:
        raise RuntimeError(f'掩码保存失败: {mask_out}')


def save_overlay(image_path: str, seg_map: np.ndarray, overlay_out: str, task: str, alpha: float = 0.5):
    ensure_parent_dir(overlay_out)

    img = mmcv.imread(image_path)
    if img is None:
        raise RuntimeError(f'原图读取失败: {image_path}')

    classes, palette = get_palette_and_classes(task)

    color_mask = np.zeros_like(img, dtype=np.uint8)

    for class_id, color in enumerate(palette):
        color_mask[seg_map == class_id] = color

    overlay = cv2.addWeighted(img, 1 - alpha, color_mask, alpha, 0)

    ok = cv2.imwrite(overlay_out, overlay)
    if not ok:
        raise RuntimeError(f'叠加图保存失败: {overlay_out}')


def save_meta(meta_out: str, payload: dict):
    ensure_parent_dir(meta_out)
    with open(meta_out, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def print_env_info():
    print('[*] ===== 环境信息 =====')
    print(f'[*] Python: {sys.version}')
    print(f'[*] Torch: {torch.__version__}')
    print(f'[*] CUDA version: {torch.version.cuda}')
    print(f'[*] CUDA available: {torch.cuda.is_available()}')
    print(f'[*] MMCV: {mmcv.__version__}')
    try:
        import mmseg
        print(f'[*] MMSeg: {mmseg.__version__}')
    except Exception:
        pass
    if torch.cuda.is_available():
        try:
            print(f'[*] GPU: {torch.cuda.get_device_name(0)}')
        except Exception:
            pass
    print('[*] ====================')


def main():
    args = parse_args()
    start_time = time.time()

    try:
        print_env_info()

        check_exists(args.image, '输入图像')
        check_exists(args.config, '配置文件')
        check_exists(args.checkpoint, '权重文件')

        device = normalize_device(args.device)

        print(f'[*] 当前任务: {args.task}')
        print(f'[*] 输入图像: {args.image}')
        print(f'[*] 配置文件: {args.config}')
        print(f'[*] 权重文件: {args.checkpoint}')
        print(f'[*] 推理设备: {device}')

        print('[*] 正在初始化旧版 MMSeg 模型...')
        model = init_segmentor(args.config, args.checkpoint, device=device)

        print('[*] 正在执行推理...')
        result = inference_segmentor(model, args.image)

        if not isinstance(result, (list, tuple)) or len(result) == 0:
            raise RuntimeError(f'推理结果格式异常: {type(result)}')

        seg_map = result[0]
        if not isinstance(seg_map, np.ndarray):
            seg_map = np.array(seg_map)

        seg_map = seg_map.astype(np.uint8)

        print(f'[*] 分割图 shape: {seg_map.shape}')
        print(f'[*] 分割图 unique values: {np.unique(seg_map).tolist()}')

        print('[*] 正在保存掩码...')
        save_mask(seg_map, args.mask_out)
        print(f'[+] 掩码保存成功: {args.mask_out}')

        print('[*] 正在保存叠加图...')
        save_overlay(args.image, seg_map, args.overlay_out, args.task, alpha=0.5)
        print(f'[+] 叠加图保存成功: {args.overlay_out}')

        elapsed = round(time.time() - start_time, 3)

        meta = {
            'status': 'success',
            'task': args.task,
            'device': device,
            'image': args.image,
            'config': args.config,
            'checkpoint': args.checkpoint,
            'mask_out': args.mask_out,
            'overlay_out': args.overlay_out,
            'shape': list(seg_map.shape),
            'unique_values': np.unique(seg_map).tolist(),
            'elapsed_seconds': elapsed
        }
        save_meta(args.meta_out, meta)
        print(f'[+] 元数据保存成功: {args.meta_out}')
        print(f'[SUCCESS] 推理完成，用时 {elapsed} 秒')

        sys.exit(0)

    except Exception as e:
        elapsed = round(time.time() - start_time, 3)
        err_msg = str(e)

        print('\n' + '=' * 60)
        print('[CRITICAL ERROR] 推理过程崩溃:')
        print(err_msg)
        print(traceback.format_exc())
        print('=' * 60)

        try:
            fail_meta = {
                'status': 'failed',
                'task': args.task,
                'device': args.device,
                'image': args.image,
                'config': args.config,
                'checkpoint': args.checkpoint,
                'error': err_msg,
                'traceback': traceback.format_exc(),
                'elapsed_seconds': elapsed
            }
            save_meta(args.meta_out, fail_meta)
        except Exception:
            pass

        sys.exit(1)


if __name__ == '__main__':
    main()