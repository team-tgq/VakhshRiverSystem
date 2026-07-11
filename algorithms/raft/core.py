"""
RAFT (Recurrent All-Pairs Field Transforms) optical flow analysis for river surface velocity measurement.

This module provides pure algorithm functions for loading the RAFT model and running
optical flow analysis on video frame pairs to estimate surface flow velocity.
All functions are independent of any GUI framework.
"""

import os
import math
import csv
from pathlib import Path
import numpy as np
import torch
import cv2

from algorithms.raft.raft_model import RAFT
from algorithms.raft.utils.utils import InputPadder
from algorithms.raft.utils import flow_viz


DEFAULT_MODEL_PATH = Path(__file__).with_name("raft-sintel.pth")


class Args:
    """Simple namespace for RAFT model configuration parameters."""

    def __init__(self, small=False, mixed_precision=False, alternate_corr=False, dropout=0.0):
        self.small = small
        self.mixed_precision = mixed_precision
        self.alternate_corr = alternate_corr
        self.dropout = dropout

    def __contains__(self, key):
        return hasattr(self, key)


def load_raft_model(model_path, device="cpu"):
    """Load a pretrained RAFT model from a checkpoint file.

    Args:
        model_path: Path to the .pth checkpoint file.
        device: Torch device string ("cpu" or "cuda").

    Returns:
        Tuple of (RAFT model, Args namespace).
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"RAFT model checkpoint not found: {model_path}")

    args = Args()
    model = RAFT(args)
    state_dict = torch.load(model_path, map_location=device)
    # Handle legacy "module." prefix from DataParallel
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    return model, args


def _prepare_image(frame, device="cpu"):
    """Convert an OpenCV BGR frame to a normalized RAFT-compatible tensor."""
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
    return img_tensor.unsqueeze(0).to(device)


def _process_raft_pair(model, frame1, frame2, device="cpu", iters=20):
    """Run RAFT dense optical flow on a single frame pair.

    Args:
        model: Loaded RAFT model in eval mode.
        frame1: First frame (numpy array, BGR).
        frame2: Second frame (numpy array, BGR).
        device: Torch device.
        iters: Number of RAFT iterations.

    Returns:
        Dict with keys: velocity (m/s), pts_count, flow_rgb (numpy array for viz),
        valid_angles (degrees), valid_pixel_distances, old_pts, new_pts.
        Returns None if no valid flow points found.
    """
    img1 = _prepare_image(frame1, device)
    img2 = _prepare_image(frame2, device)

    padder = InputPadder(img1.shape)
    img1_pad, img2_pad = padder.pad(img1, img2)

    with torch.no_grad():
        _, flow_up = model(img1_pad, img2_pad, iters=iters, test_mode=True)

    flow_up = padder.unpad(flow_up)
    flow_np = flow_up[0].permute(1, 2, 0).cpu().numpy()
    flow_rgb = flow_viz.flow_to_image(flow_np)

    flow_u = flow_np[..., 0]
    flow_v = flow_np[..., 1]
    pixel_distances = np.sqrt(flow_u ** 2 + flow_v ** 2)
    angles = np.mod(np.degrees(np.arctan2(flow_v, flow_u)), 360)

    dist_mask = pixel_distances > 0.2
    valid_angles_all = angles[dist_mask]

    if len(valid_angles_all) == 0:
        return None

    median_angle = np.median(valid_angles_all)
    angle_diffs = np.abs(angles - median_angle)
    angle_diffs = np.minimum(angle_diffs, 360 - angle_diffs)
    angle_mask = angle_diffs < 45.0
    final_mask = dist_mask & angle_mask
    valid_count = int(np.sum(final_mask))

    if valid_count == 0:
        return None

    return {
        "flow_rgb": flow_rgb,
        "pixel_distances": pixel_distances[final_mask],
        "valid_angles": angles[final_mask],
        "valid_count": valid_count,
        "flow_u": flow_u[final_mask],
        "flow_v": flow_v[final_mask],
        "full_flow_np": flow_np,
    }


def _calculate_physical_velocity(pixel_distances, frame_shape, height_m, fov_deg, tilt_deg, fps):
    """Convert pixel displacements to physical velocity in m/s.

    Uses pinhole camera model with known mounting height, FOV, and tilt angle.

    Args:
        pixel_distances: Array of pixel displacement magnitudes.
        frame_shape: Tuple (height, width) of the frame.
        height_m: Camera mounting height in meters.
        fov_deg: Camera horizontal field of view in degrees.
        tilt_deg: Camera tilt angle (0=horizontal, 90=straight down) in degrees.
        fps: Video frame rate.

    Returns:
        Mean physical velocity in m/s.
    """
    frame_height, frame_width = frame_shape[:2]
    focal_length = (frame_width / 2.0) / math.tan(math.radians(fov_deg / 2.0))
    pitch_rad = math.radians(tilt_deg)

    # Use the center pixel for a representative pixel-to-meter scale
    center_y = frame_height / 2.0
    y_offset = center_y
    alpha_y = np.arctan(y_offset / focal_length)
    gamma = pitch_rad - alpha_y
    gamma = max(gamma, 0.05)
    Z = height_m / np.tan(gamma)

    # Scale: at distance Z, each pixel subtends Z/focal_length meters
    meters_per_pixel = Z / focal_length
    avg_pixel_dist = float(np.mean(pixel_distances))
    velocity_m_s = avg_pixel_dist * meters_per_pixel / (1.0 / fps)
    return velocity_m_s


def run_raft_analysis(
    video_path,
    height_m=4.0,
    fov_deg=60.0,
    tilt_deg=35.0,
    start_frame=2,
    total_frames=10,
    model_path=None,
    device=None,
    progress_callback=None,
):
    """Run RAFT optical flow analysis on a video for surface velocity measurement.

    This is the main algorithm entry point. It is GUI-agnostic and can be called
    from any context (plugin widget, script, background thread).

    Args:
        video_path: Path to the input video file (mp4/avi/mov).
        height_m: Camera mounting height in meters.
        fov_deg: Camera horizontal field of view in degrees.
        tilt_deg: Camera tilt angle in degrees (0=horizontal, 90=straight down).
        start_frame: Frame index to start from (1-based, default 2).
        total_frames: Number of frames to extract.
        model_path: Path to RAFT .pth checkpoint. Defaults to algorithms/raft/raft-sintel.pth.
        device: Torch device string. Auto-detected if None.
        progress_callback: Optional callable(i, total, status_message) for progress updates.

    Returns:
        Dict with keys:
            status: "success" or "error"
            velocity: Median surface velocity in m/s
            all_angles: List of valid flow angles across all frame pairs
            flow_rgb: RAFT flow visualization image (numpy array) of first valid pair
            fps: Detected video frame rate
            device: Device used for inference
            message: Human-readable summary
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = str(model_path or DEFAULT_MODEL_PATH)

    # Validate video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"status": "error", "message": "无法打开视频文件"}

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if not video_fps or video_fps <= 0:
        video_fps = 30.0
    cap.release()

    # Extract frames
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame - 1)
    frames = []
    for _ in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if len(frames) < 2:
        return {"status": "error", "message": "提取的有效帧数不足，无法分析"}

    # Load model
    if progress_callback:
        progress_callback(0, len(frames) - 1, "正在加载 RAFT 模型...")

    try:
        model, _ = load_raft_model(model_path, device)
    except FileNotFoundError as e:
        return {"status": "error", "message": str(e)}
    except Exception as e:
        return {"status": "error", "message": f"RAFT模型加载失败: {str(e)}"}

    # Process frame pairs
    velocities = []
    all_angles = []
    first_flow_rgb = None
    MIN_POINTS = 1000

    for i in range(len(frames) - 1):
        if progress_callback:
            progress_callback(i + 1, len(frames) - 1,
                              f"正在处理帧对 {i+1}/{len(frames)-1} ...")

        res = _process_raft_pair(model, frames[i], frames[i + 1], device)
        if res and res["valid_count"] >= MIN_POINTS:
            vel = _calculate_physical_velocity(
                res["pixel_distances"], frames[i].shape,
                height_m, fov_deg, tilt_deg, video_fps
            )
            velocities.append(vel)
            all_angles.extend(res["valid_angles"].tolist())
            if first_flow_rgb is None:
                first_flow_rgb = res["flow_rgb"]

    if not velocities:
        return {
            "status": "error",
            "message": "(RAFT) 未提取到有效数据，特征点数可能低于阈值"
        }

    final_vel = float(np.median(velocities))

    return {
        "status": "success",
        "velocity": final_vel,
        "all_angles": all_angles,
        "flow_rgb": first_flow_rgb,
        "fps": video_fps,
        "device": device,
        "frame_count": len(frames),
        "valid_pairs": len(velocities),
        "summary": f"RAFT 测速完成 {final_vel:.4f} m/s (基于{len(velocities)}个有效帧对)",
    }


def run_raft_video_full(
    video_path,
    output_csv,
    *,
    visualization_video=None,
    model_path=None,
    device=None,
    max_dimension=640,
    iters=12,
    progress_callback=None,
):
    """Process every adjacent frame pair and report uncalibrated pixel velocity."""
    source = Path(video_path)
    if not source.is_file() or source.stat().st_size == 0:
        raise FileNotFoundError(source)
    if max_dimension <= 0:
        raise ValueError("max_dimension 必须为正数")
    if iters <= 0:
        raise ValueError("iters 必须为正数")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_file = Path(model_path or DEFAULT_MODEL_PATH)
    model, _ = load_raft_model(str(model_file), device)

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"无法打开视频: {source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"视频 FPS 无效，不能伪造默认值: {source}")
    declared_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    original_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if original_width <= 0 or original_height <= 0:
        raise ValueError(f"视频尺寸无效: {source}")
    scale = min(1.0, float(max_dimension) / max(original_width, original_height))
    working_width = max(8, int(round(original_width * scale / 8.0)) * 8)
    working_height = max(8, int(round(original_height * scale / 8.0)) * 8)
    scale_x = working_width / original_width
    scale_y = working_height / original_height

    def resize(frame):
        if frame.shape[1] == working_width and frame.shape[0] == working_height:
            return frame
        return cv2.resize(frame, (working_width, working_height), interpolation=cv2.INTER_AREA)

    ok, previous_original = capture.read()
    if not ok:
        capture.release()
        raise ValueError(f"视频没有可读取帧: {source}")
    previous = resize(previous_original)
    del previous_original

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    visualization_path = Path(visualization_video) if visualization_video else None
    writer = None
    if visualization_path is not None:
        visualization_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(visualization_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (working_width, working_height),
        )
        if not writer.isOpened():
            capture.release()
            raise RuntimeError(f"无法创建光流可视化视频: {visualization_path}")

    fieldnames = [
        "frame_index",
        "next_frame_index",
        "timestamp_s",
        "velocity_px_frame",
        "velocity_m_s",
        "valid_pixel_count",
        "confidence",
        "source_video",
    ]
    processed_pairs = 0
    valid_pairs = 0
    read_frames = 1
    pair_pixel_velocities = []
    direction_sin_sum = 0.0
    direction_cos_sum = 0.0
    direction_sample_count = 0
    try:
        with output_path.open("w", encoding="utf-8-sig", newline="") as file:
            csv_writer = csv.DictWriter(file, fieldnames=fieldnames)
            csv_writer.writeheader()
            while True:
                ok, current_original = capture.read()
                if not ok:
                    break
                current = resize(current_original)
                del current_original
                frame_index = read_frames - 1
                next_frame_index = read_frames
                result = _process_raft_pair(model, previous, current, device=device, iters=iters)
                row = {
                    "frame_index": frame_index,
                    "next_frame_index": next_frame_index,
                    "timestamp_s": f"{next_frame_index / fps:.6f}",
                    "velocity_px_frame": "",
                    "velocity_m_s": "",
                    "valid_pixel_count": 0,
                    "confidence": "0.000000",
                    "source_video": str(source.resolve()),
                }
                if result is not None:
                    # Preserve displacement in source-frame pixels after isotropic downscaling.
                    original_pixel_velocity = np.sqrt(
                        (result["flow_u"] / scale_x) ** 2 + (result["flow_v"] / scale_y) ** 2
                    )
                    pair_velocity = float(np.median(original_pixel_velocity))
                    row["velocity_px_frame"] = f"{pair_velocity:.6f}"
                    row["valid_pixel_count"] = int(result["valid_count"])
                    row["confidence"] = f"{float(result['valid_count'] / (working_width * working_height)):.6f}"
                    pair_pixel_velocities.append(pair_velocity)
                    valid_angles = np.radians(result["valid_angles"])
                    direction_sin_sum += float(np.sin(valid_angles).sum())
                    direction_cos_sum += float(np.cos(valid_angles).sum())
                    direction_sample_count += int(valid_angles.size)
                    valid_pairs += 1
                    if writer is not None:
                        writer.write(cv2.cvtColor(result["flow_rgb"], cv2.COLOR_RGB2BGR))
                elif writer is not None:
                    writer.write(np.zeros((working_height, working_width, 3), dtype=np.uint8))
                csv_writer.writerow(row)
                processed_pairs += 1
                read_frames += 1
                previous = current
                if progress_callback and (
                    processed_pairs == 1
                    or processed_pairs % 25 == 0
                    or (declared_frames > 0 and read_frames >= declared_frames)
                ):
                    progress_callback(processed_pairs, max(declared_frames - 1, processed_pairs))
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    if processed_pairs == 0:
        raise RuntimeError("视频不足两帧，未生成任何帧对结果")
    mean_direction_deg = None
    if direction_sample_count > 0:
        mean_direction_deg = float(
            np.degrees(np.arctan2(direction_sin_sum, direction_cos_sum)) % 360.0
        )
    return {
        "source_video": str(source.resolve()),
        "output_csv": str(output_path.resolve()),
        "visualization_video": str(visualization_path.resolve()) if visualization_path else None,
        "declared_frame_count": declared_frames,
        "read_frame_count": read_frames,
        "processed_pair_count": processed_pairs,
        "valid_pair_count": valid_pairs,
        "median_velocity_px_frame": (
            float(np.median(pair_pixel_velocities)) if pair_pixel_velocities else None
        ),
        "mean_flow_direction_deg": mean_direction_deg,
        "direction_sample_count": direction_sample_count,
        "fps": fps,
        "original_size": [original_width, original_height],
        "working_size": [working_width, working_height],
        "device": device,
        "model_path": str(model_file.resolve()),
        "iters": iters,
        "physical_calibration": False,
        "velocity_m_s_status": "blank because no verified spatial calibration was provided",
    }
