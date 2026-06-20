"""
RAFT (Recurrent All-Pairs Field Transforms) optical flow analysis for river surface velocity measurement.

This module provides pure algorithm functions for loading the RAFT model and running
optical flow analysis on video frame pairs to estimate surface flow velocity.
All functions are independent of any GUI framework.
"""

import os
import math
import numpy as np
import torch
import cv2

from algorithms.raft.raft_model import RAFT
from algorithms.raft.utils.utils import InputPadder
from algorithms.raft.utils import flow_viz


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
    model_path="raft-sintel.pth",
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
        model_path: Path to RAFT .pth checkpoint.
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
