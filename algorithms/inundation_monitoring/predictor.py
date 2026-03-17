import os
from typing import Dict, Any

import cv2
import numpy as np
import rasterio
import torch
import torch.nn.functional as F

from .unet_model import UNet


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_flood_unet.pth")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
IMG_SIZE = 256

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FloodPredictor:
    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self.model = None
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    def load_model(self):
        if self.model is not None:
            return

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"未找到模型权重文件: {self.model_path}")

        self.model = UNet(in_channels=2, out_channels=2).to(DEVICE)
        self.model.load_state_dict(torch.load(self.model_path, map_location=DEVICE))
        self.model.eval()

    def predict(self, img_path: str, thresh: float = 0.5) -> Dict[str, Any]:
        self.load_model()

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"未找到输入影像: {img_path}")

        with rasterio.open(img_path) as src:
            image = src.read().astype(np.float32)

        image = (image - image.min()) / (image.max() - image.min() + 1e-6)

        if image.ndim == 2:
            image = image[np.newaxis, :, :]
        elif image.shape[0] > 2:
            image = image[:2, :, :]

        if image.shape[0] == 1:
            image = np.concatenate([image, image], axis=0)

        orig_h, orig_w = image.shape[1], image.shape[2]

        image_resized = cv2.resize(
            image.transpose(1, 2, 0),
            (IMG_SIZE, IMG_SIZE)
        ).transpose(2, 0, 1)

        input_tensor = torch.tensor(
            image_resized,
            dtype=torch.float32
        ).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = self.model(input_tensor)
            probs = F.softmax(output, dim=1)[0, 1].cpu().numpy()
            pred_mask = (probs > thresh).astype(np.uint8)

        pred_mask = cv2.resize(
            pred_mask,
            (orig_w, orig_h),
            interpolation=cv2.INTER_NEAREST
        )

        orig_vis = image[0]
        overlay = self.overlay_mask(orig_vis, pred_mask)

        image_name = os.path.splitext(os.path.basename(img_path))[0]
        mask_out = os.path.join(OUTPUT_DIR, f"{image_name}_flood_mask.png")
        overlay_out = os.path.join(OUTPUT_DIR, f"{image_name}_flood_overlay.png")

        cv2.imwrite(mask_out, (pred_mask * 255).astype(np.uint8))
        cv2.imwrite(overlay_out, overlay)

        return {
            "original": orig_vis,
            "mask": pred_mask,
            "overlay": overlay,
            "mask_path": mask_out,
            "overlay_path": overlay_out,
            "device": str(DEVICE),
            "threshold": thresh,
        }

    @staticmethod
    def overlay_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        image = (image * 255).astype(np.uint8)
        color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        color[mask == 1] = [255, 0, 0]
        return color