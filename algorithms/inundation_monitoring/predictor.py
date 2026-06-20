from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
DEFAULT_WEIGHT = BASE_DIR / "mitb2_Seg7C_HandBalanced_best.pth"


class FloodPredictor:
    """Plugin-facing adapter for the SegFormer 7-channel inundation model."""

    def __init__(self, weight_path: str | os.PathLike[str] = DEFAULT_WEIGHT) -> None:
        self.weight_path = Path(weight_path)
        self.model = None
        self.device = None
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def load_model(self) -> None:
        if self.model is not None:
            return
        if not self.weight_path.exists():
            raise FileNotFoundError(f"Model weight file not found: {self.weight_path}")
        from .inundation_inference import load_model

        self.model, self.device = load_model(weight_path=self.weight_path)

    def predict(self, img_path: str, thresh: float = 0.5) -> dict[str, Any]:
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Input image not found: {img_path}")
        if not (0.0 <= float(thresh) <= 1.0):
            raise ValueError("Threshold must be between 0 and 1.")

        self.load_model()
        from .inundation_inference import predict_mask

        rgb, mask, overlay, prob = predict_mask(
            img_path,
            model=self.model,
            device=self.device,
            threshold=float(thresh),
            weight_path=self.weight_path,
        )

        image_name = Path(img_path).stem
        mask_out = OUTPUT_DIR / f"{image_name}_inundation_mask.png"
        overlay_out = OUTPUT_DIR / f"{image_name}_inundation_overlay.png"

        Image.fromarray((mask * 255).astype(np.uint8)).save(mask_out)
        Image.fromarray(overlay).save(overlay_out)

        return {
            "original": rgb,
            "mask": mask,
            "overlay": overlay,
            "probability": prob,
            "mask_path": str(mask_out),
            "overlay_path": str(overlay_out),
            "device": str(self.device),
            "threshold": float(thresh),
            "water_ratio": float(mask.sum()) / max(mask.size, 1),
        }
