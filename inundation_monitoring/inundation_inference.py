import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

try:
    import rasterio
except Exception:
    rasterio = None

from transformers import SegformerModel


os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

VIS_MEAN = np.array(
    [1576.1502, 1341.0074, 1301.2256, 1151.9066, 1364.7650, 2150.6904, 0.0023],
    dtype=np.float32,
)
VIS_STD = np.array(
    [645.2054, 684.8786, 686.2879, 811.1312, 724.8964, 918.0886, 0.1135],
    dtype=np.float32,
)

DEFAULT_WEIGHT = Path(__file__).with_name("mitb2_Seg7C_HandBalanced_best.pth")
LOCAL_BACKBONE_CANDIDATES = [
    Path(r"D:\yfcode\PI_GAT_WF\wf\mit-b2"),
    Path(r"D:\yfcode\PI-GAT\20260515\mit-b2"),
    Path(r"E:\PI-GAT\mit-b2"),
]


def default_backbone_name():
    for path in LOCAL_BACKBONE_CANDIDATES:
        if path.is_dir():
            return str(path)
    return "nvidia/mit-b2"


class SegFormerNet(nn.Module):
    def __init__(self, vis_in=7, hidden_channels=64, backbone_name="nvidia/mit-b2"):
        super().__init__()
        self.vis_stem = nn.Conv2d(vis_in, 3, kernel_size=1)
        self.backbone = SegformerModel.from_pretrained(
            backbone_name,
            output_hidden_states=True,
        )
        self.decoder_conv = nn.ModuleList(
            [nn.Conv2d(channels, hidden_channels, 1) for channels in [64, 128, 320, 512]]
        )
        self.upsample_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
                    nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                ),
                nn.Sequential(
                    nn.Upsample(scale_factor=8, mode="bilinear", align_corners=False),
                    nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                ),
                nn.Sequential(
                    nn.Upsample(scale_factor=16, mode="bilinear", align_corners=False),
                    nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                ),
                nn.Sequential(
                    nn.Upsample(scale_factor=32, mode="bilinear", align_corners=False),
                    nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                ),
            ]
        )
        self.vis_fusion = nn.Conv2d(hidden_channels * 4, hidden_channels, 3, padding=1)
        self.classifier = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 1),
        )

    def forward(self, x_vis):
        original_size = x_vis.shape[-2:]
        outputs = self.backbone(self.vis_stem(x_vis))
        upsampled = []
        for i, feature in enumerate(outputs.hidden_states):
            feature = self.decoder_conv[i](feature)
            feature = self.upsample_layers[i](feature)
            upsampled.append(feature)
        min_h = min(feature.shape[-2] for feature in upsampled)
        min_w = min(feature.shape[-1] for feature in upsampled)
        upsampled = [feature[..., :min_h, :min_w] for feature in upsampled]
        logits = self.classifier(self.vis_fusion(torch.cat(upsampled, dim=1)))
        if logits.shape[-2:] != original_size:
            logits = F.interpolate(logits, size=original_size, mode="bilinear", align_corners=False)
        return logits


def load_model(weight_path=DEFAULT_WEIGHT, device=None, backbone_name=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone_name = backbone_name or default_backbone_name()
    model = SegFormerNet(vis_in=7, backbone_name=backbone_name).to(device)
    state = torch.load(str(weight_path), map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    state = {key.replace("module.", "", 1): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, device


def read_remote_image(image_path):
    image_path = str(image_path)
    if rasterio is not None:
        try:
            with rasterio.open(image_path) as src:
                arr = src.read().astype(np.float32)
            return arr
        except Exception:
            pass

    rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32)
    return np.transpose(rgb, (2, 0, 1))


def make_rgb_display(raw_bands):
    bands = raw_bands.astype(np.float32)
    if bands.shape[0] >= 4:
        rgb = np.stack([bands[3], bands[2], bands[1]], axis=-1)
    elif bands.shape[0] >= 3:
        rgb = np.transpose(bands[:3], (1, 2, 0))
    else:
        rgb = np.repeat(bands[0][..., None], 3, axis=2)

    out = np.zeros_like(rgb, dtype=np.float32)
    for idx in range(3):
        channel = rgb[..., idx]
        lo, hi = np.nanpercentile(channel, [2, 98])
        out[..., idx] = np.clip((channel - lo) / (hi - lo + 1e-6), 0, 1)
    return (out * 255).astype(np.uint8)


def build_7c_feature(raw_bands):
    bands = raw_bands.astype(np.float32)
    if bands.shape[0] >= 6:
        six = bands[:6]
        green = six[1]
        swir = six[4]
    elif bands.shape[0] >= 3:
        rgb = bands[:3]
        six = np.stack([rgb[0], rgb[1], rgb[2], rgb[0], rgb[1], rgb[2]], axis=0)
        green = rgb[1]
        swir = rgb[2]
    else:
        gray = bands[0]
        six = np.repeat(gray[None], 6, axis=0)
        green = gray
        swir = gray

    mndwi = (green - swir) / (green + swir + 1e-6)
    feat = np.concatenate([six, mndwi[None]], axis=0)
    feat = (feat - VIS_MEAN[:, None, None]) / (VIS_STD[:, None, None] + 1e-8)
    return feat.astype(np.float32)


def pad_to_multiple(tensor, multiple=32):
    _, _, h, w = tensor.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h or pad_w:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
    return tensor, h, w


@torch.no_grad()
def predict_mask(image_path, model=None, device=None, threshold=0.50, weight_path=DEFAULT_WEIGHT):
    if model is None:
        model, device = load_model(weight_path=weight_path, device=device)
    elif device is None:
        device = next(model.parameters()).device

    raw = read_remote_image(image_path)
    rgb = make_rgb_display(raw)
    feat = build_7c_feature(raw)
    x = torch.from_numpy(feat).unsqueeze(0).to(device)
    x, h, w = pad_to_multiple(x)
    prob = torch.sigmoid(model(x))[0, 0, :h, :w].detach().cpu().numpy()
    mask = prob >= threshold
    overlay = overlay_mask(rgb, mask)
    return rgb, mask.astype(np.uint8), overlay, prob


def overlay_mask(rgb, mask, color=(255, 0, 0), alpha=0.45):
    rgb = rgb.astype(np.float32)
    mask = mask.astype(bool)
    color_arr = np.array(color, dtype=np.float32)
    overlay = rgb.copy()
    overlay[mask] = rgb[mask] * (1.0 - alpha) + color_arr * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)


def save_result(image_path, output_path, threshold=0.50, weight_path=DEFAULT_WEIGHT):
    _, _, overlay, _ = predict_mask(image_path, threshold=threshold, weight_path=weight_path)
    Image.fromarray(overlay).save(output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Seg7C inundation mask inference")
    parser.add_argument("image", help="input remote-sensing image, such as tif/png/jpg")
    parser.add_argument("-o", "--output", default="inundation_overlay.png", help="output overlay image")
    parser.add_argument("--weight", default=str(DEFAULT_WEIGHT), help="model weight path")
    parser.add_argument("--threshold", type=float, default=0.50, help="mask threshold")
    args = parser.parse_args()
    save_result(args.image, args.output, threshold=args.threshold, weight_path=args.weight)
    print(f"saved overlay to {args.output}")


if __name__ == "__main__":
    main()
