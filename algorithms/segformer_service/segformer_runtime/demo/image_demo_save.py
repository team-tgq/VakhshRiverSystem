import argparse
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SEGFORMER_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
if SEGFORMER_DIR not in sys.path:
    sys.path.insert(0, SEGFORMER_DIR)

from ..mmseg.apis import inference_segmentor, init_segmentor
import mmcv


def parse_args():
    parser = argparse.ArgumentParser(description="SegFormer single image inference and save result")
    parser.add_argument("image_path", help="input image path")
    parser.add_argument("config_path", help="config file path")
    parser.add_argument("checkpoint_path", help="checkpoint file path")
    parser.add_argument("output_path", help="output image path")
    parser.add_argument("--device", default="cpu", help="device, e.g. cpu or cuda:0")
    parser.add_argument("--palette", default="cityscapes", help="palette name")
    parser.add_argument("--opacity", type=float, default=0.5, help="mask opacity")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"[INFO] image_path={args.image_path}")
    print(f"[INFO] config_path={args.config_path}")
    print(f"[INFO] checkpoint_path={args.checkpoint_path}")
    print(f"[INFO] output_path={args.output_path}")
    print(f"[INFO] device={args.device}")

    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Input image not found: {args.image_path}")
    if not os.path.exists(args.config_path):
        raise FileNotFoundError(f"Config file not found: {args.config_path}")
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint_path}")

    out_dir = os.path.dirname(args.output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    model = init_segmentor(args.config_path, args.checkpoint_path, device=args.device)
    result = inference_segmentor(model, args.image_path)

    # 明确保存，不依赖隐式行为
    vis = model.show_result(
        args.image_path,
        result,
        palette=args.palette,
        show=False,
        out_file=None,
        opacity=args.opacity
    )

    if vis is None:
        raise RuntimeError("models.show_result returned None, visualization failed.")

    mmcv.imwrite(vis, args.output_path)

    if not os.path.exists(args.output_path):
        raise RuntimeError(f"Output file was not created: {args.output_path}")

    print(f"[INFO] saved={args.output_path}")


if __name__ == "__main__":
    main()