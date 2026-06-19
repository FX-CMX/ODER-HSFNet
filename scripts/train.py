#!/usr/bin/env python
"""Train ODER-HSFNet."""

import argparse
from pathlib import Path

import yaml

from ultralytics import YOLO


def resolve_scaled_yaml(model_yaml: str, scale: str | None) -> str:
    """Create a temporary YAML with the requested YOLO scale."""
    if not scale or scale.lower() in {"none", "default"}:
        return model_yaml

    src = Path(model_yaml)
    with src.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    scales = cfg.get("scales", {})
    if scale not in scales:
        raise ValueError(f"{model_yaml} does not define scale={scale}. Available scales: {list(scales)}")

    cfg["scale"] = scale
    out_dir = Path(".tmp_scaled_models")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{src.stem}_{scale}.yaml"
    with out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return str(out)


def parse_args():
    parser = argparse.ArgumentParser(description="Train ODER-HSFNet")
    parser.add_argument("--data", required=True, help="YOLO dataset YAML")
    parser.add_argument("--model", default="models/oder_hsfnet.yaml", help="Model YAML")
    parser.add_argument("--scale", default="s", help="Model scale: n/s/l/x, or none")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32, help="Batch size passed to Ultralytics; reduce it if GPU/host memory is insufficient")
    parser.add_argument("--workers", type=int, default=2, help="Workers per rank in DDP")
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/ODER-HSFNet")
    parser.add_argument("--name", default=None)
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained weights when supported")
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    model_yaml = resolve_scaled_yaml(args.model, args.scale)
    run_name = args.name or f"{Path(args.model).stem}_{args.scale}"

    model = YOLO(model_yaml)
    model.train(
        data=args.data,
        pretrained=args.pretrained,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        scale=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.1,
        workers=args.workers,
        device=args.device,
        project=args.project,
        name=run_name,
        exist_ok=True,
        deterministic=False,
        plots=args.plots,
    )


if __name__ == "__main__":
    main()
