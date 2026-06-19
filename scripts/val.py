#!/usr/bin/env python
"""Validate an ODER-HSFNet checkpoint."""

import argparse

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Validate ODER-HSFNet")
    parser.add_argument("--weights", required=True, help="Checkpoint path, e.g. best.pt")
    parser.add_argument("--data", required=True, help="YOLO dataset YAML")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--split", default="val")
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)
    model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        split=args.split,
        plots=args.plots,
    )


if __name__ == "__main__":
    main()
