#!/usr/bin/env python
"""Run prediction with an ODER-HSFNet checkpoint."""

import argparse

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Predict with ODER-HSFNet")
    parser.add_argument("--weights", required=True, help="Checkpoint path, e.g. best.pt")
    parser.add_argument("--source", required=True, help="Image/video path, directory, URL, or camera index")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default="0")
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)
    model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save=args.save,
    )


if __name__ == "__main__":
    main()
