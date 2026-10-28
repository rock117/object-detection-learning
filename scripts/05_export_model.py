"""
入门 5：把 YOLO 权重导出为 ONNX（部署常用格式之一）。

用法:
  python scripts/05_export_model.py
  python scripts/05_export_model.py --model runs/detect/train_coco8/weights/best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export YOLO model")
    parser.add_argument("--model", default="yolo11n.pt", help="要导出的 .pt 权重")
    parser.add_argument(
        "--format",
        default="onnx",
        choices=["onnx", "torchscript", "openvino"],
        help="导出格式",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="ONNX 动态输入尺寸（部分部署场景需要）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists() and not str(args.model).endswith(".pt"):
        raise FileNotFoundError(f"模型不存在: {args.model}")

    model = YOLO(args.model)
    exported = model.export(
        format=args.format,
        imgsz=args.imgsz,
        dynamic=args.dynamic,
    )
    print(f"导出完成: {exported}")


if __name__ == "__main__":
    main()
