"""
入门 2：对视频做目标检测并保存结果。

用法:
  python scripts/02_predict_video.py --source data/videos/demo.mp4
  python scripts/02_predict_video.py --source data/videos/demo.mp4 --model yolo11s.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO video prediction demo")
    parser.add_argument(
        "--model",
        default="yolo11n.pt",
        help="预训练权重名或本地路径",
    )
    parser.add_argument(
        "--source",
        required=True,
        help="视频文件路径，例如 data/videos/demo.mp4",
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--project",
        default=str(ROOT / "runs" / "detect"),
    )
    parser.add_argument("--name", default="predict_video")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    if not source.exists():
        print(f"找不到视频: {source}")
        print("请把视频放到 data/videos/ 后再运行。")
        sys.exit(1)

    model = YOLO(args.model)
    model.predict(
        source=str(source),
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        project=args.project,
        name=args.name,
        save=True,
        stream=True,  # 逐帧处理，降低内存占用
        exist_ok=True,
    )

    print(f"结果已保存到: {Path(args.project) / args.name}")


if __name__ == "__main__":
    main()
