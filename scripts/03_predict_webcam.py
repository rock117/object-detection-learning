"""
入门 3：摄像头实时目标检测。

用法:
  python scripts/03_predict_webcam.py
  python scripts/03_predict_webcam.py --camera 0 --conf 0.35

按 q 退出窗口。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO webcam prediction demo")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--camera", type=int, default=0, help="摄像头索引")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--project",
        default=str(ROOT / "runs" / "detect"),
    )
    parser.add_argument("--name", default="predict_webcam")
    parser.add_argument(
        "--save",
        action="store_true",
        help="是否把画面保存为视频（默认只实时显示）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)

    # show=True 弹出窗口；source 为摄像头索引
    model.predict(
        source=args.camera,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        project=args.project,
        name=args.name,
        show=True,
        save=args.save,
        stream=True,
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
