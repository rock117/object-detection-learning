"""
入门 1：使用预训练 YOLO 对图片做目标检测。

用法（在项目根目录）:
  python scripts/01_predict_image.py
  python scripts/01_predict_image.py --source data/images/demo.jpg --model yolo11n.pt
  python scripts/01_predict_image.py --conf 0.4 --iou 0.5
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO image prediction demo")
    parser.add_argument(
        "--model",
        default="yolo11n.pt",
        help="预训练权重名或本地路径（首次会自动下载）",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="图片路径；默认使用 Ultralytics 自带示例图",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU 阈值")
    parser.add_argument(
        "--device",
        default=None,
        help="如 0 / cpu；默认自动选择",
    )
    parser.add_argument(
        "--project",
        default=str(ROOT / "runs" / "detect"),
        help="结果保存根目录",
    )
    parser.add_argument("--name", default="predict_image", help="本次实验子目录名")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)

    # 未指定 source 时用官方示例（ultralytics 会解析内置资源名）
    source = args.source or "https://ultralytics.com/images/bus.jpg"

    results = model.predict(
        source=source,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        project=args.project,
        name=args.name,
        save=True,
        exist_ok=True,
    )

    for i, result in enumerate(results):
        boxes = result.boxes
        print(f"\n=== image [{i}] {result.path} ===")
        print(f"detected: {len(boxes)} objects")
        if boxes is None or len(boxes) == 0:
            continue
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            name = result.names[cls_id]
            print(f"  - {name:15s} conf={conf:.3f} xyxy={[round(v, 1) for v in xyxy]}")

    out_dir = Path(args.project) / args.name
    print(f"\n结果已保存到: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
