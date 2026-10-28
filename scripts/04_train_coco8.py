"""
入门 4：用官方迷你数据集 coco8 跑通训练流程。

coco8 只有 8 张图，目的是验证「环境 + 训练代码」可用，不是冲高 mAP。

用法:
  python scripts/04_train_coco8.py
  python scripts/04_train_coco8.py --epochs 20 --batch 4 --device cpu
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO on coco8")
    parser.add_argument("--model", default="yolo11n.pt", help="起点权重")
    parser.add_argument(
        "--data",
        default="coco8.yaml",
        help='数据集 yaml。默认用 Ultralytics 内置 "coco8.yaml"（自动下载）；'
        "自定义数据可传 configs/my_dataset.yaml",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument(
        "--device",
        default=None,
        help="0 / 0,1 / cpu；默认自动",
    )
    parser.add_argument(
        "--project",
        default=str(ROOT / "runs" / "detect"),
    )
    parser.add_argument("--name", default="train_coco8")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True,
        pretrained=True,
    )

    save_dir = Path(results.save_dir)
    best = save_dir / "weights" / "best.pt"
    print("\n训练完成。")
    print(f"输出目录: {save_dir}")
    if best.exists():
        print(f"最佳权重: {best}")
        print("可用最佳权重再推理，例如:")
        print(f'  python scripts/01_predict_image.py --model "{best}"')


if __name__ == "__main__":
    main()
