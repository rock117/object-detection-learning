"""Shared helpers for multi-task YOLO demos."""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE = "https://ultralytics.com/images/bus.jpg"


def add_predict_args(
    parser: argparse.ArgumentParser,
    *,
    default_model: str,
    default_name: str,
    default_source: str | None = DEFAULT_IMAGE,
    project_subdir: str = "detect",
) -> argparse.ArgumentParser:
    parser.add_argument("--model", default=default_model, help="权重名或本地路径")
    parser.add_argument(
        "--source",
        default=default_source,
        help="图片/视频路径、URL，或摄像头索引如 0",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值（部分任务忽略）")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU（分类/深度等可忽略）")
    parser.add_argument("--imgsz", type=int, default=None, help="推理尺寸；默认跟模型")
    parser.add_argument("--device", default=None, help="0 / cpu；默认自动")
    parser.add_argument(
        "--project",
        default=str(ROOT / "runs" / project_subdir),
        help="结果根目录",
    )
    parser.add_argument("--name", default=default_name, help="本次运行子目录名")
    parser.add_argument("--show", action="store_true", help="弹窗显示（摄像头常用）")
    return parser


def predict_kwargs(args: argparse.Namespace) -> dict:
    kwargs = {
        "source": _parse_source(args.source),
        "project": args.project,
        "name": args.name,
        "save": True,
        "exist_ok": True,
        "show": args.show,
    }
    if args.device is not None:
        kwargs["device"] = args.device
    if args.imgsz is not None:
        kwargs["imgsz"] = args.imgsz
    # conf/iou：对 detect/seg/pose/obb 有意义；其它任务多数字段会被忽略
    kwargs["conf"] = args.conf
    kwargs["iou"] = args.iou
    return kwargs


def _parse_source(source: str | None):
    if source is None:
        return DEFAULT_IMAGE
    if source.isdigit():
        return int(source)
    return source
