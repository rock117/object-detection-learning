"""
入门 6：YOLO + ByteTrack 多目标跟踪（视频 / 摄像头）。

文档: docs/07-bytetrack.md

用法（在项目根目录）:
  python scripts/06_track_bytetrack.py --source data/videos/demo.mp4
  python scripts/06_track_bytetrack.py --source 0 --show
  python scripts/06_track_bytetrack.py --source data/videos/demo.mp4 --classes 0
  python scripts/06_track_bytetrack.py --source data/videos/demo.mp4 --tracker configs/bytetrack.example.yaml
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEMO = "https://ultralytics.com/assets/decelera_landscape_min.mov"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO + ByteTrack demo")
    parser.add_argument(
        "--model",
        default="yolo11n.pt",
        help="检测/分割/姿态权重，或自训 best.pt",
    )
    parser.add_argument(
        "--source",
        default=DEFAULT_DEMO,
        help="视频路径、URL，或摄像头索引如 0",
    )
    parser.add_argument(
        "--tracker",
        default="bytetrack.yaml",
        help="跟踪器配置：bytetrack.yaml 或 configs/bytetrack.example.yaml",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="检测置信度阈值")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU")
    parser.add_argument(
        "--classes",
        nargs="*",
        type=int,
        default=None,
        help="只跟踪这些类别 id，例如 --classes 0（COCO person）",
    )
    parser.add_argument("--device", default=None, help="如 0 / cpu；默认自动")
    parser.add_argument(
        "--project",
        default=str(ROOT / "runs" / "track"),
        help="结果根目录",
    )
    parser.add_argument("--name", default="bytetrack", help="本次运行子目录名")
    parser.add_argument("--show", action="store_true", help="弹窗显示（摄像头常用）")
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="不保存可视化视频/图",
    )
    return parser.parse_args()


def resolve_source(raw: str) -> str | int:
    if raw.isdigit():
        return int(raw)
    path = Path(raw)
    if path.exists():
        return str(path)
    if raw.startswith(("http://", "https://")):
        return raw
    print(f"找不到本地源: {raw}")
    print("请把视频放到 data/videos/，或传 URL / 摄像头索引 0。")
    sys.exit(1)


def main() -> None:
    args = parse_args()
    source = resolve_source(args.source)

    tracker = args.tracker
    tracker_path = Path(tracker)
    if tracker_path.exists():
        tracker = str(tracker_path.resolve())

    model = YOLO(args.model)
    kwargs: dict = {
        "source": source,
        "tracker": tracker,
        "persist": True,
        "conf": args.conf,
        "iou": args.iou,
        "project": args.project,
        "name": args.name,
        "save": not args.no_save,
        "stream": True,
        "exist_ok": True,
        "show": args.show,
        "verbose": False,
    }
    if args.device is not None:
        kwargs["device"] = args.device
    if args.classes is not None:
        kwargs["classes"] = args.classes

    # id -> 出现帧数；顺带打印前几帧明细
    frames_per_id: dict[int, int] = defaultdict(int)
    frame_idx = 0

    print(f"model={args.model}  tracker={tracker}  source={source}")
    for result in model.track(**kwargs):
        frame_idx += 1
        n = 0 if result.boxes is None else len(result.boxes)
        ids = (
            result.boxes.id.int().tolist()
            if result.boxes is not None and result.boxes.id is not None
            else []
        )
        for tid in ids:
            frames_per_id[tid] += 1

        if frame_idx <= 3 or frame_idx % 30 == 0:
            id_preview = ids[:8]
            print(f"[frame {frame_idx}] objects={n} ids={id_preview}")

    print("---")
    print(f"总帧数约: {frame_idx}")
    print(f"唯一 track id 数: {len(frames_per_id)}")
    if frames_per_id:
        top = sorted(frames_per_id.items(), key=lambda x: x[1], reverse=True)[:10]
        print("出现帧数最多的 id（前 10）:")
        for tid, n_frames in top:
            print(f"  id={tid:4d}  frames={n_frames}")
    out = Path(args.project) / args.name
    print(f"结果目录: {out}")


if __name__ == "__main__":
    main()
