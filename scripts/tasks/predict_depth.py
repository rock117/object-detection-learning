"""单目深度估计入门推理（YOLO26 -depth）。文档: docs/tasks/04-depth.md"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import add_predict_args, predict_kwargs


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO depth estimation demo")
    add_predict_args(
        parser,
        default_model="yolo26n-depth.pt",
        default_name="predict_depth",
        project_subdir="depth",
    )
    parser.set_defaults(imgsz=768)
    args = parser.parse_args()

    print("提示: 深度估计需要较新的 ultralytics（YOLO26）。若报错请: pip install -U ultralytics")
    model = YOLO(args.model)
    results = model.predict(**predict_kwargs(args))

    for i, r in enumerate(results):
        depth = getattr(r, "depth", None)
        if depth is None:
            print(f"[{i}] 无 depth 字段，请确认模型与库版本支持 depth 任务")
            continue
        arr = depth.data.cpu().numpy().astype(np.float32)
        print(
            f"[{i}] {r.path} depth shape={arr.shape} "
            f"min={arr.min():.3f}m mean={arr.mean():.3f}m max={arr.max():.3f}m"
        )


if __name__ == "__main__":
    main()
