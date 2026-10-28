"""目标检测入门推理。文档: docs/tasks/01-detect.md"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import add_predict_args, predict_kwargs


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO detect demo")
    add_predict_args(
        parser,
        default_model="yolo11n.pt",
        default_name="predict_detect",
        project_subdir="detect",
    )
    args = parser.parse_args()

    model = YOLO(args.model)
    results = model.predict(**predict_kwargs(args))

    for i, r in enumerate(results):
        n = 0 if r.boxes is None else len(r.boxes)
        print(f"[{i}] {r.path} -> {n} objects")
        if not n:
            continue
        for box in r.boxes:
            cid = int(box.cls[0])
            print(f"  {r.names[cid]:15s} conf={float(box.conf[0]):.3f}")


if __name__ == "__main__":
    main()
