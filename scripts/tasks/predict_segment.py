"""实例分割入门推理。文档: docs/tasks/02-instance-segment.md"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import add_predict_args, predict_kwargs


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO instance segment demo")
    add_predict_args(
        parser,
        default_model="yolo11n-seg.pt",
        default_name="predict_segment",
        project_subdir="segment",
    )
    args = parser.parse_args()

    model = YOLO(args.model)
    results = model.predict(**predict_kwargs(args))

    for i, r in enumerate(results):
        n_box = 0 if r.boxes is None else len(r.boxes)
        n_mask = 0 if r.masks is None else len(r.masks)
        print(f"[{i}] {r.path} -> boxes={n_box}, masks={n_mask}")
        if r.masks is None:
            continue
        for j, mask in enumerate(r.masks.data):
            area = int(mask.sum().item())
            cid = int(r.boxes.cls[j]) if r.boxes is not None else -1
            name = r.names.get(cid, str(cid))
            conf = float(r.boxes.conf[j]) if r.boxes is not None else 0.0
            print(f"  #{j} {name:15s} conf={conf:.3f} mask_pixels={area}")


if __name__ == "__main__":
    main()
