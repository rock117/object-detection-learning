"""人体姿态估计入门推理。文档: docs/tasks/06-pose.md"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import add_predict_args, predict_kwargs


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO pose demo")
    add_predict_args(
        parser,
        default_model="yolo11n-pose.pt",
        default_name="predict_pose",
        project_subdir="pose",
    )
    args = parser.parse_args()

    model = YOLO(args.model)
    results = model.predict(**predict_kwargs(args))

    for i, r in enumerate(results):
        if r.keypoints is None:
            print(f"[{i}] {r.path} -> 无人/无关键点")
            continue
        xy = r.keypoints.xy
        print(f"[{i}] {r.path} -> {len(xy)} person(s), kpts_per_person={xy.shape[1]}")
        for pi, person in enumerate(xy):
            # 打印前 5 个关键点，避免刷屏
            pts = person.cpu().tolist()[:5]
            print(f"  person#{pi} first5_xy={[ [round(a, 1), round(b, 1)] for a, b in pts ]} ...")


if __name__ == "__main__":
    main()
