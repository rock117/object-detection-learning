"""旋转框 OBB 入门推理。文档: docs/tasks/07-obb.md"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import add_predict_args, predict_kwargs


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO OBB demo")
    add_predict_args(
        parser,
        default_model="yolo11n-obb.pt",
        default_name="predict_obb",
        # 航拍船只示例图更适合观察旋转框
        default_source="https://ultralytics.com/images/boats.jpg",
        project_subdir="obb",
    )
    args = parser.parse_args()

    model = YOLO(args.model)
    results = model.predict(**predict_kwargs(args))

    for i, r in enumerate(results):
        if r.obb is None:
            print(f"[{i}] {r.path} -> 无 OBB 结果")
            continue
        print(f"[{i}] {r.path} -> {len(r.obb)} oriented boxes")
        for j in range(len(r.obb)):
            cid = int(r.obb.cls[j])
            conf = float(r.obb.conf[j])
            # xywhr: cx, cy, w, h, angle
            xywhr = r.obb.xywhr[j].cpu().tolist()
            print(
                f"  {r.names[cid]:15s} conf={conf:.3f} "
                f"cx={xywhr[0]:.1f} cy={xywhr[1]:.1f} "
                f"w={xywhr[2]:.1f} h={xywhr[3]:.1f} angle={xywhr[4]:.3f}"
            )


if __name__ == "__main__":
    main()
