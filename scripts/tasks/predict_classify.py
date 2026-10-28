"""图像分类入门推理。文档: docs/tasks/05-classify.md"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import add_predict_args, predict_kwargs


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO classify demo")
    add_predict_args(
        parser,
        default_model="yolo11n-cls.pt",
        default_name="predict_classify",
        project_subdir="classify",
    )
    parser.set_defaults(imgsz=224)
    args = parser.parse_args()

    model = YOLO(args.model)
    results = model.predict(**predict_kwargs(args))

    for i, r in enumerate(results):
        if r.probs is None:
            print(f"[{i}] 无分类结果")
            continue
        top5 = r.probs.top5
        print(f"[{i}] {r.path}")
        for rank, cid in enumerate(top5, start=1):
            name = r.names[int(cid)]
            conf = float(r.probs.data[int(cid)])
            print(f"  top{rank}: {name:30s} {conf:.3f}")


if __name__ == "__main__":
    main()
