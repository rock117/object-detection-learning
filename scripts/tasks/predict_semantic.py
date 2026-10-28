"""语义分割入门推理（YOLO26 -sem）。文档: docs/tasks/03-semantic-segment.md"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import add_predict_args, predict_kwargs


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO semantic segmentation demo")
    add_predict_args(
        parser,
        default_model="yolo26n-sem.pt",
        default_name="predict_semantic",
        project_subdir="semantic",
    )
    parser.set_defaults(imgsz=1024)
    args = parser.parse_args()

    print("提示: 语义分割需要较新的 ultralytics（YOLO26）。若报错请: pip install -U ultralytics")
    model = YOLO(args.model)
    results = model.predict(**predict_kwargs(args))

    for i, r in enumerate(results):
        semantic = getattr(r, "semantic_mask", None)
        if semantic is None:
            print(f"[{i}] 无 semantic_mask，请确认模型与库版本支持 semantic 任务")
            continue
        data = semantic.data.cpu().numpy()
        flat = data.reshape(-1)
        counts = Counter(flat.tolist())
        total = flat.size
        print(f"[{i}] {r.path} class_map shape={data.shape}")
        # 打印占比最高的若干类
        for cid, cnt in counts.most_common(10):
            name = r.names.get(int(cid), str(cid)) if r.names else str(cid)
            print(f"  class {cid:>3} ({name}): {cnt / total * 100:5.1f}%")


if __name__ == "__main__":
    main()
