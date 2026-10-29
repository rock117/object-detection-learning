"""
把标注好的图片 + 标签按比例划分成 YOLO 的 train/val（按类别分层抽样）。

输入（标注完成后的样子）：
  datasets/fruit/images/raw/durian/*.jpg  +  datasets/fruit/labels/raw/durian/*.txt
  datasets/fruit/images/raw/hami_melon/*.jpg + datasets/fruit/labels/raw/hami_melon/*.txt

输出：
  datasets/fruit/images/train/*.jpg
  datasets/fruit/images/val/*.jpg
  datasets/fruit/labels/train/*.txt
  datasets/fruit/labels/val/*.txt
  文件名加类别前缀：durian_0001.jpg ↔ durian_0001.txt

用法：
  # 先看会怎么分（不动文件）
  python scripts/split_dataset.py --dry-run
  # 正式划分（默认复制，保留原始数据）
  python scripts/split_dataset.py --val-ratio 0.2
  # 移动而非复制（省磁盘，raw 里的图会被移走）
  python scripts/split_dataset.py --move

划分策略：
  先按「这张图里出现的类别组合」分组（例如只有 durian、durian+hami_melon），
  再在每组内随机按比例切分，保证 val 里每类都有样本，不会出现某类只在 train。
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

ROOT = Path(__file__).resolve().parents[1]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def label_signature(label_path: Path) -> frozenset[int]:
    if not label_path.exists():
        return frozenset()
    ids = set()
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if parts:
            try:
                ids.add(int(float(parts[0])))
            except ValueError:
                continue
    return frozenset(ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="划分 YOLO 数据集为 train/val")
    parser.add_argument("--src-images", default=str(ROOT / "datasets/fruit/images/raw"))
    parser.add_argument("--src-labels", default=str(ROOT / "datasets/fruit/labels/raw"))
    parser.add_argument("--dst", default=str(ROOT / "datasets/fruit"))
    parser.add_argument("--val-ratio", type=float, default=0.2, help="验证集比例")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--move", action="store_true", help="移动而不是复制")
    parser.add_argument("--dry-run", action="store_true", help="只报告，不动文件")
    parser.add_argument(
        "--require-label",
        action="store_true",
        help="只划分有标签的图（默认所有图都参与，没有标签的当纯背景图）",
    )
    args = parser.parse_args()

    src_images = Path(args.src_images)
    src_labels = Path(args.src_labels)
    dst = Path(args.dst)
    rng = random.Random(args.seed)

    if not src_images.is_dir():
        raise SystemExit(f"图片目录不存在: {src_images}")

    # 收集「图片 + 标签」配对，忽略 _review/_rejected 等以下划线开头的目录
    pairs: list[tuple[Path, Path | None]] = []
    for cls_dir in sorted(d for d in src_images.iterdir() if d.is_dir() and not d.name.startswith("_")):
        lbl_dir = src_labels / cls_dir.name
        for img in sorted(cls_dir.iterdir()):
            if img.suffix.lower() not in IMG_EXTS:
                continue
            lbl = lbl_dir / f"{img.stem}.txt"
            if args.require_label and not lbl.exists():
                continue
            pairs.append((img, lbl if lbl.exists() else None))

    if not pairs:
        raise SystemExit(f"{src_images} 下没有可用图片（或都被 --require-label 过滤掉了）")

    # 按类别组合分层
    groups: dict[frozenset[int], list[tuple[Path, Path | None]]] = defaultdict(list)
    for img, lbl in pairs:
        groups[label_signature(lbl) if lbl else frozenset()].append((img, lbl))

    plan: dict[str, list[tuple[Path, Path | None]]] = {"train": [], "val": []}
    for sig, items in sorted(groups.items(), key=lambda kv: str(sorted(kv[0]))):
        rng.shuffle(items)
        n_val = max(1, round(len(items) * args.val_ratio)) if len(items) > 1 else 0
        plan["val"] += items[:n_val]
        plan["train"] += items[n_val:]

    total = len(pairs)
    print(f"源: {src_images}")
    print(f"标签: {src_labels}")
    print(f"共 {total} 对（有标签 {sum(1 for _, l in pairs if l)}，无标签 {sum(1 for _, l in pairs if not l)}）")
    print(f"分层: {len(groups)} 组（按图内类别组合）")
    print(f"train {len(plan['train'])} / val {len(plan['val'])}  ≈ {1 - args.val_ratio:.0%} / {args.val_ratio:.0%}")

    # 类别覆盖检查
    for split in ("train", "val"):
        cnt = Counter()
        for _, lbl in plan[split]:
            for cid in label_signature(lbl) if lbl else frozenset():
                cnt[cid] += 1
        print(f"  {split} 类别 id 覆盖: {dict(sorted(cnt.items()))}")
    val_ids = {
        cid for _, lbl in plan["val"] for cid in (label_signature(lbl) if lbl else frozenset())
    }
    train_ids = {
        cid for _, lbl in plan["train"] for cid in (label_signature(lbl) if lbl else frozenset())
    }
    missing_in_val = train_ids - val_ids
    if missing_in_val:
        print(f"  ⚠️ 类别 {sorted(missing_in_val)} 只在 train 出现，val 无法评估它们")

    if args.dry_run:
        print("\n[dry-run] 未写入任何文件")
        return

    for split in ("train", "val"):
        (dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / split).mkdir(parents=True, exist_ok=True)

    copied = 0
    for split, items in plan.items():
        for img, lbl in items:
            cls_prefix = img.parent.name
            stem = f"{cls_prefix}_{img.stem}" if not img.stem.startswith(cls_prefix) else img.stem
            dst_img = dst / "images" / split / f"{stem}{img.suffix.lower()}"
            dst_lbl = dst / "labels" / split / f"{stem}.txt"
            if dst_img.exists():
                raise SystemExit(f"目标已存在，先清空输出目录再跑: {dst_img}")
            if args.move:
                shutil.move(str(img), str(dst_img))
            else:
                shutil.copy2(img, dst_img)
            if lbl:
                if args.move:
                    shutil.move(str(lbl), str(dst_lbl))
                else:
                    shutil.copy2(lbl, dst_lbl)
            else:
                dst_lbl.write_text("", encoding="utf-8")  # 背景图：空标签
            copied += 1

    print(f"\n完成：{copied} 对写入 {dst}")
    print("下一步: python scripts/check_seg_labels.py --data configs/fruit-seg.yaml")


if __name__ == "__main__":
    main()
