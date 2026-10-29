"""
YOLO 实例分割标注质检：检查 images/ 与 labels/ 是否一一对应、多边形是否合法。

能查出这些常见坑：
  - 图片没有标签 / 标签没有图片
  - 标签列数不对（把检测框 5 列格式误放进分割数据集）
  - 点数不足（<3 个点）、坐标个数为奇数
  - 坐标超出 0~1（忘了归一化，或被工具导出成像素坐标）
  - 类别 id 超出 names 范围
  - 多边形自相交（标注时手抖画成"8"字，训练时 mask 会错）
  - 多边形面积过小（几乎是一条线）/ 覆盖整图
  - 同一张图的重复多边形

两种用法：
  # 1) 按 data.yaml（检查 train / val 两个 split）
  python scripts/check_seg_labels.py --data configs/fruit-seg.yaml

  # 2) 直接指定目录（例如标注完还没划分时）
  python scripts/check_seg_labels.py --images datasets/fruit/images/raw/annotated \
      --labels datasets/fruit/labels/raw_annotated --nc 2

有错误时退出码为 1，方便接入 CI / 预处理流水线。
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import yaml

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

ROOT = Path(__file__).resolve().parents[1]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# --------------------------------------------------------------------------- #
# 几何小工具
# --------------------------------------------------------------------------- #
def polygon_area(pts: list[tuple[float, float]]) -> float:
    """鞋带公式，返回归一化面积（0~1）。"""
    s = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2


def _seg_cross(a, b, c, d) -> bool:
    """线段 ab 与 cd 是否严格相交（不含端点相接）。"""
    def cross(o, p, q):
        return (p[0] - o[0]) * (q[1] - o[1]) - (p[1] - o[1]) * (q[0] - o[0])

    d1, d2 = cross(c, d, a), cross(c, d, b)
    d3, d4 = cross(a, b, c), cross(a, b, d)
    return (d1 * d2 < 0) and (d3 * d4 < 0)


def self_intersects(pts: list[tuple[float, float]]) -> bool:
    n = len(pts)
    if n < 4:
        return False
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            c, d = pts[j], pts[(j + 1) % n]
            if _seg_cross(a, b, c, d):
                return True
    return False


# --------------------------------------------------------------------------- #
# 单个 split 的检查
# --------------------------------------------------------------------------- #
def check_split(
    img_dir: Path,
    lbl_dir: Path,
    names: dict[int, str],
    min_area: float,
    report: dict,
) -> None:
    label = f"[{img_dir.name}]"
    print(f"\n=== 检查 {label} 图片: {img_dir}")

    images = {p.stem: p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS} if img_dir.is_dir() else {}
    labels = {p.stem: p for p in lbl_dir.glob("*.txt")} if lbl_dir.is_dir() else {}

    if not images:
        print(f"  [错误] 目录里没有图片: {img_dir}")
        report["errors"].append(f"{img_dir}: 没有图片")
        return
    if not lbl_dir.is_dir():
        print(f"  [错误] 标签目录不存在: {lbl_dir}")
        report["errors"].append(f"{lbl_dir}: 不存在")
        return

    missing = sorted(set(images) - set(labels))
    orphan = sorted(set(labels) - set(images))
    empty = []
    if missing:
        print(f"  [错误] {len(missing)} 张图缺少标签，例如: {', '.join(missing[:5])}")
        report["errors"].append(f"{img_dir}: {len(missing)} 张图无标签")
    if orphan:
        print(f"  [错误] {len(orphan)} 个标签没有对应图片，例如: {', '.join(orphan[:5])}")
        report["errors"].append(f"{lbl_dir}: {len(orphan)} 个孤儿标签")

    clip_err = 0
    bad_cols = 0
    bad_class = 0
    few_points = 0
    tiny = 0
    huge = 0
    selfx = 0
    dup_poly = 0
    per_class = Counter()
    objs_per_img = []
    areas = []

    for stem, img_path in sorted(images.items()):
        lp = labels.get(stem)
        if lp is None:
            continue
        rows = []
        with lp.open(encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                rows.append((lineno, line.split()))

        if not rows:
            empty.append(stem)
            objs_per_img.append(0)
            continue

        seen_sig = set()
        n_objs = 0
        for lineno, parts in rows:
            if len(parts) < 7 or len(parts) % 2 == 0:
                # 7 = cls + 3 点(x,y)*3；偶数个字段说明坐标不成对
                print(f"  [错误] {lp.name}:{lineno} 字段数 {len(parts)} 不合法（应为 cls + 偶数个坐标，且 >=7）")
                bad_cols += 1
                continue
            try:
                cls_id = int(float(parts[0]))
                coords = [float(v) for v in parts[1:]]
            except ValueError:
                print(f"  [错误] {lp.name}:{lineno} 存在非数字字段")
                bad_cols += 1
                continue
            if cls_id not in names:
                print(f"  [错误] {lp.name}:{lineno} 类别 id {cls_id} 不在 names（0~{len(names)-1}）内")
                bad_class += 1
                continue

            pts = list(zip(coords[0::2], coords[1::2]))
            if any(c < 0 or c > 1 for c in coords):
                clip_err += 1
                print(f"  [错误] {lp.name}:{lineno} 坐标超出 0~1（是否忘了归一化？）")
                continue
            if len(pts) < 3:
                few_points += 1
                print(f"  [错误] {lp.name}:{lineno} 只有 {len(pts)} 个点，至少需要 3 个")
                continue

            a = polygon_area(pts)
            areas.append(a)
            if a < min_area:
                tiny += 1
                print(f"  [警告] {lp.name}:{lineno} 多边形面积过小 {a:.5f}（阈值 {min_area}）")
            elif a > 0.98:
                huge += 1
                print(f"  [警告] {lp.name}:{lineno} 多边形几乎覆盖整图 {a:.3f}")

            if self_intersects(pts):
                selfx += 1
                print(f"  [警告] {lp.name}:{lineno} 多边形自相交（画成了 8 字形/打了结）")

            sig = (cls_id, round(sum(x for x, _ in pts) / len(pts), 3), round(a, 3))
            if sig in seen_sig:
                dup_poly += 1
                print(f"  [警告] {lp.name}:{lineno} 疑似重复多边形（同类别 + 同中心 + 同面积）")
            seen_sig.add(sig)

            per_class[names[cls_id]] += 1
            n_objs += 1
        objs_per_img.append(n_objs)

    # ---------- 汇总 ----------
    total_objs = sum(per_class.values())
    print(f"  图片 {len(images)} 张，标签 {len(labels)} 个，目标实例 {total_objs} 个")
    if per_class:
        print("  每类实例数: " + ", ".join(f"{k}={v}" for k, v in per_class.most_common()))
    if objs_per_img:
        zero = sum(1 for v in objs_per_img if v == 0)
        print(
            f"  每图实例数: 平均 {total_objs / max(1, len(objs_per_img)):.1f}，"
            f"最多 {max(objs_per_img)}，空标签图 {zero} 张"
        )
    if areas:
        areas_sorted = sorted(areas)
        print(
            f"  多边形面积(归一化): 中位 {areas_sorted[len(areas_sorted) // 2]:.4f}，"
            f"最小 {areas_sorted[0]:.5f}，最大 {areas_sorted[-1]:.4f}"
        )
    if empty:
        print(f"  [提示] {len(empty)} 张图是空标签（背景图，可保留少量）")

    report["counts"][str(img_dir)] = {
        "images": len(images),
        "instances": total_objs,
        "per_class": dict(per_class),
        "empty": len(empty),
    }
    for key, n, msg in (
        ("bad_cols", bad_cols, "字段数/格式错误"),
        ("bad_class", bad_class, "类别 id 越界"),
        ("coord_out_of_range", clip_err, "坐标超出 0~1"),
        ("few_points", few_points, "点数不足"),
        ("tiny", tiny, "面积过小"),
        ("huge", huge, "近乎覆盖整图"),
        ("self_intersect", selfx, "自相交"),
        ("duplicate", dup_poly, "疑似重复"),
    ):
        if n:
            report["warnings" if key in ("tiny", "huge", "self_intersect", "duplicate") else "errors"].append(
                f"{img_dir}: {msg} ×{n}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO 实例分割标签质检")
    parser.add_argument("--data", help="data.yaml 路径（读取 path/train/val/names）")
    parser.add_argument("--images", help="图片目录（与 --labels 搭配）")
    parser.add_argument("--labels", help="标签目录（与 --images 搭配）")
    parser.add_argument("--nc", type=int, default=2, help="类别数（仅 --images 模式使用）")
    parser.add_argument("--min-area", type=float, default=0.0005, help="最小归一化面积阈值")
    args = parser.parse_args()

    report = {"errors": [], "warnings": [], "counts": {}}

    if args.data:
        data_path = Path(args.data)
        if not data_path.is_absolute():
            data_path = (ROOT / data_path).resolve()
        cfg = yaml.safe_load(data_path.read_text(encoding="utf-8"))
        root = Path(cfg.get("path", "."))
        if not root.is_absolute():
            root = (data_path.parent / root).resolve()
        names = {int(k): str(v) for k, v in (cfg.get("names") or {}).items()}
        print(f"数据集: {root}\n类别: {names}")
        for split_key in ("train", "val"):
            if split_key not in cfg:
                continue
            img_dir = root / cfg[split_key]
            lbl_dir = root / str(cfg[split_key]).replace("images", "labels", 1)
            check_split(img_dir, lbl_dir, names, args.min_area, report)
    elif args.images and args.labels:
        names = {i: f"class_{i}" for i in range(args.nc)}
        imgs_root, lbls_root = Path(args.images), Path(args.labels)
        has_direct_images = imgs_root.is_dir() and any(
            p.suffix.lower() in IMG_EXTS for p in imgs_root.iterdir() if p.is_file()
        )
        subdirs = (
            [d for d in sorted(imgs_root.iterdir()) if d.is_dir() and not d.name.startswith("_")]
            if imgs_root.is_dir()
            else []
        )
        if not has_direct_images and subdirs:
            # 形如 images/raw/<类别>/ 的多类别目录，逐个类别检查
            for d in subdirs:
                check_split(d, lbls_root / d.name, names, args.min_area, report)
        else:
            check_split(imgs_root, lbls_root, names, args.min_area, report)
    else:
        parser.error("请提供 --data，或同时提供 --images 与 --labels")

    print("\n" + "=" * 52)
    if report["errors"]:
        print(f"❌ 发现 {len(report['errors'])} 类错误：")
        for e in report["errors"]:
            print("   -", e)
    else:
        print("✅ 没有发现错误")
    if report["warnings"]:
        print(f"⚠️  {len(report['warnings'])} 类警告（不阻塞训练，但建议抽查）：")
        for w in report["warnings"]:
            print("   -", w)
    print("=" * 52)
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
