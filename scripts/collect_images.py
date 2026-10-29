"""
图片采集工具：整理 datasets/<数据集>/images/raw/<类别>/ 下的原始图片。

功能：
  1. 校验可读性（坏图、0 字节、非图片文件）
  2. 去重（md5 + 感知哈希 aHash，能抓到缩放/重压缩过的重复图）
  3. 统一转 JPEG、长边限制（默认 1920），带 EXIF 方向纠正
  4. 重命名为 <类别>_0001.jpg ...

用法（在项目根目录）:
  python scripts/collect_images.py                                  # 处理 datasets/fruit/images/raw
  python scripts/collect_images.py --raw datasets/fruit/images/raw --max-side 1600
  python scripts/collect_images.py --dry-run                        # 只看报告，不动文件
  python scripts/collect_images.py --keep-duplicates                # 只报告，不删重复

处理结果：
  - 合格图片留在 raw/<类别>/（已重命名、已转 jpg）
  - 坏图与重复图移到 raw/_rejected/ 下（方便复查，不会直接删除）
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".avif", ".gif"}


def ahash(img: Image.Image, size: int = 8) -> int:
    """平均哈希：对缩放、重压缩不敏感，用来抓近重复图。"""
    g = img.convert("L").resize((size, size), Image.Resampling.LANCZOS)
    pixels = list(g.getdata())
    avg = sum(pixels) / len(pixels)
    bits = 0
    for i, p in enumerate(pixels):
        if p >= avg:
            bits |= 1 << i
    return bits


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def move_to_rejected(path: Path, rejected_dir: Path, reason: str) -> None:
    rejected_dir.mkdir(parents=True, exist_ok=True)
    target = rejected_dir / f"{reason}__{path.name}"
    n = 1
    while target.exists():
        target = rejected_dir / f"{reason}__{path.stem}_{n}{path.suffix}"
        n += 1
    shutil.move(str(path), str(target))


def process_class(
    class_dir: Path,
    rejected_dir: Path,
    max_side: int,
    dry_run: bool,
    keep_duplicates: bool,
    min_side: int,
) -> dict:
    stats = {"total": 0, "kept": 0, "bad": 0, "dup": 0, "small": 0}
    seen_md5: dict[str, Path] = {}
    seen_hash: list[tuple[int, Path]] = []
    kept_images: list[Path] = []

    files = sorted(p for p in class_dir.iterdir() if p.is_file() and not p.name.startswith("."))
    stats["total"] = len(files)

    for path in files:
        if path.suffix.lower() not in EXTS:
            continue
        if path.stat().st_size == 0:
            stats["bad"] += 1
            if not dry_run:
                move_to_rejected(path, rejected_dir, "empty")
            continue

        digest = md5_of(path)
        if digest in seen_md5:
            stats["dup"] += 1
            if not dry_run and not keep_duplicates:
                move_to_rejected(path, rejected_dir, "dup_md5")
            continue

        try:
            with Image.open(path) as im:
                im.load()
                w, h = im.size
                if min(w, h) < min_side:
                    stats["small"] += 1
                    if not dry_run:
                        move_to_rejected(path, rejected_dir, "small")
                    continue

                hv = ahash(im)
                near = next(
                    (p for h0, p in seen_hash if hamming(h0, hv) <= 4),
                    None,
                )
                if near is not None:
                    stats["dup"] += 1
                    if not dry_run and not keep_duplicates:
                        move_to_rejected(path, rejected_dir, "dup_ahash")
                    continue

                im = ImageOps.exif_transpose(im)
                if max(im.size) > max_side:
                    im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
                if im.mode not in ("RGB", "L"):
                    im = im.convert("RGB")
                out = path.with_suffix(".jpg")
                tmp = out.with_name(out.stem + ".__tmp__.jpg")
                if not dry_run:
                    im.save(tmp, "JPEG", quality=92, optimize=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  [坏图] {path.name}: {exc}")
            stats["bad"] += 1
            if not dry_run:
                move_to_rejected(path, rejected_dir, "bad")
            continue

        seen_md5[digest] = path
        seen_hash.append((hv, path))
        stats["kept"] += 1
        kept_images.append(path)

        if not dry_run:
            if tmp.exists():
                path.unlink()
                tmp.rename(out)
            kept_images[-1] = out

    # 重命名：<类别>_0001.jpg
    if not dry_run:
        kept_images = sorted(p for p in class_dir.iterdir() if p.is_file() and not p.name.startswith("."))
        for i, p in enumerate(kept_images, 1):
            if p.suffix.lower() != ".jpg":
                continue
            target = class_dir / f"{class_dir.name}_{i:04d}.jpg"
            if p != target:
                if target.exists():
                    target.unlink()
                p.rename(target)

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="校验/去重/规范化原始采集图片")
    parser.add_argument(
        "--raw",
        default=str(ROOT / "datasets" / "fruit" / "images" / "raw"),
        help="原始图片根目录，其下每个子目录是一个类别",
    )
    parser.add_argument("--max-side", type=int, default=1920, help="长边上限，超过则等比缩小")
    parser.add_argument("--min-side", type=int, default=320, help="短边下限，低于此值视为无效小图")
    parser.add_argument("--dry-run", action="store_true", help="只报告，不改动文件")
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="重复图只统计不移动（保留原位，便于人工比对）",
    )
    args = parser.parse_args()

    raw = Path(args.raw)
    if not raw.exists():
        raise SystemExit(f"目录不存在: {raw}")

    class_dirs = sorted(d for d in raw.iterdir() if d.is_dir() and not d.name.startswith("_"))
    if not class_dirs:
        raise SystemExit(f"{raw} 下没有类别子目录（应为 {raw}/<类别名>/）")

    print(f"扫描: {raw}")
    print(f"{'类别':<16}{'总计':>6}{'保留':>6}{'坏图':>6}{'重复':>6}{'过小':>6}")
    grand = {"total": 0, "kept": 0, "bad": 0, "dup": 0, "small": 0}
    for class_dir in class_dirs:
        stats = process_class(
            class_dir,
            raw / "_rejected" / class_dir.name,
            args.max_side,
            args.dry_run,
            args.keep_duplicates,
            args.min_side,
        )
        for k in grand:
            grand[k] += stats[k]
        print(
            f"{class_dir.name:<16}{stats['total']:>6}{stats['kept']:>6}"
            f"{stats['bad']:>6}{stats['dup']:>6}{stats['small']:>6}"
        )

    print("-" * 46)
    print(
        f"{'合计':<16}{grand['total']:>6}{grand['kept']:>6}"
        f"{grand['bad']:>6}{grand['dup']:>6}{grand['small']:>6}"
    )
    if args.dry_run:
        print("\n[dry-run] 未修改任何文件")
    else:
        print(f"\n重复/坏图已移至: {raw / '_rejected'}")
    print("\n下一步：标注（labelImg / Label Studio）→ 再按约 8:2 划分到 images/train 与 images/val")


if __name__ == "__main__":
    main()
