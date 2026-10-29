"""
生成「联系表」预览图，方便快速人工复查采集到的图片。

把每个类别的图片拼成若干张网格图（每张 30 张），输出到 <raw>/_review/ 下，
用图片查看器逐张翻看，把混进来的错图直接删掉即可。

用法：
  python scripts/make_contact_sheet.py                        # 默认 datasets/fruit/images/raw
  python scripts/make_contact_sheet.py --per-sheet 25 --cols 5
  python scripts/make_contact_sheet.py --class durian --sheets 3
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

ROOT = Path(__file__).resolve().parents[1]


def make_sheets(
    files: list[Path],
    out_dir: Path,
    per_sheet: int,
    cols: int,
    cell: int,
    sheets: int,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    made: list[Path] = []
    rows = (per_sheet + cols - 1) // cols
    for s in range(sheets):
        chunk = files[s * per_sheet : (s + 1) * per_sheet]
        if not chunk:
            break
        canvas = Image.new("RGB", (cols * cell, rows * cell), (24, 24, 24))
        draw = ImageDraw.Draw(canvas)
        for i, p in enumerate(chunk):
            try:
                with Image.open(p) as im:
                    im = im.convert("RGB")
                    im.thumbnail((cell - 4, cell - 4), Image.Resampling.LANCZOS)
            except Exception:  # noqa: BLE001
                continue
            x = (i % cols) * cell + (cell - im.width) // 2
            y = (i // cols) * cell + (cell - im.height) // 2
            canvas.paste(im, (x, y))
            # 左上角标序号，便于对照文件名
            draw.rectangle([(i % cols) * cell, (i // cols) * cell,
                            (i % cols) * cell + 34, (i // cols) * cell + 14],
                           fill=(0, 0, 0))
            draw.text(((i % cols) * cell + 2, (i // cols) * cell + 1), str(i + 1), fill=(255, 255, 0))
        name = f"{out_dir.name}_{s + 1:02d}.jpg"
        path = out_dir / name
        canvas.save(path, quality=88)
        made.append(path)
        # 索引文件：序号 ↔ 文件名
        if s == 0:
            idx = out_dir / "index.txt"
            with idx.open("a", encoding="utf-8") as f:
                f.write(f"# sheet {name}\n")
                for i, p in enumerate(chunk, 1):
                    f.write(f"{i:03d}\t{p.name}\n")
    return made


def main() -> None:
    parser = argparse.ArgumentParser(description="生成图片联系表，用于快速人工复查")
    parser.add_argument("--raw", default=str(ROOT / "datasets" / "fruit" / "images" / "raw"))
    parser.add_argument("--class", dest="classes", action="append", help="类别名，可重复")
    parser.add_argument("--per-sheet", type=int, default=30, help="每张联系表放几张图")
    parser.add_argument("--cols", type=int, default=6, help="每行几张")
    parser.add_argument("--cell", type=int, default=220, help="每格像素")
    parser.add_argument("--sheets", type=int, default=99, help="每个类别最多生成几张表")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    raw = Path(args.raw)
    classes = args.classes or sorted(d.name for d in raw.iterdir() if d.is_dir() and not d.name.startswith("_"))
    rng = random.Random(args.seed)

    for cls in classes:
        files = sorted((raw / cls).glob("*.jpg"))
        if not files:
            print(f"{cls}: 没有图片，跳过")
            continue
        rng.shuffle(files)
        n_sheets = min(args.sheets, (len(files) + args.per_sheet - 1) // args.per_sheet)
        out_dir = raw / "_review" / cls
        if out_dir.exists():
            for old in out_dir.glob("*.jpg"):
                old.unlink()
            idx = out_dir / "index.txt"
            if idx.exists():
                idx.unlink()
        made = make_sheets(files, out_dir, args.per_sheet, args.cols, args.cell, n_sheets)
        print(f"{cls}: {len(files)} 张 → {len(made)} 张联系表 → {out_dir}")


if __name__ == "__main__":
    main()
