"""
水果图片采集器：从开放图库批量下载「榴莲 / 哈密瓜」图片到 datasets/fruit/images/raw/<类别>/

数据源（都不需要 API key）：
  wikimedia    Wikimedia Commons —— 优先走「分类」(人工整理过，干净)，再用搜索补充
  inaturalist  iNaturalist 观察记录照片（CC 授权，真实场景，量大）

用法（在项目根目录）:
  python scripts/fetch_fruit_images.py --dry-run                  # 只统计候选数量
  python scripts/fetch_fruit_images.py --class durian --limit 20  # 小批量试跑
  python scripts/fetch_fruit_images.py --class durian --limit 500
  python scripts/fetch_fruit_images.py --class hami_melon --limit 500

输出：
  datasets/fruit/images/raw/<类别>/<类别>_<来源>_0001.jpg
  datasets/fruit/images/raw/<类别>/_meta.csv     # 来源链接 / 许可证 / 署名

下载后请运行清洗：
  python scripts/collect_images.py
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

ROOT = Path(__file__).resolve().parents[1]
UA = "fruit-dataset-builder/1.0 (educational object-detection project)"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
INAT_API = "https://api.inaturalist.org/v1/observations"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
INAT_LICENSES = "cc0,cc-by,cc-by-sa,cc-by-nc,cc-by-nc-sa"

# 明显不是照片/不是目标主体的文件，按标题关键词排除
BAD_TITLE_WORDS = (
    "logo", "sign", "map", "icon", "flag", "catalog", "poster", "painting",
    "drawing", "illustration", "diagram", "chart", "stamp", "coin", "cartoon",
    "clipart", "svg", "text", "book", "cover", "advert", "label", "menu",
)

# 每个类别的采集配方
CLASS_SPEC: dict[str, dict] = {
    "durian": {
        # Commons 分类（人工归类，质量最高）
        "categories": ["Category:Durian", "Category:Durio zibethinus"],
        # Commons 全文搜索（补充），标题需命中 keywords
        "search": [
            "durian fruit",
            "durian flesh",
            "durian stall",
            "durian tree",
            "durian market",
            "durian seller",
            "durian husk",
            "durian seeds",
            "durian Thailand",
            "durian Malaysia",
            "Musang King durian",
        ],
        "keywords": ("durian", "durio"),
        "inat": ["Durio zibethinus"],
    },
    "hami_melon": {
        # 哈密瓜英文常归入 muskmelon / cantaloupe（网纹甜瓜）
        "categories": [
            "Category:Cucumis melo cantaloupe group",
            "Category:Cucumis melo",
            "Category:Cucumis melo inodorus group",
        ],
        "search": [
            "hami melon",
            "cantaloupe melon",
            "muskmelon",
            "netted melon",
            "rockmelon",
            "cantaloupe halves",
            "melon market fruit",
        ],
        "keywords": ("melon", "cantaloup", "muskmelon", "cucumis"),
        "inat": ["Cucumis melo"],
    },
}

_lock = threading.Lock()
_local = threading.local()


def session() -> requests.Session:
    """每个线程一个 Session，复用 TCP 连接（比 urllib 快很多）。"""
    s = getattr(_local, "sess", None)
    if s is None:
        s = requests.Session()
        s.headers.update({"User-Agent": UA})
        _local.sess = s
    return s


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
def fetch_bytes(url: str, params: dict | None = None, timeout: int = 40) -> bytes:
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    last: Exception | None = None
    for attempt in range(4):
        try:
            resp = session().get(url, timeout=timeout)
            if resp.status_code == 200:
                return resp.content
            last = RuntimeError(f"HTTP {resp.status_code}")
            if resp.status_code == 429:  # 限流：退避后重试
                time.sleep(4 + attempt * 4)
                continue
            if resp.status_code in (401, 403, 404):
                raise last
            time.sleep(1.5 * (attempt + 1))
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise last or RuntimeError("request failed")


def get_json(url: str, params: dict) -> dict:
    return json.loads(fetch_bytes(url, params).decode("utf-8", "replace"))


def strip_html(s: str) -> str:
    out, depth = [], 0
    for ch in s:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth = max(0, depth - 1)
        elif depth == 0:
            out.append(ch)
    return " ".join("".join(out).split())


# --------------------------------------------------------------------------- #
# Wikimedia Commons
# --------------------------------------------------------------------------- #
def _commons_item(page: dict, query: str) -> dict | None:
    info = (page.get("imageinfo") or [{}])[0]
    title = page.get("title", "")
    ext = Path(title).suffix.lower()
    if ext not in IMG_EXTS:
        return None
    url = info.get("thumburl") or info.get("url")
    if not url:
        return None
    meta = info.get("extmetadata") or {}
    lic = (meta.get("LicenseShortName") or {}).get("value", "unknown")
    artist = strip_html((meta.get("Artist") or {}).get("value", ""))[:200]
    return {
        "source": "wikimedia",
        "url": url,
        "page": info.get("descriptionurl", ""),
        "license": lic,
        "attribution": artist,
        "query": query,
        "name": title,
        "weight": 0,  # 0 = 分类（优先），1 = 搜索
    }


def commons_categories(categories: list[str], width: int, max_pages: int = 5) -> list[dict]:
    out: list[dict] = []
    for cat in categories:
        cont: dict[str, str] = {}
        n_before = len(out)
        pages_done = 0
        while pages_done < max_pages:
            pages_done += 1
            params = {
                "action": "query",
                "format": "json",
                "formatversion": 2,
                "generator": "categorymembers",
                "gcmtitle": cat,
                "gcmtype": "file",
                "gcmlimit": 100,
                "prop": "imageinfo",
                "iiprop": "url|extmetadata",
                "iiurlwidth": width,
            }
            params.update(cont)
            try:
                j = get_json(COMMONS_API, params)
            except Exception as exc:  # noqa: BLE001
                print(f"    [commons:分类] {cat}: {exc}")
                break
            pages = (j.get("query") or {}).get("pages") or []
            for p in pages:
                item = _commons_item(p, cat)
                if item:
                    out.append(item)
            cont = j.get("continue") or {}
            if not cont:
                break
            time.sleep(0.5)
        print(f"    [commons:分类] {cat}: +{len(out) - n_before}")
        time.sleep(0.5)
    return out


def commons_search(
    queries: list[str], width: int, keywords: tuple[str, ...], max_pages: int = 6
) -> list[dict]:
    out: list[dict] = []
    for q in queries:
        cont: dict[str, str] = {}
        n_before = len(out)
        pages_done = 0
        while pages_done < max_pages and len(out) - n_before < 120:
            pages_done += 1
            params = {
                "action": "query",
                "format": "json",
                "formatversion": 2,
                "generator": "search",
                "gsrsearch": q,
                "gsrnamespace": 6,
                "gsrlimit": 50,
                "prop": "imageinfo",
                "iiprop": "url|extmetadata",
                "iiurlwidth": width,
            }
            params.update(cont)
            try:
                j = get_json(COMMONS_API, params)
            except Exception as exc:  # noqa: BLE001
                print(f"    [commons:搜索] {q!r}: {exc}")
                break
            pages = (j.get("query") or {}).get("pages") or []
            for p in pages:
                item = _commons_item(p, q)
                if not item:
                    continue
                title = item["name"].lower()
                if not any(k in title for k in keywords):  # 标题必须沾边
                    continue
                if any(b in title for b in BAD_TITLE_WORDS):
                    continue
                item["weight"] = 1
                out.append(item)
            cont = j.get("continue") or {}
            if not cont:
                break
            time.sleep(0.5)
        print(f"    [commons:搜索] {q!r}: +{len(out) - n_before}")
        time.sleep(0.5)
    return out


# --------------------------------------------------------------------------- #
# iNaturalist
# --------------------------------------------------------------------------- #
def inat_items(taxa: list[str], want: int) -> list[dict]:
    out: list[dict] = []
    for taxon in taxa:
        page, n_before = 1, len(out)
        while len(out) - n_before < want and page <= 25:
            try:
                j = get_json(
                    INAT_API,
                    {
                        "taxon_name": taxon,
                        "per_page": 100,
                        "page": page,
                        "photo_license": INAT_LICENSES,
                        "quality_grade": "research",
                    },
                )
            except Exception as exc:  # noqa: BLE001
                print(f"    [inaturalist] {taxon}: {exc}")
                break
            results = j.get("results") or []
            if not results:
                break
            for obs in results:
                for ph in (obs.get("photos") or [])[:3]:  # 每条观察最多取 3 张
                    url = ph.get("url", "")
                    if not url:
                        continue
                    out.append(
                        {
                            "source": "inaturalist",
                            "url": url.replace("/square.", "/large.")
                            .replace("/small.", "/large.")
                            .replace("/thumb.", "/large."),
                            "page": f"https://www.inaturalist.org/observations/{obs.get('id')}",
                            "license": ph.get("license_code") or "unknown",
                            "attribution": (ph.get("attribution") or "")[:200],
                            "query": taxon,
                            "name": f"obs{obs.get('id')}_{ph.get('id')}",
                            "weight": 2,  # 兜底来源，优先级最低
                        }
                    )
            page += 1
            time.sleep(0.4)
        print(f"    [inaturalist] {taxon}: +{len(out) - n_before}")
    return out


def read_meta_urls(meta_path: Path) -> set[str]:
    """已下载过的图片 URL / 来源页集合，用于断点续采。"""
    if not meta_path.exists():
        return set()
    out: set[str] = set()
    with meta_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for key in ("url", "source_page"):
                v = row.get(key)
                if v:
                    out.add(v.split("?")[0])
    return out


META_FIELDS = [
    "file",
    "class",
    "source",
    "url",
    "source_page",
    "license",
    "attribution",
    "query",
    "bytes",
]


def append_meta(meta_path: Path, row: dict) -> None:
    with _lock:
        write_header = not meta_path.exists() or meta_path.stat().st_size == 0
        with meta_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=META_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)


# --------------------------------------------------------------------------- #
# 下载
# --------------------------------------------------------------------------- #
def download(item: dict, dest: Path, idx: int) -> dict | None:
    path = dest / f"{dest.name}_{item['source']}_{idx:04d}.jpg"
    if path.exists():
        return None
    try:
        data = fetch_bytes(item["url"], timeout=60)
    except Exception as exc:  # noqa: BLE001
        print(f"    [失败] {item['name'][:55]}: {exc}")
        return None
    if len(data) < 5000:  # 太小基本是占位图/错误页
        return None
    path.write_bytes(data)
    time.sleep(0.05)
    return {
        "file": path.name,
        "class": dest.name,
        "source": item["source"],
        "url": item["url"].split("?")[0],
        "source_page": item["page"],
        "license": item["license"],
        "attribution": item["attribution"],
        "query": item["query"],
        "bytes": len(data),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="从开放图库采集水果图片")
    parser.add_argument(
        "--class",
        dest="classes",
        action="append",
        choices=sorted(CLASS_SPEC),
        help="要采集的类别，可重复：--class durian --class hami_melon",
    )
    parser.add_argument("--limit", type=int, default=500, help="每个类别最多下载多少张")
    parser.add_argument("--width", type=int, default=1280, help="Commons 缩略图宽度")
    parser.add_argument("--sources", default="wikimedia,inaturalist", help="逗号分隔的数据源")
    parser.add_argument(
        "--out",
        default=str(ROOT / "datasets" / "fruit" / "images" / "raw"),
        help="输出根目录",
    )
    parser.add_argument("--workers", type=int, default=2, help="并发下载线程数（别调太高，易被限流）")
    parser.add_argument("--seed", type=int, default=0, help="打乱候选顺序的随机种子")
    parser.add_argument("--dry-run", action="store_true", help="只统计候选数量，不下载")
    args = parser.parse_args()

    classes = args.classes or ["durian", "hami_melon"]
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    out_root = Path(args.out)
    rng = random.Random(args.seed)

    print(f"数据源: {sources}   每类上限: {args.limit}   图宽: {args.width}")
    total_new = 0

    for cls in classes:
        spec = CLASS_SPEC[cls]
        dest = out_root / cls
        dest.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {cls} → {dest} ===")

        pool: list[dict] = []
        if "wikimedia" in sources:
            pool += commons_categories(spec["categories"], args.width)
            pool += commons_search(spec["search"], args.width, spec["keywords"])
        if "inaturalist" in sources:
            pool += inat_items(spec["inat"], args.limit * 2)

        # 去重（URL / 标题），再按优先级分组、组内打乱
        uniq: dict[str, dict] = {}
        for it in pool:
            key = it["url"].split("?")[0]
            if key not in uniq:
                uniq[key] = it
        items = list(uniq.values())
        rng.shuffle(items)
        items.sort(key=lambda it: it["weight"])

        print(f"  候选合计 {len(items)}（去重后）")
        if args.dry_run:
            print("  [dry-run] 未下载")
            continue

        # 目标 = 目录里已有多少张（含手工放的图）→ 本轮只补差额
        existing = len(list(dest.glob("*.jpg")))
        need = max(0, args.limit - existing)
        if need == 0:
            print(f"  已有 {existing} 张，达到 --limit {args.limit}，跳过")
            continue

        meta_path = dest / "_meta.csv"
        done_urls = read_meta_urls(meta_path)
        items = [
            it
            for it in items
            if it["url"].split("?")[0] not in done_urls and it["page"] not in done_urls
        ]
        print(f"  断点续采：跳过已下载 {len(done_urls)} 个 URL，剩候选 {len(items)}")

        # 预留 50% 余量给下载失败/重复
        budget = min(len(items), int(need * 1.5) + 10)
        start_idx = 1 + max(
            [int(p.stem.rsplit("_", 1)[-1]) for p in dest.glob("*.jpg") if p.stem.rsplit("_", 1)[-1].isdigit()]
            or [0]
        )
        print(f"  已有 {existing} 张，本轮目标再补 {need} 张（候选 {budget}）")

        rows: list[dict] = []
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(download, it, dest, start_idx + i) for i, it in enumerate(items[:budget])]
            for fut in as_completed(futures):
                row = fut.result()
                if row:
                    rows.append(row)
                    if len(rows) % 25 == 0:
                        print(
                            f"    ... {len(rows)}/{need} 张，已用 {time.time() - t0:.0f}s",
                            flush=True,
                        )
                    append_meta(meta_path, rows[-1])
                if len(rows) >= need:
                    for f in futures:
                        f.cancel()
                    break
        by_src: dict[str, int] = {}
        for r in rows:
            by_src[r["source"]] = by_src.get(r["source"], 0) + 1
        elapsed = time.time() - t0
        print(f"  本轮下载 {len(rows)} 张 / 用时 {elapsed:.0f}s，来源分布: {by_src}")
        total_new += len(rows)

    if args.dry_run:
        print("\n[dry-run] 未下载任何文件")
        return

    total = len(list(out_root.glob("*/*.jpg")))
    print(f"\n本轮新增 {total_new} 张，raw 目录现有图片总数: {total}")
    print("下一步: python scripts/collect_images.py --dry-run")


if __name__ == "__main__":
    main()
