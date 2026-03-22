import argparse
import json
import html
import re
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
AUTHOR_ARTICLES_DIR = BASE_DIR / "data" / "raw" / "author_articles"
SOURCE_ARTICLES_DIR = BASE_DIR / "data" / "raw" / "source_articles"


TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")


def strip_html(text: str) -> str:
    unescaped = html.unescape(text)
    without_tags = TAG_RE.sub("", unescaped)
    normalized = WHITESPACE_RE.sub(" ", without_tags).strip()
    return normalized


def process_article(path: Path) -> None:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    content = data.get("content")
    if isinstance(content, str):
        data["content"] = strip_html(content)

        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def process_dir(dir_path: Path) -> None:
    if not dir_path.exists():
        return

    for json_path in sorted(dir_path.glob("article-*.json")):
        process_article(json_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strip HTML / normalize whitespace in article JSON `content` fields."
    )
    parser.add_argument(
        "--author-only",
        action="store_true",
        help="Only process data/raw/author_articles",
    )
    parser.add_argument(
        "--source-only",
        action="store_true",
        help="Only process data/raw/source_articles",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.author_only and args.source_only:
        raise SystemExit(
            "Use only one of --author-only or --source-only, or neither for both."
        )
    if args.author_only:
        process_dir(AUTHOR_ARTICLES_DIR)
        return
    if args.source_only:
        process_dir(SOURCE_ARTICLES_DIR)
        return
    process_dir(AUTHOR_ARTICLES_DIR)
    process_dir(SOURCE_ARTICLES_DIR)


if __name__ == "__main__":
    main()
