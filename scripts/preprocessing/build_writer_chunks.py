from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

"""
本模組負責將作者的原始範文 (Raw Articles) 處理成適合 RAG 使用的片段 (Chunks)。
主要流程：
1. 讀取原始 JSON 檔案。
2. 進行初步文字清洗。
3. 根據段落與字數限制進行智慧切分 (Chunking)。
4. 產出帶有 Metadata 的 JSONL 檔案，供後續建立向量索引。
"""

def read_json(path: Path) -> dict[str, Any]:
    """讀取單一 JSON 檔案。"""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Path) -> None:
    """確保目錄路徑存在。"""
    path.mkdir(parents=True, exist_ok=True)


def clean_text(text: str) -> str:
    """基礎文字清洗：統一換行符號、壓縮空格與過多換行。"""
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_paragraphs(text: str) -> list[str]:
    """
    將整篇文章切分為段落。
    
    策略：
    1. 優先尋找雙換行 (\n\n) 作為自然段落分隔。
    2. 若全文無明顯換行，則退而求其次，根據句號/感嘆號/問號進行「虛擬段落」切分。
    3. 確保每個段落的長度適中。
    """
    text = clean_text(text)
    if not text:
        return []

    # 1. 嘗試自然段落切分
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if len(paragraphs) > 1:
        return paragraphs

    # 2. 如果只有一段，則嘗試按句切分並重新組合成適當長度
    sentences = re.split(r"(?<=[。！？；])", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return [text]

    pseudo_paragraphs: list[str] = []
    buffer = ""

    for sentence in sentences:
        if len(buffer) + len(sentence) <= 120:
            buffer += sentence
        else:
            if buffer:
                pseudo_paragraphs.append(buffer.strip())
            buffer = sentence

    if buffer:
        pseudo_paragraphs.append(buffer.strip())

    return pseudo_paragraphs


def split_long_paragraph(paragraph: str, max_chars: int) -> list[str]:
    """
    若單一段落過長，則強制將其切分為更細的片段，避免超過 RAG 推薦字數。
    """
    if len(paragraph) <= max_chars:
        return [paragraph]

    parts: list[str] = []
    sentences = re.split(r"(?<=[。！？；])", paragraph)
    buffer = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(buffer) + len(sentence) <= max_chars:
            buffer += sentence
        else:
            if buffer:
                parts.append(buffer.strip())
                buffer = sentence
            else:
                # 若單一句子就超過 max_chars，則執行硬切分 (Hard Split)
                for i in range(0, len(sentence), max_chars):
                    parts.append(sentence[i : i + max_chars].strip())
                buffer = ""

    if buffer:
        parts.append(buffer.strip())

    return [p for p in parts if p]


def build_chunks_from_paragraphs(
    paragraphs: list[str],
    min_chars: int = 150,
    max_chars: int = 350,
) -> list[dict[str, Any]]:
    """
    核心切分邏輯：將多個段落組合成適當大小的 Chunk。
    
    規則：
    1. 若一個段落太長，先把它拆細。
    2. 將相鄰段落合併，直到接近 max_chars。
    3. 處理過短的「孤兒片段」，嘗試將其併入前一個 Chunk。
    """
    normalized_paragraphs: list[str] = []

    # 處理過長段落
    for paragraph in paragraphs:
        normalized_paragraphs.extend(
            split_long_paragraph(paragraph, max_chars=max_chars)
        )

    chunks: list[dict[str, Any]] = []
    buffer: list[str] = []
    char_count = 0

    # 合併段落
    for paragraph in normalized_paragraphs:
        paragraph_len = len(paragraph)

        if not buffer:
            buffer = [paragraph]
            char_count = paragraph_len
            continue

        # 如果加上這一段還沒爆量，就繼續塞
        if char_count + 1 + paragraph_len <= max_chars:
            buffer.append(paragraph)
            char_count += 1 + paragraph_len
            continue

        # 否則就結案一個 Chunk
        chunks.append(
            {
                "text": "\n\n".join(buffer).strip(),
                "char_count": len("\n\n".join(buffer).strip()),
                "paragraph_count": len(buffer),
            }
        )

        buffer = [paragraph]
        char_count = paragraph_len

    # 處理剩餘的 buffer
    if buffer:
        chunks.append(
            {
                "text": "\n\n".join(buffer).strip(),
                "char_count": len("\n\n".join(buffer).strip()),
                "paragraph_count": len(buffer),
            }
        )

    # 進階優化：合併過短的片段 (如只有一行結尾的話)
    merged_chunks: list[dict[str, Any]] = []
    for chunk in chunks:
        # 若當前片段太短，且前一個片段還有空間，則合併
        if (
            merged_chunks
            and chunk["char_count"] < min_chars
            and merged_chunks[-1]["char_count"] + 2 + chunk["char_count"]
            <= max_chars + 80 # 稍微放寬上限以容納短句
        ):
            merged_chunks[-1][
                "text"
            ] = f'{merged_chunks[-1]["text"]}\n\n{chunk["text"]}'.strip()
            merged_chunks[-1]["char_count"] = len(merged_chunks[-1]["text"])
            merged_chunks[-1]["paragraph_count"] += chunk["paragraph_count"]
        else:
            merged_chunks.append(chunk)

    return merged_chunks


def extract_article_text(article: dict[str, Any]) -> str:
    """從 JSON 字典中尋找可能的內文欄位。"""
    if isinstance(article.get("content"), str) and article["content"].strip():
        return article["content"].strip()

    candidate_fields = [
        "cleaned_text", "cleaned_source_text", "text", "body", "article",
    ]

    for field in candidate_fields:
        value = article.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()

    return ""


def extract_article_title(article: dict[str, Any], fallback: str) -> str:
    """從 JSON 字典中尋找標題欄位。"""
    if isinstance(article.get("title"), str) and article["title"].strip():
        return article["title"].strip()

    candidate_fields = ["source_title", "headline"]

    for field in candidate_fields:
        value = article.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()

    return fallback


def build_chunk_records(
    article_path: Path,
    article_json: dict[str, Any],
    min_chars: int,
    max_chars: int,
) -> list[dict[str, Any]]:
    """將單一 JSON 文章檔案轉換為多個 Chunk 紀錄字典。"""
    article_id = (
        article_json.get("article_id") or article_json.get("id") or article_path.stem
    )
    title = extract_article_title(article_json, fallback=article_path.stem)
    article_text = extract_article_text(article_json)

    if not article_text:
        return []

    # 1. 切分段落
    paragraphs = split_paragraphs(article_text)
    if not paragraphs:
        return []

    # 2. 執行核心切分邏輯
    chunks = build_chunks_from_paragraphs(
        paragraphs=paragraphs,
        min_chars=min_chars,
        max_chars=max_chars,
    )

    created_at = datetime.now().isoformat(timespec="seconds")
    results: list[dict[str, Any]] = []

    # 3. 封裝為最終格式
    for index, chunk in enumerate(chunks):
        results.append(
            {
                "chunk_id": f"{article_id}_chunk_{index:03d}",
                "article_id": article_id,
                "source_file": str(article_path.as_posix()),
                "title": title,
                "chunk_index": index,
                "text": chunk["text"],
                "char_count": chunk["char_count"],
                "paragraph_count": chunk["paragraph_count"],
                "created_at": created_at,
            }
        )

    return results


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """寫入 JSONL 格式檔案。"""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="將作者的文章切分為適合 RAG 用的風格片段 (Style Chunks)。"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/raw/author_articles",
        help="作者原始 JSON 文章目錄。",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="data/processed/writer_chunks/writer_chunks.jsonl",
        help="輸出的 JSONL 檔案路徑。",
    )
    parser.add_argument("--min-chars", type=int, default=150, help="片段最小字數預期。")
    parser.add_argument("--max-chars", type=int, default=350, help="片段最大字數上限。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_file = Path(args.output_file)

    article_paths = sorted(input_dir.glob("*.json"))
    all_chunks: list[dict[str, Any]] = []

    # 遍歷所有範文並執行切分
    for article_path in article_paths:
        article_json = read_json(article_path)
        article_chunks = build_chunk_records(
            article_path=article_path,
            article_json=article_json,
            min_chars=args.min_chars,
            max_chars=args.max_chars,
        )
        all_chunks.extend(article_chunks)

    # 儲存結果
    write_jsonl(output_file, all_chunks)

    # 輸出統計摘要
    print(
        json.dumps(
            {
                "status": "ok",
                "input_dir": str(input_dir),
                "output_file": str(output_file),
                "article_count": len(article_paths),
                "chunk_count": len(all_chunks),
                "min_chars": args.min_chars,
                "max_chars": args.max_chars,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
