from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

"""
本模組負責執行 Phase 3: RAG + Self-Correction (檢索增強生成與自我修正)。
這是專案中邏輯最複雜的生成階段，主要包含：
1. 語意檢索 (RAG)：根據來源新聞標題與前文，從資料庫中搜尋風格相近的作者範文。
2. 智慧挑選：根據新聞類型 (市場、產品等) 挑選最合適的範文片段。
3. 自我修正 (Self-Correction Loop)：生成後自動檢查字數、格式與「數字保留率」。
4. 數字強化：若檢查不通過，觸發 Refine 流程，強制模型保留原始新聞中的關鍵數據。
"""

# 若非作為套件執行，則將專案根目錄加入系統路徑
if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.utils.io_utils import read_text, write_json
from scripts.utils.ollama_client import OllamaGenerator
from scripts.utils.prompt_builder import build_prompt
from scripts.utils.retrieval_utils import (
    build_rag_context,
    retrieve_style_chunks,
    classify_article_type,
)
from scripts.utils.text_cleaner import normalize_source_text
from scripts.utils.generation_utils import (
    clean_generated_text,
    build_metrics,
    build_ollama_meta,
    build_source_stats,
    make_source_id,
    get_dynamic_length_range,
)

from scripts.evaluation.scorer import extract_numbers
from configs.style_specs import PHASE3_STYLE_SPEC

# 來源目錄 (Phase 3 預設基於 Phase 2 的結果進行優化，或直接使用原始文章)
FIXED_INPUT_PATH = "outputs/generations/phase2"

# 修正閾值與參數
PHASE3_PARAGRAPH_LINES_MIN = 4  # 標題(1) + 內文至少 3 段
PHASE3_PARAGRAPH_LINES_MAX = 7  # 標題(1) + 內文至多 6 段
MIN_NUMERIC_OVERLAP_RATIO = 0.25 # 最低數字保留率 (確保事實真實性)


def _has_bullet_list(text: str) -> bool:
    """檢查文字中是否含有列點格式。"""
    bullet_patterns = [
        r"^\s*[-*]\s+",
        r"^\s*\d+[\.\)]\s+",
        r"^\s*•\s+",
    ]
    import re
    return any(re.search(p, text, flags=re.MULTILINE) for p in bullet_patterns)


def _has_section_header(text: str) -> bool:
    """檢查文字中是否含有 Markdown 標題或特定小標題格式。"""
    header_patterns = [
        r"^\s*#{1,6}\s+",
        r"^\s*【.+】\s*$",
    ]
    import re
    return any(re.search(p, text, flags=re.MULTILINE) for p in header_patterns)


def _numeric_overlap_ratio(source_text: str, output_text: str) -> float:
    """
    計算「數字保留率」。
    比較來源文章與生成文章中阿拉伯數字的重合度，
    這是財經新聞事實查核 (Fact-checking) 的核心指標。
    """
    source_numbers = extract_numbers(source_text)
    if not source_numbers:
        return 1.0 # 來源無數字則視為完全保留

    output_numbers = extract_numbers(output_text)
    if not output_numbers:
        return 0.0 # 生成無數字則保留率為 0

    overlap = len(source_numbers & output_numbers) / len(source_numbers)
    return overlap


def needs_rewrite(
    generated_text: str, metrics: dict[str, Any], source_text: str, target_min: int, target_max: int
) -> bool:
    """
    判斷生成內容是否需要進入「重寫修正」流程。
    
    觸發條件：
    1. 含有禁止的列點格式。
    2. 含有禁止的小標題。
    3. 產生了虛假日期/地點 (Hallucination)。
    4. 數字保留率過低 (代表遺漏關鍵財經資訊)。
    5. 字數或段落數不符規範。
    """
    output_char_count = metrics.get("output_char_count", 0)
    paragraph_count = metrics.get("paragraph_count", 0)

    within_target_length = target_min <= output_char_count <= target_max
    paragraph_count_valid = PHASE3_PARAGRAPH_LINES_MIN <= paragraph_count <= PHASE3_PARAGRAPH_LINES_MAX

    numeric_overlap_ratio = _numeric_overlap_ratio(source_text, generated_text)
    numeric_retention_valid = numeric_overlap_ratio >= MIN_NUMERIC_OVERLAP_RATIO

    return (
        _has_bullet_list(generated_text)
        or _has_section_header(generated_text)
        or metrics.get("has_hallucinated_dateline", False)
        or not numeric_retention_valid
        or not within_target_length
        or not paragraph_count_valid
    )


def build_rewrite_prompt(draft_text: str, source_text: str, target_min: int, target_max: int) -> str:
    """
    建立修正專用的 Prompt (Refine Prompt)。
    特別強調數字的保留，並將來源文章中的所有數字列出，要求模型強制保留。
    """
    source_numbers = sorted(extract_numbers(source_text), key=len, reverse=True)
    source_numbers_preview = ", ".join(source_numbers[:50])

    return f"""Refine the following draft into a proper Traditional Chinese financial news article. 

Your goal is to fix the FORMAT and TONE while preserving ALL factual information.

Strict Rules:
1. DO NOT drop, change, or rewrite any numbers, dates, or financial figures.
2. YOU MUST keep these specific numeric values found in the source: {source_numbers_preview}
3. If the draft contains markdown (headings #, bullet points -), REMOVE them and convert into plain paragraphs.
4. Output must be exactly: One title line, followed by 3 to 5 plain text paragraphs.
5. Maintain a neutral, objective financial news tone.
6. Target length is {target_min} to {target_max} characters.

Draft to refine:
{draft_text}
"""


def read_json(path: Path) -> dict[str, Any]:
    """讀取 JSON。"""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_source_text(article_json: dict[str, Any]) -> str:
    """從不同階段的 JSON 格式中兼容地提取原始文章內容。"""
    fields = [
        "cleaned_source_text", "cleaned_text", "source_text", "text", "content", "body",
    ]
    for field in fields:
        value = article_json.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def select_prompt_chunks(
    chunks: list[dict[str, Any]],
    query_type: str,
    max_chunks: int = 2,
) -> list[dict[str, Any]]:
    """
    從 RAG 檢索到的多個片段中，根據當前文章類型挑選最適合的「風格範文」。
    例如：若目前是市場分析新聞，則排除掉純產品介紹的範文片段。
    """
    preferred: list[dict[str, Any]] = []
    fallback: list[dict[str, Any]] = []

    for chunk in chunks:
        if chunk.get("is_list_heavy", False):
            continue

        if query_type == "market":
            if chunk.get("is_product_heavy", False) or chunk.get("is_event_heavy", False):
                continue

        if not chunk.get("is_product_heavy", False) and not chunk.get("is_event_heavy", False):
            preferred.append(chunk)
        else:
            fallback.append(chunk)

    selected = preferred[:max_chunks]

    # 若 Preferred 不足且非市場類新聞，則允許使用 Fallback 片段補足
    if query_type != "market" and len(selected) < max_chunks:
        selected.extend(fallback[: max_chunks - len(selected)])

    return selected


def extract_source_title(article_json: dict[str, Any], fallback: str) -> str:
    """提取原始新聞標題。"""
    fields = ["source_title", "title", "headline"]
    for field in fields:
        value = article_json.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return fallback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="執行 Phase 3: RAG + 自我修正生成流程。")
    parser.add_argument("--input", type=str, default=FIXED_INPUT_PATH)
    parser.add_argument("--prompt", type=str, default="dataset/prompts/phase3_rag_prompt.txt")
    parser.add_argument("--output-dir", type=str, default="outputs/generations/phase3")
    parser.add_argument("--log-dir", type=str, default="outputs/logs/phase3")
    parser.add_argument("--retrieval-dir", type=str, default="outputs/retrieval/phase3")
    parser.add_argument("--model", type=str, default="gemma2:9b")
    parser.add_argument("--host", type=str, default="http://localhost:11434")
    parser.add_argument("--top-k", type=int, default=3, help="RAG 初步檢索數量")
    parser.add_argument("--per-article-limit", type=int, default=1, help="同一範文限制引用次數")
    parser.add_argument("--collection-name", type=str, default="writer_style_chunks")
    parser.add_argument("--db-path", type=str, default="data/indexes/writer_style_index")
    parser.add_argument("--embed-model", type=str, default="nomic-embed-text-v2-moe")
    parser.add_argument("--prompt-version", type=str, default="rag_v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.input = FIXED_INPUT_PATH

    # 初始化路徑
    input_dir = Path(args.input)
    prompt_file = Path(args.prompt)
    output_dir = Path(args.output_dir)
    log_dir = Path(args.log_dir)
    retrieval_dir = Path(args.retrieval_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    retrieval_dir.mkdir(parents=True, exist_ok=True)

    prompt_template = read_text(prompt_file)
    generator = OllamaGenerator(
        model=args.model,
        host=args.host,
        temperature=0.1, # 極低溫度，因為 Phase 3 追求精準度
        num_predict=800,
        top_p=0.9,
    )
    source_paths = sorted(input_dir.glob("*.json"))

    for source_path in source_paths:
        # 1. 準備來源資料
        source_json = read_json(source_path)
        source_title = extract_source_title(source_json, fallback=source_path.stem)
        source_text = extract_source_text(source_json)
        cleaned_title = normalize_source_text(source_title)
        cleaned_source_text = normalize_source_text(source_text)
        source_id = source_path.stem

        # 2. 執行 RAG 檢索
        # 使用標題加上前 400 字作為查詢字串，以獲取最精準的風格匹配
        MAX_QUERY_CHARS = 400
        query_body = cleaned_source_text[:MAX_QUERY_CHARS]
        query_text = f"{cleaned_title}\n{query_body}".strip()

        retrieved_chunks = retrieve_style_chunks(
            query_text=query_text,
            top_k=args.top_k,
            per_article_limit=args.per_article_limit,
            db_path=args.db_path,
            collection_name=args.collection_name,
            embed_model=args.embed_model,
            host=args.host,
        )

        # 挑選最契合的範文片段並組裝上下文
        query_type = classify_article_type(query_text)
        selected_chunks = select_prompt_chunks(retrieved_chunks, query_type, max_chunks=2)
        rag_context = build_rag_context(selected_chunks)
        
        # 3. 第一次生成
        target_min, target_max = get_dynamic_length_range(cleaned_source_text)
        prompt = build_prompt(
            template=prompt_template,
            title=cleaned_title,
            content=cleaned_source_text,
            rag_context=rag_context,
            style_spec=PHASE3_STYLE_SPEC,
            target_min=target_min,
            target_max=target_max,
        )

        ollama_response = generator.generate(prompt)
        generated_text = clean_generated_text(ollama_response.get("response", ""))
        metrics = build_metrics(generated_text)

        # 4. 自我修正檢查 (Rewriting Loop)
        # 檢查生成的初稿是否符合各項嚴格指標
        if needs_rewrite(generated_text, metrics, cleaned_source_text, target_min, target_max):
            rewrite_prompt = build_rewrite_prompt(generated_text, cleaned_source_text, target_min, target_max)
            rewrite_response = generator.generate(rewrite_prompt)
            rewritten_text = clean_generated_text(rewrite_response.get("response", ""))
            rewritten_metrics = build_metrics(rewritten_text)

            # 驗證修正後的結果是否真的變好了
            numeric_overlap_ratio = _numeric_overlap_ratio(cleaned_source_text, rewritten_text)
            numeric_retention_valid = numeric_overlap_ratio >= MIN_NUMERIC_OVERLAP_RATIO
            paragraph_count_valid = PHASE3_PARAGRAPH_LINES_MIN <= rewritten_metrics.get("paragraph_count", 0) <= PHASE3_PARAGRAPH_LINES_MAX

            # 若修正版符合基本規範，則採用修正版；否則保留原版避免越改越爛
            if (
                target_min <= rewritten_metrics.get("output_char_count", 0) <= target_max
                and not rewritten_metrics["has_markdown"]
                and not rewritten_metrics["has_bullet_list"]
                and not rewritten_metrics["has_hallucinated_dateline"]
                and paragraph_count_valid
                and numeric_retention_valid
            ):
                generated_text = rewritten_text
                metrics = rewritten_metrics
                ollama_response = rewrite_response # 更新元數據

        # 5. 儲存結果與元數據
        created_at = datetime.now().isoformat(timespec="seconds")
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        source_stats = build_source_stats(
            raw_title=cleaned_title, raw_text=cleaned_source_text,
            cleaned_title=cleaned_title, cleaned_text=cleaned_source_text,
        )

        output_payload = {
            "run_id": run_id,
            "phase": "phase3",
            "source_id": source_id,
            "source_file": str(source_path.as_posix()),
            "model": args.model,
            "host": args.host,
            "prompt_file": str(prompt_file.as_posix()),
            "prompt_version": args.prompt_version,
            "prompt": prompt,
            "source_title": source_title,
            "source_text": source_text,
            "cleaned_source_text": cleaned_source_text,
            "rag_context": rag_context,
            "retrieval_meta": {
                "top_k": args.top_k,
                "per_article_limit": args.per_article_limit,
                "collection_name": args.collection_name,
                "db_path": args.db_path,
                "embed_model": args.embed_model,
                "query_max_chars": MAX_QUERY_CHARS,
            },
            "retrieval_query_text": query_text,
            "retrieved_chunks": retrieved_chunks,
            "generated_text": generated_text,
            "source_stats": source_stats,
            "metrics": metrics,
            "ollama_meta": build_ollama_meta(ollama_response),
            "created_at": created_at,
            "prompt_chunks": selected_chunks,
            "query_type": query_type,
        }

        # 分別存入檢索紀錄、最終輸出與 Log 目錄
        write_json(output_dir / f"{source_id}.json", output_payload)
        write_json(retrieval_dir / f"{source_id}.json", {
            "run_id": run_id, "source_id": source_id, "query_text": query_text,
            "retrieval_meta": output_payload["retrieval_meta"],
            "retrieved_chunks": retrieved_chunks, "created_at": created_at,
        })
        write_json(log_dir / f"{source_id}.json", output_payload)

        print(f"Phase 3 OK: {source_id} (Retrieved: {len(retrieved_chunks)})")


if __name__ == "__main__":
    main()
