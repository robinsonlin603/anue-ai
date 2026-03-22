from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

"""
本模組負責執行 Phase 1: Baseline 生成任務。
主要目標：
1. 建立「零樣本 (Zero-shot)」的基準模型輸出。
2. 僅提供簡單指令，不加入詳細風格規範或範文，用於評估模型原生轉譯財經新聞的能力。
3. 作為後續 Phase 2 (Style) 與 Phase 3 (RAG) 的對照組。
"""

# 若非作為套件執行，則將專案根目錄加入系統路徑，以便引用內部模組
if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.utils.generation_utils import (
    build_metrics,
    build_ollama_meta,
    build_source_stats,
    clean_generated_text,
    list_input_files,
    load_source_fields,
    make_source_id,
    get_dynamic_length_range,
)
from scripts.utils.io_utils import append_jsonl, read_text, write_json
from scripts.utils.logger_utils import setup_logger
from scripts.utils.ollama_client import OllamaGenerator
from scripts.utils.prompt_builder import build_prompt
from scripts.utils.text_cleaner import normalize_source_text

# 固定的來源新聞目錄
FIXED_INPUT_PATH = "data/raw/source_articles"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="執行 Phase 1 基準生成 (Baseline Generation)")
    parser.add_argument(
        "--input",
        type=str,
        default=FIXED_INPUT_PATH,
        help="輸入檔案或目錄路徑。",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="dataset/prompts/phase1_baseline_prompt.txt",
        help="Prompt 模板路徑。",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/generations/phase1",
        help="生成結果輸出目錄。",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="outputs/logs/phase1",
        help="日誌紀錄目錄。",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemma2:9b",
        help="Ollama 模型名稱。",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="http://localhost:11434",
        help="Ollama API 位址。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.input = FIXED_INPUT_PATH

    # 初始化路徑與 ID
    input_path = Path(args.input)
    prompt_path = Path(args.prompt)
    output_dir = Path(args.output_dir)
    log_dir = Path(args.log_dir)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logger(log_dir / f"run_{run_id}.log")
    jsonl_log_path = log_dir / f"run_{run_id}.jsonl"

    # 1. 載入模板與初始化生成器
    prompt_template = read_text(prompt_path)
    generator = OllamaGenerator(
        model=args.model,
        host=args.host,
        temperature=0.3, # 較低溫度以維持專業性
        num_predict=800,
        top_p=0.9,
    )

    # 2. 搜尋待處理檔案
    input_files = list_input_files(input_path)
    logger.info("找到 %s 個輸入檔案。", len(input_files))

    # 3. 遍歷並處理每一篇新聞
    for file_path in input_files:
        try:
            # 載入原始標題與內文
            raw_title, raw_text = load_source_fields(file_path)

            # 文字標準化清洗 (處理 HTML 字元、空白等)
            cleaned_title = normalize_source_text(raw_title)
            cleaned_source_text = normalize_source_text(raw_text)

            # 動態計算目標生成的字數範圍
            target_min, target_max = get_dynamic_length_range(cleaned_source_text)

            # 組裝完整 Prompt
            prompt = build_prompt(
                template=prompt_template,
                title=cleaned_title,
                content=cleaned_source_text,
                target_min=target_min,
                target_max=target_max,
            )

            source_id = make_source_id(file_path, cleaned_title, cleaned_source_text)

            # 執行 Ollama 生成
            logger.info("正在為其生成文章: %s", file_path.name)
            response = generator.generate(prompt)
            
            # 清理生成內容 (移除 Markdown, 處理格式)
            generated_text = clean_generated_text(response.get("response", ""))

            # 蒐集統計數據與指標
            source_stats = build_source_stats(
                raw_title=raw_title,
                raw_text=raw_text,
                cleaned_title=cleaned_title,
                cleaned_text=cleaned_source_text,
            )
            metrics = build_metrics(generated_text)
            ollama_meta = build_ollama_meta(response)

            # 4. 儲存詳細結果 (JSON)
            output_record: dict[str, Any] = {
                "run_id": run_id,
                "phase": "phase1",
                "source_id": source_id,
                "source_file": str(file_path),
                "model": args.model,
                "host": args.host,
                "prompt_file": str(prompt_path),
                "prompt": prompt,
                "source_title": cleaned_title,
                "source_text": raw_text,
                "cleaned_source_text": cleaned_source_text,
                "generated_text": generated_text,
                "source_stats": source_stats,
                "metrics": metrics,
                "ollama_meta": ollama_meta,
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }

            output_path = output_dir / f"{file_path.stem}.json"
            write_json(output_path, output_record)

            # 5. 紀錄到 JSONL 日誌 (便於後續大批次掃描分析)
            append_jsonl(
                jsonl_log_path,
                {
                    "run_id": run_id,
                    "source_file": str(file_path),
                    "output_file": str(output_path),
                    "source_id": source_id,
                    "status": "success",
                    "source_stats": source_stats,
                    "metrics": metrics,
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                },
            )

            logger.info("已將輸出儲存至 %s", output_path)

        except Exception as exc:
            logger.exception("處理檔案失敗: %s", file_path)
            append_jsonl(
                jsonl_log_path,
                {
                    "run_id": run_id,
                    "source_file": str(file_path),
                    "status": "failed",
                    "error": str(exc),
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                },
            )


if __name__ == "__main__":
    main()
