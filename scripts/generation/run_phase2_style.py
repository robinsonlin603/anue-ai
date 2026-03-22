from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

"""
本模組負責執行 Phase 2: Style 風格化生成任務。
主要目標：
1. 引入詳細的「風格規範說明 (Style Specification)」。
2. 透過 Prompt Engineering 引導模型使用特定的財經語氣、結構與禁止事項 (如禁止 Markdown)。
3. 相比 Phase 1，此階段開始嘗試模擬特定作者的文字風格，但尚未引入 RAG 範文。
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

# 載入預定義的風格規範
from configs.style_specs import PHASE2_STYLE_SPEC

# 固定的來源新聞目錄
FIXED_INPUT_PATH = "data/raw/source_articles"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="執行 Phase 2 風格化生成 (Style Generation)")
    parser.add_argument(
        "--input",
        type=str,
        default=FIXED_INPUT_PATH,
        help="輸入檔案或目錄路徑。",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="dataset/prompts/phase2_style_prompt.txt",
        help="Prompt 模板路徑。",
    )
    parser.add_argument(
        "--prompt-version",
        type=str,
        default="style_v1",
        help="Prompt 版本標籤 (用於實驗追蹤)。",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/generations/phase2",
        help="生成結果輸出目錄。",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="outputs/logs/phase2",
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
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="採樣溫度 (預設 0.3)。",
    )
    parser.add_argument(
        "--num-predict",
        type=int,
        default=800,
        help="最大生成 Token 數。",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p 採樣門檻 (預設 0.9)。",
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
        temperature=args.temperature,
        num_predict=args.num_predict,
        top_p=args.top_p,
    )

    # 2. 搜尋待處理檔案
    input_files = list_input_files(input_path)
    logger.info("找到 %s 個輸入檔案。", len(input_files))

    # 3. 遍歷並處理每一篇新聞
    for file_path in input_files:
        try:
            # 載入原始數據
            raw_title, raw_text = load_source_fields(file_path)

            # 文字標準化清洗
            cleaned_title = normalize_source_text(raw_title)
            cleaned_source_text = normalize_source_text(raw_text)

            # 動態計算目標生成的字數範圍
            target_min, target_max = get_dynamic_length_range(cleaned_source_text)

            # 組裝完整 Prompt (在此階段注入 style_spec)
            prompt = build_prompt(
                template=prompt_template,
                title=cleaned_title,
                content=cleaned_source_text,
                style_spec=PHASE2_STYLE_SPEC, # 注入風格規範
                target_min=target_min,
                target_max=target_max,
            )

            source_id = make_source_id(file_path, cleaned_title, cleaned_source_text)

            # 執行文字生成
            logger.info("正在生成 Phase 2 風格化文章: %s", file_path.name)
            response = generator.generate(prompt)
            
            # 清理生成內容
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
                "phase": "phase2",
                "source_id": source_id,
                "source_file": str(file_path),
                "model": args.model,
                "host": args.host,
                "prompt_file": str(prompt_path),
                "prompt_version": args.prompt_version,
                "prompt": prompt,
                "style_spec": PHASE2_STYLE_SPEC,
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

            # 5. 紀錄到 JSONL 日誌
            append_jsonl(
                jsonl_log_path,
                {
                    "run_id": run_id,
                    "phase": "phase2",
                    "source_file": str(file_path),
                    "output_file": str(output_path),
                    "source_id": source_id,
                    "prompt_version": args.prompt_version,
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
                    "phase": "phase2",
                    "source_file": str(file_path),
                    "prompt_version": args.prompt_version,
                    "status": "failed",
                    "error": str(exc),
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                },
            )


if __name__ == "__main__":
    main()
