from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

"""
本模組負責 Phase 1 到 Phase 4 所有生成結果的綜合評估與比較。
主要功能：
1. 讀取各階段 (Baseline, Style, RAG, LoRA) 針對同一篇新聞產出的結果。
2. 調用 Scorer 模組進行自動化評分 (包含事實保留、長度控制、風格等指標)。
3. 自動選出表現最好的版本 (Winner Phase)。
4. 產出綜合比較 JSON 檔案，作為 Phase 4 微調資料集的篩選依據。
"""

# 若非作為套件執行，則將專案根目錄加入系統路徑
if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.utils.generation_utils import clean_generated_text, load_source_fields
from scripts.utils.io_utils import write_json
from scripts.utils.text_cleaner import normalize_source_text

from scripts.evaluation.scorer import evaluate

# 設定固定的輸入輸出路徑
FIXED_INPUT_PATH = "data/raw/source_articles"
FIXED_OUTPUT_PATH = "outputs/comparisons"
FIXED_MODEL = "gemma2:9b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="綜合比較各階段 (Phase 1-4) 的生成品質。")
    parser.add_argument(
        "--phase1-prompt", type=str, default="dataset/prompts/phase1_baseline_prompt.txt",
    )
    parser.add_argument(
        "--phase2-prompt", type=str, default="dataset/prompts/phase2_style_prompt.txt",
    )
    parser.add_argument(
        "--phase2-prompt-version", type=str, default="style_v1",
    )
    parser.add_argument(
        "--input", type=str, default=FIXED_INPUT_PATH,
    )
    parser.add_argument("--output", type=str, default=FIXED_OUTPUT_PATH)
    parser.add_argument("--model", type=str, default=FIXED_MODEL)
    parser.add_argument(
        "--host", type=str, default="http://localhost:11434",
    )
    return parser.parse_args()


def list_input_files(input_path: Path) -> list[Path]:
    """獲取待評估的新聞檔案列表。"""
    if input_path.is_file():
        return [input_path]
    return sorted(input_path.glob("article-*.json"))


def resolve_output_path(output_arg: str, input_file: Path, is_batch: bool) -> Path:
    """決定比較結果 JSON 的儲存路徑。"""
    output_path = Path(output_arg)
    if is_batch:
        output_dir = (
            output_path if output_path.suffix != ".json" else output_path.parent
        )
        return output_dir / f"{input_file.stem}_compare.json"

    if output_path.suffix == ".json":
        return output_path
    return output_path / f"{input_file.stem}_compare.json"


def compare_one(
    input_path: Path,
    output_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """
    針對單篇新聞，比較各個階段的輸出。
    """
    # 載入原始資料
    raw_title, raw_text = load_source_fields(input_path)
    cleaned_title = normalize_source_text(raw_title)
    cleaned_source_text = normalize_source_text(raw_text)

    # 各階段結果的存放目錄
    phase1_dir = Path("outputs/generations/phase1")
    phase2_dir = Path("outputs/generations/phase2")
    phase3_dir = Path("outputs/generations/phase3")
    phase4_dir = Path("outputs/generations/phase4")
    phase4_1_dir = Path("outputs/generations/phase4_1") # 實驗用的額外變體
    article_stem = input_path.stem

    # 組裝各階段輸出檔案的路徑
    phase1_output_path = phase1_dir / f"{article_stem}.json"
    phase2_output_path = phase2_dir / f"{article_stem}.json"
    phase3_output_path = phase3_dir / f"{article_stem}.json"
    phase4_output_path = phase4_dir / f"{article_stem}.json"
    phase4_1_output_path = phase4_1_dir / f"{article_stem}.json"

    # 檢查檔案完整性
    for p in [phase1_output_path, phase2_output_path, phase3_output_path, phase4_output_path]:
        if not p.exists():
            raise FileNotFoundError(f"缺少階段輸出檔案: {p}")

    import json

    # 讀取生成結果
    phase1_data = json.loads(phase1_output_path.read_text(encoding="utf-8"))
    phase2_data = json.loads(phase2_output_path.read_text(encoding="utf-8"))
    phase3_data = json.loads(phase3_output_path.read_text(encoding="utf-8"))
    phase4_data = json.loads(phase4_output_path.read_text(encoding="utf-8"))
    
    # phase4_1 是可選的，不存在則設為空
    phase4_1_data = {}
    if phase4_1_output_path.exists():
        phase4_1_data = json.loads(phase4_1_output_path.read_text(encoding="utf-8"))

    # 清洗文字以便精準評分
    phase1_text = clean_generated_text(phase1_data.get("generated_text", ""))
    phase2_text = clean_generated_text(phase2_data.get("generated_text", ""))
    phase3_text = clean_generated_text(phase3_data.get("generated_text", ""))
    phase4_text = clean_generated_text(phase4_data.get("generated_text", "") or phase4_data.get("generated_content", ""))
    phase4_1_text = clean_generated_text(phase4_1_data.get("generated_text", ""))

    # 核心評分環節：調用 scorer.py 的 evaluate 函式
    phase1_scores = evaluate(cleaned_source_text, phase1_text)
    phase2_scores = evaluate(cleaned_source_text, phase2_text)
    phase3_scores = evaluate(cleaned_source_text, phase3_text)
    phase4_scores = evaluate(cleaned_source_text, phase4_text)
    phase4_1_scores = evaluate(cleaned_source_text, phase4_1_text) if phase4_1_text else {"total": 0}

    # 提取 Phase 3 的特殊 RAG 資訊
    phase3_query_type = phase3_data.get("query_type", "")
    phase3_retrieved_chunks = phase3_data.get("retrieved_chunks", [])

    print(f"[Scores] {article_stem} -> P1:{phase1_scores['total']} P2:{phase2_scores['total']} P3:{phase3_scores['total']} P4:{phase4_scores['total']}")

    # 優勝者邏輯 (Winner Logic)：選出總分最高者
    scores_dict = {
        "phase1": phase1_scores["total"],
        "phase2": phase2_scores["total"],
        "phase3": phase3_scores["total"],
        "phase4": phase4_scores["total"]
    }
    if phase4_1_text:
        scores_dict["phase4.1"] = phase4_1_scores["total"]
        
    winner = max(scores_dict, key=scores_dict.get)

    # 組裝詳細的綜合報告
    output = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_file": str(input_path),
        "source_title": cleaned_title,
        "cleaned_source_text": cleaned_source_text,
        "phase1": {
            "generated_text": phase1_text,
            "scores": phase1_scores,
        },
        "phase2": {
            "generated_text": phase2_text,
            "scores": phase2_scores,
        },
        "phase3": {
            "generated_text": phase3_text,
            "scores": phase3_scores,
            "query_type": phase3_query_type,
            "retrieved_chunks": phase3_retrieved_chunks,
        },
        "phase4": {
            "generated_text": phase4_text,
            "scores": phase4_scores,
        },
        "winner": winner,
        "all_scores": scores_dict
    }

    # 儲存報告
    write_json(output_path, output)
    
    return {
        "article_stem": input_path.stem,
        "phase1_total": phase1_scores.get("total"),
        "phase2_total": phase2_scores.get("total"),
        "phase3_total": phase3_scores.get("total"),
        "phase4_total": phase4_scores.get("total"),
        "winner": winner,
    }


def main() -> None:
    args = parse_args()
    # 強制使用預設路徑以確保專案一致性
    args.input = FIXED_INPUT_PATH
    args.output = FIXED_OUTPUT_PATH
    args.model = FIXED_MODEL

    input_path = Path(args.input)
    input_files = list_input_files(input_path)
    if not input_files:
        raise FileNotFoundError(f"找不到輸入檔案: {input_path}")

    # 批次處理統計資訊
    winner_counter: dict[str, int] = {"phase1": 0, "phase2": 0, "phase3": 0, "phase4": 0}
    per_article_rows: list[dict[str, Any]] = []
    
    for file_path in input_files:
        try:
            output_path = resolve_output_path(args.output, file_path, True)
            row = compare_one(file_path, output_path, args)
            winner = row["winner"]
            winner_counter[winner] = winner_counter.get(winner, 0) + 1
            per_article_rows.append(row)
        except Exception as e:
            print(f"[Error] 跳過 {file_path.name}: {e}")

    # 輸出最終總表摘要
    print("\n" + "=" * 30)
    print("=== 各階段勝出統計 (Winner Summary) ===")
    print("=" * 30)
    for k, v in winner_counter.items():
        print(f"{k:10}: {v} 篇")

    # 輸出每篇文章的詳細對比表
    per_article_rows.sort(key=lambda r: r["article_stem"])
    print("\n=== 每篇詳細得分 (Per-Article Summary) ===")
    print("文章 ID\t\tP1\tP2\tP3\tP4\t優勝者")
    print("-" * 60)
    for row in per_article_rows:
        print(
            f'{row["article_stem"]:15}\t'
            f'{row["phase1_total"]}\t'
            f'{row["phase2_total"]}\t'
            f'{row["phase3_total"]}\t'
            f'{row["phase4_total"]}\t'
            f'{row["winner"]}'
        )


if __name__ == "__main__":
    main()
