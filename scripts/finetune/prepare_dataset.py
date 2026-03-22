from __future__ import annotations

import json
import sys
from pathlib import Path
from datetime import datetime

"""
本模組負責準備微調 (Fine-tuning) 所需的訓練資料集。
主要流程：
1. 讀取 Phase 1~3 的評估比較結果 (outputs/comparisons)。
2. 自動挑選出每個樣本中表現最好的版本 (Winner Phase)。
3. 建立符合「指令微調 (Instruction Fine-tuning)」格式的資料對。
4. 產出 train.jsonl 供 Unsloth 或其他訓練框架使用。
"""

# 若非作為套件執行，則將專案根目錄加入系統路徑，以便引用內部模組
if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# 定義路徑
COMPARISONS_DIR = Path("outputs/comparisons") # 存放之前階段比較結果的目錄
OUTPUT_JSONL_DIR = Path("dataset/jsonl")      # 訓練資料集輸出目錄
OUTPUT_FILE = OUTPUT_JSONL_DIR / "train.jsonl" # 最終訓練檔案

def extract_winner_data(file_path: Path) -> dict | None:
    """
    從比較的 JSON 檔案中提取來源 (Input) 與優勝者的輸出內容 (Winner Output)。
    """
    try:
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        
        # 取得系統標記的優勝階段 (例如: "phase3")
        winner_phase = data.get("scores", {}).get("winner")
        if not winner_phase:
            print(f"  [跳過] {file_path.name} 中找不到優勝者標記。")
            return None
        
        # 提取原始新聞內容作為 Input
        source_text = data.get("cleaned_source_text")
        
        # 提取該優勝階段生成的文字作為訓練目標 (Target)
        winner_data = data.get(winner_phase, {})
        target_text = winner_data.get("generated_text")
        
        if not source_text or not target_text:
            print(f"  [跳過] {file_path.name} 缺少來源或目標文字。")
            return None
        
        # 根據來源長度計算目標動態範圍 (與生產環境的 Prompt 保持一致)
        from scripts.utils.generation_utils import get_dynamic_length_range
        target_min, target_max = get_dynamic_length_range(source_text)
        
        # 組合微調指令 (Instruction)
        # 這段文字必須與我們在 Phase 1~3 中使用的風格一致，模型才會學得更好。
        instruction = f"你是一位專業的財經新聞編輯。請將以下原始新聞內容改寫為專業、客觀且具備特定作者風格的繁體中文財經報導。要求：標題一行，內文3至5段，總字數約 {target_min}–{target_max} 字，嚴禁使用 Markdown 或列點。"

        return {
            "instruction": instruction,
            "input": source_text,
            "output": target_text,
            "metadata": {
                "source_file": data.get("source_file"),
                "winner_phase": winner_phase,
                "score": data.get("scores", {}).get(winner_phase)
            }
        }
    except Exception as e:
        print(f"  [錯誤] 處理 {file_path.name} 時發生異常: {e}")
        return None

def main():
    print("=== Phase 4: 正在準備微調資料集 (Dataset Preparation) ===")
    
    # 檢查必要目錄
    if not COMPARISONS_DIR.exists():
        print(f"錯誤：找不到比較目錄 {COMPARISONS_DIR}")
        return

    OUTPUT_JSONL_DIR.mkdir(parents=True, exist_ok=True)
    
    # 尋找所有比較檔案
    compare_files = list(COMPARISONS_DIR.glob("*_compare.json"))
    print(f"找到 {len(compare_files)} 個比較檔案。")

    dataset = []
    # 統計各階段勝出的數量，用於評估資料來源分布
    winner_counts = {"phase1": 0, "phase2": 0, "phase3": 0}

    for file_path in sorted(compare_files):
        entry = extract_winner_data(file_path)
        if entry:
            dataset.append(entry)
            phase = entry["metadata"]["winner_phase"]
            winner_counts[phase] += 1

    # 寫入 JSONL 格式 (每行一個字典)
    # 這種格式被 Unsloth, LLaMA-Factory 等工具廣泛支援。
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for entry in dataset:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")

    # 輸出訓練統計資訊
    print("\n=== 資料集摘要 (Dataset Summary) ===")
    print(f"總訓練筆數: {len(dataset)}")
    print(f"Phase 1 勝出: {winner_counts['phase1']}")
    print(f"Phase 2 勝出: {winner_counts['phase2']}")
    print(f"Phase 3 勝出: {winner_counts['phase3']}")
    print(f"\n訓練資料集已儲存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
