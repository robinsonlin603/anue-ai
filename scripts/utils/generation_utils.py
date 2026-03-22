from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

"""
本模組提供文字生成流程中所需的各種通用工具，包含：
1. 檔案讀取與載入 (Input/Source Loading)
2. 動態生成參數計算 (Length Scaling)
3. 生成結果清理與格式化 (Post-processing/Cleaning)
4. 指標檢測 (Metrics & Hallucination Detection)
5. Ollama 元數據封裝 (Metadata Packaging)
"""

def list_input_files(input_path: Path) -> list[Path]:
    """
    掃描指定路徑下的所有合法輸入檔案。
    
    Args:
        input_path: 檔案或目錄路徑。
        
    Returns:
        Sorted list of Path objects (.txt, .md, .json)。
    """
    if input_path.is_file():
        return [input_path]

    supported_exts = {".txt", ".md", ".json"}
    return sorted(
        p
        for p in input_path.glob("**/*")
        if p.is_file() and p.suffix.lower() in supported_exts
    )


def load_source_fields(file_path: Path) -> tuple[str, str]:
    """
    載入來源文章的標題與內文。支援 JSON 格式或純文字檔案。
    
    Args:
        file_path: 檔案路徑。
        
    Returns:
        tuple[str, str]: (標題, 內文)。
    """
    if file_path.suffix.lower() == ".json":
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        # 優先尋找 JSON 中的特定欄位
        title = data.get("title") or file_path.stem
        content = data.get("content") or data.get("source_text") or data.get("text")

        if not isinstance(title, str):
            title = file_path.stem
        if not isinstance(content, str):
            raise ValueError(f"JSON 缺少必要的文字欄位 'content': {file_path}")

        return title, content

    # 若非 JSON，則以檔名為標題，檔案內容為內文
    return file_path.stem, file_path.read_text(encoding="utf-8")


def make_source_id(file_path: Path, source_title: str, source_text: str) -> str:
    """
    產生成來源文章的唯一識別碼 (ID)。目前預設使用檔案名稱。
    """
    return file_path.stem


def get_dynamic_length_range(source_text: str) -> tuple[int, int]:
    """
    根據來源文章的長度，動態計算目標生成的字數區間 (Dynamic Length Scaling)。
    
    邏輯：
    1. 基礎區間設定為來源長度的 0.7 倍至 1.1 倍。
    2. 下限 (Min) 至少 320 字，避免過短。
    3. 上限 (Max) 至多 550 字，避免過長導致模型灌水。
    """
    src_len = len(source_text.strip())
    target_min = max(320, int(src_len * 0.7))
    target_max = min(550, int(src_len * 1.1))

    # 確保最小值小於最大值，若重疊則強制增加 100 字區間
    if target_min >= target_max:
        target_max = target_min + 100

    return target_min, target_max


def count_chars(text: str) -> int:
    """計算文字總字數 (去除首尾空格)。"""
    return len(text.strip())


def clean_generated_text(text: str) -> str:
    """
    對模型生成的原始文字進行後處理與清理。
    
    包含：
    1. 移除 Markdown 粗體符號 (**)。
    2. 移除 Markdown 標題符號 (#)。
    3. 移除列表符號 (- 或 * 或 數字列表)。
    4. 壓縮過多的連續換行。
    """
    text = text.strip()
    text = text.replace("**", "")

    lines = text.splitlines()
    cleaned_lines: list[str] = []

    for line in lines:
        stripped = line.strip()

        if not stripped:
            cleaned_lines.append("")
            continue

        # 移除標題符號 (# 標題)
        if re.match(r"^#{1,6}\s+", stripped):
            stripped = re.sub(r"^#{1,6}\s+", "", stripped).strip()

        # 移除列表符號 (- 或 * 列表)
        if re.match(r"^[-*]\s+", stripped):
            stripped = re.sub(r"^[-*]\s+", "", stripped).strip()

        # 移除數字列表符號 (1. 列表)
        if re.match(r"^\d+\.\s+", stripped):
            stripped = re.sub(r"^\d+\.\s+", "", stripped).strip()

        cleaned_lines.append(stripped)

    # 重新組合成文字並壓縮換行 (最多連續兩個換行)
    cleaned_text = "\n".join(cleaned_lines)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)

    return cleaned_text.strip()


def detect_markdown(text: str) -> bool:
    """偵測文字中是否殘留 Markdown 格式 (標題、粗體、列表)。"""
    markdown_patterns = [
        r"^\s*#{1,6}\s+",
        r"\*\*.+?\*\*",
        r"^\s*[-*]\s+",
    ]
    return any(
        re.search(pattern, text, flags=re.MULTILINE) for pattern in markdown_patterns
    )


def detect_bullet_list(text: str) -> bool:
    """偵測文字中是否殘留列表格式。"""
    bullet_list_patterns = [
        r"^\s*[-*]\s+",
        r"^\s*\d+[\.\)]\s+",
        r"^\s*•\s+",
    ]
    return any(
        re.search(pattern, text, flags=re.MULTILINE) for pattern in bullet_list_patterns
    )


def detect_section_header(text: str) -> bool:
    """偵測是否使用了小標題 (如 # 標題 或 【標題】 格式)。"""
    section_header_patterns = [
        r"^\s*#{1,6}\s+",
        r"^\s*【.+】\s*$",
    ]
    return any(
        re.search(pattern, text, flags=re.MULTILINE)
        for pattern in section_header_patterns
    )


def detect_dateline(text: str) -> bool:
    """
    偵測模型是否產生了「虛假的地點/日期標記」(Dateline Hallucination)。
    
    例如：台北，2026年3月22日訊、記者 OOO 報導等。
    這在風格轉移中通常需要被移除。
    """
    dateline_patterns = [
        r"^\s*[\u4e00-\u9fff]{2,10}[，,]\d{4}年\d{1,2}月\d{1,2}日",
        r"^\s*[\u4e00-\u9fff]{2,10}[訊報導]",
        r"^\s*記者[\u4e00-\u9fff]{1,10}",
    ]
    return any(
        re.search(pattern, text, flags=re.MULTILINE) for pattern in dateline_patterns
    )


def build_source_stats(
    raw_title: str,
    raw_text: str,
    cleaned_title: str,
    cleaned_text: str,
) -> dict[str, int]:
    """建立來源文章的字數統計資料。"""
    return {
        "raw_title_char_count": count_chars(raw_title),
        "raw_char_count": count_chars(raw_text),
        "cleaned_title_char_count": count_chars(cleaned_title),
        "cleaned_char_count": count_chars(cleaned_text),
    }


def build_metrics(generated_text: str) -> dict[str, Any]:
    """
    對生成內容進行全面的指標分析。
    
    包含：
    - 字數與段落數
    - 格式錯誤偵測 (Markdown, Bullet List, Headers)
    - 幻覺偵測 (Dateline)
    - 是否符合目標字數與段落區間。
    """
    output_char_count = count_chars(generated_text)
    paragraph_count = len([p for p in generated_text.splitlines() if p.strip()])
    return {
        "output_char_count": output_char_count,
        "paragraph_count": paragraph_count,
        "has_markdown": detect_markdown(generated_text),
        "has_bullet_list": detect_bullet_list(generated_text),
        "has_section_header": detect_section_header(generated_text),
        "has_hallucinated_dateline": detect_dateline(generated_text),
        "within_target_length": 320 <= output_char_count <= 450,
        "paragraph_count_valid": 4 <= paragraph_count <= 10,
    }


def build_ollama_meta(response: dict[str, Any]) -> dict[str, Any]:
    """將 Ollama API 回傳的原始數據封裝為結構化的元數據 (Metadata)。"""
    return {
        "model": response.get("model"),
        "created_at": response.get("created_at"),
        "done": response.get("done"),
        "done_reason": response.get("done_reason"),
        "total_duration": response.get("total_duration"),
        "load_duration": response.get("load_duration"),
        "prompt_eval_count": response.get("prompt_eval_count"),
        "prompt_eval_duration": response.get("prompt_eval_duration"),
        "eval_count": response.get("eval_count"),
        "eval_duration": response.get("eval_duration"),
    }
