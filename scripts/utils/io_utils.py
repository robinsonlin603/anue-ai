from __future__ import annotations

import json
from pathlib import Path
from typing import Any

"""
本模組提供檔案讀寫相關的基礎工具 (I/O Utilities)。
主要功能：
1. 確保目錄存在 (ensure_dir)。
2. 提供統一的文字檔案與 JSON 讀寫介面。
3. 支援 JSONL 格式的資料追加 (Append)，主要用於執行日誌 (Logs) 的紀錄。
"""

def ensure_dir(path: Path) -> None:
    """
    確保指定的路徑目錄存在，若不存在則自動建立所有父層目錄。
    """
    path.mkdir(parents=True, exist_ok=True)


def read_text(path: Path, encoding: str = "utf-8") -> str:
    """
    讀取純文字檔案內容。
    """
    return path.read_text(encoding=encoding).strip()


def write_text(path: Path, content: str, encoding: str = "utf-8") -> None:
    """
    將文字寫入檔案，若父目錄不存在則自動建立。
    """
    ensure_dir(path.parent)
    path.write_text(content, encoding=encoding)


def write_json(path: Path, data: dict[str, Any], encoding: str = "utf-8") -> None:
    """
    將字典轉換為美化後的 JSON 格式並寫入檔案。
    
    Args:
        path: 檔案路徑。
        data: 待寫入的資料字典。
        encoding: 檔案編碼 (預設 utf-8)。
    """
    ensure_dir(path.parent)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding=encoding,
    )


def append_jsonl(path: Path, record: dict[str, Any], encoding: str = "utf-8") -> None:
    """
    以 JSONL (JSON Lines) 格式將單筆紀錄追加到檔案末尾。
    常用於 Phase 1~3 的執行進度與狀態紀錄。
    
    Args:
        path: JSONL 檔案路徑。
        record: 待追加的單筆資料字典。
    """
    ensure_dir(path.parent)
    with path.open("a", encoding=encoding) as f:
        # 使用 ensure_ascii=False 確保中文能正確顯示，而非轉為 Unicode 逃逸字元
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
