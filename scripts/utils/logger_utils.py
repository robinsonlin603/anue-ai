from __future__ import annotations

import logging
from pathlib import Path

"""
本模組提供日誌 (Logging) 的基礎工具。
主要功能：
1. 建立並配置日誌記錄器 (Logger)。
2. 同時將日誌輸出到控制台 (Console) 與實體檔案。
3. 統一各階段執行過程的日誌格式。
"""

def setup_logger(log_file: Path) -> logging.Logger:
    """
    設定並回傳一個日誌記錄器。

    Args:
        log_file: 日誌檔案存放的路徑。

    Returns:
        logging.Logger: 配置完成的 Logger 物件。
    """
    # 確保日誌目錄存在
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # 取得 Logger 物件，並將檔案路徑作為 ID 以避免重複
    logger = logging.getLogger(str(log_file))
    logger.setLevel(logging.INFO)
    
    # 清除已有的 Handler，防止在同一個進程中重複運行時出現重複日誌
    logger.handlers.clear()

    # 定義日誌顯示格式：時間 | 層級 | 訊息
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    # 1. 檔案 Handler：負責將日誌寫入檔案
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    # 2. 控制台 Handler：負責將日誌輸出到螢幕
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    # 將配置好的 Handler 加入 Logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger
