from __future__ import annotations

import html
import re

"""
本模組負責原始新聞內容 (Source Text) 的標準化與清洗。
主要功能：
1. 修正 HTML 跳脫字元 (如 &nbsp; 轉為空格)。
2. 統一空白字元與換行符號。
3. 移除不必要的免責聲明或廣告字樣。
"""

# 用於匹配並移除新聞末尾常見的「免責聲明」
DISCLAIMER_PATTERNS = [
    r"※免責聲明[:：].*",
]


def normalize_source_text(text: str) -> str:
    """
    對原始來源文章進行標準化清洗。

    Args:
        text: 原始讀取的文章文字。

    Returns:
        str: 清洗後乾淨的文字。
    """
    # 1. 處理 HTML 轉義字元 (例如 &amp; -> &)
    text = html.unescape(text)

    # 2. 處理各種特殊空白符號：
    # \u3000: 全形空格
    # \xa0: Non-breaking space
    text = text.replace("\u3000", " ")
    text = text.replace("\xa0", " ")

    # 3. 統一換行符號 (\r\n 轉為 \n)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 4. 移除免責聲明等不必要的文字內容
    for pattern in DISCLAIMER_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.DOTALL)

    # 5. 壓縮過多的連續換行 (最多只保留兩個換行)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 6. 壓縮行內多餘的空格或 Tab
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()
