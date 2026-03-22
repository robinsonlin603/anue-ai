from __future__ import annotations

import re

"""
本模組負責對生成的財經新聞進行自動化評分。
評分維度包含：
1. 事實真實性 (Factuality, 40%)：檢查關鍵數字與股票代碼的保留程度。
2. 資訊覆蓋率 (Coverage, 30%)：檢查原始新聞中的關鍵詞與數據是否完整傳達。
3. 結構規範 (Structure, 15%)：檢查標題與段落數是否符合 3-5 段的要求。
4. 風格符合度 (Style, 15%)：檢查是否殘留 Markdown、列點或非客觀的禁忌詞。
"""

# 定義正規表達式用於提取數據與代碼
NUMBER_PATTERN = r"\d{1,3}(?:,\d{3})*(?:\.\d+)?%?"
CODE_PATTERN = r"\b[A-Z]{2,}(?:\s?[A-Z]{2,})?\b|\b\d{3,5}(?:-TW)?\b"

# 禁止出現的非客觀或過度主觀詞彙
STYLE_BAD_WORDS = [
    "建議", "應該", "最好", "一定要", "值得注意的是", "最佳策略",
]

# 偵測 Markdown 格式的規則
MARKDOWN_PATTERNS = [
    r"^\s*#{1,6}\s+", # 標題
    r"\*\*.+?\*\*",    # 粗體
    r"^\s*[-*]\s+",    # 列表
]

# 偵測列點格式的規則 (財經報導通常要求純文字段落)
BULLET_PATTERNS = [
    r"^\s*[-*]\s+",
    r"^\s*\d+\.\s+",
]


def extract_numbers(text: str) -> set[str]:
    """提取文字中所有的數字、百分比。"""
    return set(re.findall(NUMBER_PATTERN, text))


def extract_codes(text: str) -> set[str]:
    """提取文字中的股票代碼 (如 2330-TW) 或英文縮寫。"""
    return set(re.findall(CODE_PATTERN, text))


def extract_candidate_keywords(text: str) -> list[str]:
    """
    從來源文章中提取關鍵詞作為參考指標。
    排除掉常見的停止詞，並按長度排序（長詞通常更具代表性）。
    """
    chinese_phrases = re.findall(r"[\u4e00-\u9fff]{2,8}", text)
    
    stopwords = {
        "表示", "指出", "說明", "公司", "市場", "投資", "相關", "進一步",
        "以及", "其中", "台股", "今年", "去年", "今日", "可以", "因為",
        "因此", "主要", "內容", "標題", "新聞素材", "透過", "布局", "持續",
        "提供", "對於", "不需要", "具有", "加上"
    }

    keywords = [
        token for token in chinese_phrases if token not in stopwords
    ]
    
    # 移除重複並取長度前 40 個最長詞彙
    unique_keywords = []
    for k in sorted(keywords, key=len, reverse=True):
        if k not in unique_keywords:
            unique_keywords.append(k)
    
    return unique_keywords[:40]


def score_factuality(source: str, output: str) -> int:
    """
    評分：事實真實性 (滿分 40)。
    計算數字與代碼的交集率 (Overlap Ratio)。
    """
    source_numbers = extract_numbers(source)
    output_numbers = extract_numbers(output)
    
    source_codes = extract_codes(source)
    output_codes = extract_codes(output)

    if not source_numbers and not source_codes:
        return 40 # 若來源無數據則直接給滿分

    num_overlap = len(source_numbers & output_numbers) / len(source_numbers) if source_numbers else 1.0
    code_overlap = len(source_codes & output_codes) / len(source_codes) if source_codes else 1.0
    
    # 綜合重合度：數字佔 60%，股票代碼佔 40% (權重分配可依需求調整)
    combined_overlap = (num_overlap * 0.6) + (code_overlap * 0.4)

    if combined_overlap >= 0.85: return 40
    if combined_overlap >= 0.65: return 32
    if combined_overlap >= 0.45: return 24
    return 16


def score_coverage(source: str, output: str) -> int:
    """
    評分：資訊覆蓋率 (滿分 30)。
    檢查關鍵數據與關鍵詞是否被帶入生成的新聞中。
    """
    source_numbers = extract_numbers(source)
    source_codes = extract_codes(source)
    source_ref_keywords = extract_candidate_keywords(source)

    output_numbers = extract_numbers(output)
    output_codes = extract_codes(output)
    output_keywords = set(re.findall(r"[\u4e00-\u9fff]{2,8}", output))

    num_score = 0
    code_score = 0
    keyword_score = 0

    if source_numbers:
        num_overlap = len(source_numbers & output_numbers) / len(source_numbers)
        num_score = min(12, round(num_overlap * 12))

    if source_codes:
        code_overlap = len(source_codes & output_codes) / len(source_codes)
        code_score = min(10, round(code_overlap * 10))

    if source_ref_keywords:
        # 檢查來源文章前 20 個最重要關鍵詞在輸出中出現的比例
        top_ref = source_ref_keywords[:20]
        hits = sum(1 for k in top_ref if k in output_keywords)
        key_overlap = hits / len(top_ref)
        keyword_score = min(8, round(key_overlap * 8))

    return num_score + code_score + keyword_score

def score_structure(text: str) -> int:
    """
    評分：結構規範 (滿分 15)。
    檢查標題行 + 內文段落數。
    理想結構：1 標題 + 3~5 段內文 = 4~6 行文字。
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    if not lines:
        return 0

    if 4 <= len(lines) <= 7:
        return 15
    if 3 <= len(lines) <= 8:
        return 10
    return 5


def score_style(text: str) -> int:
    """
    評分：風格符合度 (滿分 15)。
    採扣分制：
    - 出現 Markdown 格式、列點、或禁忌詞彙。
    """
    penalty = 0

    # 檢查 Markdown 殘留
    for pattern in MARKDOWN_PATTERNS:
        if re.search(pattern, text, flags=re.MULTILINE):
            penalty += 4

    # 檢查禁止的列點
    for pattern in BULLET_PATTERNS:
        if re.search(pattern, text, flags=re.MULTILINE):
            penalty += 4

    # 檢查主觀禁忌詞
    for word in STYLE_BAD_WORDS:
        if word in text:
            penalty += 2

    return max(15 - penalty, 5)


def evaluate(source: str, output: str) -> dict[str, int]:
    """
    綜合評估函式：執行各項評分並加總。
    """
    factuality = score_factuality(source, output)
    coverage = score_coverage(source, output)
    structure = score_structure(output)
    style = score_style(output)

    total = factuality + coverage + structure + style

    return {
        "factuality": factuality,
        "coverage": coverage,
        "structure": structure,
        "style": style,
        "total": total,
    }
