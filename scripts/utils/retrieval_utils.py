from __future__ import annotations

from typing import Any

import chromadb
import re

from scripts.utils.embedding_client import OllamaEmbedder

"""
本模組提供 RAG (檢索增強生成) 的核心工具，負責從向量資料庫中搜尋風格相近的範文片段。
主要功能包含：
1. 文字分類 (Classify Article Type)：判斷新聞屬於市場、營收、產品或事件類。
2. 風格過濾 (Style Filtering)：過濾掉太過雜亂（如純數字列表）的片段。
3. 語意檢索 (Semantic Retrieval)：調用 Ollama 向量模型並透過 ChromaDB 進行相似度搜尋。
4. 候選排序與多樣性控制：確保檢索結果不會全部來自同一篇文章。
"""

def _normalize_score(distance: float) -> float:
    """
    將 ChromaDB 回傳的 Distance (距離) 轉換為更直觀的 Similarity Score (相似度分數)。
    公式：1 / (1 + distance)，距離越小分數越高。
    """
    return round(1 / (1 + max(distance, 0.0)), 6)


def retrieve_style_chunks(
    query_text: str,
    top_k: int = 3,
    per_article_limit: int = 1,
    db_path: str = "data/indexes/writer_style_index",
    collection_name: str = "writer_style_chunks",
    embed_model: str = "nomic-embed-text-v2-moe",
    host: str = "http://localhost:11434",
    fetch_k_multiplier: int = 4,
) -> list[dict[str, Any]]:
    """
    執行語意檢索，從資料庫中找出風格最匹配的範文片段。

    Args:
        query_text: 查詢文字 (通常是目前的來源新聞)。
        top_k: 最終返回的片段數量。
        per_article_limit: 限制來自同一篇文章的片段數量 (確保多樣性)。
        db_path: 向量資料庫路徑。
        collection_name: 資料表名稱。
        embed_model: 使用的向量模型。
        host: Ollama API 位址。
        fetch_k_multiplier: 初步搜尋的倍數 (因後續會過濾，故初步先抓多一點)。

    Returns:
        list[dict[str, Any]]: 經過過濾與排序後的候選片段。
    """
    if not query_text.strip():
        return []

    # 1. 初始化資料庫與向量生成器
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection(name=collection_name)

    embedder = OllamaEmbedder(model=embed_model, host=host)
    query_embedding = embedder.embed([query_text])[0]

    # 2. 進行初步語意搜尋
    initial_k = max(top_k * fetch_k_multiplier, top_k)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=initial_k,
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    # 3. 候選片段處理與分類
    primary_candidates: list[dict[str, Any]] = []
    fallback_candidates: list[dict[str, Any]] = []
    
    # 判斷查詢內容的類型 (市場型、產品型等)
    query_type = classify_article_type(query_text)

    for document, metadata, distance in zip(documents, metadatas, distances):
        if not metadata:
            continue

        candidate = _build_candidate(metadata, document, distance)

        # 規則 A：過濾掉含有過多代碼或純數字的列表片段 (不利於學風格)
        if candidate["is_list_heavy"]:
            continue

        # 規則 B：依照類型進行初步匹配
        # 如果查詢是「市場分析」，則優先排除掉「純產品發表」或「純活動展覽」的範文
        if query_type == "market":
            if candidate["is_product_heavy"] or candidate["is_event_heavy"]:
                fallback_candidates.append(candidate)
            else:
                primary_candidates.append(candidate)
        else:
            primary_candidates.append(candidate)

    # 4. 最終篩選 (包含多樣性檢查)
    selected: list[dict[str, Any]] = []
    article_counter: dict[str, int] = {}

    # 先從最匹配的 Primary 候選中挑選
    for candidate in primary_candidates:
        article_id = str(candidate.get("article_id", "") or "")
        if not _can_add_article(article_counter, article_id, per_article_limit):
            continue

        selected.append(candidate)
        _mark_article_used(article_counter, article_id)

        if len(selected) >= top_k:
            return selected

    # 若 Primary 不足，再從 Fallback 中補足
    for candidate in fallback_candidates:
        article_id = str(candidate.get("article_id", "") or "")
        if not _can_add_article(article_counter, article_id, per_article_limit):
            continue

        selected.append(candidate)
        _mark_article_used(article_counter, article_id)

        if len(selected) >= top_k:
            return selected

    return selected


def build_rag_context(chunks: list[dict[str, Any]]) -> str:
    """
    將檢索到的多個片段組合成 Prompt 中使用的上下文文字。
    格式：
    [Style Example 1]
    內容...
    """
    if not chunks:
        return ""

    parts: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(
            "\n".join(
                [
                    f"[Style Example {i}]",
                    chunk["text"].strip(),
                ]
            ).strip()
        )

    return "\n\n".join(parts).strip()


def is_list_heavy_chunk(text: str) -> bool:
    """
    判斷片段是否為「重度列表型」文字。
    
    判斷基準：
    - 含有過多股票代碼 (e.g., (2330-TW))
    - 含有過多標點符號 (、，；)
    - 數字比例過高
    
    這類片段通常是數據列舉，對於學習作者的「敘事風格」幫助不大。
    """
    stock_code_count = len(re.findall(r"\(\d{4,6}[A-Z-]*-TW\)", text))
    comma_like_count = len(re.findall(r"[、，；]", text))
    number_count = len(re.findall(r"\d+(?:\.\d+)?", text))

    if stock_code_count >= 4:
        return True
    if stock_code_count >= 3 and comma_like_count >= 6:
        return True
    if number_count >= 12 and comma_like_count >= 8:
        return True

    return False


def is_product_heavy(text: str) -> bool:
    """
    判斷片段是否為「產品/技術介紹型」文字。
    關鍵字包含：模組、電感、水冷電源、量產、驗證等。
    """
    keywords = [
        "產品", "模組", "平台", "設計", "開發", "量產", "送樣",
        "驗證", "電感", "供電", "伺服器用", "水冷電源", "客戶驗證",
        "試樣", "導入量產",
    ]
    hit_count = sum(1 for kw in keywords if kw in text)
    return hit_count >= 3


def is_event_heavy(text: str) -> bool:
    """
    判斷片段是否為「活動/展覽型」文字。
    關鍵字包含：展出、大會、GTC、簽約、發表等。
    """
    keywords = [
        "大會", "展出", "發布", "揭露", "亮相", "展示", "GTC",
        "代表團", "最新進展", "簽約", "發表",
    ]
    hit_count = sum(1 for kw in keywords if kw in text)
    return hit_count >= 2


def classify_article_type(text: str) -> str:
    """
    簡單基於關鍵字的文字分類器，將內容分為四大類：
    1. market: 市場/ETF/大盤趨勢
    2. earnings: 財報/獲利/配息/法說會
    3. product: 產品研發/量產/驗證
    4. event: 活動/大會/簽約
    """
    text = text or ""

    market_keywords = [
        "ETF", "指數", "權重", "市值型", "法人", "台股", "大盤",
        "含息報酬率", "加權股價指數",
    ]
    earnings_keywords = [
        "EPS", "配息", "法說", "董事會", "稅後純益", "年增", "獲利", "營收",
    ]
    product_keywords = [
        "產品", "模組", "平台", "設計", "開發", "量產", "送樣",
        "驗證", "導入", "電感", "供電", "伺服器用", "水冷電源", "客戶驗證", "試樣",
    ]
    event_keywords = [
        "大會", "展出", "發布", "揭露", "亮相", "GTC", "代表團", "最新進展",
    ]

    def count_hits(keywords):
        return sum(1 for kw in keywords if kw in text)

    scores = {
        "market": count_hits(market_keywords),
        "earnings": count_hits(earnings_keywords),
        "product": count_hits(product_keywords),
        "event": count_hits(event_keywords),
    }

    # 返回得分最高的一類
    return max(scores, key=scores.get)


def _build_candidate(
    metadata: dict[str, Any],
    document: str,
    distance: float,
) -> dict[str, Any]:
    """
    內部函式：將檢索到的原始數據組合為候選片段物件，並添加進階特徵分析。
    """
    doc_type = classify_article_type(document)
    product_heavy = is_product_heavy(document)
    event_heavy = is_event_heavy(document)
    list_heavy = is_list_heavy_chunk(document)

    return {
        "chunk_id": metadata.get("chunk_id"),
        "article_id": metadata.get("article_id"),
        "title": metadata.get("title"),
        "source_file": metadata.get("source_file"),
        "chunk_index": metadata.get("chunk_index"),
        "char_count": metadata.get("char_count"),
        "score": _normalize_score(distance),
        "distance": distance,
        "text": document,
        "document_type": doc_type,
        "is_product_heavy": product_heavy,
        "is_event_heavy": event_heavy,
        "is_list_heavy": list_heavy,
    }


def _can_add_article(
    article_counter: dict[str, int],
    article_id: str,
    per_article_limit: int,
) -> bool:
    """內部函式：檢查同一篇文章的引用次數是否已達上限。"""
    current_count = article_counter.get(article_id, 0)
    return current_count < per_article_limit


def _mark_article_used(
    article_counter: dict[str, int],
    article_id: str,
) -> None:
    """內部函式：標記文章已被引用。"""
    article_counter[article_id] = article_counter.get(article_id, 0) + 1
