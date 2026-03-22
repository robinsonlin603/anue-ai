from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection

"""
本模組負責建立範文片段的向量索引 (Vector Indexing)。
主要流程：
1. 讀取預處理好的 JSONL 片段檔案。
2. 初始化 ChromaDB 本地資料庫。
3. 使用 Ollama 向量模型 (如 nomic-embed-text) 將文字片段轉換為高維度向量。
4. 將文字、向量與相關元數據 (Metadata) 存入資料庫，供 Phase 3 的 RAG 檢索使用。
"""

# 若非作為套件執行，則將專案根目錄加入系統路徑，以便引用內部模組
if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.utils.embedding_client import OllamaEmbedder


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """讀取 JSONL 格式的資料行，並轉換為字典列表。"""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def batch_iter(items: list[Any], batch_size: int) -> list[list[Any]]:
    """將列表切分為指定大小的批次 (Batches)，用於批次處理 API 請求。"""
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def get_or_create_collection(
    db_path: Path,
    collection_name: str,
) -> Collection:
    """初始化 ChromaDB 客戶端並取得或建立指定的 Collection (資料表)。"""
    client = chromadb.PersistentClient(path=str(db_path))
    return client.get_or_create_collection(name=collection_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="將作者範文片段建立為 Chroma 向量索引。"
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default="data/processed/writer_chunks/writer_chunks.jsonl",
        help="輸入的 JSONL 片段路徑。",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/indexes/writer_style_index",
        help="向量資料庫存放目錄。",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default="writer_style_chunks",
        help="資料表 (Collection) 名稱。",
    )
    parser.add_argument(
        "--embed-model",
        type=str,
        default="nomic-embed-text-v2-moe",
        help="使用的 Ollama 向量模型名稱。",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="http://localhost:11434",
        help="Ollama API 伺服器位址。",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="每批次處理的向量化請求數量。",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="在開始索引前，先刪除並重建現有的 Collection。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_file = Path(args.input_file)
    db_path = Path(args.db_path)
    db_path.mkdir(parents=True, exist_ok=True)

    # 1. 載入資料
    chunks = read_jsonl(input_file)
    client = chromadb.PersistentClient(path=str(db_path))

    # 2. 如果指定了 --reset，則先清理舊資料
    if args.reset:
        try:
            client.delete_collection(args.collection_name)
        except Exception:
            pass

    # 3. 初始化資料庫 Collection 與 Ollama 向量生成器
    collection = client.get_or_create_collection(name=args.collection_name)
    embedder = OllamaEmbedder(model=args.embed_model, host=args.host)

    # 4. 批次處理與寫入
    total = 0
    for batch in batch_iter(chunks, args.batch_size):
        ids = [row["chunk_id"] for row in batch]
        documents = [row["text"] for row in batch]
        
        # 調用 Ollama API 生成文字向量 (Embeddings)
        embeddings = embedder.embed(documents)

        # 整理元數據 (Metadata)，這些資訊會在檢索時一併回傳
        metadatas = []
        for row in batch:
            metadatas.append(
                {
                    "chunk_id": row["chunk_id"],
                    "article_id": row["article_id"],
                    "title": row["title"],
                    "source_file": row["source_file"],
                    "chunk_index": row["chunk_index"],
                    "char_count": row["char_count"],
                    "paragraph_count": row["paragraph_count"],
                    "created_at": row["created_at"],
                }
            )

        # 將資料寫入向量資料庫 (upsert 表示若 ID 存在則更新，不存在則新增)
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        total += len(batch)

    # 5. 輸出執行結果
    print(
        json.dumps(
            {
                "status": "ok",
                "input_file": str(input_file),
                "db_path": str(db_path),
                "collection_name": args.collection_name,
                "embed_model": args.embed_model,
                "indexed_count": total,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
