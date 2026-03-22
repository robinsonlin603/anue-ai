from __future__ import annotations

import requests
from typing import Any


class OllamaEmbedder:
    """
    Ollama 向量生成客戶端 (Embedding Client)。
    專門對接本地運行的 Ollama API，將文字轉換為高維度向量 (Embeddings)，
    主要用於 Phase 3 的 RAG 檢索與向量資料庫 (ChromaDB) 的建立。
    """

    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        timeout: int = 120,
        max_text_chars: int = 2000,
    ) -> None:
        """
        初始化 Ollama 向量生成器。

        Args:
            model: 使用的向量模型名稱 (例如: nomic-embed-text-v2-moe)。
            host: Ollama API 的基礎路徑，預設為本地端。
            timeout: API 請求超時時間 (秒)。
            max_text_chars: 單次向量化文字的最大長度上限，避免模型因文字過長而崩潰或截斷。
        """
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.max_text_chars = max_text_chars

    def _truncate_text(self, text: str) -> str:
        """
        內部函式：截斷過長的文字，確保其長度在 max_text_chars 範圍內。
        """
        text = text.strip()
        if len(text) <= self.max_text_chars:
            return text
        return text[: self.max_text_chars]

    def _embed_one(self, text: str) -> list[float]:
        """
        內部函式：調用 Ollama API 為單一文字片段生成向量。
        """
        safe_text = self._truncate_text(text)

        url = f"{self.host}/api/embeddings"
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": safe_text,
        }

        # 發送 POST 請求至 Ollama
        response = requests.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()

        data = response.json()
        embedding = data.get("embedding")

        # 驗證回傳格式是否正確
        if not isinstance(embedding, list):
            raise ValueError("Invalid embedding response from Ollama.")

        return embedding

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        公開函式：批次將多段文字轉換為向量列表。

        Args:
            texts: 待轉換的文字列表。

        Returns:
            list[list[float]]: 對應的向量列表。
        """
        if not texts:
            return []

        embeddings: list[list[float]] = []
        for text in texts:
            # 遍歷文字片段並逐一生成向量
            embeddings.append(self._embed_one(text))

        return embeddings
