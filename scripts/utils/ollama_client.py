from __future__ import annotations

from typing import Any

from ollama import Client

"""
本模組提供與 Ollama API 互動的客戶端封裝。
主要負責：
1. 管理 Ollama 的連線與生成參數 (Temperature, Top_p 等)。
2. 執行推理 (Inference) 並將回應轉換為可序列化的 Python 字典 (dict)。
"""

class OllamaGenerator:
    """
    Ollama 文字生成器。
    用於調用本地運行的大型語言模型 (LLM)，如 Gemma 2。
    """

    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        temperature: float = 0.3,
        num_predict: int = 800,
        top_p: float = 0.9,
    ) -> None:
        """
        初始化生成器參數。

        Args:
            model: 模型名稱 (例如: gemma2:9b)。
            host: Ollama 服務位址。
            temperature: 隨機性控制。較低的值 (如 0.3) 會使輸出更穩定、更具事實性。
            num_predict: 最大生成的 Token 數量限制。
            top_p: 核採樣 (Nucleus Sampling) 參數，控制詞彙的多樣性。
        """
        self.model = model
        self.client = Client(host=host)
        self.options = {
            "temperature": temperature,
            "num_predict": num_predict,
            "top_p": top_p,
        }

    def generate(self, prompt: str) -> dict[str, Any]:
        """
        執行文字生成。

        Args:
            prompt: 完整的 Prompt 字串。

        Returns:
            dict[str, Any]: 包含生成文字與模型元數據 (Metadata) 的字典。
        """
        # 發送生成請求至 Ollama (非串流模式)
        raw_response = self.client.generate(
            model=self.model,
            prompt=prompt,
            stream=False,
            options=self.options,
        )

        # 處理回應格式：
        # `ollama.Client.generate()` 回傳的是 Pydantic 模型 (GenerateResponse)，
        # 為了後續能方便存成 JSON 檔案，這裡統一把物件轉成標準 dict。
        
        if isinstance(raw_response, dict):
            return raw_response

        # 優先嘗試 Pydantic v2 的 model_dump 方法
        model_dump = getattr(raw_response, "model_dump", None)
        if callable(model_dump):
            return model_dump()

        # 嘗試舊版 Pydantic 的 dict 方法
        dict_method = getattr(raw_response, "dict", None)
        if callable(dict_method):
            return dict_method()

        # 如果以上方法都失敗，則手動提取核心欄位進行保底轉換
        return {
            "response": getattr(raw_response, "response", ""),
            "model": getattr(raw_response, "model", self.model),
            "done": getattr(raw_response, "done", True),
            "created_at": getattr(raw_response, "created_at", None),
        }
