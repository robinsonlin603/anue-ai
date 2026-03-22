# Anue AI: 財經新聞作者風格轉移專案 (Writer Style Transfer)

本專案旨在利用大型語言模型 (LLM) 技術，將一般的財經新聞素材自動轉譯為具備特定作者筆感、語氣與結構的高品質報導。專案採用 Google Gemma 2 9B 模型作為基礎，並透過 RAG 與 LoRA 微調技術不斷優化生成品質。

## 🚀 專案發展階段 (Phases)

本專案共分為四個主要研發階段，每個階段都代表了技術路徑的演進：

### Phase 1: 基準生成 (Baseline)
*   **目標**：建立零樣本 (Zero-shot) 的基本轉譯能力。
*   **技術**：直接調用 Ollama 運行的 Gemma 2 9B 模型，提供基礎指令進行改寫。
*   **價值**：作為後續優化效果的對照組。

### Phase 2: 顯性風格化 (Style Specification)
*   **目標**：引入專業財經新聞的結構規範。
*   **技術**：在 Prompt 中注入詳細的「風格規範說明」，包含段落要求、禁止 Markdown 格式、禁忌詞彙過濾等。
*   **價值**：大幅提升了輸出的專業度與格式穩定性。

### Phase 3: RAG 檢索增強與自我修正 (RAG + Self-Correction)
*   **目標**：讓模型「模仿」真實的作者範文。
*   **技術**：
    *   **語意檢索**：使用 `nomic-embed-text-v2-moe` 將作者範文切片並建立向量索引。
    *   **智慧匹配**：根據來源新聞類型（如市場、營收、產品）自動挑選最契合的風格範文片段。
    *   **自我修正循環**：生成後自動檢測「數字保留率」與格式，若不符規範則觸發 Refine 流程。
*   **價值**：生成結果在風格深度與事實真實性 (Factuality) 上達到了人工可用的水準。

### Phase 4: LoRA 微調與知識內化 (Fine-tuning)
*   **目標**：將風格與穩定性直接植入模型權重中。
*   **技術**：
    *   **資料蒸餾**：從 Phase 1~3 的結果中選出評分最高的 19 條「黃金樣本」。
    *   **微調框架**：採用 Unsloth (Gemma 2 9B 4-bit LoRA) 進行訓練。
    *   **技術突破**：實作了針對 Gemma 2 的 Logit Softcapping Patch，成功避開 VRAM 檢查並實現穩定收斂。
*   **價值**：模型不再需要冗長的 RAG 上下文即可產出具備特定風格的高品質內容，且推論更穩定。

---

## 🛠️ 技術關鍵點

1.  **動態字數管理 (Dynamic Length Scaling)**：
    所有階段均採用動態計算，確保生成字數與來源新聞長度成合理比例（預設 0.7x ~ 1.1x），避免短文灌水或長文漏訊。
2.  **數字保留率檢測**：
    開發了專用的 Scorer 系統，強制監控阿拉伯數字、百分比、股票代碼的保留程度，確保財經新聞的事實查核準確性。
3.  **混合推論模式 (Hybrid Mode)**：
    Phase 4.1 支援 LoRA 權重與風格指令的混合使用，追求極致的控制力。

## 📂 檔案結構說明

*   `scripts/utils/`: 核心工具組（Ollama 客戶端、RAG 檢索、文字清洗）。
*   `scripts/preprocessing/`: 範文切分與預處理。
*   `scripts/indexing/`: 建立 ChromaDB 向量索引。
*   `scripts/finetune/`: 微調資料準備與訓練腳本。
*   `scripts/generation/`: Phase 1~4 的執行主程式。
*   `scripts/evaluation/`: 自動化評分與各階段對比系統。

## ⚙️ 快速開始

1.  安裝依賴：`pip install -r requirements.txt`
2.  建立索引：`python scripts/indexing/build_writer_index.py`
3.  執行生成：`python scripts/generation/run_phase3_rag.py`
4.  微調訓練：`python scripts/finetune/train.py`

---
**Anue AI Team** - 致力於打造最專業的財經新聞自動化風格轉換系統。
