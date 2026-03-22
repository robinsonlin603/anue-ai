from __future__ import annotations

from typing import Optional

"""
本模組負責組裝最終發送給 LLM 的 Prompt (提示詞)。
主要功能：
1. 替換模板中的占位符 (Placeholders)，如 {{TITLE}}, {{CONTENT}} 等。
2. 動態注入 RAG 檢索到的範文上下文。
3. 動態更新字數限制 (Length Constraint)，確保 Prompt 中的要求與計算出的目標一致。
"""

def build_prompt(
    template: str,
    title: str,
    content: str,
    style_spec: Optional[str] = None,
    rag_context: Optional[str] = None,
    target_min: Optional[int] = None,
    target_max: Optional[int] = None,
) -> str:
    """
    根據模板與提供的資料組裝 Prompt。

    Args:
        template: Prompt 原始模板。
        title: 來源文章標題。
        content: 來源文章內文。
        style_spec: 選用的風格規範說明 (Style Specification)。
        rag_context: RAG 檢索到的範文片段文字。
        target_min: 目標生成字數下限。
        target_max: 目標生成字數上限。

    Returns:
        str: 組裝完成的完整 Prompt 字串。
    """
    import re
    
    # 1. 以模板作為基礎
    prompt = template
    
    # 2. 動態字數替換：
    # 如果有提供 target_min/max，則尋找 Prompt 中如 "320–450 Chinese characters" 的字樣並替換。
    # 這能確保即使模板寫死了一個範圍，程式計算出的「動態長度」也能正確傳達給模型。
    if target_min is not None and target_max is not None:
        pattern = r"\d+[-–]\d+\s+Chinese characters"
        replacement = f"{target_min}–{target_max} Chinese characters"
        
        prompt = re.sub(pattern, replacement, prompt)
        if style_spec:
            style_spec = re.sub(pattern, replacement, style_spec)

    # 3. 執行標準占位符替換
    # {{TITLE}}: 原始新聞標題
    # {{CONTENT}}: 原始新聞內文
    # {{RAG_CONTEXT}}: Phase 3 加入的範文片段
    # {{STYLE_SPEC}}: Phase 2 加入的詳細風格規範
    prompt = prompt.replace("{{TITLE}}", title)
    prompt = prompt.replace("{{CONTENT}}", content)
    prompt = prompt.replace("{{RAG_CONTEXT}}", rag_context or "")
    prompt = prompt.replace("{{STYLE_SPEC}}", style_spec or "")
    
    return prompt
