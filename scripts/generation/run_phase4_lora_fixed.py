import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from pathlib import Path
import json
import re
from tqdm import tqdm

"""
本模組是 Phase 4 LoRA 生成的「修正與強化版」(Fixed Version)。
相較於原始版，此版本針對生產環境進行了多項優化：
1. 精確的 Token 截斷：確保只獲取模型新生成的文字，而不包含原始 Prompt。
2. 強大的雜訊過濾：針對模型可能產出的 HTML 標籤、重複內容或對話標籤進行智慧清除。
3. 參數微調：提高了 `repetition_penalty` 以徹底解決微調模型容易產生的「跳針」問題。
4. 穩定性載入：使用原生 Transformers 配置，確保在 4-bit 模式下推論最穩定。
"""

# 將當前目錄加入 path 以便引用 utils
sys.path.append(os.getcwd())

from scripts.utils.generation_utils import get_dynamic_length_range, clean_generated_text

# 1. 配置設定
model_name = "unsloth/gemma-2-9b-it-bnb-4bit"
lora_adapter_path = "models/adapters/writer-style-lora"
source_dir = Path("data/raw/source_articles")
output_dir = Path("outputs/generations/phase4")
output_dir.mkdir(parents=True, exist_ok=True)

# 2. 載入模型與 LoRA 適配器
print(f"=== 正在載入模型 (原生 Transformers 4-bit 穩定模式) ===")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
)

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(f"=== 正在掛載 LoRA 適配器: {lora_adapter_path} ===")
model = PeftModel.from_pretrained(base_model, lora_adapter_path)
model.eval()

# 3. 核心工具：文字精煉與去噪
def extract_first_generation(text: str) -> str:
    """
    智慧截斷函式：如果模型輸出了重複內容、對話標籤或 HTML 雜訊，只截取第一部分有效的新聞內容。
    """
    # 定義模型可能產出的無效停止字元或標籤
    stop_tokens = [
        "<start_of_turn>", 
        "<end_of_turn>", 
        "model", 
        "user", 
        "<unused4>", 
        "改寫如下",
        "</textarea>",
        "</button>",
        "<html>",
        "</body>",
        "});",
        "<eos>",
        "<|im_end|>"
    ]
    
    # 尋找第一個出現以上標籤的位置，並在那裡「切斷」文字
    earliest_pos = len(text)
    for token in stop_tokens:
        pos = text.find(token)
        if pos != -1 and pos < earliest_pos:
            earliest_pos = pos
            
    content = text[:earliest_pos].strip()
    
    # 檢查自我重複現象：
    # 如果文章後面又出現了一次一模一樣的標題，代表模型「寫上癮了」，需切掉重複部分。
    lines = content.splitlines()
    if len(lines) > 5:
        first_line = lines[0].strip()
        for i in range(2, len(lines)):
            # 若第 i 行內容與第 1 行相同，且字數足夠，視為重複起點
            if lines[i].strip() == first_line and len(first_line) > 5:
                content = "\n".join(lines[:i]).strip()
                break
                
    return content

# 4. 執行批次生成
source_files = sorted(list(source_dir.glob("article-*.json")))
print(f"=== 找到 {len(source_files)} 篇新聞待處理 ===")

instruction_template = "你是一位專業的財經新聞編輯。請將以下原始新聞內容改寫為專業、客觀且具備特定作者風格的繁體中文財經報導。要求：標題一行，內文3至5段，總字數約 {min_len}-{max_len} 字，嚴禁使用 Markdown 或列點。"

prompt_format = """<start_of_turn>user
{}
{}<end_of_turn>
<start_of_turn>model
"""

for file_path in tqdm(source_files, desc="Phase 4 生成中 (穩定版)"):
    output_file = output_dir / f"{file_path.stem}.json"
    
    with open(file_path, "r", encoding="utf-8") as f:
        article_data = json.load(f)
        source_content = article_data.get("content", "")
    
    # 計算動態字數
    min_len, max_len = get_dynamic_length_range(source_content)
    instruction = instruction_template.format(min_len=min_len, max_len=max_len)
    
    full_prompt = prompt_format.format(instruction, source_content)
    
    # 將文字轉為 GPU 張量
    inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")
    input_len = inputs["input_ids"].shape[1] # 紀錄輸入的長度，以便後續截斷
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=1024,
            temperature=0.4, # 較低溫度以維持專業性
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.5, # 關鍵：設定較高的懲罰值，防止微調模型產生跳針文字
            pad_token_id=tokenizer.eos_token_id
        )
    
    # 5. 精確後處理
    # 只解碼模型「新產出」的 Token 部分
    generated_tokens = outputs[0][input_len:]
    raw_text = tokenizer.decode(generated_tokens, skip_special_tokens=False).strip()
    
    # 執行智慧截斷 (移除雜訊與重複)
    final_text = extract_first_generation(raw_text)
    
    # 基礎格式清理 (移除 Markdown 符號)
    final_text = clean_generated_text(final_text)

    # 6. 儲存結果
    result = {
        "article_id": file_path.stem,
        "source_file": str(file_path),
        "target_length_range": f"{min_len}-{max_len}",
        "generated_text": final_text,
        "model": "gemma-2-9b-it-lora-distilled"
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

print(f"\n=== Phase 4 批次生成完成！結果已儲存至 {output_dir} ===")
