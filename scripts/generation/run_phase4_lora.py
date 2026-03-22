import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from pathlib import Path
import json
from tqdm import tqdm

"""
本模組負責執行 Phase 4: LoRA 微調模型的批次生成任務。
主要功能：
1. 載入微調後的 LoRA Adapter 並掛載到 Gemma 2 基礎模型上。
2. 針對 data/raw/source_articles 下的所有原始新聞進行批次風格轉換。
3. 採用「動態長度管理」確保生成字數符合來源新聞比例。
4. 產出 JSON 格式結果，供後續評估與比較。
"""

# 將當前目錄加入 Python Path 以便引用內部的 scripts.utils
sys.path.append(os.getcwd())

from scripts.utils.generation_utils import get_dynamic_length_range, clean_generated_text

# 1. 配置設定
model_name = "unsloth/gemma-2-9b-it-bnb-4bit"
lora_adapter_path = "models/adapters/writer-style-lora" # 指向 Phase 4 訓練好的權重
source_dir = Path("data/raw/source_articles")
output_dir = Path("outputs/generations/phase4")
output_dir.mkdir(parents=True, exist_ok=True)

# 2. 載入模型與 LoRA 適配器
print(f"=== 正在以 4-bit 模式載入基礎模型 (Native Transformers) ===")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
)

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto", # 自動平衡顯示卡資源
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(f"=== 正在掛載微調權重: {lora_adapter_path} ===")
model = PeftModel.from_pretrained(base_model, lora_adapter_path)
model.eval() # 設定為評估模式

# 3. 準備批次生成
source_files = sorted(list(source_dir.glob("article-*.json")))
print(f"=== 找到 {len(source_files)} 篇新聞待處理 ===")

# 指令模板：需與訓練階段保持高度一致
instruction_template = "你是一位專業的財經新聞編輯。請將以下原始新聞內容改寫為專業、客觀且具備特定作者風格的繁體中文財經報導。要求：標題一行，內文3至5段，總字數約 {min_len}-{max_len} 字，嚴禁使用 Markdown 或列點。"

# Gemma 2 標準對話格式
prompt_format = """<start_of_turn>user
{}
{}<end_of_turn>
<start_of_turn>model
"""

# 4. 迴圈處理每一篇新聞
for file_path in tqdm(source_files, desc="Phase 4 生成中"):
    output_file = output_dir / f"{file_path.stem}.json"
    
    # 若檔案已存在則跳過 (斷點續傳)
    if output_file.exists():
        continue
        
    with open(file_path, "r", encoding="utf-8") as f:
        article_data = json.load(f)
        source_content = article_data.get("content", "")
    
    # 計算動態字數要求
    min_len, max_len = get_dynamic_length_range(source_content)
    instruction = instruction_template.format(min_len=min_len, max_len=max_len)
    
    # 組裝完整 Prompt
    full_prompt = prompt_format.format(instruction, source_content)
    
    # 執行模型推理 (Inference)
    inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=1024,
            temperature=0.7, # 稍微調高隨機性以增加文筆流暢度
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1, # 避免重複跳針
            pad_token_id=tokenizer.eos_token_id
        )
    
    # 將 Token 轉回文字
    generated_full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 擷取模型回答部分：只取 <start_of_turn>model 之後的內容
    if "model" in generated_full:
        final_text = generated_full.split("model")[-1].strip()
    else:
        # 保底方案：移除 Prompt 原始文字
        final_text = generated_full.replace(full_prompt, "").strip()
        
    # 文字後處理清洗
    final_text = clean_generated_text(final_text)

    # 5. 封裝結果並儲存
    result = {
        "article_id": file_path.stem,
        "source_file": str(file_path),
        "target_length_range": f"{min_len}-{max_len}",
        "generated_content": final_text,
        "model": "gemma-2-9b-it-lora-distilled"
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

print(f"\n=== Phase 4 批次生成完成！結果已儲存至 {output_dir} ===")
