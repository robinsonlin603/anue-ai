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
本模組負責執行 Phase 4.1: Hybrid (混合模式) 生成任務。
主要特色：
1. LoRA Adapter + 顯性風格規範 (Phase 2 Spec)：
   這是一個實驗性的變體，不但使用了微調後的模型權重，還在 Prompt 中保留了 Phase 2 的詳細風格指令。
2. 雙重約束：透過模型內在權重 (LoRA) 與外在指令 (Spec) 共同夾擊，旨在達成極致的風格轉移效果。
3. 採用與 Fixed 版本一致的穩定推論配置與後處理邏輯。
"""

# 將當前目錄加入 path 以便引用工具與配置
sys.path.append(os.getcwd())

from scripts.utils.generation_utils import get_dynamic_length_range, clean_generated_text
from configs.style_specs import PHASE2_STYLE_SPEC

# 1. 配置設定
model_name = "unsloth/gemma-2-9b-it-bnb-4bit"
lora_adapter_path = "models/adapters/writer-style-lora"
source_dir = Path("data/raw/source_articles")
output_dir = Path("outputs/generations/phase4_1") # 儲存到獨立的 phase4_1 目錄
output_dir.mkdir(parents=True, exist_ok=True)

# 2. 載入模型與 LoRA 適配器
print(f"=== 正在載入混合模式模型 (LoRA + Spec) ===")
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

print(f"=== 正在掛載微調權重: {lora_adapter_path} ===")
model = PeftModel.from_pretrained(base_model, lora_adapter_path)
model.eval()

# 3. 準備混合指令模板
# 這裡將 Phase 2 的詳細規範直接嵌入到 Instruction 中
instruction_template = f"""You are a financial news editor.
Rewrite the given news content into a Chinese financial news article that mimics a specific journalist style.

{PHASE2_STYLE_SPEC}

Target Length: {{min_len}}–{{max_len}} Chinese characters.
"""

prompt_format = """<start_of_turn>user
{}

Title:
{}

Content:
{}<end_of_turn>
<start_of_turn>model
"""

def extract_first_generation(text: str) -> str:
    """擷取模型輸出的第一部分，並智慧過濾掉雜訊。"""
    stop_tokens = [
        "<start_of_turn>", "<end_of_turn>", "model", "user", "<unused4>", 
        "</textarea>", "</button>", "<html>", "</body>", "});", "<eos>", "<|im_end|>"
    ]
    earliest_pos = len(text)
    for token in stop_tokens:
        pos = text.find(token)
        if pos != -1 and pos < earliest_pos:
            earliest_pos = pos
    content = text[:earliest_pos].strip()
    return content

# 4. 執行批次混合生成
source_files = sorted(list(source_dir.glob("article-*.json")))
print(f"=== 找到 {len(source_files)} 篇新聞進行 Phase 4.1 混合生成 ===")

for file_path in tqdm(source_files, desc="Phase 4.1 生成中"):
    output_file = output_dir / f"{file_path.stem}.json"
    
    # 讀取來源
    with open(file_path, "r", encoding="utf-8") as f:
        article_data = json.load(f)
        source_title = article_data.get("title", "")
        source_content = article_data.get("content", "")
    
    # 計算動態長度
    min_len, max_len = get_dynamic_length_range(source_content)
    instruction = instruction_template.format(min_len=min_len, max_len=max_len)
    
    # 組裝 Prompt
    full_prompt = prompt_format.format(instruction, source_title, source_content)
    
    # 模型推理
    inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")
    input_len = inputs["input_ids"].shape[1]
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=1024,
            temperature=0.4, 
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.5,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # 後處理與清洗
    generated_tokens = outputs[0][input_len:]
    raw_text = tokenizer.decode(generated_tokens, skip_special_tokens=False).strip()
    final_text = extract_first_generation(raw_text)
    final_text = clean_generated_text(final_text)

    # 儲存結果 (標記模型為 hybrid-v4.1)
    result = {
        "article_id": file_path.stem,
        "source_file": str(file_path),
        "target_length_range": f"{min_len}-{max_len}",
        "generated_text": final_text,
        "model": "gemma-2-9b-it-lora-hybrid-v4.1"
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

print(f"\n=== Phase 4.1 混合生成完成！結果已儲存至 {output_dir} ===")
