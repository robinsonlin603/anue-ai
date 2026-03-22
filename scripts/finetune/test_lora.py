import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from pathlib import Path
import json

"""
本模組用於測試微調後的 LoRA 模型效果 (Evaluation Script)。
主要特色：
1. 使用原生 Hugging Face `transformers` 與 `peft` 庫，而非 Unsloth 進行推論。
2. 目的是避開 Unsloth 在某些 GPU 上推論時可能出現的格式錯誤或顯存 Bug。
3. 針對單一來源新聞 (預設為 article-020) 進行風格轉換測試，並輸出結果供人工檢閱。
"""

# 1. 配置設定
model_name = "unsloth/gemma-2-9b-it-bnb-4bit"
lora_adapter_path = "models/adapters/writer-style-lora" # 剛才訓練好的 LoRA 路徑
test_article_path = "data/raw/source_articles/article-020.json" # 測試用的原始新聞
output_dir = "outputs/generations/phase4_lora"
Path(output_dir).mkdir(parents=True, exist_ok=True)

# 2. 載入基礎模型與 LoRA 適配器
print(f"=== 正在載入模型 (使用原生 Transformers 模式) ===")
# 使用 4-bit 量化載入以節省顯存
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
)

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto", # 自動分配 GPU 資源
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 載入我們訓練好的 LoRA 權重
print(f"=== 正在掛載 LoRA 適配器: {lora_adapter_path} ===")
model = PeftModel.from_pretrained(base_model, lora_adapter_path)
model.eval() # 設定為評估模式 (關閉 Dropout 等)

# 3. 準備測試資料與 Prompt
with open(test_article_path, "r", encoding="utf-8") as f:
    article_data = json.load(f)
    source_content = article_data.get("content", "")

# 使用與訓練時完全相同的 Instruction
instruction = "你是一位專業的財經新聞編輯。請將以下原始新聞內容改寫為專業、客觀且具備特定作者風格的繁體中文財經報導。要求：標題一行，內文3至5段，總字數約 500-700 字，嚴禁使用 Markdown 或列點。"

# 符合 Gemma 2 的對話格式模板
prompt_format = """<start_of_turn>user
{}
{}<end_of_turn>
<start_of_turn>model
"""

full_prompt = prompt_format.format(instruction, source_content)

# 4. 執行文字生成
print("=== 模型正在進行風格轉換生成中... ===")
inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=1024, # 限制最大長度
        temperature=0.7,     # 稍微提高隨機性，讓文筆更自然
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.1, # 懲罰重複字眼，增加流暢度
        pad_token_id=tokenizer.eos_token_id
    )

# 將輸出的 Token 轉回文字
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# 後處理：移除 Prompt 部分，只保留模型生成的新聞內容
final_output = generated_text
if "model" in generated_text:
    # 根據 Gemma 2 的格式切分，取 model 標籤之後的內容
    parts = generated_text.split("model")
    if len(parts) > 1:
        final_output = parts[-1].strip()

# 移除模型可能產生的過渡性文字
final_output = final_output.replace("改寫如下：", "").strip()

# 5. 儲存結果與預覽
output_file = Path(output_dir) / "article-020_lora.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(final_output)

print(f"\n=== 測試完成！ ===\n")
print(f"生成結果已儲存至: {output_file}\n")
print("-" * 30)
print("文章預覽 (前 500 字)：")
print("-" * 30)
print(final_output[:500] + "...")
print("-" * 30)
