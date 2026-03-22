import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

"""
本模組是 Phase 4 微調流程的核心腳本。
主要功能：
1. 使用 Unsloth 框架載入 Gemma 2 9B 4-bit 模型。
2. 配置 LoRA (Low-Rank Adaptation) 參數，針對特定的寫作風格進行模型微調。
3. 包含針對 Gemma 2 模型在特定環境下的 Fused Loss 暴力 Patch，解決 VRAM 檢查報錯。
4. 執行訓練並將 LoRA 適配器 (Adapter) 儲存至 models/adapters/。
"""

# 0. 暴力 Monkey Patch: 徹底取代 Unsloth 的 Fused Loss 以避開 VRAM 檢查
# 原因：在某些硬體或版本下，Unsloth 的 Fused Loss 會觸發顯存不足報錯或與 Gemma 2 不相容。
# 這裡強行用原生 PyTorch 的 Cross Entropy 取代它，雖然慢一點點，但保證能跑成功。
import unsloth.models.llama
import unsloth_zoo.fused_losses.cross_entropy_loss

def dummy_fused_ce_loss(*args, **kwargs):
    # 從參數中提取必要的隱藏狀態、權重與標籤
    hidden_states = kwargs.get("hidden_states")
    lm_head_weight = kwargs.get("lm_head_weight")
    labels = kwargs.get("labels")
    
    # 若參數是從 args 傳入則手動解析
    if hidden_states is None and len(args) >= 2:
        hidden_states = args[0]
        lm_head_weight = args[1]
        labels = args[2] if len(args) >= 3 else labels

    if hidden_states is None or lm_head_weight is None:
        logits = kwargs.get("logits", args[0] if len(args) > 0 else None)
    else:
        # 手動計算 Logits: hidden_states 與 lm_head_weight 的轉置相乘
        logits = torch.matmul(hidden_states, lm_head_weight.t())
        
        # 特別處理 Gemma 2 的 Logit Softcapping (數值限制在 30.0 以內)
        logit_softcapping = kwargs.get("logit_softcapping", 30.0)
        if logit_softcapping > 0:
            logits = logits / logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * logit_softcapping

    if logits is None or labels is None:
        raise RuntimeError(f"Monkey patch failed. Args: {len(args)}, Kwargs: {list(kwargs.keys())}")

    # 標準 LLM 損失計算 (Shifted Cross Entropy)
    # 將預測值向右移一位，與標籤對齊
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return loss

# 強行覆蓋 Unsloth 內部的 Fused Loss 進入點
unsloth.models.llama.unsloth_fused_ce_loss = dummy_fused_ce_loss
unsloth_zoo.fused_losses.cross_entropy_loss.unsloth_fused_ce_loss = dummy_fused_ce_loss

# 1. 配置設定
model_name = "unsloth/gemma-2-9b-it-bnb-4bit" # 使用 Unsloth 優化的 4-bit 版本
max_seq_length = 1024 # 最大序列長度，足以容納新聞內文
dataset_path = "dataset/jsonl/train.jsonl" # 剛才準備好的訓練集
output_dir = "models/adapters/writer-style-lora" # 產出 Adapter 的位置

# 2. 載入模型與分詞器 (Tokenizer)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    load_in_4bit = True, # 強制 4-bit 載入以省顯存
)

# 3. 加入 LoRA 適配器配置
# LoRA 能讓我們只訓練模型不到 1% 的參數，就能學會新的風格。
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Rank：控制適配器的複雜度
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], # 針對核心層進行微調
    lora_alpha = 32,
    lora_dropout = 0, # Unsloth 推薦 LoRA Dropout 為 0 以獲得最佳速度
    bias = "none",
    use_gradient_checkpointing = "unsloth", # 開啟優化的梯度檢查點，大幅降低顯存消耗
)

# 4. 準備資料格式 (Prompt Formatting)
# 必須與微調時使用的對話模板 (Gemma 2 格式) 完全一致。
prompt_format = """<start_of_turn>user
{}
{}<end_of_turn>
<start_of_turn>model
{}<end_of_turn>"""

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input_text, output in zip(instructions, inputs, outputs):
        text = prompt_format.format(instruction, input_text, output)
        texts.append(text)
    return { "text" : texts }

# 載入 JSONL 資料集並套用格式化
dataset = load_dataset("json", data_files=dataset_path, split="train")
dataset = dataset.map(formatting_prompts_func, batched = True)

# 5. 設定訓練參數
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    args = TrainingArguments(
        per_device_train_batch_size = 1, # 顯存受限時設為 1，並搭配梯度累積
        gradient_accumulation_steps = 8, # 每 8 步才更新一次權重，效果等同 batch_size=8
        warmup_steps = 5,
        max_steps = 60, # 財經風格轉移通常不需要太多步就能學會，避免過擬合 (Overfitting)
        learning_rate = 2e-4, # 學習率，LoRA 微調的標準設定
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit", # 使用 8-bit Optimizer 以省顯存
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407, # 幸運隨機數種子
        output_dir = "outputs/logs/finetune",
        report_to = "none",
    ),
)

# 6. 開始訓練
print("=== 正在開始微調訓練 (Gemma 2 已套用自定義 Patch) ===")
trainer.train()

# 7. 儲存訓練好的 LoRA Adapter
# 注意：這只會存幾十 MB 的小檔案 (Adapter)，不會存整個 9B 的大模型
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"=== 微調完成！ LoRA 適配器已儲存至: {output_dir} ===")
