"""專案早期集中設定：路徑、預設模型、prompt 檔對照。

部分新腳本已改為 argparse / 環境變數；此檔仍可作為常數參考或舊程式 import 來源。
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

CHAT_MODEL = "gemma:7b"
EMBED_MODEL = "nomic-embed-text-v2-moe"
OLLAMA_HOST = "http://localhost:11434"

PROMPTS_DIR = BASE_DIR / "dataset" / "prompts"
OUTPUTS_DIR = BASE_DIR / "outputs" / "generations"

PROMPT_FILES = {
    "phase1": PROMPTS_DIR / "phase1_basic.txt",
    "phase2_style_v1": PROMPTS_DIR / "phase2_style_v1.txt",
    "phase2_style_v2": PROMPTS_DIR / "phase2_style_v2.txt",
}

DEFAULT_PROMPT_VERSION = {
    "phase1": "phase1_basic",
    "phase2": "phase2_style_v1",
}
