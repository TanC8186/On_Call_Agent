"""FastAPI Web 服务启动器

启动主 FastAPI 应用，提供 Web 界面和 API 服务。
运行方式: python servers/web_server.py
"""

import os
import sys
from pathlib import Path

# ── 必须在所有 HF/huggingface 相关模块导入前设置 ──
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENDPOINT"] = "https://hf-mirror.com"

# 确保项目根目录在 sys.path 中
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=9900,
        log_level="info",
    )
