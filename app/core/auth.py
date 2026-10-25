"""可选 API Key 鉴权。

默认不设置 API_KEY 时保持本地开发体验；一旦设置，所有 /api 路由都要求
`X-API-Key` 请求头。生产/公网部署必须设置。
"""

import os

from fastapi import Header, HTTPException

API_KEY = os.getenv("API_KEY", "")


async def require_api_key(x_api_key: str = Header(default="")) -> None:
    if not API_KEY:
        return
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")
