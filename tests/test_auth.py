"""可选 API Key 鉴权测试。"""

import pytest
from fastapi import HTTPException

from app.core import auth


@pytest.mark.asyncio
async def test_auth_disabled_when_no_key(monkeypatch):
    monkeypatch.setattr(auth, "API_KEY", "")
    await auth.require_api_key("")  # 不应抛出


@pytest.mark.asyncio
async def test_auth_rejects_wrong_key(monkeypatch):
    monkeypatch.setattr(auth, "API_KEY", "secret")
    with pytest.raises(HTTPException) as exc:
        await auth.require_api_key("wrong")
    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_auth_accepts_correct_key(monkeypatch):
    monkeypatch.setattr(auth, "API_KEY", "secret")
    await auth.require_api_key("secret")  # 不应抛出
