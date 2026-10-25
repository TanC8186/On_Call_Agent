"""LLM 工厂测试：provider 选择 + 第三方网关兼容补丁。

不依赖网络也能跑：用一个最小的伪事件对象验证 `_CompatChatAnthropic` 会把
dict 形式的 `context_management` 摘掉（否则 langchain-anthropic 会抛
`AttributeError: 'dict' object has no attribute 'model_dump'`）。
"""

from types import SimpleNamespace

from app.config import config
from app.core.llm_factory import _CompatChatAnthropic, create_chat_model


def test_openai_provider_builds_chat_openai(monkeypatch):
    monkeypatch.setattr(config, "llm_provider", "openai")
    monkeypatch.setattr(config, "llm_base_url", "https://api.deepseek.com")
    monkeypatch.setattr(config, "llm_model", "deepseek-chat")
    llm = create_chat_model(streaming=False)
    assert type(llm).__name__ == "ChatOpenAI"
    assert "deepseek" in str(llm.openai_api_base)


def test_anthropic_provider_uses_compat_subclass(monkeypatch):
    monkeypatch.setattr(config, "llm_provider", "anthropic")
    monkeypatch.setattr(config, "llm_base_url", "https://example.test/api/v1/anthropic")
    monkeypatch.setattr(config, "llm_model", "some-model")
    llm = create_chat_model(streaming=False)
    assert isinstance(llm, _CompatChatAnthropic)


def test_unknown_provider_raises(monkeypatch):
    monkeypatch.setattr(config, "llm_provider", "nope")
    try:
        create_chat_model()
    except ValueError as exc:
        assert "LLM_PROVIDER" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("未知 provider 应当报错")


def test_compat_patch_removes_dict_context_management():
    """dict 字段要被摘掉，pydantic 对象则保持原样。"""
    llm = _CompatChatAnthropic(
        model="m",
        anthropic_api_key="k",
        anthropic_api_url="https://example.test",
        max_tokens=16,
    )
    calls = []

    def fake_super(event, **kwargs):
        calls.append((event.context_management, event.container))
        return None, None

    llm.__class__.__bases__[0]._make_message_chunk_from_anthropic_event = staticmethod(fake_super)
    try:
        event = SimpleNamespace(
            type="message_delta",
            context_management={"applied_edits": []},
            container={"x": 1},
        )
        llm._make_message_chunk_from_anthropic_event(event)
        assert calls[0] == (None, None)
    finally:
        del llm.__class__.__bases__[0]._make_message_chunk_from_anthropic_event


def test_factory_is_the_only_construction_site():
    """业务代码不应再直接 new ChatOpenAI / ChatAnthropic。"""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "app"
    offenders = []
    for path in root.rglob("*.py"):
        if path.name == "llm_factory.py":
            continue
        text = path.read_text(encoding="utf-8")
        if "ChatOpenAI(" in text or "ChatAnthropic(" in text:
            offenders.append(str(path.relative_to(root)))
    assert not offenders, f"这些文件绕过了工厂: {offenders}"
