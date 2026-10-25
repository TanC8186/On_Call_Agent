"""LLM 工厂：唯一的模型构造入口。

两个 provider：

- `anthropic`（默认）：走 Messages API，支持 tool calling、流式、token 用量。
  第三方网关（AutoDL 等）只在字段序列化上和官方 SDK 有差异，这里做一次兼容补丁。
- `openai`：任意 OpenAI 兼容端点（DeepSeek、DashScope、本地 vLLM 等）。

业务代码不应该再直接 `ChatOpenAI(...)`，否则换 provider 时要改五处。
"""

from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from loguru import logger

from app.config import config


class _CompatChatAnthropic(ChatAnthropic):
    """兼容第三方 Anthropic 网关。

    AutoDL 的 `/v1/messages` 在 `message_delta` 里返回
    `context_management: {...}`（dict），而 langchain-anthropic 1.7.6 假设它是
    pydantic 对象、直接调 `.model_dump()`，导致流式请求抛
    `AttributeError: 'dict' object has no attribute 'model_dump'`。

    这里在解析前把 dict 形式的字段摘掉——它只是元信息，不影响内容和 usage。
    """

    def _make_message_chunk_from_anthropic_event(self, event, **kwargs):  # type: ignore[override]
        for field in ("context_management", "container"):
            if isinstance(getattr(event, field, None), dict):
                try:
                    object.__setattr__(event, field, None)
                except Exception:  # noqa: BLE001 - 只做尽力而为的兼容
                    pass
        return super()._make_message_chunk_from_anthropic_event(event, **kwargs)


def _create_chat_model(
    model: str | None = None,
    temperature: float = 0.7,
    streaming: bool = True,
    max_tokens: int | None = None,
    **_: object,
) -> BaseChatModel:
    provider = (config.llm_provider or "anthropic").lower()
    model = model or config.llm_model

    if provider == "anthropic":
        llm: BaseChatModel = _CompatChatAnthropic(
            model=model,
            anthropic_api_key=config.llm_api_key,
            anthropic_api_url=config.llm_base_url.rstrip("/"),
            max_tokens=max_tokens or config.llm_max_tokens,
            temperature=temperature,
            streaming=streaming,
        )
    elif provider == "openai":
        llm = ChatOpenAI(
            model=model,
            api_key=config.llm_api_key,
            base_url=config.llm_base_url,
            temperature=temperature,
            streaming=streaming,
        )
    else:
        raise ValueError(f"未知的 LLM_PROVIDER: {provider}（支持 anthropic / openai）")

    logger.info(
        f"LLM 初始化: provider={provider}, model={model}, "
        f"base_url={config.llm_base_url}, streaming={streaming}"
    )
    return llm


class LLMFactory:
    """保持旧调用方式 `llm_factory.create_chat_model(...)` 可用。"""

    @staticmethod
    def create_chat_model(**kwargs: object) -> BaseChatModel:
        return _create_chat_model(**kwargs)


llm_factory = LLMFactory()
create_chat_model = _create_chat_model
