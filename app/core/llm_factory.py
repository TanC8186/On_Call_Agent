"""LLM 工厂类

使用 LangChain ChatOpenAI 通过 OpenAI 兼容模式调用各种 LLM 提供商。
支持 DeepSeek / 阿里云 DashScope / OpenAI / 及其他兼容 API。

默认使用 DeepSeek，通过 .env 中的 LLM_BASE_URL / LLM_API_KEY / LLM_MODEL 配置。
"""

from langchain_openai import ChatOpenAI
from app.config import config
from loguru import logger


class LLMFactory:
    """LLM 工厂类 - 使用 OpenAI 兼容模式，支持多种提供商"""

    @staticmethod
    def create_chat_model(
        model: str | None = None,
        temperature: float = 0.7,
        streaming: bool = True,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> ChatOpenAI:
        """
        创建 ChatOpenAI 实例（OpenAI 兼容模式）

        通过 .env 配置切换提供商：
        - DeepSeek:  LLM_BASE_URL=https://api.deepseek.com  LLM_MODEL=deepseek-chat
        - DashScope: LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1  LLM_MODEL=qwen-max
        - OpenAI:    LLM_BASE_URL=https://api.openai.com/v1  LLM_MODEL=gpt-4o

        Args:
            model: 模型名称，默认使用 config.llm_model
            temperature: 温度参数
            streaming: 是否流式输出
            base_url: API 地址，默认使用 config.llm_base_url
            api_key: API Key，默认使用 config.llm_api_key
        """
        model = model or config.llm_model
        base_url = base_url or config.llm_base_url
        api_key = api_key or config.llm_api_key

        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            streaming=streaming,
            base_url=base_url,
            api_key=api_key,
        )

        logger.info(
            f"LLM 初始化: model={model}, base_url={base_url}, streaming={streaming}"
        )
        return llm


# 全局 LLM 工厂实例
llm_factory = LLMFactory()
