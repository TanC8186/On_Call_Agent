"""配置管理模块

使用 Pydantic Settings 实现类型安全的配置管理
"""

import os
from pathlib import Path
from typing import Dict, Any
from pydantic_settings import BaseSettings, SettingsConfigDict

# .env 文件路径（相对于项目根目录，不依赖当前工作目录）
_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    """应用配置"""

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # 应用配置
    app_name: str = "SuperBizAgent"
    app_version: str = "1.0.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 9900

    # ====== LLM 通用配置（支持 DeepSeek / DashScope / OpenAI 等） ======
    llm_api_key: str = ""
    llm_base_url: str = "https://api.deepseek.com"
    llm_model: str = "deepseek-chat"

    # Embedding 配置
    embedding_provider: str = "dashscope"  # dashscope | openai | local
    embedding_api_key: str = ""
    embedding_model: str = "text-embedding-v4"
    embedding_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # 向后兼容旧字段名 (DASHSCOPE_* 环境变量仍可用)
    @property
    def dashscope_api_key(self) -> str:
        """兼容旧代码：优先用 LLM_API_KEY，回退到 DASHSCOPE_API_KEY"""
        return self.llm_api_key

    @property
    def dashscope_model(self) -> str:
        """兼容旧代码"""
        return self.llm_model

    @property
    def dashscope_embedding_model(self) -> str:
        """兼容旧代码"""
        return self.embedding_model

    @property
    def rag_model(self) -> str:
        """RAG 使用的模型"""
        return self.llm_model

    # Milvus 配置
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_timeout: int = 10000  # 毫秒

    # RAG 配置
    rag_top_k: int = 3

    # 文档分块配置
    chunk_max_size: int = 800
    chunk_overlap: int = 100

    # 腾讯云 API 凭证
    tencentcloud_appid: str = ""
    tencentcloud_secret_id: str = ""
    tencentcloud_secret_key: str = ""
    tencentcloud_region: str = "ap-beijing"

    # MCP 服务配置
    mcp_cls_transport: str = "streamable-http"
    mcp_cls_url: str = "http://localhost:8003/mcp"
    mcp_monitor_transport: str = "streamable-http"
    mcp_monitor_url: str = "http://localhost:8004/mcp"
    mcp_ssh_transport: str = "streamable-http"
    mcp_ssh_url: str = "http://localhost:8005/mcp"

    @property
    def mcp_servers(self) -> Dict[str, Dict[str, Any]]:
        """获取完整的 MCP 服务器配置"""
        servers = {
            "cls": {
                "transport": self.mcp_cls_transport,
                "url": self.mcp_cls_url,
            },
            "monitor": {
                "transport": self.mcp_monitor_transport,
                "url": self.mcp_monitor_url,
            },
            "ssh": {
                "transport": self.mcp_ssh_transport,
                "url": self.mcp_ssh_url,
            }
        }
        return servers


# 全局配置实例
config = Settings()
