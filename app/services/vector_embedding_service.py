"""向量嵌入服务模块 - 多提供商 Embedding 支持

支持 local / dashscope / openai 三种 Embedding 提供商切换。
- local: 本地 sentence-transformers 模型（无需 API Key）
"""

import os
# 国内 HuggingFace 镜像，必须在任何 huggingface 导入前设置
_HF_MIRROR = "https://hf-mirror.com"
os.environ.setdefault("HF_ENDPOINT", _HF_MIRROR)

from typing import List

from langchain_core.embeddings import Embeddings
from openai import OpenAI
from loguru import logger

from app.config import config


class _ZeroVectorEmbeddings(Embeddings):
    """零向量 Embedding（占位，所有 provider 不可用时的最后回退）"""
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[0.0] * 384 for _ in texts]
    def embed_query(self, text: str) -> List[float]:
        return [0.0] * 384


def _create_openai_embeddings(
    api_key: str,
    base_url: str,
    model: str,
    dimensions: int = 1024,
) -> "BaseOpenAIEmbeddings":
    """通过 OpenAI 兼容 API 创建 Embeddings 实例"""
    class OpenAICompatibleEmbeddings(Embeddings):
        """OpenAI 兼容的 Embeddings 实现"""

        def __init__(self, api_key: str, base_url: str, model: str, dimensions: int):
            if not api_key or api_key in ("your-api-key-here", "your-dashscope-api-key-here", "sk-your-deepseek-api-key-here", "sk-your-openai-key"):
                raise ValueError(
                    f"请设置 EMBEDDING_API_KEY 环境变量。"
                    f"当前 provider={config.embedding_provider}，base_url={base_url}"
                )
            self.client = OpenAI(api_key=api_key, base_url=base_url)
            self.model = model
            self.dimensions = dimensions
            masked = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 8 else "***"
            logger.info(f"Embeddings 初始化: provider={config.embedding_provider}, model={model}, dim={dimensions}, key={masked}")

        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            if not texts:
                return []
            try:
                logger.info(f"批量嵌入 {len(texts)} 个文档")
                kwargs = dict(model=self.model, input=texts, encoding_format="float")
                if self.dimensions:
                    kwargs["dimensions"] = self.dimensions
                response = self.client.embeddings.create(**kwargs)
                return [item.embedding for item in response.data]
            except Exception as e:
                logger.error(f"批量嵌入失败: {e}")
                raise RuntimeError(f"批量嵌入失败: {e}") from e

        def embed_query(self, text: str) -> List[float]:
            if not text or not text.strip():
                raise ValueError("查询文本不能为空")
            try:
                kwargs = dict(model=self.model, input=text, encoding_format="float")
                if self.dimensions:
                    kwargs["dimensions"] = self.dimensions
                response = self.client.embeddings.create(**kwargs)
                return response.data[0].embedding
            except Exception as e:
                logger.error(f"查询嵌入失败: {e}")
                raise RuntimeError(f"查询嵌入失败: {e}") from e

    return OpenAICompatibleEmbeddings(api_key, base_url, model, dimensions)


def create_embedding_service() -> Embeddings:
    """根据配置创建 Embedding 服务（容错：所有 provider 失败时返回零向量 stub）"""
    provider = config.embedding_provider

    if provider == "local":
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            model_name = config.embedding_model or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            mirror = os.environ.get("HF_ENDPOINT", "default")
            logger.info(f"使用本地 Embedding 模型: {model_name} (mirror: {mirror})")
            try:
                logger.info("尝试离线加载 Embedding 模型...")
                return HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={"device": "cpu", "local_files_only": True},
                    encode_kwargs={"normalize_embeddings": True},
                )
            except Exception as offline_err:
                logger.warning(f"离线加载失败，尝试在线模式: {offline_err}")
                return HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True},
                )
        except Exception as e:
            logger.warning(f"本地 Embedding 模型加载失败: {e}，尝试 DashScope")
            provider = "dashscope"

    if provider == "openai":
        try:
            return _create_openai_embeddings(
                api_key=config.embedding_api_key or config.llm_api_key,
                base_url=config.embedding_base_url or "https://api.openai.com/v1",
                model=config.embedding_model or "text-embedding-3-small",
                dimensions=1536,
            )
        except Exception as e:
            logger.warning(f"OpenAI Embedding 初始化失败: {e}")

    # dashscope
    try:
        return _create_openai_embeddings(
            api_key=config.embedding_api_key or config.llm_api_key,
            base_url=config.embedding_base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1",
            model=config.embedding_model or "text-embedding-v4",
            dimensions=1024,
        )
    except Exception as e:
        logger.warning(f"DashScope Embedding 初始化失败: {e}，使用零向量 stub")

    # 最终回退：零向量 stub（避免应用崩溃，但 RAG 功能不可用）
    logger.warning("⚠️  所有 Embedding provider 均不可用，RAG 知识库功能将返回空结果")
    return _ZeroVectorEmbeddings()


# 懒加载全局单例（避免 import 时因模型缺失导致整个应用崩溃）
_vector_embedding_service = None

def get_embedding_service() -> Embeddings:
    global _vector_embedding_service
    if _vector_embedding_service is None:
        _vector_embedding_service = create_embedding_service()
    return _vector_embedding_service

# 兼容旧代码的模块级 __getattr__（Python 3.7+）
def __getattr__(name: str):
    if name == "vector_embedding_service":
        return get_embedding_service()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
