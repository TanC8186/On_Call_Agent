"""向量存储管理器 - 封装 Milvus VectorStore 操作（懒加载）"""

from typing import List

from langchain_core.documents import Document
from langchain_milvus import Milvus
from loguru import logger

from app.config import config
from app.core.milvus_client import milvus_manager
from app.services.vector_embedding_service import vector_embedding_service


# 统一使用 biz collection
COLLECTION_NAME = "biz"


class VectorStoreManager:
    """向量存储管理器（懒加载，Milvus 不可用时不影响对话功能）"""

    def __init__(self):
        """初始化向量存储管理器"""
        self.vector_store = None
        self._initialized = False
        self._init_error = None
        self.collection_name = COLLECTION_NAME

    def _ensure_initialized(self):
        """懒加载：首次调用时才连接 Milvus"""
        if self._initialized:
            if self._init_error:
                raise self._init_error
            return True
        self._initialized = True
        try:
            _ = milvus_manager.connect()

            connection_args = {
                "host": config.milvus_host,
                "port": config.milvus_port,
            }

            self.vector_store = Milvus(
                embedding_function=vector_embedding_service,
                collection_name=self.collection_name,
                connection_args=connection_args,
                auto_id=False,
                drop_old=False,
                text_field="content",
                vector_field="vector",
                primary_field="id",
                metadata_field="metadata",
            )

            logger.info(
                f"VectorStore 初始化成功: {config.milvus_host}:{config.milvus_port}, "
                f"collection: {self.collection_name}"
            )

        except Exception as e:
            logger.warning(f"VectorStore 初始化失败（Milvus 不可用？）: {e}")
            self._init_error = e

    def add_documents(self, documents: List[Document]) -> List[str]:
        """批量添加文档到向量存储"""
        self._ensure_initialized()
        if self.vector_store is None:
            raise RuntimeError("VectorStore 未初始化，无法添加文档")
        try:
            import uuid
            ids = [str(uuid.uuid4()) for _ in documents]
            return self.vector_store.add_documents(documents, ids=ids)
        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            raise

    def delete_by_source(self, file_path: str) -> int:
        """删除指定文件的所有文档"""
        self._ensure_initialized()
        if self.vector_store is None:
            return 0
        try:
            collection = milvus_manager.get_collection()
            expr = f'metadata["_source"] == "{file_path}"'
            result = collection.delete(expr)
            deleted_count = result.delete_count if hasattr(result, "delete_count") else 0
            logger.info(f"删除文件旧数据: {file_path}, 删除数量: {deleted_count}")
            return deleted_count
        except Exception as e:
            logger.warning(f"删除旧数据失败 (可能是首次索引): {e}")
            return 0

    def get_vector_store(self):
        """获取 VectorStore 实例（如果未初始化则返回 None）"""
        try:
            self._ensure_initialized()
        except Exception:
            pass
        return self.vector_store

    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """相似度搜索"""
        self._ensure_initialized()
        if self.vector_store is None:
            return []
        try:
            docs = self.vector_store.similarity_search(query, k=k)
            logger.debug(f"相似度搜索: query='{query}', 结果数={len(docs)}")
            return docs
        except Exception as e:
            logger.error(f"相似度搜索失败: {e}")
            return []


# 全局单例（懒加载，不会在导入时崩溃）
vector_store_manager = VectorStoreManager()
