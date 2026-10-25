"""回归测试：RAG 未命中时必须返回空上下文，而不是被上层当成命中。"""

from app.tools import knowledge_tool


class _EmptyRetriever:
    def invoke(self, _query):
        return []


class _EmptyVectorStore:
    def as_retriever(self, **_kwargs):
        return _EmptyRetriever()


def test_miss_returns_empty_context(monkeypatch):
    monkeypatch.setattr(
        knowledge_tool.vector_store_manager,
        "get_vector_store",
        lambda: _EmptyVectorStore(),
    )
    context, documents = knowledge_tool.retrieve_knowledge.func("不存在的知识")
    assert context == ""
    assert documents == []
