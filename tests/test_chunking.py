"""回归测试：分块不得超出 embedding 模型的上下文窗口。"""

from pathlib import Path

import pytest

from app.config import config
from app.services.document_splitter_service import document_splitter_service

DOCS_DIR = Path(__file__).resolve().parents[1] / "aiops-docs"


def _all_chunks():
    chunks = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        chunks.extend(
            document_splitter_service.split_document(
                path.read_text(encoding="utf-8"), str(path)
            )
        )
    return chunks


def test_chunks_do_not_exceed_configured_char_budget():
    chunks = _all_chunks()
    assert chunks, "知识库没有产生任何分块"
    oversized = [doc for doc in chunks if len(doc.page_content) > config.chunk_max_size]
    assert not oversized, f"{len(oversized)} 个分块超过 {config.chunk_max_size} 字符"


def test_chunks_do_not_exceed_embedding_token_limit():
    """只有本地模型可用时才跑；CI 没有模型缓存时自动跳过。"""
    transformers = pytest.importorskip("transformers")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            config.embedding_model, local_files_only=True
        )
    except Exception:
        pytest.skip("本地 embedding tokenizer 不可用")

    chunks = _all_chunks()
    too_long = [
        (len(tokenizer.encode(doc.page_content)), doc.metadata.get("_file_name"))
        for doc in chunks
        if len(tokenizer.encode(doc.page_content)) > 128
    ]
    assert not too_long, f"{len(too_long)} 个分块超过 128 tokens: {too_long[:5]}"


def test_merge_does_not_cross_section_boundaries():
    """合并小分片时不能把两个 H2 段落拼进同一个 chunk。"""
    for path in sorted(DOCS_DIR.glob("*.md")):
        chunks = document_splitter_service.split_document(
            path.read_text(encoding="utf-8"), str(path)
        )
        for doc in chunks:
            h2_count = sum(
                1 for line in doc.page_content.splitlines() if line.startswith("## ")
            )
            if h2_count > 1:
                pytest.fail(f"{path.name}: 一个 chunk 跨越了 {h2_count} 个 H2 段落")
