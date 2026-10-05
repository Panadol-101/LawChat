from __future__ import annotations

import uuid

import pytest

from indexing import (
    BM25Settings,
    LexicalChunk,
    TantivyBM25Index,
    normalize_legal_identifier,
)
from retrieval import SparseSearchFilter, TantivySparseSearcher
from retrieval.sparse import _build_query


class MemoryLexicalSource:
    def __init__(self, chunks, *, fetch_size=2):
        self.chunks = sorted(chunks, key=lambda item: item.database_id)
        self.fetch_size = fetch_size

    def iter_batches(self, *, collection, after=None, limit=None):
        rows = [
            item
            for item in self.chunks
            if after is None or item.database_id > after
        ]
        if limit is not None:
            rows = rows[:limit]
        for offset in range(0, len(rows), self.fetch_size):
            yield rows[offset:offset + self.fetch_size]

    def count_expected(self, *, collection):
        return len(self.chunks)


def _chunk(
    number: int,
    *,
    document_number: str,
    text: str,
    article: str = "5",
    clause: str = "2",
) -> LexicalChunk:
    return LexicalChunk(
        database_id=uuid.UUID(int=number),
        point_id=f"point-{number}",
        chunk_id=f"doc-{number}::article_{article}::clause_{clause}",
        doc_id=f"doc-{number}",
        document_number=document_number,
        document_type="Nghị định",
        authority="Chính phủ",
        legal_field="Giao thông",
        title=f"Nghị định {document_number}",
        article=article,
        clause=clause,
        point=None,
        retrieval_text=text,
    )


def test_bm25_build_is_resumable_and_alias_requires_completeness(tmp_path):
    settings = BM25Settings(
        root=tmp_path,
        index_name="legal-test-v1",
        alias="legal-current",
        source_collection="dense-v2",
        writer_heap_size=20_000_000,
        writer_threads=1,
    )
    source = MemoryLexicalSource(
        [
            _chunk(
                1,
                document_number="100/2019/NĐ-CP",
                text="Xử phạt vi phạm hành chính giao thông đường bộ.",
            ),
            _chunk(
                2,
                document_number="123/2024/NĐ-CP",
                text="Điều kiện kinh doanh vận tải hành khách.",
            ),
            _chunk(
                3,
                document_number="45/2019/QH14",
                text="Quyền đơn phương chấm dứt hợp đồng lao động.",
            ),
        ]
    )
    index = TantivyBM25Index(source, settings)

    partial = index.build(limit=2)

    assert partial.indexed_points == 2
    assert not partial.complete
    with pytest.raises(ValueError, match="complete manifest"):
        index.promote_alias()

    complete = index.build(promote_alias=True)

    assert complete.indexed_points == 3
    assert complete.newly_indexed_points == 1
    assert complete.complete
    assert settings.alias_path.exists()

    searcher = TantivySparseSearcher.from_settings(settings)
    exact = searcher.search(
        "Nghị định 100 quy định gì?",
        limit=3,
        filters=SparseSearchFilter(document_numbers=("100/2019/NĐ-CP",)),
    )
    semantic = searcher.search("chấm dứt hợp đồng lao động", limit=1)

    assert exact[0].point_id == "point-1"
    assert semantic[0].point_id == "point-3"


def test_legal_identifier_normalization_preserves_legal_separators():
    assert normalize_legal_identifier(" 100/2019/NĐ–CP ") == "100/2019/nđ-cp"


def test_bm25_can_require_document_article_and_clause(tmp_path):
    settings = BM25Settings(
        root=tmp_path,
        index_name="structured-v1",
        alias="structured-current",
        source_collection="dense-v2",
        writer_heap_size=20_000_000,
        writer_threads=1,
    )
    source = MemoryLexicalSource(
        [
            _chunk(
                11,
                document_number="100/2019/NĐ-CP",
                text="Quy định cần tìm.",
                article="5",
                clause="2",
            ),
            _chunk(
                12,
                document_number="100/2019/NĐ-CP",
                text="Quy định cần tìm.",
                article="36",
                clause="3",
            ),
        ]
    )
    TantivyBM25Index(source, settings).build(promote_alias=True)
    searcher = TantivySparseSearcher.from_settings(settings)

    results = searcher.search(
        "Quy định cần tìm",
        limit=10,
        filters=SparseSearchFilter(
            document_numbers=("100/2019/NĐ-CP",),
            article_hints=("5",),
            clause_hints=("2",),
            require_structure=True,
        ),
    )

    assert [item.chunk_id for item in results] == ["doc-11::article_5::clause_2"]


def test_bm25_query_analyzer_boosts_vietnamese_legal_phrases():
    query = _build_query(
        "Người lao động có được đơn phương chấm dứt hợp đồng không?",
        SparseSearchFilter(),
    )

    assert '"Người lao động"^10' in query
    assert '"đơn phương chấm dứt hợp đồng"^10' in query
    assert '"đơn phương chấm dứt hợp đồng lao động"^20' in query
    assert " AND " in query
    assert "(Người lao động có được đơn phương" in query
