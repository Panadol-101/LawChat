import pytest

from chunking import (
    ChunkingConfig,
    LegalChunker,
)
from chunking.utils import (
    approximate_token_count,
    configure_token_counter,
)


def test_default_profile_matches_bge_m3_chunk_policy():
    config = ChunkingConfig()

    assert config.target_min_tokens == 600
    assert config.target_tokens == 700
    assert config.target_max_tokens == 800
    assert config.max_indexable_tokens == 1200
    assert config.overlap_ratio == 0.125
    assert config.min_overlap_tokens == 60
    assert config.base_indexable_tokens == 1200


class _InflatingTokenCounter:
    def count_tokens(self, text):
        return approximate_token_count(text) * 3

    def count_many_tokens(self, texts):
        return [self.count_tokens(text) for text in texts]

    def split_text(self, text, *, max_tokens):
        words = text.split()
        size = max(1, max_tokens // 3)
        return [
            " ".join(words[index:index + size])
            for index in range(0, len(words), size)
        ]

    def tail_text(self, text, *, max_tokens):
        return self.split_text(text, max_tokens=max_tokens)[-1]


def test_model_token_counter_creates_safe_fragments_after_semantic_chunking():
    parsed = {
        "doc_id": "doc-model-limit",
        "structure_type": "article_based",
        "preamble": "",
        "body": [
            {
                "type": "article",
                "article": "1",
                "title": "Nội dung dài",
                "text": " ".join(f"noidung{index}" for index in range(500)),
                "clauses": [],
                "points": [],
                "path": {},
            }
        ],
    }
    configure_token_counter(_InflatingTokenCounter())
    try:
        chunks = LegalChunker().chunk_document(parsed, {"id": "doc-model-limit"})
    finally:
        configure_token_counter(None)

    assert any("parent" in chunk["chunk_type"] for chunk in chunks)
    assert any("fragment" in chunk["chunk_type"] for chunk in chunks)
    assert all(
        chunk["approx_token_count"] <= 1200
        for chunk in chunks
        if chunk["is_indexable"]
    )


def metadata():
    return {
        "id": "doc-123",
        "title": "Nghị định thử nghiệm",
        "so_ky_hieu": "123/2026/NĐ-CP",
        "loai_van_ban": "Nghị định",
        "co_quan_ban_hanh": "Chính phủ",
        "ngay_ban_hanh": "2025-12-10",
        "ngay_co_hieu_luc": "2026-01-01",
        "ngay_het_hieu_luc": None,
        "tinh_trang_hieu_luc": "Còn hiệu lực",
        "linh_vuc": "Doanh nghiệp",
    }


def test_short_article_with_clauses_indexes_whole_article():
    parsed = {
        "doc_id": "doc-123",
        "structure_type": "article_based",
        "preamble": "",
        "body": [
            {
                "type": "article",
                "article": "15",
                "title": "Quyền của cá nhân",
                "text": "",
                "clauses": [
                    {
                        "clause": "1",
                        "text": "Cá nhân có quyền thực hiện hoạt động.",
                        "points": [],
                    },
                    {
                        "clause": "2",
                        "text": "Cá nhân có các quyền sau:",
                        "points": [
                            {
                                "point": "a",
                                "text": "Quyền thứ nhất.",
                            },
                            {
                                "point": "b",
                                "text": "Quyền thứ hai.",
                            },
                        ],
                    },
                ],
                "points": [],
                "path": {
                    "chapter": "III",
                    "section": "2",
                },
            }
        ],
    }

    chunks = LegalChunker().chunk_document(
        parsed,
        metadata(),
    )

    assert [item["chunk_type"] for item in chunks] == ["article"]
    article = chunks[0]
    assert article["article"] == "15"
    assert "1. Cá nhân có quyền" in article["text"]
    assert "a) Quyền thứ nhất." in article["text"]
    assert "b) Quyền thứ hai." in article["text"]
    assert "Điều 15: Quyền của cá nhân" in article["retrieval_text"]
    assert article["effective_date"] == "2026-01-01"


def test_article_without_clause_is_indexable_article():
    parsed = {
        "doc_id": "doc-123",
        "structure_type": "article_based",
        "preamble": "",
        "body": [
            {
                "type": "article",
                "article": "1",
                "title": "Phạm vi điều chỉnh",
                "text": "Văn bản này quy định về phạm vi điều chỉnh.",
                "clauses": [],
                "points": [],
                "path": {},
            }
        ],
    }

    chunks = LegalChunker().chunk_document(
        parsed,
        metadata(),
    )

    assert len(chunks) == 1
    assert chunks[0]["chunk_type"] == "article"
    assert chunks[0]["is_indexable"] is True


def test_oversized_clause_prefers_point_boundaries():
    config = ChunkingConfig(
        target_min_tokens=18,
        target_tokens=24,
        target_max_tokens=32,
        max_child_tokens=24,
        freeform_target_tokens=18,
        freeform_max_tokens=24,
        max_indexable_tokens=60,
        min_tail_tokens=0,
    )

    parsed = {
        "doc_id": "doc-123",
        "structure_type": "article_based",
        "preamble": "",
        "body": [
            {
                "type": "article",
                "article": "8",
                "title": "Trách nhiệm",
                "text": "",
                "clauses": [
                    {
                        "clause": "2",
                        "text": "Chủ thể có các trách nhiệm sau:",
                        "points": [
                            {
                                "point": "a",
                                "text": "Thực hiện trách nhiệm thứ nhất theo quy định.",
                            },
                            {
                                "point": "b",
                                "text": "Thực hiện trách nhiệm thứ hai theo quy đh.",
                            },
                        ],
                    }
                ],
                "points": [],
                "path": {},
            }
        ],
    }

    chunks = LegalChunker(
        config
    ).chunk_document(
        parsed,
        metadata(),
    )

    types = [
        item["chunk_type"]
        for item in chunks
    ]

    assert "clause_parent" in types
    assert types.count("point") == 2

    points = [
        item
        for item in chunks
        if item["chunk_type"] == "point"
    ]

    assert all(
        "Chủ thể có các trách nhiệm sau:"
        in item["retrieval_text"]
        for item in points
    )


def test_oversized_clause_without_points_uses_fragments():
    config = ChunkingConfig(
        target_min_tokens=10,
        target_tokens=16,
        target_max_tokens=24,
        max_child_tokens=12,
        freeform_target_tokens=10,
        freeform_max_tokens=12,
        max_indexable_tokens=45,
        min_tail_tokens=0,
    )

    parsed = {
        "doc_id": "doc-123",
        "structure_type": "article_based",
        "preamble": "",
        "body": [
            {
                "type": "article",
                "article": "2",
                "title": "Nội dung",
                "text": "",
                "clauses": [
                    {
                        "clause": "1",
                        "text": (
                            "Câu thứ nhất có nội dung tương đối dài. "
                            "Câu thứ hai tiếp tục quy định một nội dung khác. "
                            "Câu thứ ba bổ sung quy định."
                        ),
                        "points": [],
                    }
                ],
                "points": [],
                "path": {},
            }
        ],
    }

    chunks = LegalChunker(
        config
    ).chunk_document(
        parsed,
        metadata(),
    )

    assert any(
        item["chunk_type"] == "clause_parent"
        for item in chunks
    )

    assert sum(
        item["chunk_type"] == "clause_fragment"
        for item in chunks
    ) >= 2


def test_hierarchical_document_without_articles_preserves_section_parent():
    parsed = {
        "doc_id": "doc-123",
        "structure_type": "hierarchical_no_articles",
        "preamble": "",
        "body": [
            {
                "type": "chapter",
                "chapter": "I",
                "title": "TỔ CHỨC THỰC HIỆN",
                "children": [
                    {
                        "type": "section",
                        "section": "1",
                        "title": "TRÁCH NHIỆM",
                        "children": [
                            {
                                "type": "paragraph",
                                "text": "Bộ A có trách nhiệm thực hiện nhiệm vụ.",
                            },
                            {
                                "type": "list_item",
                                "marker": "1.",
                                "text": "Rà soát quy định.",
                            },
                        ],
                    }
                ],
            }
        ],
    }

    chunks = LegalChunker().chunk_document(
        parsed,
        metadata(),
    )

    types = [
        item["chunk_type"]
        for item in chunks
    ]

    assert "chapter_parent" in types
    assert "section_parent" in types
    assert "hierarchical_block" in types

    block = next(
        item
        for item in chunks
        if item["chunk_type"] == "hierarchical_block"
    )

    assert block["parent_type"] == "section"
    assert block["section"] == "1"


def test_semistructured_numbered_items_are_not_clauses():
    parsed = {
        "doc_id": "doc-123",
        "structure_type": "semi_structured",
        "preamble": "",
        "body": [
            {
                "type": "heading",
                "text": "CÔNG VĂN",
            },
            {
                "type": "paragraph",
                "text": "Kính gửi: Sở Tư pháp.",
            },
            {
                "type": "list_item",
                "marker": "1.",
                "text": "Tiếp tục rà soát.",
            },
            {
                "type": "list_item",
                "marker": "2.",
                "text": "Báo cáo kết quả.",
            },
        ],
    }

    chunks = LegalChunker().chunk_document(
        parsed,
        metadata(),
    )

    assert all(
        item["clause"] is None
        for item in chunks
    )

    assert any(
        item["chunk_type"] == "structured_block"
        for item in chunks
    )


def test_free_form_uses_document_parent_and_blocks():
    parsed = {
        "doc_id": "doc-123",
        "structure_type": "free_form",
        "preamble": "",
        "body": [
            {
                "type": "paragraph",
                "text": "Đoạn văn thứ nhất.",
            },
            {
                "type": "paragraph",
                "text": "Đoạn văn thứ hai.",
            },
            {
                "type": "paragraph",
                "text": "Đoạn văn thứ ba.",
            },
        ],
    }

    chunks = LegalChunker().chunk_document(
        parsed,
        metadata(),
    )

    assert chunks[0]["chunk_type"] == "document_parent"
    assert chunks[0]["is_indexable"] is False

    assert any(
        item["chunk_type"] == "freeform_block"
        and item["is_indexable"]
        for item in chunks
    )


def test_appendix_is_hard_boundary():
    parsed = {
        "doc_id": "doc-123",
        "structure_type": "mixed_article_appendix",
        "preamble": "",
        "body": [
            {
                "type": "article",
                "article": "1",
                "title": "Ban hành danh mục",
                "text": "Ban hành kèm theo danh mục.",
                "clauses": [],
                "points": [],
                "path": {},
            },
            {
                "type": "appendix",
                "appendix": "I",
                "title": "DANH MỤC",
                "children": [
                    {
                        "type": "list_item",
                        "marker": "1.",
                        "text": "Thủ tục A",
                    },
                    {
                        "type": "list_item",
                        "marker": "2.",
                        "text": "Thủ tục B",
                    },
                ],
            },
        ],
    }

    chunks = LegalChunker().chunk_document(
        parsed,
        metadata(),
    )

    assert any(
        item["chunk_type"] == "article"
        for item in chunks
    )

    assert any(
        item["chunk_type"] == "appendix_parent"
        for item in chunks
    )

    appendix_block = next(
        item
        for item in chunks
        if item["chunk_type"] == "appendix_block"
    )

    assert appendix_block["parent_type"] == "appendix"
    assert appendix_block["appendix"] == "I"


def test_table_appendix_repeats_header_per_block():
    parsed = {
        "doc_id": "doc-123",
        "structure_type": "appendix_or_form",
        "preamble": "",
        "body": [
            {
                "type": "appendix",
                "appendix": "I",
                "title": "MỨC PHẠT",
                "children": [
                    {
                        "type": "table",
                        "headers": [
                            "STT",
                            "Hành vi",
                            "Mức phạt",
                        ],
                        "rows": [
                            [
                                "1",
                                "Hành vi A",
                                "1 triệu",
                            ],
                            [
                                "2",
                                "Hành vi B",
                                "2 triệu",
                            ],
                        ],
                    }
                ],
            }
        ],
    }

    chunks = LegalChunker().chunk_document(
        parsed,
        metadata(),
    )

    table_blocks = [
        item
        for item in chunks
        if item["chunk_type"] == "table_block"
    ]

    assert table_blocks
    assert all(
        "STT | Hành vi | Mức phạt"
        in item["text"]
        for item in table_blocks
    )


def test_temporal_metadata_is_payload_not_embedding_by_default():
    parsed = {
        "doc_id": "doc-123",
        "structure_type": "free_form",
        "preamble": "",
        "body": [
            {
                "type": "paragraph",
                "text": "Nội dung quy định.",
            }
        ],
    }

    chunk = next(
        item
        for item in LegalChunker().chunk_document(
            parsed,
            metadata(),
        )
        if item["is_indexable"]
    )

    assert chunk["effective_date"] == "2026-01-01"
    assert "Ngày có hiệu lực:" not in chunk["retrieval_text"]

    config = ChunkingConfig(
        include_temporal_metadata_in_retrieval_text=True
    )

    chunk_with_temporal = next(
        item
        for item in LegalChunker(config).chunk_document(
            parsed,
            metadata(),
        )
        if item["is_indexable"]
    )

    assert (
        "Ngày có hiệu lực: 2026-01-01"
        in chunk_with_temporal["retrieval_text"]
    )


def test_parent_integrity_and_unique_ids():
    parsed = {
        "doc_id": "doc-123",
        "structure_type": "article_based",
        "preamble": "",
        "body": [
            {
                "type": "article",
                "article": "1",
                "title": "Một",
                "text": "",
                "clauses": [
                    {
                        "clause": "1",
                        "text": "Nội dung một.",
                        "points": [],
                    }
                ],
                "points": [],
                "path": {},
            },
            {
                "type": "article",
                "article": "2",
                "title": "Hai",
                "text": "",
                "clauses": [
                    {
                        "clause": "1",
                        "text": "Nội dung hai.",
                        "points": [],
                    }
                ],
                "points": [],
                "path": {},
            },
        ],
    }

    chunks = LegalChunker().chunk_document(
        parsed,
        metadata(),
    )

    ids = {
        item["chunk_id"]
        for item in chunks
    }

    assert len(ids) == len(chunks)

    for item in chunks:
        parent = item["parent_chunk_id"]
        if parent is not None:
            assert parent in ids

    assert all(
        item["retrieval_text"] == ""
        and item["approx_token_count"] == 0
        for item in chunks
        if not item["is_indexable"]
    )


def test_empty_document_produces_no_chunks():
    parsed = {
        "doc_id": "doc-123",
        "structure_type": "empty",
        "preamble": "",
        "body": [],
    }

    assert LegalChunker().chunk_document(
        parsed,
        metadata(),
    ) == []


def test_mixed_document_indexes_content_outside_articles():
    parsed = {
        "doc_id": "doc-123",
        "structure_type": "hierarchical_article_based",
        "preamble": "",
        "body": [{
            "type": "chapter", "chapter": "I", "title": "QUY ĐỊNH CHUNG",
            "children": [
                {"type": "paragraph", "text": "NỘI DUNG NGOÀI ĐIỀU PHẢI ĐƯỢC INDEX."},
                {"type": "article", "article": "1", "title": "Phạm vi",
                 "text": "Nội dung Điều 1.", "clauses": [], "points": [],
                 "path": {"chapter": "I"}},
            ],
        }],
    }
    chunks = LegalChunker().chunk_document(parsed, metadata())
    assert any(c["is_indexable"] and "NỘI DUNG NGOÀI ĐIỀU" in c["text"] for c in chunks)


def test_content_coverage_allows_section_split_across_multiple_strategies():
    parsed = {
        "doc_id": "doc-mixed-section",
        "structure_type": "hierarchical_article_based",
        "preamble": "",
        "body": [{
            "type": "section",
            "section": "II",
            "title": "XỬ LÝ VI PHẠM HÀNH CHÍNH",
            "children": [
                {
                    "type": "paragraph",
                    "text": "Đoạn văn trước Điều lồng phải được bảo toàn.",
                },
                {
                    "type": "article",
                    "article": "8",
                    "title": "của Nghị định quy định về xử phạt",
                    "text": "Nội dung Điều lồng được xử lý bởi article strategy.",
                    "clauses": [],
                    "points": [],
                    "path": {"section": "II"},
                },
                {
                    "type": "paragraph",
                    "text": "Đoạn văn sau Điều lồng cũng phải được bảo toàn.",
                },
            ],
        }],
    }

    chunks = LegalChunker().chunk_document(
        parsed, {"id": "doc-mixed-section"}
    )
    canonical_text = "\n".join(chunk["text"] for chunk in chunks)

    assert "Mục II XỬ LÝ VI PHẠM HÀNH CHÍNH" in canonical_text
    assert "Đoạn văn trước Điều lồng" in canonical_text
    assert "Nội dung Điều lồng" in canonical_text
    assert "Đoạn văn sau Điều lồng" in canonical_text
    assert {chunk["strategy"] for chunk in chunks} >= {
        "article", "semi_structured"
    }


def test_article_lead_is_indexable_when_article_has_clauses():
    parsed = {
        "doc_id": "doc-123", "structure_type": "article_based", "preamble": "",
        "body": [{"type": "article", "article": "1", "title": "Trách nhiệm",
                  "text": "PHẦN DẪN CỦA ĐIỀU PHẢI ĐƯỢC INDEX.",
                  "clauses": [{"clause": "1", "text": "Cơ quan A thực hiện nhiệm vụ.", "points": []}],
                  "points": [], "path": {}}],
    }
    chunks = LegalChunker().chunk_document(parsed, metadata())
    assert any(c["is_indexable"] and "PHẦN DẪN CỦA ĐIỀU" in c["text"] for c in chunks)


def test_nested_article_in_appendix_is_indexed():
    parsed = {
        "doc_id": "doc-123", "structure_type": "mixed_hierarchical", "preamble": "",
        "body": [{"type": "appendix", "appendix": "I", "title": "PHỤ LỤC THỬ NGHIỆM",
                  "children": [{"type": "chapter", "chapter": "I", "title": "NỘI DUNG",
                                "children": [{"type": "article", "article": "1", "title": "Quy định",
                                              "text": "ĐIỀU LỒNG TRONG PHỤ LỤC.", "clauses": [],
                                              "points": [], "path": {"appendix": "I", "chapter": "I"}}]}]}],
    }
    chunks = LegalChunker().chunk_document(parsed, metadata())
    assert any(c["is_indexable"] and "ĐIỀU LỒNG TRONG PHỤ LỤC" in c["text"] for c in chunks)


def test_all_indexable_chunks_respect_final_retrieval_limit():
    config = ChunkingConfig(max_child_tokens=40, freeform_target_tokens=30,
                            freeform_max_tokens=40, max_indexable_tokens=90,
                            min_tail_tokens=0)
    parsed = {
        "doc_id": "doc-123", "structure_type": "article_based",
        "preamble": "Từ " * 200,
        "body": [{"type": "article", "article": "1", "title": "Điều rất dài",
                  "text": "Nội dung " * 300, "clauses": [], "points": [], "path": {}}],
    }
    chunks = LegalChunker(config).chunk_document(parsed, {"id": "doc-123"})
    assert all(approximate_token_count(c["retrieval_text"]) <= config.max_indexable_tokens
               for c in chunks if c["is_indexable"])


def test_single_punctuation_run_cannot_escape_final_limit():
    config = ChunkingConfig(max_child_tokens=20, freeform_target_tokens=15,
                            freeform_max_tokens=20, max_indexable_tokens=40,
                            min_tail_tokens=0)
    parsed = {"doc_id": "doc-123", "structure_type": "free_form", "preamble": "",
              "body": [{"type": "paragraph", "text": "." * 200}]}
    chunks = LegalChunker(config).chunk_document(parsed, {"id": "doc-123"})
    assert all(c["approx_token_count"] <= config.max_indexable_tokens
               for c in chunks if c["is_indexable"])


def test_semistructured_table_repeats_header_once_per_block():
    parsed = {
        "doc_id": "doc-123",
        "structure_type": "semi_structured",
        "preamble": "",
        "body": [
            {
                "type": "table",
                "headers": ["STT", "Tên"],
                "rows": [["1", "A"], ["2", "B"]],
            }
        ],
    }

    chunks = LegalChunker().chunk_document(parsed, metadata())
    block = next(c for c in chunks if c["chunk_type"] == "structured_block")

    assert block["text"].count("STT | Tên") == 1


def test_final_budget_prefers_point_boundaries_over_generic_fragments():
    config = ChunkingConfig(
        max_child_tokens=700,
        freeform_target_tokens=40,
        freeform_max_tokens=70,
        max_indexable_tokens=40,
        min_tail_tokens=0,
    )
    parsed = {
        "doc_id": "doc-123",
        "structure_type": "article_based",
        "preamble": "",
        "body": [{
            "type": "article",
            "article": "5",
            "title": "Trách nhiệm của cơ quan",
            "text": "",
            "clauses": [{
                "clause": "1",
                "text": "Cơ quan có các trách nhiệm sau:",
                "points": [
                    {"point": "a", "text": "Thực hiện nhiệm vụ thứ nhất theo quy định pháp luật."},
                    {"point": "b", "text": "Thực hiện nhiệm vụ thứ hai theo quy định pháp luật."},
                    {"point": "c", "text": "Báo cáo kết quả thực hiện cho cơ quan có thẩm quyền."},
                ],
            }],
            "points": [],
            "path": {},
        }],
    }

    chunks = LegalChunker(config).chunk_document(parsed, {"id": "doc-123"})
    indexable = [chunk for chunk in chunks if chunk["is_indexable"]]

    assert {chunk["point"] for chunk in indexable} == {"a", "b", "c"}
    assert all(
        "Trách nhiệm của cơ quan" in chunk["retrieval_text"]
        for chunk in indexable
    )
    assert all("::overflow_" not in chunk["chunk_id"] for chunk in chunks)


def test_leaf_article_fragments_repeat_article_heading():
    config = ChunkingConfig(
        max_child_tokens=80,
        freeform_target_tokens=45,
        freeform_max_tokens=80,
        max_indexable_tokens=100,
        min_tail_tokens=0,
    )
    parsed = {
        "doc_id": "doc-123",
        "structure_type": "article_based",
        "preamble": "",
        "body": [{
            "type": "article",
            "article": "1",
            "title": "Phạm vi điều chỉnh",
            "text": "\n".join(
                f"Đoạn {index}. Nội dung quy định độc lập."
                for index in range(1, 30)
            ),
            "clauses": [],
            "points": [],
            "path": {},
        }],
    }

    chunks = LegalChunker(config).chunk_document(parsed, {"id": "doc-123"})
    fragments = [
        chunk for chunk in chunks
        if chunk["chunk_type"] == "article_fragment"
    ]

    assert len(fragments) > 1
    assert all(
        "Điều 1: Phạm vi điều chỉnh" in chunk["retrieval_text"]
        for chunk in fragments
    )
    assert all(chunk["approx_token_count"] <= 100 for chunk in fragments)


def test_long_point_fragments_repeat_legal_context():
    config = ChunkingConfig(
        max_child_tokens=70,
        freeform_target_tokens=40,
        freeform_max_tokens=70,
        max_indexable_tokens=90,
        min_tail_tokens=0,
    )
    parsed = {
        "doc_id": "doc-123",
        "structure_type": "article_based",
        "preamble": "",
        "body": [{
            "type": "article",
            "article": "2",
            "title": "Nghĩa vụ báo cáo",
            "text": "",
            "clauses": [{
                "clause": "1",
                "text": "Cơ quan thực hiện các nghĩa vụ sau:",
                "points": [{
                    "point": "a",
                    "text": " ".join(
                        f"Nội dung báo cáo thứ {index}."
                        for index in range(1, 40)
                    ),
                }],
            }],
            "points": [],
            "path": {},
        }],
    }

    chunks = LegalChunker(config).chunk_document(parsed, {"id": "doc-123"})
    fragments = [
        chunk for chunk in chunks
        if chunk["chunk_type"] == "point_fragment"
    ]

    assert len(fragments) > 1
    assert all("Nghĩa vụ báo cáo" in chunk["retrieval_text"] for chunk in fragments)
    assert all("Cơ quan thực hiện" in chunk["retrieval_text"] for chunk in fragments)
    assert all("Điểm a" in chunk["retrieval_text"] for chunk in fragments)
    assert all(chunk["approx_token_count"] <= 90 for chunk in fragments)


def test_pathological_table_header_is_separated_from_rows():
    config = ChunkingConfig(
        max_child_tokens=90,
        freeform_target_tokens=50,
        freeform_max_tokens=90,
        max_indexable_tokens=120,
        min_tail_tokens=0,
    )
    parsed = {
        "doc_id": "doc-123",
        "structure_type": "appendix_or_form",
        "preamble": "",
        "body": [{
            "type": "appendix",
            "appendix": "I",
            "title": "DANH MỤC",
            "children": [{
                "type": "table",
                "headers": [" ".join(f"Cột {index}." for index in range(100))],
                "rows": [["1", "Nội dung A"], ["2", "Nội dung B"]],
            }],
        }],
    }

    chunks = LegalChunker(config).chunk_document(parsed, {"id": "doc-123"})
    indexable = [chunk for chunk in chunks if chunk["is_indexable"]]

    assert any(chunk["chunk_type"] == "table_header_fragment" for chunk in indexable)
    assert any(
        chunk["chunk_type"] == "table_block" and "Nội dung A" in chunk["text"]
        for chunk in indexable
    )
    assert all(chunk["approx_token_count"] <= 120 for chunk in indexable)


def test_consecutive_headings_are_preserved_as_bounded_context():
    parsed = {
        "doc_id": "doc-123",
        "structure_type": "semi_structured",
        "preamble": "",
        "body": [
            {"type": "heading", "text": "HEADING ONE"},
            {"type": "heading", "text": "HEADING TWO"},
            {"type": "paragraph", "text": "Nội dung của heading hai."},
            {"type": "heading", "text": "TRAILING HEADING"},
        ],
    }

    chunks = LegalChunker().chunk_document(parsed, {"id": "doc-123"})
    indexable_texts = [
        chunk["text"] for chunk in chunks if chunk["is_indexable"]
    ]

    assert any(
        "HEADING ONE" in text
        and "HEADING TWO" in text
        and "Nội dung của heading hai" in text
        for text in indexable_texts
    )
    assert any(text == "TRAILING HEADING" for text in indexable_texts)


def test_semantic_overlap_is_bounded_and_kept_out_of_canonical_text():
    parsed = {
        "doc_id": "doc-123",
        "structure_type": "article_based",
        "preamble": "",
        "body": [{
            "type": "article",
            "article": "1",
            "title": "Phạm vi điều chỉnh",
            "text": "\n".join(
                f"Đoạn {index}. Nội dung pháp lý độc lập cần được bảo toàn đầy đủ."
                for index in range(1, 100)
            ),
            "clauses": [],
            "points": [],
            "path": {},
        }],
    }

    chunks = LegalChunker().chunk_document(parsed, {"id": "doc-123"})
    fragments = [
        chunk for chunk in chunks
        if chunk["chunk_type"] == "article_fragment"
    ]

    assert len(fragments) > 1
    assert fragments[0]["overlap_text"] is None
    assert fragments[1]["overlap_token_count"] >= 30
    assert fragments[1]["overlap_from_chunk_id"] == fragments[0]["chunk_id"]
    assert fragments[1]["overlap_text"] not in fragments[1]["text"]
    assert "[Ngữ cảnh tiếp nối]" in fragments[1]["retrieval_text"]
    assert all(chunk["approx_token_count"] <= 1200 for chunk in fragments)


def test_overlap_never_crosses_article_or_document_boundary():
    def article(number):
        return {
            "type": "article",
            "article": number,
            "title": f"Điều thử nghiệm {number}",
            "text": "\n".join(
                f"Đoạn {index}. Nội dung của Điều {number} cần được giữ độc lập."
                for index in range(1, 90)
            ),
            "clauses": [],
            "points": [],
            "path": {},
        }

    first_document = {
        "doc_id": "doc-123",
        "structure_type": "article_based",
        "preamble": "",
        "body": [article("1"), article("2")],
    }
    second_document = {
        "doc_id": "doc-456",
        "structure_type": "article_based",
        "preamble": "",
        "body": [article("1")],
    }

    chunker = LegalChunker()
    first_chunks = chunker.chunk_document(first_document, {"id": "doc-123"})
    second_chunks = chunker.chunk_document(second_document, {"id": "doc-456"})

    article_two_fragments = [
        chunk for chunk in first_chunks
        if chunk["chunk_type"] == "article_fragment"
        and chunk["article"] == "2"
    ]
    second_doc_fragments = [
        chunk for chunk in second_chunks
        if chunk["chunk_type"] == "article_fragment"
    ]

    assert article_two_fragments[0]["overlap_text"] is None
    assert second_doc_fragments[0]["overlap_text"] is None
    assert all(
        not chunk["overlap_from_chunk_id"]
        or chunk["overlap_from_chunk_id"].startswith(chunk["doc_id"] + "::")
        for chunk in [*first_chunks, *second_chunks]
    )


def test_overlap_fallback_supplies_minimum_for_one_long_sentence():
    parsed = {
        "doc_id": "doc-123",
        "structure_type": "article_based",
        "preamble": "",
        "body": [{
            "type": "article",
            "article": "1",
            "title": "Quy định thử nghiệm",
            "text": "\n".join(
                " ".join(f"từ{line}_{index}" for index in range(160))
                for line in range(8)
            ),
            "clauses": [],
            "points": [],
            "path": {},
        }],
    }

    chunks = LegalChunker().chunk_document(parsed, {"id": "doc-123"})
    fragments = [
        chunk for chunk in chunks
        if chunk["chunk_type"] == "article_fragment"
    ]

    assert len(fragments) > 1
    assert all(
        fragments[index]["overlap_token_count"] >= 30
        for index in range(1, len(fragments))
        if approximate_token_count(fragments[index - 1]["text"]) >= 130
    )


def _article_at_final_token_count(token_count: int, number: str = "1"):
    # With no document metadata the breadcrumb and repeated source heading use
    # exactly eight regex tokens: "Điều 1: T" + "Điều 1. T".
    return {
        "doc_id": "doc-boundary",
        "structure_type": "article_based",
        "preamble": "",
        "body": [{
            "type": "article",
            "article": number,
            "title": "T",
            "text": " ".join(["x"] * (token_count - 8)),
            "clauses": [],
            "points": [],
            "path": {},
        }],
    }


@pytest.mark.parametrize("token_count", [50, 300, 600, 800, 1000, 1199])
def test_complete_articles_up_to_1199_tokens_remain_one_legal_unit(token_count):
    chunks = LegalChunker().chunk_document(
        _article_at_final_token_count(token_count), {"id": "doc-boundary"}
    )

    assert len(chunks) == 1
    assert chunks[0]["chunk_type"] == "article"
    assert chunks[0]["approx_token_count"] == token_count


@pytest.mark.parametrize("token_count", [1201, 2500])
def test_articles_over_hard_limit_are_split(token_count):
    chunks = LegalChunker().chunk_document(
        _article_at_final_token_count(token_count), {"id": "doc-boundary"}
    )
    indexable = [chunk for chunk in chunks if chunk["is_indexable"]]

    assert chunks[0]["chunk_type"] == "article_parent"
    assert len(indexable) >= 2
    assert all(chunk["approx_token_count"] <= 1200 for chunk in indexable)
    assert all("::fragment_" in chunk["chunk_id"] for chunk in indexable)


def test_long_article_packs_adjacent_clauses_into_deterministic_ranges():
    clauses = [
        {
            "clause": str(index),
            "text": " ".join([f"clause{index}"] * 300),
            "points": [],
        }
        for index in range(1, 6)
    ]
    parsed = {
        "doc_id": "doc-groups",
        "structure_type": "article_based",
        "preamble": "",
        "body": [{
            "type": "article", "article": "10", "title": "Trách nhiệm",
            "text": "", "clauses": clauses, "points": [], "path": {},
        }],
    }

    first = LegalChunker().chunk_document(parsed, {"id": "doc-groups"})
    second = LegalChunker().chunk_document(parsed, {"id": "doc-groups"})
    leaves = [chunk for chunk in first if chunk["is_indexable"]]

    assert [chunk["clause"] for chunk in leaves] == ["1-2", "3-4", "5"]
    assert [chunk["chunk_type"] for chunk in leaves] == [
        "clause_group", "clause_group", "clause"
    ]
    assert [chunk["chunk_id"] for chunk in first] == [
        chunk["chunk_id"] for chunk in second
    ]


def test_long_clause_packs_adjacent_points_and_repeats_only_lead_in_retrieval():
    points = [
        {
            "point": chr(96 + index),
            "text": " ".join([f"point{index}"] * 300),
        }
        for index in range(1, 6)
    ]
    parsed = {
        "doc_id": "doc-points",
        "structure_type": "article_based",
        "preamble": "",
        "body": [{
            "type": "article", "article": "5a", "title": "Nghĩa vụ báo cáo",
            "text": "", "points": [], "path": {},
            "clauses": [{
                "clause": "1", "text": "Cơ quan có các nghĩa vụ sau:",
                "points": points,
            }],
        }],
    }

    chunks = LegalChunker().chunk_document(parsed, {"id": "doc-points"})
    leaves = [chunk for chunk in chunks if chunk["is_indexable"]]

    assert [chunk["point"] for chunk in leaves] == ["a-b", "c-d", "e"]
    assert all("Điều 5a: Nghĩa vụ báo cáo" in chunk["retrieval_text"] for chunk in leaves)
    assert all("Cơ quan có các nghĩa vụ sau:" in chunk["retrieval_text"] for chunk in leaves)
    assert all(chunk["approx_token_count"] <= 1200 for chunk in leaves)


def test_numbering_citations_amendments_and_ocr_newlines_are_not_rewritten():
    legal_text = (
        "Sửa đổi Điều 5b theo Điều 5 của Luật này.\n"
        "Chủ thể  thực hiện nghĩa vụ khi có đủ điều kiện;\n"
        "không được tách mức phạt 2.000.000 đồng khỏi hành vi."
    )
    parsed = {
        "doc_id": "doc-preserve",
        "structure_type": "article_based",
        "preamble": "",
        "body": [{
            "type": "article", "article": "5b", "title": "Sửa đổi, bổ sung",
            "text": legal_text, "clauses": [], "points": [], "path": {},
        }],
    }

    chunks = LegalChunker().chunk_document(parsed, {"id": "doc-preserve"})

    assert chunks[0]["article"] == "5b"
    assert legal_text in chunks[0]["text"]
    assert "theo Điều 5" in chunks[0]["retrieval_text"]
    assert "2.000.000 đồng" in chunks[0]["retrieval_text"]
