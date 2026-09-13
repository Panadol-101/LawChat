from __future__ import annotations

import re
from dataclasses import replace
from datetime import date

import pytest

from lawchat.rag import RAGContextBuilder, TokenBudget
from lawchat.retrieval import (
    LegalCitation,
    LegalGraphEdge,
    LegalGraphDocument,
    RetrievalResponse,
    RetrievedLegalChunk,
    LegalIssue,
)


class WordCounter:
    def count_tokens(self, text: str) -> int:
        return len(re.findall(r"\S+", text))


class CharacterCounter:
    def count_tokens(self, text: str) -> int:
        return len(text)


def _result(
    chunk_id: str,
    text: str,
    *,
    document_id: str = "doc-a",
    document_number: str = "01/2025/QH15",
    article: str = "5",
    clause: str | None = None,
    score: float = 0.5,
    reranker_score: float | None = None,
    parent_text: str | None = None,
    metadata: dict | None = None,
) -> RetrievedLegalChunk:
    citation = LegalCitation(
        document_id=document_id,
        title=f"Văn bản {document_id}",
        document_number=document_number,
        article=article,
        clause=clause,
        point=None,
        source_url=f"https://example.test/{document_id}",
        as_of=date(2026, 8, 31),
        status="EFFECTIVE",
    )
    source_scores = {"dense": score}
    if reranker_score is not None:
        source_scores["reranker"] = reranker_score
    return RetrievedLegalChunk(
        point_id=f"point-{chunk_id}",
        chunk_id=chunk_id,
        score=reranker_score if reranker_score is not None else score,
        text=text,
        context_text=text,
        chunk_type="clause" if clause else "article",
        citation=citation,
        source_scores=source_scores,
        metadata=metadata or {},
        parent_text=parent_text,
    )


def _response(
    results: tuple[RetrievedLegalChunk, ...],
    *,
    query: str = "Điều 5 văn bản 01/2025/QH15 quy định gì?",
    seed_documents: tuple[LegalGraphDocument, ...] = (),
    related_documents: tuple[LegalGraphDocument, ...] = (),
    graph_edges: tuple[LegalGraphEdge, ...] = (),
) -> RetrievalResponse:
    return RetrievalResponse(
        query=query,
        semantic_query=query,
        as_of=date(2026, 8, 31),
        results=results,
        searched_candidates=len(results),
        rejected_candidates=0,
        seed_documents=seed_documents,
        related_documents=related_documents,
        graph_edges=graph_edges,
    )


def test_only_official_graph_edges_are_rendered_as_graph_evidence():
    official = LegalGraphEdge(
        "new", "old", "REPLACES", "02/2026/QH15", "01/2020/QH14",
        direction="incoming", source_title="Văn bản mới", target_title="Văn bản cũ",
        source_status="EFFECTIVE", source_url="https://official.test/new",
        citation_text="Văn bản mới thay thế văn bản cũ", is_official=True,
    )
    unverified = LegalGraphEdge(
        "x", "y", "AMENDS", source_url="https://aggregate.test/x"
    )
    packed = RAGContextBuilder(WordCounter()).build(
        _response((), graph_edges=(official, unverified))
    )
    assert [item.evidence_id for item in packed.evidence] == ["G1"]
    assert "REPLACED_BY" not in packed.rendered_context
    assert "Văn bản mới thay thế văn bản cũ" in packed.rendered_context
    assert "official" in packed.rendered_context
    assert "Nội dung:\nVăn bản mới thay thế văn bản cũ" in packed.rendered_context
    assert packed.evidence[0].token_count > 0


def test_relationship_query_reserves_graph_evidence_before_text_evidence():
    edge = LegalGraphEdge(
        "new", "old", "REPEALS", "02/2026/QH15", "01/2020/QH14",
        direction="incoming", source_title="Văn bản mới", target_title="Văn bản cũ",
        source_status="EFFECTIVE", source_url="https://official.test/new",
        citation_text="02/2026/QH15 bãi bỏ 01/2020/QH14", is_official=True,
    )
    packed = RAGContextBuilder(
        WordCounter(), max_evidence_total=1, max_graph_evidence=1
    ).build(_response(
        (_result("text", "Nội dung phụ."),),
        query="Văn bản 02/2026/QH15 bãi bỏ văn bản nào?",
        graph_edges=(edge,),
    ))

    assert [item.evidence_id for item in packed.evidence] == ["G1"]
    assert packed.evidence[0].text in packed.rendered_context


def test_context_is_deterministic_grouped_and_prioritizes_exact_role_and_reranker():
    unrelated = _result(
        "other::article_6",
        "Nội dung Điều 6 không được hỏi.",
        document_id="doc-other",
        document_number="02/2025/QH15",
        article="6",
        score=0.99,
    )
    exact_second = _result(
        "doc-a::article_5::clause_2",
        "Khoản 2 quy định nghĩa vụ báo cáo.",
        clause="2",
        reranker_score=0.8,
    )
    exact_first = _result(
        "doc-a::article_5::clause_1",
        "Khoản 1 quy định phạm vi áp dụng.",
        clause="1",
        reranker_score=0.9,
    )
    current_authority = _result(
        "doc-current::article_7",
        "Quy định hiện hành liên quan.",
        document_id="doc-current",
        document_number="03/2026/QH15",
        article="7",
        score=0.7,
    )
    seed = LegalGraphDocument(
        "doc-a", "01/2025/QH15", "Văn bản A", "EFFECTIVE", "queried_document"
    )
    related = LegalGraphDocument(
        "doc-current",
        "03/2026/QH15",
        "Văn bản hiện hành",
        "EFFECTIVE",
        "current_authority",
    )
    response = _response(
        (unrelated, exact_second, current_authority, exact_first),
        seed_documents=(seed,),
        related_documents=(related,),
    )
    builder = RAGContextBuilder(WordCounter())

    first = builder.build(response)
    second = builder.build(response)

    assert first == second
    assert [item.evidence_id for item in first.evidence] == ["E1", "E2", "E3", "E4"]
    assert [item.chunk_id for item in first.evidence] == [
        exact_first.chunk_id,
        exact_second.chunk_id,
        current_authority.chunk_id,
        unrelated.chunk_id,
    ]
    assert [item.role for item in first.evidence] == [
        "queried_document",
        "queried_document",
        "current_authority",
        "retrieved_document",
    ]
    assert first.rendered_context.count("## 01/2025/QH15 — Điều 5") == 1
    assert all(
        evidence.citation.document_id
        == next(
            result.citation.document_id
            for result in response.results
            if result.chunk_id == evidence.chunk_id
        )
        for evidence in first.evidence
    )


def test_exact_provision_priority_requires_matching_document_number():
    wrong_document = _result(
        "wrong::article_5",
        "Điều 5 của văn bản khác.",
        document_id="wrong",
        document_number="99/2025/QH15",
        article="5",
        reranker_score=1.0,
    )
    expected = _result(
        "doc-a::article_5",
        "Điều 5 của văn bản được hỏi.",
        article="5",
        reranker_score=0.1,
    )

    packed = RAGContextBuilder(WordCounter()).build(
        _response((wrong_document, expected))
    )

    assert packed.evidence[0].chunk_id == expected.chunk_id


def test_context_deduplicates_equal_contained_and_declared_overlap_text():
    preferred = _result(
        "doc-a::article_5::clause_1",
        "Người sử dụng lao động phải báo cáo định kỳ cho cơ quan có thẩm quyền.",
        clause="1",
        reranker_score=0.9,
    )
    duplicate = _result(
        "doc-a::article_5::fragment_2",
        "Người sử dụng lao động phải báo cáo định kỳ cho cơ quan có thẩm quyền.",
        reranker_score=0.1,
    )
    with_overlap = _result(
        "doc-a::article_5::clause_2",
        "Câu lặp từ chunk trước. Nghĩa vụ mới của cơ quan quản lý.",
        clause="2",
        metadata={"overlap_text": "Câu lặp từ chunk trước."},
    )

    packed = RAGContextBuilder(WordCounter()).build(
        _response((duplicate, with_overlap, preferred))
    )

    assert [item.chunk_id for item in packed.evidence] == [
        preferred.chunk_id,
        with_overlap.chunk_id,
    ]
    assert duplicate.chunk_id in packed.dropped_chunk_ids
    assert "Câu lặp từ chunk trước" not in packed.evidence[1].text


def test_context_never_exceeds_budget_and_drops_whole_leaf_instead_of_truncating():
    oversized = _result("oversized", "A" * 400, reranker_score=1.0)
    small = _result("small", "Nội dung ngắn.", article="6", score=0.1)
    budget = TokenBudget(
        model_context_window=260,
        system_prompt_tokens=0,
        answer_reserve_tokens=0,
    )

    packed = RAGContextBuilder(
        CharacterCounter(),
        budget=budget,
        max_ancestor_tokens=0,
    ).build(_response((oversized, small)))

    assert packed.used_tokens <= packed.token_budget == 260
    assert [item.chunk_id for item in packed.evidence] == ["small"]
    assert packed.evidence[0].text == "Nội dung ngắn."
    assert "oversized" in packed.dropped_chunk_ids


def test_ancestor_excerpt_keeps_complete_sentences_within_its_own_budget():
    result = _result(
        "with-parent",
        "Khoản 1. Nội dung cần trích dẫn.",
        parent_text="Một hai ba. Bốn năm sáu.",
    )

    packed = RAGContextBuilder(
        WordCounter(),
        max_ancestor_tokens=5,
    ).build(_response((result,)))

    assert "Một hai ba." in packed.evidence[0].text
    assert "Bốn năm sáu." not in packed.evidence[0].text
    assert packed.evidence[0].text.endswith(result.text)


def test_ancestor_does_not_repeat_leaf_content():
    leaf = "Khoản 1. Nội dung cần trích dẫn."
    result = _result(
        "parent-contains-leaf",
        leaf,
        parent_text=f"Điều 5. Quy định chung. {leaf}",
    )

    packed = RAGContextBuilder(WordCounter()).build(_response((result,)))

    assert packed.evidence[0].text.count(leaf) == 1
    assert "Điều 5. Quy định chung." in packed.evidence[0].text


def test_default_budget_reserves_prompt_and_answer_tokens():
    assert TokenBudget().evidence_tokens == 11_000
    with pytest.raises(ValueError, match="evidence token budget"):
        TokenBudget(
            model_context_window=4_000,
            system_prompt_tokens=2_000,
            answer_reserve_tokens=2_000,
        )


def test_rendered_context_translates_status_for_the_llm():
    packed = RAGContextBuilder(WordCounter()).build(
        _response((_result("partial", "Nội dung.", article="5"),))
    )

    assert "Đang có hiệu lực (EFFECTIVE)" in packed.rendered_context


def test_issue_fair_packing_serves_each_issue_before_secondary_evidence():
    results = (
        _result("i1-a", "Nguồn chính vấn đề một.", metadata={
            "issue_ids": ["I1"], "issue_scores": {"I1": 0.99},
        }),
        _result("i1-b", "Nguồn phụ vấn đề một.", article="6", metadata={
            "issue_ids": ["I1"], "issue_scores": {"I1": 0.98},
        }),
        _result("i2-a", "Nguồn chính vấn đề hai.", article="7", metadata={
            "issue_ids": ["I2"], "issue_scores": {"I2": 0.50},
        }),
    )
    response = _response(results)
    response = replace(
        response,
        legal_issues=(
            LegalIssue("I1", "Một?", "một"),
            LegalIssue("I2", "Hai?", "hai"),
        ),
    )
    packed = RAGContextBuilder(
        WordCounter(), max_evidence_total=2, max_evidence_per_issue=2
    ).build(response)

    assert [item.chunk_id for item in packed.evidence] == ["i1-a", "i2-a"]
    assert "i1-b" in packed.dropped_chunk_ids


def test_deduplicated_evidence_preserves_all_issue_assignments():
    first = _result("same-i1", "Cùng một căn cứ pháp lý đủ dài để được loại trùng.", metadata={
        "issue_ids": ["I1"], "issue_scores": {"I1": 0.9},
    })
    second = _result("same-i2", first.text, metadata={
        "issue_ids": ["I2"], "issue_scores": {"I2": 0.8},
    })
    response = replace(
        _response((first, second)),
        legal_issues=(
            LegalIssue("I1", "Một?", "một"),
            LegalIssue("I2", "Hai?", "hai"),
        ),
    )

    packed = RAGContextBuilder(WordCounter()).build(response)

    assert len(packed.evidence) == 1
    assert packed.evidence[0].issue_ids == ("I1", "I2")


def test_issue_extras_prefer_a_distinct_legal_basis():
    same_basis = _result("same-basis", "Chi tiết phụ cùng Điều 5.", metadata={
        "issue_ids": ["I1"], "issue_scores": {"I1": 0.8},
    })
    primary = _result("primary", "Căn cứ chính tại Điều 5.", reranker_score=0.9, metadata={
        "issue_ids": ["I1"], "issue_scores": {"I1": 0.9},
    })
    distinct = _result("distinct", "Căn cứ bổ sung tại Điều 6.", article="6", metadata={
        "issue_ids": ["I1"], "issue_scores": {"I1": 0.7},
    })
    response = replace(
        _response((primary, same_basis, distinct)),
        legal_issues=(LegalIssue("I1", "Một?", "một"),),
    )

    packed = RAGContextBuilder(
        WordCounter(), max_evidence_total=2, max_evidence_per_issue=2
    ).build(response)

    assert [item.chunk_id for item in packed.evidence] == ["primary", "distinct"]
    assert "same-basis" in packed.dropped_chunk_ids


def test_context_caps_total_evidence_and_prefers_effective_source():
    expired = _result("expired", "Nguồn đã hết hiệu lực.", reranker_score=0.9)
    expired = replace(expired, citation=replace(expired.citation, status="EXPIRED"))
    effective = _result("effective", "Nguồn đang có hiệu lực.", article="6", reranker_score=0.8)
    packed = RAGContextBuilder(WordCounter(), max_evidence_total=1).build(
        _response((expired, effective))
    )

    assert [item.chunk_id for item in packed.evidence] == ["effective"]
    assert "expired" in packed.dropped_chunk_ids
