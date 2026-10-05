from datetime import date

import pytest

from retrieval import AmbiguousTemporalQuery, LegalQueryParser


def test_parser_extracts_vietnamese_date_document_and_structure():
    parser = LegalQueryParser(today=lambda: date(2030, 1, 1))

    result = parser.parse(
        "Ngày 01/01/2025, Điều 36 khoản 2 Nghị định 123/2024/NĐ-CP áp dụng thế nào?"
    )

    assert result.as_of == date(2025, 1, 1)
    assert result.document_numbers == ("123/2024/NĐ-CP",)
    assert result.referenced_articles == ("36",)
    assert result.referenced_clauses == ("2",)
    assert result.semantic_query == "áp dụng thế nào"


def test_parser_defaults_to_injected_today_and_accepts_explicit_override():
    parser = LegalQueryParser(today=lambda: date(2030, 1, 1))

    assert parser.parse("Quy định hiện hành").as_of == date(2030, 1, 1)
    assert parser.parse(
        "So sánh ngày 01/01/2020 và 01/01/2025",
        as_of=date(2025, 1, 1),
    ).as_of == date(2025, 1, 1)


def test_parser_rejects_ambiguous_or_invalid_dates():
    parser = LegalQueryParser()

    with pytest.raises(AmbiguousTemporalQuery, match="multiple dates"):
        parser.parse("So sánh 01/01/2020 và 01/01/2025")
    with pytest.raises(ValueError, match="invalid date"):
        parser.parse("Quy định ngày 31/02/2025")


def test_parser_removes_temporal_and_structural_boilerplate_for_retrieval():
    parser = LegalQueryParser()

    parsed = parser.parse(
        "Tại ngày 27/08/2026, quy định đang áp dụng về hợp đồng điện tử là gì?"
    )
    exact = parser.parse(
        "Điều 20 Khoản 3 của văn bản 68/2020/QH14 quy định gì?"
    )

    assert parsed.semantic_query == "hợp đồng điện tử"
    assert exact.semantic_query == "nội dung quy định pháp luật"


def test_parser_can_keep_raw_query_for_ablation():
    parser = LegalQueryParser(clean_semantic_query=False)
    query = "Tại ngày 27/08/2026, quy định đang áp dụng về hợp đồng điện tử là gì?"

    assert parser.parse(query).semantic_query == query


def test_parser_preserves_natural_semantic_and_scenario_queries():
    parser = LegalQueryParser()
    query = "Nếu phát sinh tranh chấp hợp đồng, quyền của các bên được xác định thế nào?"

    assert parser.parse(query).semantic_query == query


def test_parser_detects_outgoing_and_incoming_legal_relationships():
    parser = LegalQueryParser()

    outgoing = parser.parse(
        "Văn bản 07/2025/QĐ-UBND thay thế văn bản nào?"
    )
    incoming = parser.parse(
        "Văn bản nào bãi bỏ 18/2024/QĐ-UBND?"
    )

    assert outgoing.relationship_types == ("REPLACES",)
    assert outgoing.relationship_direction == "outgoing"
    assert incoming.relationship_types == ("REPEALS",)
    assert incoming.relationship_direction == "incoming"


def test_parser_normalizes_spaces_around_document_number_slashes():
    parsed = LegalQueryParser().parse(
        "Văn bản 11 /2016/QĐ-UBND thay thế văn bản nào?"
    )

    assert parsed.document_numbers == ("11/2016/QĐ-UBND",)
    assert parsed.relationship_direction == "outgoing"


def test_parser_supports_document_numbers_without_year_and_cleans_status_lookup():
    parsed = LegalQueryParser().parse(
        "Văn bản 464/QĐ-UBND từng quy định về quyết định này có hiệu lực "
        "kể từ ngày ký hiện còn hiệu lực không?"
    )

    assert parsed.document_numbers == ("464/QĐ-UBND",)
    assert parsed.semantic_query == "quyết định này có hiệu lực kể từ ngày ký"
