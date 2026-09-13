from frontend.components import answer_for_display, deduplicate_citations


def test_deduplicate_citations_keeps_first_source_for_document_number():
    citations = [
        {
            "document_id": "140152",
            "document_number": "100/2019/NĐ-CP",
            "article": "47",
        },
        {
            "document_id": "140152",
            "document_number": " 100/2019/nđ–cp ",
            "article": "5",
        },
        {
            "document_id": "other",
            "document_number": "168/2024/NĐ-CP",
            "article": "1",
        },
    ]

    result = deduplicate_citations(citations)

    assert result == [citations[0], citations[2]]


def test_deduplicate_citations_keeps_sources_without_document_number():
    citations = [
        {"document_id": "first", "document_number": None},
        {"document_id": "second", "document_number": ""},
    ]

    assert deduplicate_citations(citations) == citations


def test_answer_for_display_removes_trailing_evidence_markers():
    answer = (
        "Câu trả lời thứ nhất. [E1]\n\n"
        "Câu trả lời thứ hai. [E2] [G1]\n"
        "Nội dung có ký hiệu [E3] ở giữa câu thì được giữ nguyên."
    )

    assert answer_for_display(answer) == (
        "Câu trả lời thứ nhất.\n\n"
        "Câu trả lời thứ hai.\n"
        "Nội dung có ký hiệu [E3] ở giữa câu thì được giữ nguyên."
    )


def test_answer_for_display_does_not_change_regular_bracketed_text():
    answer = "Áp dụng theo khoản [1] và Phụ lục [A]."

    assert answer_for_display(answer) == answer
