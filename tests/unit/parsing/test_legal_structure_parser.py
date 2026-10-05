from parsing.legal_structure_parser import LegalStructureParser


def parser() -> LegalStructureParser:
    return LegalStructureParser()


def _find_first(node_list, node_type):
    for node in node_list:
        if node["type"] == node_type:
            return node

        child = _find_first(
            node.get("children", []),
            node_type,
        )

        if child is not None:
            return child

    return None


def test_article_clause_point_structure():
    text = """Điều 12. Điều kiện kinh doanh

1. Doanh nghiệp phải đáp ứng điều kiện.

2. Doanh nghiệp có trách nhiệm:

a) Thực hiện nghĩa vụ thứ nhất.

b) Thực hiện nghĩa vụ thứ hai.
"""

    result = parser().parse_document(
        text,
        doc_id="doc-1",
    )

    assert result["structure_type"] == "article_based"
    assert result["stats"]["articles"] == 1
    assert result["stats"]["clauses"] == 2
    assert result["stats"]["points"] == 2

    article = result["body"][0]

    assert article["article"] == "12"
    assert article["title"] == "Điều kiện kinh doanh"

    assert article["clauses"][0] == {
        "clause": "1",
        "text": "Doanh nghiệp phải đáp ứng điều kiện.",
        "points": [],
        "line_start": 3,
    }

    clause_2 = article["clauses"][1]

    assert clause_2["clause"] == "2"
    assert clause_2["text"] == "Doanh nghiệp có trách nhiệm:"
    assert clause_2["points"][0]["point"] == "a"
    assert clause_2["points"][1]["point"] == "b"


def test_full_hierarchy():
    text = """PHẦN I
QUY ĐỊNH CHUNG

CHƯƠNG I
NHỮNG QUY ĐỊNH CHUNG

MỤC 1
PHẠM VI VÀ ĐỐI TƯỢNG

Điều 1. Phạm vi điều chỉnh
1. Văn bản này quy định...
a) Nội dung A
b) Nội dung B
"""

    result = parser().parse_document(text)

    assert (
        result["structure_type"]
        == "hierarchical_article_based"
    )

    part = result["body"][0]
    assert part["type"] == "part"
    assert part["part"] == "I"
    assert part["title"] == "QUY ĐỊNH CHUNG"

    chapter = part["children"][0]
    assert chapter["type"] == "chapter"
    assert chapter["chapter"] == "I"

    section = chapter["children"][0]
    assert section["type"] == "section"
    assert section["section"] == "1"

    article = section["children"][0]
    assert article["type"] == "article"
    assert article["article"] == "1"

    assert article["path"] == {
        "part": "I",
        "chapter": "I",
        "section": "1",
    }


def test_split_article_title_is_inferred():
    text = """Điều 5
Điều kiện cấp giấy phép

1. Tổ chức phải đáp ứng điều kiện.
"""

    result = parser().parse_document(text)

    article = result["body"][0]

    assert article["title"] == "Điều kiện cấp giấy phép"
    assert article["title_inferred"] is True
    assert article["clauses"][0]["clause"] == "1"


def test_article_without_clauses_is_preserved():
    text = """Điều 1. Phạm vi điều chỉnh
Văn bản này quy định về phạm vi, đối tượng và trách nhiệm thực hiện.
"""

    result = parser().parse_document(text)

    article = result["body"][0]

    assert article["article"] == "1"
    assert article["clauses"] == []
    assert "Văn bản này quy định" in article["text"]


def test_points_directly_under_article_are_supported():
    text = """Điều 3. Hồ sơ gồm:
a) Đơn đề nghị;
b) Giấy tờ chứng minh.
"""

    result = parser().parse_document(text)

    article = result["body"][0]

    assert article["clauses"] == []
    assert len(article["points"]) == 2
    assert article["points"][0]["point"] == "a"


def test_free_form_document_is_not_forced_into_articles():
    text = """CÔNG VĂN

Kính gửi: Sở Tư pháp

Thực hiện Công văn số 01/ABC, Bộ có ý kiến như sau:

1. Nội dung thứ nhất

2. Nội dung thứ hai

Trân trọng.
"""

    result = parser().parse_document(text)

    assert result["stats"]["articles"] == 0
    assert result["structure_type"] == "semi_structured"

    types = [
        node["type"]
        for node in result["body"]
    ]

    assert "heading" in types
    assert "list_item" in types


def test_numbered_list_outside_article_is_not_clause():
    text = """THÔNG BÁO

1. Nội dung thông báo thứ nhất

2. Nội dung thông báo thứ hai
"""

    result = parser().parse_document(text)

    assert result["stats"]["articles"] == 0
    assert result["stats"]["clauses"] == 0
    assert result["stats"]["list_items"] == 2


def test_appendix_without_articles():
    text = """PHỤ LỤC I
DANH MỤC THỦ TỤC HÀNH CHÍNH

1. Thủ tục A
2. Thủ tục B
"""

    result = parser().parse_document(text)

    assert result["structure_type"] == "appendix_or_form"
    assert result["stats"]["appendices"] == 1
    assert result["stats"]["articles"] == 0

    appendix = result["body"][0]

    assert appendix["type"] == "appendix"
    assert appendix["appendix"].upper() == "PHỤ LỤC I"


def test_mixed_document_with_appendix():
    text = """Điều 1. Ban hành danh mục
Ban hành kèm theo văn bản này danh mục.

PHỤ LỤC I
DANH MỤC

1. Mục A
"""

    result = parser().parse_document(text)

    assert (
        result["structure_type"]
        == "mixed_article_appendix"
    )
    assert result["stats"]["articles"] == 1
    assert result["stats"]["appendices"] == 1


def test_inserted_article_number():
    text = """Điều 12a. Quy định bổ sung
1. Nội dung bổ sung.
"""

    result = parser().parse_document(text)

    article = result["body"][0]

    assert article["article"].casefold() == "12a"
    assert article["clauses"][0]["clause"] == "1"


def test_continuation_text_stays_with_point():
    text = """Điều 7. Nghĩa vụ
1. Chủ thể có các nghĩa vụ sau:
a) Nghĩa vụ thứ nhất;
Nội dung giải thích tiếp tục của điểm a.
b) Nghĩa vụ thứ hai.
"""

    result = parser().parse_document(text)

    article = result["body"][0]
    point_a = article["clauses"][0]["points"][0]

    assert "Nội dung giải thích" in point_a["text"]


def test_empty_document():
    result = parser().parse_document(
        "",
        doc_id="empty",
    )

    assert result["structure_type"] == "empty"
    assert result["body"] == []
    assert result["stats"]["articles"] == 0


def test_roman_heading_is_recognized_before_articles():
    text = """I. QUY ĐỊNH CHUNG
Nội dung mở đầu.

Điều 1. Phạm vi
Văn bản này quy định phạm vi.
"""
    result = parser().parse_document(text)
    assert result["preamble"] == ""
    assert result["stats"]["sections"] == 1


def test_serialized_html_table_becomes_table_node():
    text = """PHỤ LỤC I
DANH MỤC
[[LAWCHAT_TABLE_START]]
[[LAWCHAT_TABLE_HEADERS]] ["STT", "Tên văn bản"]
[[LAWCHAT_TABLE_ROW]] ["1", "Văn bản A"]
[[LAWCHAT_TABLE_END]]
"""
    result = parser().parse_document(text)
    appendix = result["body"][0]
    table = next(node for node in appendix["children"] if node["type"] == "table")
    assert table["headers"] == ["STT", "Tên văn bản"]
    assert table["rows"] == [["1", "Văn bản A"]]
    assert result["stats"]["tables"] == 1


def test_signature_closes_last_article():
    text = """Điều 1. Thi hành
Văn bản này có hiệu lực.

KT. CHỦ TỊCH
PHÓ CHỦ TỊCH
Nguyễn Văn A

DANH MỤC VĂN BẢN KÈM THEO
STT
Tên văn bản
"""
    result = parser().parse_document(text)
    article = next(node for node in result["body"] if node["type"] == "article")
    assert "KT. CHỦ TỊCH" not in article["text"]
    assert any(node["type"] == "appendix" for node in result["body"])
