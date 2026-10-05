from preprocessing.html_cleaner import clean_html


def test_cleaner_uses_body_and_drops_document_title():
    html = """
    <html>
      <head><title>Document Content</title></head>
      <body><p>Điều 1. Nội dung</p></body>
    </html>
    """

    assert clean_html(html) == "Điều 1. Nội dung"


def test_cleaner_preserves_table_rows_and_cells():
    html = """
    <table>
      <tr><th>STT</th><th>Tên văn bản</th></tr>
      <tr><td>1</td><td>Văn bản A</td></tr>
      <tr><td>2</td><td>Văn bản B</td></tr>
    </table>
    """

    cleaned = clean_html(html)

    assert '[[LAWCHAT_TABLE_HEADERS]] ["STT", "Tên văn bản"]' in cleaned
    assert '[[LAWCHAT_TABLE_ROW]] ["1", "Văn bản A"]' in cleaned
    assert '[[LAWCHAT_TABLE_ROW]] ["2", "Văn bản B"]' in cleaned


def test_cleaner_collapses_punctuation_only_filler():
    cleaned = clean_html(
        "<p>Họ và tên: ........................................</p>"
        "<p>................................................................</p>"
    )

    assert "Họ và tên: [THÔNG TIN ĐỂ TRỐNG]" in cleaned
    assert "\n[THÔNG TIN ĐỂ TRỐNG]" in cleaned
    assert ".........." not in cleaned


def test_cleaner_keeps_inline_text_together():
    assert clean_html(
        "<p>Điều <strong>1</strong>. Nội dung</p>"
    ) == "Điều 1. Nội dung"


def test_nested_table_is_serialized_without_inflating_outer_cells():
    cleaned = clean_html("""
        <table>
          <tr><td>Thông tin biểu mẫu</td></tr>
          <tr><td>
            <table>
              <tr><th>STT</th><th>Nội dung</th></tr>
              <tr><td>1</td><td>Quy định A</td></tr>
            </table>
          </td></tr>
        </table>
    """)

    assert cleaned.count("[[LAWCHAT_TABLE_START]]") == 2
    assert '[[LAWCHAT_TABLE_ROW]] ["Thông tin biểu mẫu"]' in cleaned
    assert '[[LAWCHAT_TABLE_HEADERS]] ["STT", "Nội dung"]' in cleaned
    assert '[[LAWCHAT_TABLE_ROW]] ["1", "Quy định A"]' in cleaned
