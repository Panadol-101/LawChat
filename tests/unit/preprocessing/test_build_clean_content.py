from scripts.build_clean_content import classify_clean_status
from lawchat.preprocessing.html_cleaner import clean_html


def test_punctuation_only_html_is_not_classified_ok():
    html = "<p>........................................</p>"
    cleaned = clean_html(html)

    assert classify_clean_status(html, cleaned) == "no_meaningful_text"


def test_script_only_html_has_no_visible_text():
    html = "<script>hidden crawler payload</script>"

    assert classify_clean_status(html, clean_html(html)) == "no_visible_text"


def test_empty_html_table_has_no_visible_content():
    html = "<table><tr><td></td></tr></table>"
    cleaned = clean_html(html)

    assert classify_clean_status(html, cleaned) == "no_visible_text"
