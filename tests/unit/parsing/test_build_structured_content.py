from scripts.build_structured_content import assess_parse_quality


def test_quality_flags_quarantine_pathological_article():
    parsed = {
        "preamble": "",
        "body": [
            {
                "type": "article",
                "text": "x" * 60_000,
                "children": [],
            }
        ],
        "warnings": [],
    }

    flags = assess_parse_quality(parsed, 60_000)

    assert "severe:oversized_article_text" in flags
    assert "severe:single_node_dominates_document" in flags
