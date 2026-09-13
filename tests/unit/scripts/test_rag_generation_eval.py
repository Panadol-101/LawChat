from scripts.evaluate_rag import _stratified_sample


def test_stratified_generation_sample_round_robins_categories():
    cases = [
        {"case_id": "a1", "category": "a"},
        {"case_id": "a2", "category": "a"},
        {"case_id": "b1", "category": "b"},
        {"case_id": "b2", "category": "b"},
    ]

    selected = _stratified_sample(cases, 3)

    assert [item["case_id"] for item in selected] == ["a1", "b1", "a2"]
