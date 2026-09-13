from lawchat.evaluation import HistoricalGateObservation, evaluate_historical_gate


def test_historical_gate_passes_only_with_exact_versions_and_refusals():
    report = evaluate_historical_gate(
        (
            HistoricalGateObservation(
                "historical-ok", "historical", "historical", "v1", ("v1",)
            ),
            HistoricalGateObservation(
                "current-ok", "current", "current", None
            ),
            HistoricalGateObservation(
                "unavailable", "historical", None, None, (), False, "REFUSED"
            ),
        )
    )
    assert report.passed


def test_historical_gate_counts_each_failure_class():
    report = evaluate_historical_gate(
        (
            HistoricalGateObservation("a", "historical", "current", "v1", ("v2",)),
            HistoricalGateObservation("b", "current", "historical", None),
            HistoricalGateObservation("c", "historical", None, None, (), False, "VERIFIED"),
        )
    )
    assert not report.passed
    assert report.current_as_historical == 1
    assert report.historical_as_current == 1
    assert report.incorrect_version_citations == 1
    assert report.unavailable_not_refused == 1
