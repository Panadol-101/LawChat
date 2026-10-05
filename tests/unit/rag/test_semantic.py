import asyncio
import json
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from rag import (
    Evidence, PackedContext, GeneratedAnswer, GeneratedClaim, GenerationRequest,
    GroundedRAGService, FakeLegalAnswerGenerator, OpenAICompatibleSettings, IssueResolution,
    OpenAICompatibleLegalAnswerGenerator, InvalidStructuredResponseError,
)
from rag.generator import SupportingQuote
from rag.semantic import (
    ClaimCheck, CoverageCheck, SemanticReview, SemanticVerifier, OpenAICompatibleClaimJudge,
    ReviewRequest, check_supporting_quotes, repair_supporting_quotes,
)
from retrieval import LegalCitation, LegalIssue

CASES = json.loads(Path('tests/fixtures/claim_entailment_v1.json').read_text())['cases']


def example(text='Doanh nghiệp phải báo cáo định kỳ.', claim=None):
    citation = LegalCitation(document_id='synthetic', title='Văn bản giả lập',
        document_number='01/2026/QH15', article='1', clause=None, point=None,
        source_url='https://example.test/synthetic', as_of=date(2026, 9, 6), status='EFFECTIVE')
    evidence = Evidence('E1', 'synthetic', 'c1', text, citation, 'queried_document', 20)
    request = GenerationRequest('Nghĩa vụ theo văn bản giả lập là gì?',
        PackedContext((evidence,), '[E1]\n'+text, 20, 800, ()), citation.as_of)
    answer = GeneratedAnswer('', (GeneratedClaim(claim or text, ('E1',),
        (SupportingQuote('E1', text),)),), (), 'high')
    return request, answer


def review(verdict='SUPPORTED', text='Doanh nghiệp phải báo cáo định kỳ.'):
    return SemanticReview((ClaimCheck(0, verdict, 'Đối chiếu nội dung.', ({'evidence_id': 'E1', 'quote': text},)),))


class FakeJudge:
    def __init__(self, *items):
        self.items = list(items)
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        item = self.items.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


@pytest.mark.parametrize('case', CASES, ids=lambda c: c['case_id'])
def test_enforcement_rejects_negative_verdicts_and_preserves_supported_paraphrases(case):
    req, answer = example(case['evidence'], case['claim'])
    judge = FakeJudge(review(case['expected_verdict'], case['evidence']), review(case['expected_verdict'], case['evidence']))
    result = GroundedRAGService(FakeLegalAnswerGenerator([answer, answer]),
        semantic_verifier=SemanticVerifier(judge), semantic_mode='enforce').answer(req)
    assert (result.status.value == 'VERIFIED') == (case['expected_verdict'] == 'SUPPORTED')
    assert result.attempts == (1 if case['expected_verdict'] == 'SUPPORTED' else 2)
    if case['expected_verdict'] != 'SUPPORTED':
        assert result.answer.claims == ()


@pytest.mark.parametrize('async_mode', [False, True])
def test_shadow_records_contradiction_without_triggering_repair(async_mode):
    req, answer = example(claim='Doanh nghiệp không phải báo cáo định kỳ.')
    generator = FakeLegalAnswerGenerator([answer])
    service = GroundedRAGService(generator, semantic_verifier=SemanticVerifier(FakeJudge(review('CONTRADICTED'))), semantic_mode='shadow')
    result = asyncio.run(service.answer_async(req)) if async_mode else service.answer(req)
    assert result.status.value == 'VERIFIED'
    assert result.semantic_status == 'FLAGGED'
    assert result.attempts == 1
    assert len(generator.requests) == 1


@pytest.mark.parametrize('async_mode', [False, True])
def test_enforce_repairs_once_and_rechecks(async_mode):
    req, good = example()
    bad = replace(good, claims=(replace(good.claims[0], text='Doanh nghiệp không phải báo cáo định kỳ.'),))
    judge = FakeJudge(review('CONTRADICTED'), review())
    generator = FakeLegalAnswerGenerator([bad, good])
    service = GroundedRAGService(generator, semantic_verifier=SemanticVerifier(judge), semantic_mode='enforce')
    result = asyncio.run(service.answer_async(req)) if async_mode else service.answer(req)
    assert result.status.value == 'VERIFIED'
    assert result.semantic_status == 'PASSED'
    assert result.attempts == 2
    assert len(result.semantic_observations) == 2
    assert 'CONTRADICTED' in generator.requests[1].repair_instructions[0]


def test_missing_fabricated_and_wrong_source_quotes_are_flagged_even_if_judge_accepts():
    req, answer = example()
    for quotes in [(), (SupportingQuote('E1', 'Doanh nghiệp không phải báo cáo.'),), (SupportingQuote('E2', answer.claims[0].text),)]:
        bad = replace(answer, claims=(replace(answer.claims[0], supporting_quotes=quotes),))
        result = SemanticVerifier(FakeJudge(review())).review(req, bad)
        assert result.quote_issues
        assert not result.passed


def test_invalid_generator_quote_is_repaired_before_spending_a_judge_call():
    req, answer = example()
    invalid = replace(
        answer,
        claims=(replace(answer.claims[0], supporting_quotes=()),),
    )
    judge = FakeJudge(review())
    verifier = SemanticVerifier(judge)
    result = verifier.review(req, invalid)
    assert result.quote_issues
    assert judge.requests == []


def test_quote_matching_normalizes_whitespace_but_not_negation():
    req, answer = example()
    good = replace(answer, claims=(replace(answer.claims[0], supporting_quotes=(SupportingQuote('E1','Doanh nghiệp\n phải báo cáo định kỳ.'),)),))
    assert check_supporting_quotes(ReviewRequest(req, good)) == ()
    punctuation_only = replace(answer, claims=(replace(
        answer.claims[0],
        supporting_quotes=(SupportingQuote(
            'E1', 'Doanh nghiệp, phải báo cáo định kỳ!'
        ),),
    ),))
    assert check_supporting_quotes(ReviewRequest(req, punctuation_only)) == ()
    changed_negation = replace(answer, claims=(replace(
        answer.claims[0],
        supporting_quotes=(SupportingQuote(
            'E1', 'Doanh nghiệp không phải báo cáo định kỳ.'
        ),),
    ),))
    assert check_supporting_quotes(ReviewRequest(req, changed_negation))


def test_invalid_model_quote_is_replaced_by_relevant_verbatim_evidence_span():
    req, answer = example(
        text=(
            "Quyết định sửa đổi bảng giá tính thuế tài nguyên. "
            "Văn bản đã hết hiệu lực từ ngày được xác định."
        ),
        claim="Quyết định sửa đổi bảng giá tính thuế tài nguyên.",
    )
    answer = replace(answer, claims=(replace(
        answer.claims[0],
        supporting_quotes=(SupportingQuote(
            "E1", "Quyết định có sửa đổi bảng giá thuế tài nguyên."
        ),),
    ),))

    repaired = repair_supporting_quotes(req, answer)

    quote = repaired.claims[0].supporting_quotes[0].quote
    assert quote == "Quyết định sửa đổi bảng giá tính thuế tài nguyên."
    assert quote in req.context.evidence[0].text
    assert check_supporting_quotes(ReviewRequest(req, repaired)) == ()


def test_quote_repair_replaces_wrong_citation_with_best_exact_evidence():
    req, answer = example(
        text="Nội dung không liên quan đến hiệu lực văn bản.",
        claim="21/2019/QĐ-UBND đã bị bãi bỏ bởi 18/2023/QĐ-UBND.",
    )
    graph = replace(
        req.context.evidence[0],
        evidence_id="G1",
        chunk_id="graph:1",
        text="18/2023/QĐ-UBND bãi bỏ 21/2019/QĐ-UBND.",
    )
    req = replace(req, context=replace(
        req.context, evidence=(*req.context.evidence, graph)
    ))
    answer = replace(answer, claims=(replace(
        answer.claims[0],
        evidence_ids=("E1",),
        supporting_quotes=(SupportingQuote("E1", "Đoạn paraphrase không có."),),
    ),))

    repaired = repair_supporting_quotes(req, answer)

    assert repaired.claims[0].evidence_ids == ("G1",)
    assert repaired.claims[0].supporting_quotes[0].quote == graph.text


@pytest.mark.parametrize('mode', ['shadow', 'enforce'])
def test_judge_failure_is_observable_and_never_counts_as_supported(mode):
    req, answer = example()
    service = GroundedRAGService(FakeLegalAnswerGenerator([answer]),
        semantic_verifier=SemanticVerifier(FakeJudge(RuntimeError('secret'))), semantic_mode=mode)
    result = service.answer(req)
    assert result.semantic_status == 'ERROR'
    assert result.attempts == 1
    assert result.status.value == ('REFUSED' if mode == 'enforce' else 'VERIFIED')
    assert 'secret' not in str(result)


@pytest.mark.parametrize('checks', [(), (review().checks[0], review().checks[0]), (replace(review().checks[0], claim_index=1),)])
def test_missing_duplicate_or_out_of_range_judge_results_fail_closed(checks):
    req, answer = example()
    result = SemanticVerifier(FakeJudge(SemanticReview(checks))).review(req, answer)
    assert result.error == 'INCOMPLETE_OR_DUPLICATE_CLAIM_CHECKS'
    assert not result.passed


def test_judge_cannot_cite_uncited_evidence_or_fabricated_quote():
    req, answer = example()
    for key, text in [('E2', answer.claims[0].text), ('E1', 'Nội dung không có trong nguồn')]:
        check = replace(review().checks[0], evidence_quotes=({'evidence_id': key, 'quote': text},))
        result = SemanticVerifier(FakeJudge(SemanticReview((check,)))).review(req, answer)
        assert result.error == 'JUDGE_QUOTE_NOT_IN_CITED_EVIDENCE'


def test_off_and_structural_refusal_skip_judge():
    req, answer = example()
    judge = FakeJudge()
    for mode, current_req in [('off', req), ('enforce', replace(req, context=replace(req.context, evidence=())))]:
        GroundedRAGService(FakeLegalAnswerGenerator([answer]), semantic_verifier=SemanticVerifier(judge), semantic_mode=mode).answer(current_req)
    assert judge.requests == []


def test_only_entire_fenced_json_is_accepted_when_explicitly_enabled():
    req, answer = example()
    settings = OpenAICompatibleSettings('http://test', '', 'gemma', allow_fenced_json=True, response_format='json_object')
    def provider(content):
        return OpenAICompatibleLegalAnswerGenerator(settings, transport=lambda *a: {'choices':[{'message':{'content':content}, 'finish_reason':'stop'}]})
    raw = json.dumps(answer.to_dict())
    assert provider('```json\n'+raw+'\n```').generate(req).claims == answer.claims
    for content in ['Explanation\n```json\n'+raw+'\n```', '```json\n'+raw+'\n```\nextra', raw[:-2], raw+'{}']:
        with pytest.raises(InvalidStructuredResponseError):
            provider(content).generate(req)
    settings = replace(settings, allow_fenced_json=False)
    with pytest.raises(InvalidStructuredResponseError):
        provider('```json\n'+raw+'\n```').generate(req)


def test_judge_receives_only_each_claims_full_cited_evidence():
    req, answer = example()
    extra = replace(req.context.evidence[0], evidence_id='E2', text='UNRELATED SECRET')
    req = replace(req, context=replace(req.context, evidence=(*req.context.evidence, extra)))
    judge = OpenAICompatibleClaimJudge(OpenAICompatibleSettings('http://test', '', 'test'))
    payload = judge._payload(ReviewRequest(req, answer), stream=False)
    assert answer.claims[0].text in payload['messages'][1]['content']
    assert 'UNRELATED SECRET' not in payload['messages'][1]['content']
    assert 'supporting_quotes' not in payload['messages'][1]['content']
    assert 'JSON schema:' not in payload['messages'][1]['content']
    assert '"status": "EFFECTIVE"' in payload['messages'][1]['content']
    assert '"as_of": "2026-09-06"' in payload['messages'][1]['content']


def test_async_judge_cancellation_propagates():
    class CancelledJudge:
        async def generate_async(self, request):
            raise asyncio.CancelledError()
    req, answer = example()
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(SemanticVerifier(CancelledJudge()).review_async(req, answer))


def test_judge_json_contract_rejects_malformed_verdicts_and_extra_fields():
    judge = OpenAICompatibleClaimJudge(OpenAICompatibleSettings('http://test', '', 'test'))
    valid = {'checks': [{'claim_index': 0, 'verdict': 'SUPPORTED', 'reason': 'Có nguồn.'}]}
    from copy import deepcopy
    for change in [lambda d: d.update(extra=True), lambda d: d['checks'][0].update(verdict='MAYBE'),
        lambda d: d['checks'][0].update(claim_index=True), lambda d: d['checks'][0].update(reason=''),
        lambda d: d['checks'][0].update(reason='x' * 201)]:
        invalid = deepcopy(valid)
        change(invalid)
        response = {'choices': [{'message': {'content': json.dumps(invalid)}}]}
        with pytest.raises(InvalidStructuredResponseError):
            judge._answer_from_response(response)


def test_semantic_cache_reuses_only_exact_review_input():
    from rag.semantic import SemanticReviewCache

    class CountingJudge:
        def __init__(self):
            self.calls = 0

        def generate(self, request):
            self.calls += 1
            return review('SUPPORTED')

    req, answer = example()
    judge = CountingJudge()
    verifier = SemanticVerifier(judge, SemanticReviewCache(max_entries=2, ttl_seconds=60))

    first = verifier.review(req, answer)
    second = verifier.review(req, answer)
    changed = verifier.review(replace(req, question=req.question + ' khác'), answer)

    assert not first.cache_hit
    assert second.cache_hit
    assert second.telemetry is None
    assert not changed.cache_hit
    assert judge.calls == 2


def test_candidate_shadow_never_overrides_authoritative_judge():
    req, answer = example()
    answer = replace(answer, claims=(replace(
        answer.claims[0],
        supporting_quotes=(SupportingQuote('E1', req.context.evidence[0].text),),
    ),))
    service = GroundedRAGService(
        FakeLegalAnswerGenerator([answer]),
        semantic_verifier=SemanticVerifier(FakeJudge(review('SUPPORTED'))),
        shadow_semantic_verifier=SemanticVerifier(
            FakeJudge(review('CONTRADICTED'))
        ),
        semantic_mode='enforce',
    )

    result = asyncio.run(service.answer_async(req))

    assert result.status.value == 'VERIFIED'
    assert result.semantic_status == 'PASSED'
    assert [item.mode for item in result.semantic_observations] == [
        'candidate_shadow', 'enforce'
    ]


def test_runtime_uses_selected_gemma_for_generation_and_judge(monkeypatch):
    from rag.runtime import create_generation_service
    monkeypatch.setenv('LAWCHAT_SEMANTIC_MODE','shadow')
    monkeypatch.setenv('LAWCHAT_SEMANTIC_MODEL','gemma4:31b-cloud')
    service = create_generation_service(OpenAICompatibleSettings('http://ollama/v1','','gemma4:31b-cloud',response_format='json_object',allow_fenced_json=True))
    assert service.semantic_mode == 'shadow'
    assert service.semantic_verifier.judge.settings.model == 'gemma4:31b-cloud'
    assert service.semantic_verifier.judge.settings.allow_fenced_json


def test_evaluation_does_not_count_judge_errors_as_successful_detection():
    from scripts.evaluate_rag import summarize_entailment as summarize
    rows=[{'expected_verdict':'CONTRADICTED','actual_verdict':None,'structural_status':'VERIFIED',
        'passed':False,'review':{'error':'TimeoutError','total_seconds':1,'telemetry':None}}]
    metrics=summarize(rows)
    assert metrics['unsafe_claim_detection_rate'] == 0
    assert metrics['review_errors'] == 1
    assert metrics['usage_missing_calls'] == 1


@pytest.mark.parametrize('async_mode', [False, True])
def test_format_repair_and_semantic_repair_share_two_generation_attempts(async_mode):
    req, good = example()
    bad = replace(good, claims=(replace(good.claims[0], text='Doanh nghiệp không phải báo cáo định kỳ.'),))
    generator = FakeLegalAnswerGenerator([InvalidStructuredResponseError('bad JSON'), bad, good])
    service = GroundedRAGService(generator,
        semantic_verifier=SemanticVerifier(FakeJudge(review('CONTRADICTED'))), semantic_mode='enforce')
    result = asyncio.run(service.answer_async(req)) if async_mode else service.answer(req)
    assert result.status.value == 'REFUSED'
    assert result.attempts == 2
    assert len(generator.requests) == 2
    assert result.answer.claims == ()


def test_failed_json_usage_is_preserved_through_successful_retry():
    from rag.generator import GenerationTelemetry
    req, good = example()
    error = InvalidStructuredResponseError('malformed')
    error.telemetry = GenerationTelemetry(1, prompt_tokens=10, completion_tokens=5)
    good = replace(good, telemetry=GenerationTelemetry(2,prompt_tokens=20,completion_tokens=6))
    result = GroundedRAGService(FakeLegalAnswerGenerator([error, good])).answer(req)
    assert result.status.value == 'VERIFIED'
    assert result.attempts == 2
    assert sum(t.prompt_tokens for t in result.telemetry) == 30


def issue_example(*, include_second_resolution=True):
    request, answer = example()
    issues = (
        LegalIssue("I1", "Có nghĩa vụ báo cáo không?", "nghĩa vụ báo cáo"),
        LegalIssue("I2", "Thời hạn là bao lâu?", "thời hạn báo cáo"),
    )
    claim = replace(answer.claims[0], issue_ids=("I1",))
    resolutions = [IssueResolution("I1", "ANSWERED", (0,), "Đã trả lời bằng claim 0.")]
    limitations = ()
    if include_second_resolution:
        resolutions.append(IssueResolution("I2", "INSUFFICIENT_EVIDENCE", (), "Chưa có căn cứ về thời hạn."))
        limitations = ("Chưa có căn cứ về thời hạn.",)
    return (
        replace(request, issues=issues),
        replace(answer, claims=(claim,), issue_resolutions=tuple(resolutions), limitations=limitations),
    )


def coverage_review(*, plan_complete=True, second="DECLARED_UNANSWERED"):
    return SemanticReview(
        checks=review().checks,
        plan_complete=plan_complete,
        plan_reason="Hai vấn đề đã bao phủ câu hỏi." if plan_complete else "Còn thiếu vấn đề tiền lương.",
        coverage_checks=(
            CoverageCheck("I1", "COVERED_BY_CLAIMS", "Claim 0 trả lời trực tiếp.", (0,)),
            CoverageCheck("I2", second, "Đã công bố thiếu căn cứ." if second != "MISSING_OR_PARTIAL" else "Chưa trả lời thời hạn."),
        ),
    )


def test_structural_verifier_requires_exactly_one_resolution_for_every_issue():
    req, answer = issue_example(include_second_resolution=False)
    result = GroundedRAGService(FakeLegalAnswerGenerator([answer, answer])).answer(req)
    assert result.status.value == "REFUSED"
    assert result.attempts == 2
    assert any(item.code.value == "MISSING_ISSUE_RESOLUTION" for item in result.verification.issues)


def test_complete_answer_accepts_answered_and_explicitly_unanswered_issues():
    req, answer = issue_example()
    result = GroundedRAGService(
        FakeLegalAnswerGenerator([answer]),
        semantic_verifier=SemanticVerifier(FakeJudge(coverage_review())),
        semantic_mode="enforce",
    ).answer(req)
    assert result.status.value == "VERIFIED"
    assert result.semantic_status == "PASSED"


def test_service_exposes_unanswered_issue_explanation_as_limitation():
    req, answer = issue_example()
    answer = replace(answer, limitations=())
    result = GroundedRAGService(
        FakeLegalAnswerGenerator([answer]),
        semantic_verifier=SemanticVerifier(FakeJudge(coverage_review())),
        semantic_mode="enforce",
    ).answer(req)
    assert result.status.value == "VERIFIED"
    assert "Chưa có căn cứ về thời hạn." in result.answer.limitations


def test_partial_issue_triggers_one_repair_then_refusal():
    req, answer = issue_example()
    judge = FakeJudge(
        coverage_review(second="MISSING_OR_PARTIAL"),
        coverage_review(second="MISSING_OR_PARTIAL"),
    )
    result = GroundedRAGService(
        FakeLegalAnswerGenerator([answer, answer]),
        semantic_verifier=SemanticVerifier(judge),
        semantic_mode="enforce",
    ).answer(req)
    assert result.status.value == "REFUSED"
    assert result.attempts == 2
    assert any(item.code.value == "ISSUE_NOT_COVERED" for item in result.verification.issues)


def test_incomplete_issue_plan_is_refused_without_useless_generation_repair():
    req, answer = issue_example()
    generator = FakeLegalAnswerGenerator([answer, answer])
    result = GroundedRAGService(
        generator,
        semantic_verifier=SemanticVerifier(FakeJudge(coverage_review(plan_complete=False))),
        semantic_mode="enforce",
    ).answer(req)
    assert result.status.value == "REFUSED"
    assert result.attempts == 1
    assert len(generator.requests) == 1
    assert result.verification.issues[0].code.value == "ISSUE_PLAN_INCOMPLETE"


def test_coverage_review_requires_every_issue_once_and_valid_claim_indexes():
    req, answer = issue_example()
    incomplete = replace(coverage_review(), coverage_checks=coverage_review().coverage_checks[:1])
    checked = SemanticVerifier(FakeJudge(incomplete)).review(req, answer)
    assert checked.error == "INCOMPLETE_OR_DUPLICATE_COVERAGE_CHECKS"

    invalid = replace(
        coverage_review(),
        coverage_checks=(
            CoverageCheck("I1", "COVERED_BY_CLAIMS", "Sai index.", (4,)),
            coverage_review().coverage_checks[1],
        ),
    )
    checked = SemanticVerifier(FakeJudge(invalid)).review(req, answer)
    assert checked.error == "INVALID_COVERAGE_CLAIM_INDEX"
