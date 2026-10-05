from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from datetime import date
from pathlib import Path

from rag import (
    Evidence,
    FakeLegalAnswerGenerator,
    GeneratedAnswer,
    GeneratedClaim,
    GenerationRequest,
    GroundedRAGService,
    GroundingVerifier,
    OpenAICompatibleLegalAnswerGenerator,
    OpenAICompatibleSettings,
    PackedContext,
    VerificationCode,
    VerificationStatus,
)
from rag.prompts import OUTPUT_CONTRACT, SYSTEM_PROMPT, build_user_prompt
from retrieval import LegalCitation


AS_OF = date(2026, 8, 31)


def _evidence(
    evidence_id: str = "E1",
    *,
    document_number: str = "01/2025/QH15",
    article: str = "5",
    clause: str | None = "1",
    point: str | None = None,
    status: str = "EFFECTIVE",
    as_of: date = AS_OF,
) -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        document_id=f"doc-{evidence_id}",
        chunk_id=f"chunk-{evidence_id}",
        text="Người sử dụng lao động phải báo cáo định kỳ.",
        citation=LegalCitation(
            document_id=f"doc-{evidence_id}",
            title="Văn bản thử nghiệm",
            document_number=document_number,
            article=article,
            clause=clause,
            point=point,
            source_url=f"https://example.test/{evidence_id}",
            as_of=as_of,
            status=status,
        ),
        role="queried_document",
        token_count=10,
    )


def _request(
    *evidence: Evidence,
    temporal_intent: str = "current_law",
    historical_content_available: bool = True,
) -> GenerationRequest:
    items = evidence or (_evidence(),)
    return GenerationRequest(
        question="Nghĩa vụ báo cáo được quy định thế nào?",
        context=PackedContext(
            evidence=tuple(items),
            rendered_context="\n\n".join(
                f"[{item.evidence_id}]\nNội dung: {item.text}" for item in items
            ),
            used_tokens=20,
            token_budget=100,
            dropped_chunk_ids=(),
        ),
        as_of=AS_OF,
        temporal_intent=temporal_intent,
        historical_content_available=historical_content_available,
    )


def _answer(
    text: str = "Khoản 1 Điều 5 của 01/2025/QH15 quy định nghĩa vụ báo cáo.",
    evidence_ids: tuple[str, ...] = ("E1",),
) -> GeneratedAnswer:
    return GeneratedAnswer(
        answer=text,
        claims=(GeneratedClaim(text, evidence_ids),),
        limitations=(),
        confidence="high",
    )


def _codes(result) -> set[VerificationCode]:
    return {issue.code for issue in result.issues}


def test_verifier_accepts_fully_grounded_structured_answer():
    result = GroundingVerifier().verify(_request(), _answer())

    assert result.status is VerificationStatus.VERIFIED
    assert result.issues == ()


def test_graph_related_document_numbers_are_valid_answer_level_citations():
    graph = replace(
        _evidence("G1", document_number="74/2025/QH15", article="", clause=None),
        text="74/2025/QH15 thay thế 38/2013/QH13.",
        role="graph_relationship",
        related_document_numbers=("74/2025/QH15", "38/2013/QH13"),
    )
    text = "74/2025/QH15 thay thế 38/2013/QH13."
    answer = GeneratedAnswer(
        answer=text,
        claims=(GeneratedClaim(text, ("G1",)),),
        limitations=(),
        confidence="high",
    )

    result = GroundingVerifier().verify(_request(graph), answer)

    assert VerificationCode.DOCUMENT_NUMBER_NOT_IN_EVIDENCE not in _codes(result)
    assert VerificationCode.DOCUMENT_NUMBER_CITATION_MISMATCH not in _codes(result)


def test_document_number_cross_reference_is_allowed_when_verbatim_in_cited_text():
    evidence = replace(
        _evidence(document_number="21/2019/QĐ-UBND", article="2", clause=None),
        text="21/2019/QĐ-UBND sửa đổi Quyết định 30/2017/QĐ-UBND.",
    )
    text = "21/2019/QĐ-UBND sửa đổi 30/2017/QĐ-UBND."

    result = GroundingVerifier().verify(
        _request(evidence), _answer(text)
    )

    assert VerificationCode.DOCUMENT_NUMBER_NOT_IN_EVIDENCE not in _codes(result)
    assert VerificationCode.DOCUMENT_NUMBER_CITATION_MISMATCH not in _codes(result)


def test_service_adds_inactive_status_disclosure_from_cited_metadata():
    expired = _evidence(status="EXPIRED")
    answer = _answer("01/2025/QH15 quy định nghĩa vụ báo cáo.")

    result = GroundedRAGService(FakeLegalAnswerGenerator([answer])).answer(
        _request(expired)
    )

    assert result.status is VerificationStatus.VERIFIED
    assert any("đã hết hiệu lực" in item for item in result.answer.limitations)


def test_all_curated_case_specific_prediction_prompts_fail_preflight():
    fixture = json.loads(Path(
        "tests/fixtures/legal_retrieval_benchmark_bge_m3_v1.json"
    ).read_text(encoding="utf-8"))
    questions = [
        item["query"] for item in fixture["cases"]
        if item["category"] == "insufficient_evidence"
    ]

    assert len(questions) == 10
    for question in questions:
        request = replace(_request(), question=question)
        result = GroundingVerifier().verify(request, _answer())
        assert result.status is VerificationStatus.REFUSED, question
        assert VerificationCode.CASE_SPECIFIC_PREDICTION_UNSUPPORTED in _codes(result)


def test_general_legal_deadline_question_is_not_prediction_blocked():
    request = replace(
        _request(),
        question="Nghị định quy định thời hạn giải quyết hồ sơ đăng ký là bao lâu?",
    )

    result = GroundingVerifier().verify(request, _answer())

    assert VerificationCode.CASE_SPECIFIC_PREDICTION_UNSUPPORTED not in _codes(result)


def test_oversized_quote_is_safely_trimmed_for_json_object_providers():
    claim = GeneratedClaim.from_dict({
        "text": "Nghĩa vụ được quy định.",
        "evidence_ids": ["E1"],
        "supporting_quotes": [{"evidence_id": "E1", "quote": "nội dung " * 200}],
        "issue_ids": ["I1"],
    })

    assert 0 < len(claim.supporting_quotes[0].quote) <= 1200


def test_every_legal_claim_requires_an_existing_evidence_id():
    answer = GeneratedAnswer(
        answer="Có hai nghĩa vụ.",
        claims=(
            GeneratedClaim("Nghĩa vụ thứ nhất.", ()),
            GeneratedClaim("Nghĩa vụ thứ hai.", ("E404",)),
        ),
        limitations=(),
        confidence="medium",
    )

    result = GroundingVerifier().verify(_request(), answer)

    assert result.status is VerificationStatus.REPAIR_REQUIRED
    assert VerificationCode.MISSING_CITATION in _codes(result)
    assert VerificationCode.UNKNOWN_EVIDENCE_ID in _codes(result)


def test_answer_cannot_hide_legal_conclusion_outside_claims():
    answer = GeneratedAnswer(
        answer="Pháp luật yêu cầu phải báo cáo.",
        claims=(),
        limitations=(),
        confidence="high",
    )

    result = GroundingVerifier().verify(_request(), answer)

    assert VerificationCode.MISSING_CLAIMS in _codes(result)
    assert result.status is VerificationStatus.REPAIR_REQUIRED


def test_verifier_rejects_hallucinated_number_provision_status_and_url():
    text = (
        "Khoản 9 Điều 99 của 88/2026/NĐ-CP đã hết hiệu lực; "
        "xem https://hallucinated.test/source."
    )

    result = GroundingVerifier().verify(_request(), _answer(text))

    assert result.status is VerificationStatus.REPAIR_REQUIRED
    assert {
        VerificationCode.DOCUMENT_NUMBER_NOT_IN_EVIDENCE,
        VerificationCode.DOCUMENT_NUMBER_CITATION_MISMATCH,
        VerificationCode.PROVISION_CITATION_MISMATCH,
        VerificationCode.STATUS_CITATION_MISMATCH,
        VerificationCode.UNTRUSTED_URL,
    } <= _codes(result)


def test_inline_evidence_id_and_answer_level_provision_are_also_verified():
    answer = GeneratedAnswer(
        answer="Điều 77 đặt ra nghĩa vụ này [E404].",
        claims=(GeneratedClaim("Có nghĩa vụ báo cáo.", ("E1",)),),
        limitations=(),
        confidence="high",
    )

    result = GroundingVerifier().verify(_request(), answer)

    assert VerificationCode.UNKNOWN_EVIDENCE_ID in _codes(result)
    assert VerificationCode.PROVISION_CITATION_MISMATCH in _codes(result)


def test_status_evidence_must_be_resolved_at_requested_as_of():
    stale = _evidence(as_of=date(2025, 1, 1))

    result = GroundingVerifier().verify(_request(stale), _answer())

    assert VerificationCode.AS_OF_MISMATCH in _codes(result)


def test_cross_referenced_provision_is_allowed_only_when_present_in_cited_text():
    evidence = replace(
        _evidence(article="143", clause=None),
        text=(
            "Người chưa đủ 13 tuổi chỉ được làm công việc theo quy định "
            "tại Khoản 3 Điều 145 của Bộ luật này."
        ),
    )
    text = "Khoản 3 Điều 145 quy định danh mục công việc liên quan."

    supported = GroundingVerifier().verify(
        _request(evidence),
        _answer(text),
    )
    hallucinated = GroundingVerifier().verify(
        _request(evidence),
        _answer("Khoản 9 Điều 999 quy định danh mục công việc liên quan."),
    )

    assert supported.status is VerificationStatus.VERIFIED
    assert VerificationCode.PROVISION_CITATION_MISMATCH in _codes(hallucinated)


def test_non_effective_cited_document_status_must_be_disclosed():
    expired = _evidence(status="EXPIRED")

    missing = GroundingVerifier().verify(_request(expired), _answer())
    disclosed = GroundingVerifier().verify(
        _request(expired),
        _answer("Khoản 1 Điều 5 của 01/2025/QH15 đã hết hiệu lực."),
    )

    assert VerificationCode.STATUS_DISCLOSURE_MISSING in _codes(missing)
    assert disclosed.status is VerificationStatus.VERIFIED


def test_non_effective_status_may_be_disclosed_once_in_answer_summary():
    expired = _evidence(status="EXPIRED")
    answer = GeneratedAnswer(
        answer="Văn bản 01/2025/QH15 đã hết hiệu lực tại thời điểm 2026.",
        claims=(
            GeneratedClaim(
                "Người sử dụng lao động phải thực hiện báo cáo định kỳ.",
                ("E1",),
            ),
        ),
        limitations=(),
        confidence="high",
    )

    result = GroundingVerifier().verify(_request(expired), answer)

    assert result.status is VerificationStatus.VERIFIED
    assert VerificationCode.PROVISION_CITATION_MISMATCH not in _codes(result)


def test_historical_query_without_historical_content_is_refused_and_disclosed():
    request = _request(
        temporal_intent="historical",
        historical_content_available=False,
    )
    generator = FakeLegalAnswerGenerator([_answer()])

    result = GroundedRAGService(generator).answer(request)

    assert result.status is VerificationStatus.REFUSED
    assert result.attempts == 0
    assert generator.requests == []
    assert result.answer.claims == ()
    assert "phiên bản hiện tại" in result.answer.limitations[0]
    assert VerificationCode.HISTORICAL_SOURCE_UNAVAILABLE in _codes(
        result.verification
    )
    assert VerificationCode.HISTORICAL_LIMITATION_MISSING not in _codes(
        result.verification
    )


def test_current_law_refuses_unresolved_partial_provision_before_generation(monkeypatch):
    monkeypatch.setenv("LAWCHAT_FLAG_FAIL_CLOSED_PARTIAL", "true")
    partial = replace(
        _evidence(status="PARTIALLY_EFFECTIVE"),
        citation=replace(
            _evidence(status="PARTIALLY_EFFECTIVE").citation,
            status_scope="document",
        ),
    )
    generator = FakeLegalAnswerGenerator([_answer()])

    result = GroundedRAGService(generator).answer(_request(partial))

    assert result.status is VerificationStatus.REFUSED
    assert result.attempts == 0
    assert generator.requests == []
    assert VerificationCode.UNRESOLVED_PROVISION_STATUS in _codes(
        result.verification
    )


def test_current_law_allows_partial_provision_by_default():
    partial = replace(
        _evidence(status="PARTIALLY_EFFECTIVE"),
        citation=replace(
            _evidence(status="PARTIALLY_EFFECTIVE").citation,
            status_scope="document",
        ),
    )
    generator = FakeLegalAnswerGenerator([_answer()])

    result = GroundedRAGService(generator).answer(_request(partial))

    assert result.status is VerificationStatus.VERIFIED


def test_historical_claim_requires_version_identity_and_covering_interval():
    missing_version = GroundingVerifier().verify(
        _request(temporal_intent="historical"),
        _answer(),
    )
    assert VerificationCode.HISTORICAL_VERSION_MISMATCH in _codes(missing_version)

    evidence = _evidence()
    evidence = replace(
        evidence,
        citation=replace(
            evidence.citation,
            version_id="version-1",
            content_valid_from=date(2020, 1, 1),
            content_valid_to=None,
        ),
    )
    verified = GroundingVerifier().verify(
        _request(evidence, temporal_intent="historical"),
        _answer(),
    )
    assert verified.status is VerificationStatus.VERIFIED


def test_empty_evidence_is_refused_without_calling_generator():
    request = GenerationRequest(
        question="Câu hỏi thiếu nguồn",
        context=PackedContext((), "", 0, 100, ()),
        as_of=AS_OF,
    )
    generator = FakeLegalAnswerGenerator([_answer()])

    result = GroundedRAGService(generator).answer(request)

    assert result.status is VerificationStatus.REFUSED
    assert result.attempts == 0
    assert not generator.requests


def test_service_repairs_at_most_once_then_returns_verified_answer():
    invalid = _answer(evidence_ids=("E404",))
    valid = _answer()
    generator = FakeLegalAnswerGenerator([invalid, valid])

    result = GroundedRAGService(generator).answer(_request())

    assert result.status is VerificationStatus.VERIFIED
    assert result.answer.claims == valid.claims
    assert result.answer.answer.endswith("[E1]")
    assert result.attempts == 2
    assert len(generator.requests) == 2
    assert generator.requests[1].previous_answer == invalid
    assert generator.requests[1].repair_instructions


def test_verified_response_discards_untrusted_draft_answer():
    generated = GeneratedAnswer(
        answer="Người lao động được nhận một khoản trợ cấp không có trong evidence.",
        claims=(
            GeneratedClaim(
                "Người sử dụng lao động phải báo cáo định kỳ",
                ("E1",),
            ),
        ),
        limitations=(),
        confidence="high",
    )

    result = GroundedRAGService(
        FakeLegalAnswerGenerator([generated])
    ).answer(_request())

    assert result.status is VerificationStatus.VERIFIED
    assert "trợ cấp" not in result.answer.answer
    assert result.answer.answer == "Người sử dụng lao động phải báo cáo định kỳ. [E1]"


def test_failed_single_repair_fails_closed_without_a_third_generation():
    invalid = _answer(evidence_ids=("E404",))
    generator = FakeLegalAnswerGenerator([invalid, invalid, _answer()])

    result = GroundedRAGService(generator).answer(_request())

    assert result.status is VerificationStatus.REFUSED
    assert result.answer.claims == ()
    assert result.attempts == 2
    assert len(generator.requests) == 2


def test_generator_exception_returns_safe_response_without_raw_error():
    secret = "provider-secret-should-never-leak"
    generator = FakeLegalAnswerGenerator([RuntimeError(secret)])

    result = GroundedRAGService(generator).answer(_request())

    assert result.status is VerificationStatus.REFUSED
    assert VerificationCode.GENERATION_FAILED in _codes(result.verification)
    assert secret not in result.answer.answer
    assert secret not in " ".join(result.answer.limitations)


def test_repair_provider_failure_preserves_first_verification_issues():
    invalid = _answer(evidence_ids=("E404",))
    generator = FakeLegalAnswerGenerator(
        [invalid, RuntimeError("repair failed")]
    )

    result = GroundedRAGService(generator).answer(_request())

    assert result.status is VerificationStatus.REFUSED
    assert result.attempts == 2
    assert VerificationCode.UNKNOWN_EVIDENCE_ID in _codes(result.verification)
    assert VerificationCode.GENERATION_FAILED in _codes(result.verification)


def test_truncated_provider_response_has_a_distinct_safe_failure_code():
    from rag import TruncatedGenerationError

    generator = FakeLegalAnswerGenerator(
        [TruncatedGenerationError("raw provider detail")]
    )

    result = GroundedRAGService(generator).answer(_request())

    assert result.status is VerificationStatus.REFUSED
    assert VerificationCode.GENERATION_TRUNCATED in _codes(result.verification)
    assert "raw provider detail" not in result.answer.answer


def test_openai_compatible_provider_requests_schema_and_parses_json():
    captured = {}

    def transport(url, payload, headers, timeout):
        captured.update(
            url=url,
            payload=payload,
            headers=headers,
            timeout=timeout,
        )
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"answer":"Có nghĩa vụ báo cáo.",'
                            '"claims":[{"text":"Có nghĩa vụ báo cáo.",'
                            '"evidence_ids":["E1"]}],'
                            '"limitations":[],"confidence":"high"}'
                        )
                    }
                }
            ]
        }

    provider = OpenAICompatibleLegalAnswerGenerator(
        OpenAICompatibleSettings(
            base_url="https://llm.example/v1/",
            api_key="test-key",
            model="test-model",
        ),
        transport=transport,
    )

    answer = provider.generate(_request())

    assert answer.claims[0].evidence_ids == ("E1",)
    assert captured["url"] == "https://llm.example/v1/chat/completions"
    assert captured["payload"]["response_format"]["type"] == "json_schema"
    assert captured["payload"]["temperature"] == 0
    assert captured["payload"]["max_tokens"] == 1024
    assert captured["payload"]["response_format"]["json_schema"]["schema"]["properties"]["claims"]["maxItems"] == 5
    assert captured["headers"]["Authorization"] == "Bearer test-key"


def test_async_streaming_provider_records_ttft_and_usage():
    import httpx

    structured = {
        "answer": "",
        "claims": [
            {"text": "Có nghĩa vụ báo cáo.", "evidence_ids": ["E1"]}
        ],
        "limitations": [],
        "confidence": "high",
    }
    content = json.dumps(structured, ensure_ascii=False)

    def handler(request):
        lines = [
            "data: " + json.dumps(
                {"choices": [{"delta": {"content": content[:20]}}]}
            ),
            "data: " + json.dumps(
                {
                    "choices": [
                        {
                            "delta": {"content": content[20:]},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 20},
                }
            ),
            "data: [DONE]",
        ]
        return httpx.Response(200, text="\n\n".join(lines))

    provider = OpenAICompatibleLegalAnswerGenerator(
        OpenAICompatibleSettings(
            base_url="http://ollama/v1",
            api_key="ollama",
            model="test-model",
        ),
        async_transport=httpx.MockTransport(handler),
    )

    answer = asyncio.run(provider.generate_async(_request()))

    assert answer.claims[0].evidence_ids == ("E1",)
    assert answer.telemetry is not None
    assert answer.telemetry.ttft_seconds is not None
    assert answer.telemetry.prompt_tokens == 10
    assert answer.telemetry.completion_tokens == 20


def test_system_prompt_contains_all_grounding_contracts():
    prompt = SYSTEM_PROMPT.casefold()

    assert "chỉ sử dụng" in prompt
    assert "evidence_id" in prompt
    assert "url" in prompt
    assert "hết hiệu lực" in prompt
    assert "nội dung lịch sử" in prompt
    assert "phải từ chối" in prompt


def test_output_contract_is_last_for_initial_generation_and_repair():
    initial = _request()
    repair = replace(
        initial,
        previous_answer=_answer(),
        repair_instructions=("Evidence ID không tồn tại.",),
    )

    initial_prompt = build_user_prompt(initial)
    repair_prompt = build_user_prompt(repair)

    assert initial_prompt.endswith(OUTPUT_CONTRACT)
    assert repair_prompt.endswith(OUTPUT_CONTRACT)
    assert repair_prompt.index("Yêu cầu repair") < repair_prompt.index(OUTPUT_CONTRACT)
    assert '"answer"' in OUTPUT_CONTRACT
    assert '"evidence_ids"' in OUTPUT_CONTRACT
