# LawChat system assessment — 2026-09-24

LawChat is a functioning legal RAG MVP with a substantial automated test foundation, a populated corpus, live verified answers, persistent conversations, and verified SSE delivery. It is not yet demonstrated ready for an untrusted multi-user deployment. Access control, historical coverage, response time, admin functionality, and broader independently reviewed answer quality remain material gaps.

## Fresh validation

| Area | Result | Evidence / limits |
|---|---|---|
| Default automated suite | 274 passed, 12 skipped; 6.65 seconds | `system_assessment_20260924.xml`; dependency-dependent cases subsequently exercised below |
| PostgreSQL integration | 10 passed; 3.39 seconds | `system_database_isolated_20260924.xml`; fresh isolated database migrated through all 10 revisions |
| Real BGE-M3 quality gate | 1 passed; 356.55 seconds | `system_semantic_20260924.xml`; 105-document, 135-query fixture; passes Hit@3 >= 0.85 and MRR >= 0.70; exact scores are not emitted by this test |
| Real Qdrant integration | Passed equivalent assertions | Created two-point assessment collection, promoted its own alias, checked payload index, count, and search result; retained collection instead of invoking deletion cleanup |
| Live service health | All six Compose services healthy after start | API health/readiness and both Streamlit health endpoints returned 200 |
| Frontend execution | Both login screens rendered without exceptions | Separate Streamlit AppTest processes; not a full browser/layout or successful account-login test |
| HTTP validation | Empty query rejected with 422 | Saved live report |
| Authentication boundary | Auth/me and admin/ping returned 401 without a cookie | Projects returned 200 without authentication; see finding below |
| Persistence | Project and conversation created; messages persisted | Assessment workspace and records retained |
| Workspace selection | Different workspace returned 404 for assessment conversation | This validates header-based separation, not authenticated ownership |
| Streaming and retry | Verified answer streamed; replay did not duplicate messages | `system_stream_recovery_20260924.json`; verification.completed preceded answer.delta; final chat.completed; four messages total for two distinct client message IDs |
| Queue behavior | Concurrent request failed with QUEUE_FULL | HTTP 200 SSE transport carried chat.failed with http_status 429; saved in `system_chat_20260924.json`; recovery succeeded after pipeline became available |

Combined: 285 pytest cases passed across the successful runs, plus the retained real-Qdrant equivalent of the remaining gated test. This is not a claim that all 286 cases ran together in one configuration.

The default suite covers parsing, cleaning, chunking, ingestion contracts, database query construction, index contracts, temporal routing, fusion, reranking, retrieval, context packing, issue decomposition, structural/semantic verification, API behavior, frontend text/citation helpers, and evaluation metrics. API integration tests in that suite inject dependencies; they are not production-model evidence by themselves.

## Live answers

All six requests returned HTTP 200. VERIFIED and REFUSED are application outcomes, not independent legal expert judgments.

| Scenario | Outcome | Seconds | Observation |
|---|---|---:|---|
| Exact provision: marriage conditions, Article 8 of 52/2014/QH13 | VERIFIED | 15.019 | 2 claims, 1 citation; semantic PASSED and coverage COMPLETE |
| Current scenario: minors working at night | VERIFIED | 95.735 | 3 claims, 2 citations; semantic PASSED and coverage COMPLETE |
| Historical marriage conditions at 2010-01-01 | REFUSED | 0.654 | Explicitly disclosed unavailable historical evidence; no current text presented as historical |
| Invented statute 9999/2099/QH99 | REFUSED | 2.154 | No exact seed or supporting evidence |
| Multiple issues: incomplete pay and a 14-year-old working nights | VERIFIED | 211.736 | 4 claims, 3 citations; semantic PASSED and coverage COMPLETE |
| Instruction to bypass verification and fabricate law / guaranteed victory | REFUSED | 52.348 | No fabricated answer exposed in this probe |

All nine claims in the three verified answers had supporting quotes and evidence IDs present in the returned citations. Full responses, evidence, timings, warnings, and verification details are retained in `system_live_20260924.json`.

These scenarios are smoke probes, not a statistically representative or independently adjudicated legal benchmark. Current-law probes explicitly used 2026-07-31, matching the configured cutoff default; they do not establish legal freshness through 2026-09-24. The multi-issue answer also includes penalty-related detail that warrants expert relevance/completeness review beyond checking its citations.

## Corpus and deployment

- PostgreSQL: 171,556 documents; 2,224,668 chunks; 2,009,485 indexable chunks after integration-test cleanup.
- Production Qdrant collection: 2,009,485 points; green status; 1024-dimensional BGE-M3 vectors.
- BM25 manifest: complete, 2,009,485 indexed/expected points, source collection matches production Qdrant.
- These counts reconcile; they do not prove the content and payload of every point match. No exhaustive per-record checksum audit was performed.
- Database migration head: `de2ba91103cb`; fresh migration chain also succeeded in the assessment database.
- Running API main.py SHA-256 matched the workspace file: `f76f5ac88e2fa6bd0ad7a781a273d2e255801bf78fc0eb499ac194f7c599079c`. Other deployed files were not exhaustively compared.
- Ollama-compatible endpoint listed `gemma4:31b-cloud`; live generation and semantic verification succeeded.
- All containers were initially stopped. Existing containers were started without rebuilding images.

## Findings and remaining work

1. **High priority — bind data access to authenticated identity.** `src/lawchat/api/main.py` uses `get_workspace_id` for project/conversation operations, accepting caller-supplied X-Workspace-ID with default `local`. The project endpoints do not require CurrentUserDependency. Live unauthenticated project/conversation creation succeeded in the assessment workspace. A different header returned 404, but a caller can select the original header. Authentication on the login UI or admin ping does not enforce ownership on these APIs. `frontend/api_client.py` defaults all clients to the same `local` workspace unless configured otherwise.
2. **High priority — historical answering remains unavailable in this deployment.** The historical probe safely refused. `HISTORICAL_INDEX_ENABLED` is false in Compose. The historical import/filter infrastructure passed isolated tests, but a historical production corpus has not been demonstrated.
3. **High priority — improve and measure latency in isolation.** The current scenario spent about 43.5 seconds hydrating and 41.0 seconds reranking; total was 95.7 seconds. Multi-issue took 211.7 seconds. Concurrent embedding and corpus-scanning database tests competed for CPU/I/O, so these are observed loaded timings, not a normal-user latency benchmark. The queue rejected a waiting chat after its configured wait; recovery worked. Sustained throughput, cancellation recovery, and capacity limits were not fully load-tested.
4. **Authentication has incomplete paths and coverage.** Static inspection shows `/auth/login` verifies a password but does not create a session. The main frontend only enters the TOTP step when requires_totp is true; for a user without TOTP, it rerenders login without establishing a session. This is a code-path finding, not a successful-account reproduction. Existing automated tests do not provide full login/TOTP/logout/role/expiry coverage. The cookie uses secure=False. Full account lifecycle testing remains necessary before remote deployment.
5. **Admin is a scaffold.** Users, Security, RAG/Data, and System pages explicitly say they will be connected next. Dashboard Online indicators are hard-coded, not live dependency checks. Login-screen rendering and admin unauthorized rejection work; management functionality is not complete.
6. **Answer quality needs a larger fresh benchmark and expert review.** Live verification and citations work, but six prompts cannot establish reliable legal correctness, completeness, or zero unsafe answers. Versioned gold labels and category-level evaluation should be used for the next quality gate.
7. **Frontend coverage is limited.** Existing unit coverage is four citation/text helper tests. Headless login rendering adds execution evidence, but visual correctness, successful login, accessibility, mobile behavior, and complete browser chat interactions remain unverified.
8. **Source resolution is only partly validated.** Mock-backed source endpoint tests passed and source-MCP service is healthy; live official-link resolution/search accuracy was not exercised in this run.

## Earlier reports — context, not fresh measurements

`RETRIEVAL_HYBRID_200.json` selects 200 cases and evaluates 190 answerable cases. Its chunk metrics use a 155-case denominator: Recall/Hit@K 77.42%, MRR 0.589, nDCG 0.634. Strict hit rate across answerable cases is 75.26%. Category hit rates include exact document number 30%, legal relationship 60%, exact provision 100%, and historical 0%. Denominators differ; these figures must not be conflated.

`RETRIEVAL_RERANKER_SAMPLE.json` reports 90% hit rate on only 10 cases. That smaller sample does not establish a gain over the 200-case report.

`RAG_GENERATION_EVAL_10.json` reports 7 verified of 10 total, 7 of 8 answerable verified (87.5%), false refusal 12.5%, unsafe answer 0% in that small sample, grounding document precision 66.7%, and p95 latency 103.91 seconds. Corpus release is `chunker-v4-bge-m3-20260830`. These reports were present before this assessment and were not rerun.

`CLAIM_ENTAILMENT.json`, dated 2026-09-20, reports 20 synthetic cases, exact verdict accuracy 90%, unsafe-claim detection 100%, no unsafe claims accepted, and no false flags. Synthetic labels are not validation of real-law answers.

## Reproducibility and retained state

No project source fixes or file deletions were performed. Existing user changes were preserved. New assessment scripts and JSON/XML/Markdown reports are under `reports/`. Pytest ran with its cache provider disabled and temporary-directory retention enabled.

The initial database attempt failed because the example password was not the deployment password (`system_database_20260924.xml`). A second run with configured credentials was interrupted because the historical test scanned the full corpus (`system_database_configured_20260924.xml`). The isolated run is the completed database result. Database test fixtures cleaned up their own temporary rows; final production document/chunk counts match the original counts.

Retained resources: database `lawchat_assessment_20260924`; Qdrant collection `lawchat_assessment_20260924` and alias `lawchat_assessment_20260924_alias`; chat workspace `assessment-20260924` with project/conversation and four messages. The six deployment services remain running.

An initial combined-process frontend AppTest produced an api_client module-name collision; rerunning each app in a separate process, as deployed, succeeded. This was a harness artifact and is not counted as a deployed frontend failure.

The lawchat-rag skill guided historical refusal, source traceability, verification-before-streaming, and separation of fresh evidence from prior benchmarks. No legal advice or independent assertion of current Vietnamese law is made by this assessment.
