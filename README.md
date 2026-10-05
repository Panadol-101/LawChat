# Vietnamese Legal RAG

This project provides a Retrieval-Augmented Generation (RAG) pipeline for Vietnamese legal documents.

The v0.4 application includes persistent projects/conversations, a verified
chat SSE endpoint and a Streamlit client. FastAPI owns all PostgreSQL access;
the frontend communicates only through HTTP/SSE.

Start the complete local application after configuring `.env`:

```bash
docker compose up -d --build postgres qdrant app frontend
```

Open `http://localhost:8501`. The API remains available at
`http://localhost:8000`, including `POST /api/v1/answer`,
`POST /api/v1/chat` and `POST /api/v1/chat/stream`.

The frontend sends a stable `X-Workspace-ID` header and never receives database
credentials. `POST /api/v1/chat/stream` emits progress events while the RAG
pipeline runs, but only emits `answer.delta` after structural and semantic
verification completes. Conversation and message schemas are managed by
Alembic, not by Streamlit.

The current workspace header provides MVP data separation, not authentication.
Do not expose the deployment to untrusted users until an authenticated identity
is bound to the workspace on the server.

### Human-readable citation sources

Citation buttons call `GET /api/v1/documents/{document_id}/open-source` instead
of opening the corpus URL. If the document already has a readable source, the
API redirects there. Otherwise it asks the configured Google Search MCP tool,
verifies the exact document number and an official domain, caches the result in
`document_sources`, and redirects to the full-text page. If MCP is unavailable
or no official match is safe to select, the API falls back to an exact Google
search page so it never sends the reader to Hugging Face.

Configure a Streamable HTTP MCP server whose search tool accepts a `query`
argument and returns structured results containing `url`/`link`, `title`, and
`snippet` fields:

```dotenv
LAWCHAT_SOURCE_MCP_URL=https://your-mcp.example/mcp
LAWCHAT_SOURCE_MCP_TOOL=google_search
LAWCHAT_SOURCE_MCP_API_KEY=
```

Historical content uses an isolated, fail-closed release path. See
[HISTORICAL_INDEXING_GUIDE.md](HISTORICAL_INDEXING_GUIDE.md); the current
Qdrant and BM25 aliases remain unchanged.

## Project structure

```text
LawChat/
├── src/          # application and domain code
│   ├── api/              # FastAPI routes and response schemas
│   ├── chat/             # persistent projects and conversations
│   ├── chunking/         # legal-document chunk strategies
│   ├── database/         # models, queries and connections
│   ├── evaluation/       # shared quality metrics
│   ├── indexing/         # Qdrant embeddings and Tantivy BM25
│   ├── ingestion/        # PostgreSQL and enrichment loaders
│   ├── rag/              # generation, context and verification
│   ├── retrieval/        # dense, sparse, graph and temporal retrieval
│   └── sources/          # official source resolution
├── scripts/              # consolidated maintenance commands
├── frontend/             # Streamlit web client
├── migrations/           # immutable Alembic history
├── tests/
├── docker/
├── pyproject.toml
├── docker-compose.yml
└── README.md
```

Consolidated maintenance entry points:

- `scripts.build_indexes`: `dense` or `sparse` indexing;
- `scripts.query`: `dense`, `hybrid`, `legal`, or full `answer` diagnostics;
- `scripts.check_chunks`: human inspection and machine quality gates;
- `scripts.legal_metadata`: enrichment and provision-status operations;
- `scripts.evaluate_retrieval` and `scripts.evaluate_rag`: quality evaluation;
- `scripts.test_answer_api`: HTTP smoke and load tests;
- `scripts.verify_deployment`: embedding and release verification.

## Notes

- `data/raw` stores original downloaded legal documents.
- `data/processed` stores cleaned or normalized data.
- `data/chunks` stores segment chunks for indexing.
- `data/evaluation` stores benchmark and evaluation datasets.
- `src/` contains the core ingestion, parsing, retrieval, and generation pipeline.

This repo structure is ready for expansion into a full legal RAG system.

## PostgreSQL legal metadata

PostgreSQL is the deterministic metadata and temporal-filter layer. Qdrant
continues to own embeddings and similarity search; `chunks` only stores its
collection/point reference.

The initial schema contains:

- `documents`: stable legal identity and commonly filtered metadata;
- `document_versions`: immutable crawl/content snapshots with SHA-256 lineage;
- `articles` and `chunks`: structure and Qdrant point mapping per snapshot;
- `effective_status`: non-overlapping legal-status history;
- `document_relationships`: amendment, replacement, repeal and citation links;
- `crawl_runs` and `crawl_records`: crawl audit and source observations.

Start PostgreSQL and apply the schema:

```bash
cp .env.example .env
docker compose up -d postgres
uv run alembic upgrade head
```

`effective_status.valid_to` is exclusive. A row with
`valid_from=2026-01-01` and `valid_to=2027-01-01` is valid throughout 2026,
but not on 2027-01-01. PostgreSQL rejects overlapping periods for the same
document with a GiST exclusion constraint, preventing ambiguous answers.

Use the shared query builder so API and workers apply exactly the same rule:

```python
from datetime import date

from database import LegalMetadataFilter, MetadataQueries

filters = LegalMetadataFilter(
    as_of=date(2026, 8, 25),
    authorities=("Chính phủ",),
    legal_fields=("Doanh nghiệp",),
)

# Feed these external IDs into Qdrant's `doc_id` payload filter.
statement = MetadataQueries.effective_document_external_ids(filters)
document_ids = session.scalars(statement).all()
```

For an exact point-level allow-list, use
`MetadataQueries.qdrant_point_ids(...)`. Only indexable chunks belonging to
the current document version are returned.

Run unit tests and optional PostgreSQL integration tests:

```bash
uv run pytest -q
TEST_DATABASE_URL="$DATABASE_URL" uv run pytest -q tests/integration/database
```

### Import the Hugging Face snapshot

Validate normalization and dependency coverage without writing anything:

```bash
uv run python -m scripts.load_postgres_metadata \
  --dry-run \
  --limit-documents 100
```

Load the complete local snapshot with its Hugging Face commit revision:

```bash
DATABASE_URL="postgresql+psycopg://lawchat:lawchat@localhost:5432/lawchat" \
uv run python -m scripts.load_postgres_metadata \
  --batch-size 20000 \
  --dataset-revision 8977887f17be2defae4c5171d55562e1cde7d695
```

The loader streams Parquet record batches into temporary staging tables with
PostgreSQL `COPY`, then performs dependency-ordered upserts. It is safe to
rerun the same snapshot: document IDs, content hashes, chunk IDs and source
relationship identities prevent duplicates. Each run is audited in
`crawl_runs`, including normalization warnings.

For partial development runs, `--limit-documents N` selects the same document
set consistently across metadata, structured content, chunks and
relationships. `--skip-structured`, `--skip-chunks` and
`--skip-relationships` allow isolated stages.

After a full import, useful reconciliation queries are:

```sql
SELECT count(*) FROM documents;
SELECT count(*), count(*) FILTER (WHERE is_current) FROM document_versions;
SELECT count(*), count(*) FILTER (WHERE is_indexable) FROM chunks;
SELECT count(*), count(*) FILTER (WHERE target_document_id IS NULL)
FROM document_relationships;

SELECT count(*) AS orphan_chunks
FROM chunks c
LEFT JOIN documents d ON d.id = c.document_id
LEFT JOIN document_versions v ON v.id = c.version_id
WHERE d.id IS NULL OR v.id IS NULL;
```

## Dense vector index with Qdrant and BGE-M3

Start the vector database:

```bash
docker compose up -d qdrant
```

The default model is `BAAI/bge-m3`, executed by SentenceTransformers/PyTorch.
It produces normalized 1024-dimensional cosine vectors. The production collection
uses on-disk vector/HNSW storage and INT8 scalar quantization. Payload indexes
are created before ingestion for `doc_id`, document metadata, legal structure
and dates. PostgreSQL remains the source of truth for temporal validity;
Qdrant answers semantic similarity.

The chunker uses the pinned BGE-M3 Hugging Face tokenizer, not a character or
regex estimate. Chunker v4 targets 600–800 tokens and enforces a hard maximum
of 1200 tokens on every indexable `retrieval_text`.

Rebuild and import a replacement chunk generation:

```bash
HF_HUB_OFFLINE=1 \
uv run python -m scripts.build_chunks

uv run python -m scripts.load_postgres_metadata \
  --batch-size 20000 \
  --dataset-revision 8977887f17be2defae4c5171d55562e1cde7d695
```

Every imported chunk carries an import-generation marker. Stale chunks are
deleted only after the complete new generation has loaded. The
`parent_chunk_id` FK is indexed so bulk replacement does not degenerate into
repeated full-table scans. Interrupted cleanup can be resumed only after an
exact generation-count check with `scripts.resume_chunk_cleanup`.

Run a small end-to-end smoke index first. A limited build is deliberately not
allowed to promote the production alias:

```bash
uv run python -m scripts.build_indexes dense \
  --fetch-size 8 \
  --embedding-batch-size 16 \
  --limit 16
```

Run the complete resumable build and atomically promote the alias only after
all current/indexable chunks are indexed:

```bash
uv run python -m scripts.build_indexes dense \
  --fetch-size 128 \
  --embedding-batch-size 64 \
  --promote-alias
```

SentenceTransformers uses PyTorch. `EMBEDDING_DEVICE=cuda` is intended for the
temporary GPU indexing VM, while `EMBEDDING_DEVICE=cpu` supports query-time
encoding after the snapshot is restored locally. Benchmark a representative
1,000–10,000 point sample before estimating the full build. The indexer commits to Qdrant first
and only then records `qdrant_point_id` in PostgreSQL. Stable UUIDv5 point IDs
make retries idempotent after a process or network failure.

### GPU indexing runtime

Install an NVIDIA driver compatible with the PyTorch build, synchronize the
locked environment, and verify both PyTorch and BGE-M3 before indexing:

```bash
.venv/bin/python -c \
  "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name())"

EMBEDDING_DEVICE=cuda \
.venv/bin/python -m scripts.verify_deployment embedding
```

Run a small smoke build before the complete index:

```bash
EMBEDDING_DEVICE=cuda \
.venv/bin/python -m scripts.build_indexes dense \
  --fetch-size 128 \
  --embedding-batch-size 8 \
  --limit 1000
```

Then run the resumable complete build. Increase the batch size only after
observing stable VRAM use on the smoke build:

```bash
EMBEDDING_DEVICE=cuda \
QDRANT_PREFER_GRPC=true \
.venv/bin/python -m scripts.build_indexes dense \
  --fetch-size 256 \
  --embedding-batch-size 16 \
  --promote-alias
```

When CUDA is explicitly requested but unavailable, the embedder raises an
error instead of silently running a multi-million-point build on CPU.

Semantic search through the stable alias after successful full promotion:

```bash
uv run python -m scripts.query dense \
  "Người sử dụng lao động có được đơn phương cho nghỉ việc không?" \
  --limit 5
```

### Temporal legal retrieval

Production retrieval treats Qdrant as the semantic candidate ranker and
PostgreSQL as the source of truth. PostgreSQL hydrates the ranked point IDs
with legal text, parent context and citation metadata, and rejects documents
that are not valid on the requested date. Qdrant's snapshot `status` payload
is deliberately not trusted for historical validity.

Audit a restored data release before serving traffic:

```bash
uv run python -m scripts.verify_deployment release
```

The audit exits unsuccessfully unless the production alias, embedding model,
1024-dimensional cosine configuration, Qdrant count, PostgreSQL marked count
and a deterministic point sample all agree.

Run temporal search from the command line:

```bash
uv run python -m scripts.query legal \
  "Ngày 01/01/2025 công ty có được cho lao động nữ mang thai nghỉ việc không?" \
  --limit 5
```

Dates in `YYYY-MM-DD`, `DD/MM/YYYY` and Vietnamese textual form are parsed
deterministically. If a question contains multiple dates, pass `--as-of`
explicitly. Search results contain the source text, parent context, effective
period and a stable legal citation. Use repeated `--status` arguments when
retrieving inactive documents; the default is `EFFECTIVE` and
`PARTIALLY_EFFECTIVE`.

Start the HTTP API after PostgreSQL and Qdrant are ready:

```bash
docker compose up -d app

curl -X POST http://localhost:8000/api/v1/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"Điều kiện đơn phương chấm dứt hợp đồng lao động", "limit":5}'
```

`GET /health` checks the process. `GET /ready` additionally verifies the
PostgreSQL connection and that `legal_chunks_current` targets the configured
physical collection.

### Hybrid Dense + BM25 retrieval

The lexical side uses a disk-based Tantivy BM25 index. Only stable point and
chunk IDs are stored in Tantivy; legal text and temporal metadata continue to
come from PostgreSQL. The BM25 build is resumable and uses an atomic alias
manifest so an incomplete index cannot receive production traffic.

Run a limited smoke build first. A limited build cannot be promoted:

```bash
uv run python -m scripts.build_indexes sparse \
  --fetch-size 1000 \
  --limit 10000
```

Resume and complete the production index, then promote its alias:

```bash
uv run python -m scripts.build_indexes sparse \
  --fetch-size 10000 \
  --promote-alias
```

Run hybrid search:

```bash
uv run python -m scripts.query hybrid \
  "Điều 5 Khoản 2 Nghị định 100/2019/NĐ-CP quy định gì?" \
  --limit 5
```

Dense and sparse retrieval run concurrently and are combined with weighted
Reciprocal Rank Fusion. PostgreSQL then hydrates the fused point IDs and
enforces the same legal-validity rule as dense-only search. If the BM25 alias
is unavailable, the API falls back to dense retrieval and returns a warning in
the response diagnostics.

Evaluate the production hybrid service against a curated fixture:

```bash
uv run python -m scripts.evaluate_retrieval hybrid \
  --fixture tests/fixtures/dense_retrieval_eval.json \
  --k 10 \
  --candidate-limit 100
```

Important BM25/RRF settings:

```ini
BM25_INDEX_ROOT=data/indexes/bm25
BM25_INDEX_NAME=legal_bm25_v1
BM25_INDEX_ALIAS=legal_bm25_current
RRF_K=60
RRF_DENSE_WEIGHT=1.0
RRF_SPARSE_WEIGHT=1.0
```

Run the curated real-model quality gate:

```bash
uv run python -m scripts.evaluate_retrieval dense \
  --k 3 \
  --min-hit-rate 0.85 \
  --min-mrr 0.70

EMBEDDING_CACHE_PATH="$PWD/data/.cache/huggingface" \
HF_HUB_OFFLINE=1 \
RUN_BGE_M3_EVAL=1 \
uv run pytest -q tests/semantic
```

The test layers are intentionally separate:

- unit tests validate stable IDs, vector shape, idempotency, ranking metrics,
  filters and alias safety using deterministic embeddings;
- PostgreSQL/Qdrant integration tests validate the real services;
- semantic tests use BGE-M3 over curated Vietnamese legal passages and
  enforce retrieval quality thresholds.

## Claim entailment review and Gemma Cloud trial

The RAG answer API now supports quoted claim evidence and semantic review in
`off`, `shadow`, or `enforce` mode. The Gemma trial uses `gemma4:31b-cloud` with
issue decomposition and enforced claim/coverage review. The supported legal
data cutoff is 2026-07-31; later `as_of` dates are rejected. See
[SEMANTIC_VERIFICATION_GUIDE.md](SEMANTIC_VERIFICATION_GUIDE.md) for setup,
response fields, evaluation commands, limitations and rollback.
