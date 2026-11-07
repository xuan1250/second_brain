# Development

Local workflow for working on the backend and frontend outside Docker. For the containerized stack see [deployment.md](deployment.md).

## Toolchain

- Python dependencies are managed **only** with Astral `uv` ([`pyproject.toml`](../backend/pyproject.toml) + `uv.lock`). Do not install with global `pip`. `torch`/`torchvision` are routed to the CPU-only PyTorch index (see `[tool.uv.sources]`) to avoid multi-GB CUDA wheels.
- Frontend uses npm; scripts are defined in [`frontend/package.json`](../frontend/package.json).
- Required versions are pinned by `requires-python` in `pyproject.toml` and by the base images in the [`Dockerfile`](../Dockerfile).

## Run locally

Start Qdrant first:
```bash
docker compose up -d qdrant                       # vector DB only
```

### Option A: Local CPU execution (Default)
By default, `uv sync` installs the lightweight CPU PyTorch wheels configured in `pyproject.toml`.

```bash
# terminal 1 — backend (CPU mode)
cd backend
uv sync --frozen
uv run uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000

# terminal 2 — frontend (http://localhost:5173)
cd frontend
npm ci
npm run dev
```

### Option B: Local NVIDIA GPU execution (CUDA)
If developing on an NVIDIA GPU workstation/laptop and wishing to run with CUDA acceleration locally outside Docker:

```bash
cd backend
uv sync --frozen

# Install CUDA 12.4 PyTorch and ONNX Runtime GPU wheels into .venv:
uv pip install --reinstall-package torch --reinstall-package torchvision \
  --index-url https://download.pytorch.org/whl/cu124 \
  "torch>=2.4.0" "torchvision>=0.19.0"
uv pip uninstall onnxruntime
uv pip install "onnxruntime-gpu==1.20.0"

# Launch backend with auto-detection (resolves to cuda):
uv run uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000
```

The backend automatically detects CUDA and logs:
```
INFO: BGE-M3 model initialized on device 'cuda' (fp16=True, sparse_mode=bge_m3)
```

The Vite dev server proxies `/api` to the backend ([`vite.config.ts`](../frontend/vite.config.ts)). Running `npm run build` produces `frontend/dist/`, which FastAPI serves at `/` on next startup — so a single backend process can serve the full app.

## Configuration

Settings are loaded by [`src/config.py`](../backend/src/config.py) (`load_settings`) from [`backend/config.yaml`](../backend/config.yaml); that file is the reference for available keys and defaults. Relative paths in it resolve against `backend/`.

**Precedence (highest first)**

1. Per-request `provider` / `model` sent by the chat UI (LLM only) or per-upload `prefer_fast_path` / `approval_mode` query params.
2. Persisted `WorkspaceSettings` in SQLite (`GET/PATCH /api/v1/settings`, synced into runtime `Settings` on startup and on live save).
3. Environment variables.
4. `config.yaml` — for the LLM, `llm.providers.<name>` supplies the default model/base URL when a non-default provider is selected.

**Secrets** go in `backend/.env` (start from [`.env.example`](../backend/.env.example)); it is loaded by [`src/rag/llm.py`](../backend/src/rag/llm.py). Values that look like template placeholders (e.g. starting with `your-`) are treated as unset. Never commit real keys.

**Environment variables read by the code**

| Variable | Effect |
|---|---|
| `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `DEEPSEEK_API_KEY` | Provider credentials |
| `LLM_API_KEY` | Credential for OpenAI-compatible endpoints (falls back to the OpenAI/DeepSeek key) |
| `LLM_PROVIDER`, `LLM_MODEL_NAME`, `LLM_BASE_URL` | Override `llm.provider`, `llm.model_name`, `llm.base_url` |
| `CONFIG_PATH` | Alternate `config.yaml` location |
| `QDRANT_URL` | Qdrant endpoint; `:memory:` runs an in-process instance |
| `DATA_DIR`, `DB_DIR`, `FRONTEND_DIST_DIR` | Override `storage.*` paths |
| `EMBEDDING_DEVICE` | Compute device for embeddings: `auto` (default), `cuda`, or `cpu` |
| `EMBEDDING_FP16` | Whether to load embedding model in fp16 on CUDA: `true` (default) or `false` |
| `EMBEDDING_SPARSE_MODE` | Sparse representation mode: `auto` (default, `bge_m3` on CUDA, `bm25` on CPU), `bge_m3`, or `bm25` |
| `EMBEDDING_MAX_LENGTH` | Maximum token sequence length for BGE-M3 (default: `8192`) |
| `INGESTION_DEVICE` | Compute device for Docling layout parser: `auto` (default), `cuda`, or `cpu` |
| `HF_HOME`, `TORCH_HOME`, `FASTEMBED_CACHE_PATH` | Model cache locations |
| `INGESTION_USE_PROCESS_POOL` | `false` runs parse/chunk in a thread instead of the spawn pool |
| `INGESTION_WORKER_IDLE_TIMEOUT_SEC` | Inactivity window in seconds before automatically shutting down the Docling/RapidOCR worker process (default: `60`, `0` to disable) |
| `INGESTION_WORKER_MAX_TASKS` | Max completed documents before recycling the worker pool between batches (default: `0` = disabled) |
| `INGESTION_CLEAR_CUDA_CACHE` | Whether to flush PyTorch CUDA scratchpad cache (`release_cuda_cache()`) after document indexing (default: `true`) |
| `LAYA_MODEL_NAME` | Model name for local multilingual domain classification (default: `convaiinnovations/laya-multilingual`) |
| `INGESTION_ENABLE_LLM_CLASSIFICATION_FALLBACK` | Whether to fall back to LLM for classification if confidence is below threshold (default: `false`) |
| `INGESTION_HIGH_CONFIDENCE_THRESHOLD` | Confidence cutoff to bypass human approval under `auto_high_confidence` (default: `0.70`) |
| `PREFER_PYMUPDF4LLM` | `true` uses `pymupdf4llm` for PDFs instead of Docling |
| `FLASHRANK_ALLOW_DOWNLOAD` | `true` lets FlashRank download its model into `data/models/flashrank`. **If the model is not cached and this is unset, FlashRank is silently skipped** and a lexical-overlap heuristic ranks candidates instead (see `_lexical_overlap_score` in [`reranker.py`](../backend/src/rag/reranker.py)). |

To find every variable the backend reads: `rg "environ" backend/src`.

Supported LLM provider names and aliases are defined in `src/rag/llm.py` (`_SUPPORTED_INIT_PROVIDERS`, `_OPENAI_COMPATIBLE_PROVIDERS`); any provider with a `base_url` is routed through the OpenAI-compatible client.

## Batch ingestion (CLI)

Besides the upload dropzone, a file or folder can be ingested from the command line. Options are defined in [`src/ingestion/cli.py`](../backend/src/ingestion/cli.py):

```bash
cd backend
uv run python -m src.ingestion.cli <path> [--force] [--approval-mode {review_all,auto_high_confidence,auto_all}]
uv run python -m src.ingestion.cli --help
```

`--stub-embedder --in-memory-qdrant --no-pool` gives an offline dry run with no model downloads.

## Tests

Tests live in [`backend/tests/`](../backend/tests/); shared fixtures (generated PDF/DOCX/XLSX/PPTX samples, stub embedder, fake LLM, in-memory Qdrant) are in [`conftest.py`](../backend/tests/conftest.py).

| Tier | Command | Needs |
|---|---|---|
| Offline (default) | `cd backend && uv run pytest` | Nothing — no network, no API keys |
| Live acceptance | `cd backend && uv run pytest -m live` | Running Qdrant (`QDRANT_URL`), downloaded `bge-m3` and FlashRank models, a valid LLM key |

The default run excludes `live` via `addopts` in `pyproject.toml`. Why two tiers: real embeddings need a ~2 GB download and real LLM calls cost money and are non-deterministic, so every change gets a fast deterministic gate while live tests verify the real integrations on purpose. Do not replace a failing live integration with a stub to make it pass.

### Concurrency & Resilience Testing

To validate worker self-healing, fast-path heuristics, pre-upsert relational barriers, compensating rollbacks, and multi-file batch torture runs:

```bash
cd backend
uv run pytest tests/unit/test_batch_resilience.py tests/integration/test_concurrency_stress.py -v
```

- [`tests/unit/test_batch_resilience.py`](../backend/tests/unit/test_batch_resilience.py): Unit tests for `BrokenProcessPool` worker crash recovery, pre-upsert SQLite existence barriers, compensating Qdrant rollback on partial failure, in-flight cancellation flags, PyMuPDF digital vs scanned fast-path heuristics, retry chunk/point pre-purge invariance, and concurrent duplicate upload `IntegrityError` race handling.
- [`tests/integration/test_concurrency_stress.py`](../backend/tests/integration/test_concurrency_stress.py): 25-file batch concurrency torture test (10 digital PDFs, 5 notes, 5 duplicate uploads, 3 poison files, and 2 in-flight deletions). Asserts zero deadlocks, zero orphan vectors, and 100% data consistency.


### Ingestion Modes & Performance Benchmark Testing

To validate user-controlled ingestion modes (`prefer_fast_path`), auto-upgrades on duplicate uploads, downgrade guards, engine-aware checkpoint invalidation, transparent Docling failover diagnostics, and $O(1)$ zero-disk-I/O listing latency (< 20ms):

```bash
cd backend
uv run pytest tests/test_ingestion_modes.py tests/test_listing_performance.py tests/test_document_schema.py -v
```

- [`tests/test_ingestion_modes.py`](../backend/tests/test_ingestion_modes.py): Complete routing test matrix for `prefer_fast_path` (`True`, `False`, `None`), duplicate-upload GPU upgrades, concurrency CAS in-flight coalescing, downgrade protections, zero-orphan vector purge, and mocked Docling failover to PyMuPDF.
- [`tests/test_listing_performance.py`](../backend/tests/test_listing_performance.py): Listing performance benchmark verifying 100 mock documents execute `GET /api/v1/documents` in < 20ms with 0 filesystem reads to `data/parsed/`.
- [`tests/test_document_schema.py`](../backend/tests/test_document_schema.py): SQLModel schema, serialization, and idempotent `init_db()` column migration tests.

### RAG Graph & Multi-Turn Conversational Testing

To validate conversational Corrective RAG routing, query rewriting, reranking, grading, and multi-turn citation persistence:

```bash
uv run --directory backend pytest tests/test_rag_graph.py tests/test_api_routes.py -v
```

- [`tests/test_rag_graph.py`](../backend/tests/test_rag_graph.py): Tests the CRAG state graph, hybrid retrieval, fallback notices, and multi-turn `AIMessage.additional_kwargs["citations"]` checkpoint persistence.
- [`tests/test_api_routes.py`](../backend/tests/test_api_routes.py): Integration tests for chat SSE streaming, thread persistence, and multi-turn citation retrieval via `GET /api/v1/chat/threads/{thread_id}`.

### Multi-Domain Knowledge Organization & Federated CRAG Testing

To validate relational taxonomy schemas, local multilingual Laya classification, human-in-the-loop approval state machines, contextual prefixing, federated scatter-gather retrieval, calibrated RRF, and epistemic synthesis:

```bash
cd backend
uv run pytest tests/test_multidomain_schema.py tests/test_laya_classifier_and_approval.py tests/test_multidomain_crag.py -v
```

- [`tests/test_multidomain_schema.py`](../backend/tests/test_multidomain_schema.py): SQLModel relational schema, SQLite idempotent taxonomy seeding (`DomainRegistry`, `SubdomainRegistry`, `DomainLexicon`), and Qdrant domain/subdomain payload indexing and backfill.
- [`tests/test_laya_classifier_and_approval.py`](../backend/tests/test_laya_classifier_and_approval.py): `LayaDomainClassifier` multilingual predictions (EN/VI), confidence scoring, Stage 1b `awaiting_approval` state machine, 3-mode approval routing (`review_all`, `auto_high_confidence`, `auto_all`), zero-starvation semaphore release, and approval REST API endpoints.
- [`tests/test_multidomain_crag.py`](../backend/tests/test_multidomain_crag.py): CRAG pre-analyzer domain classification, polysemy collision detection, federated scatter-gather retrieval across domain partitions, calibrated RRF with 0.5x general discount, soft authority prior tie-breaking ($\alpha=0.08$), and epistemic synthesis with conflict matrix.

### Developer Gotchas & Debugging Lessons

Key operational gotchas distilled from development and testing:

- **Windows Host Python & Application Control Policy**: On Windows workstations where Application Control or AppLocker blocks dynamically compiled DLLs under non-system paths (e.g. pandas C-extensions triggering `ImportError: DLL load failed while importing indexing`), run targeted backend unit tests with `--noconftest` (e.g. `uv run pytest --noconftest <test_path>`) or execute tests inside the Docker container.
- **PowerShell Script Execution Policy (`npm.cmd`)**: When running npm scripts in Windows PowerShell where script execution is disabled (`npm.ps1 cannot be loaded`), invoke `npm.cmd` explicitly (e.g. `npm.cmd run build` or `npm.cmd run dev`).
- **Pydantic Model Subclassing in Docling-Core**: Tokenizers and processors inheriting from `docling_core.transforms.chunker.hierarchical_chunker.BaseTokenizer` inherit from Pydantic `BaseModel` without `extra="allow"`. Custom wrapper parameters (such as `max_tokens` or `model_name`) must be explicitly declared as type annotations on the subclass or filtered out before calling `super().__init__()`.
- **In-Flight Ingestion Cancellation Contract**: Document deletion routes must signal `IngestionOrchestrator.cancel_document(doc_id)` before purging storage. Any in-flight worker pipeline stage checks its cancellation token before proceeding to avoid ghost writes.
- **Local Qdrant In-Memory Payload Indexing**: Local in-process Qdrant (`:memory:`) ignores payload index creation (logging `UserWarning: Payload indexes have no effect in the local Qdrant`). Multi-domain filtering logic should be verified against a live server Qdrant container.
- **React 19 Hook Ordering & Error Boundary**: Never place React hook invocations after conditional returns or in changing order. Drawer panels (e.g. `SourcesDrawer`) must declare hooks unconditionally at top level. The application is wrapped in `<ErrorBoundary>` to isolate runtime render exceptions.
