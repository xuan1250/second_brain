# Deployment

The entire system runs as two Docker Compose services defined in [`docker-compose.yml`](../docker-compose.yml): `qdrant` (hybrid vector store) and `second-brain` (FastAPI backend + compiled React SPA).

The application supports two operating modes: **CPU-only (Default)** and **NVIDIA GPU-accelerated (CUDA)**.

---

## 1. Running with CPU (Default Mode)

Recommended for all general setups, macOS, Linux, and Windows environments without an NVIDIA GPU.

### How to launch
```bash
# 1. Configure API keys
cp backend/.env.example backend/.env

# 2. Build and start the stack
docker compose up -d --build

# 3. Verify health
curl -fsS http://127.0.0.1:8000/api/v1/system/health
curl -fsS http://127.0.0.1:6333/readyz
```

### What happens under the hood
- Builds the standard lightweight [`Dockerfile`](../Dockerfile) (~2.1 GB image).
- Embeddings: `BAAI/bge-m3` runs on CPU with PyTorch CPU wheels; sparse vectors are generated via FastEmbed BM25.
- Layout & OCR: Docling runs layout parsing and RapidOCR text recognition via CPU execution (`CPUExecutionProvider`) using OpenMP threads (`OMP_NUM_THREADS=4`).
- Memory footprint: ~2 GB host RAM; zero GPU VRAM required.

---

## 2. Running with NVIDIA GPU (CUDA Acceleration)

Recommended for machines with an NVIDIA GPU (`<= 8 GB` VRAM budget, e.g. RTX 2000 Ada, RTX 30/40 series, or data-center GPUs).

### Prerequisites
- **Windows 11 / 10**: Docker Desktop with the WSL2 backend enabled, and an up-to-date NVIDIA display driver installed on the Windows host (WSL2 handles CUDA passthrough automatically).
- **Linux**: Docker Engine with Docker Compose v2 and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed (`docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi` should succeed).

### How to launch
Use the GPU Compose override [`docker-compose.gpu.yml`](../docker-compose.gpu.yml):

```bash
# 1. Configure API keys
cp backend/.env.example backend/.env

# 2. Build and start with GPU override
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build

# 3. Verify health & GPU detection
curl -fsS http://127.0.0.1:8000/api/v1/system/health
curl -fsS http://127.0.0.1:8000/api/v1/system/stats
```

### What happens under the hood
- Builds [`Dockerfile.gpu`](../Dockerfile.gpu) with CUDA 12.4 (`cu124`) PyTorch wheels and `onnxruntime-gpu==1.20.0`.
- Reserves the host GPU via Compose `deploy.resources.reservations.devices` with capabilities `[gpu]`.
- Embeddings: `BAAI/bge-m3` runs in native single-pass `torch.float16` on CUDA, outputting dense 1024-d vectors + native learned sparse vectors (via `sparse_linear.pt`) in a single forward pass.
- Layout & OCR: Docling Layout runs on `AcceleratorDevice.CUDA`, and RapidOCR runs with `CUDAExecutionProvider` using Tensor Cores.
- Performance: Ingestion is **5x–10x faster** on complex/scanned PDFs compared to CPU mode.

### VRAM footprint and allocation

| Component | Process | Runtime Model | VRAM Usage |
|---|---|---|---|
| Embedder | Main FastAPI process | `BAAI/bge-m3` in `torch.float16` (`max_length: 8192`) + `sparse_linear.pt` (single-pass dense + learned sparse) | ~1.4 GB |
| Parser | Spawned worker process (`max_workers: 1`) | Docling layout + TableFormer on `AcceleratorDevice.CUDA` | ~1.8 GB |
| OCR | Docling worker | RapidOCR (DBNet + SVTR) on `CUDAExecutionProvider` | ~0.3 GB |
| **Peak Total** | | | **~3.5 GB** (comfortably within `8 GB` budget) |

Measured resting VRAM after initialization is ~1.2 GB, peaking at ~3.5 GB during concurrent document layout parsing, OCR, and embedding passes. When CUDA is unavailable or the container runs without `--gpus`, `device: auto` logs a warning and automatically falls back to CPU + FastEmbed BM25 with zero code changes.

### Adaptive memory & VRAM reclaim

The system incorporates a three-tiered adaptive memory reclamation architecture designed to keep GPU VRAM and host RAM bounded while preserving high throughput across multi-file batch uploads:

1. **Post-Indexing PyTorch CUDA Cache Flushing (`clear_cuda_cache_after_indexing: true`)**:
   - Flushes unmapped PyTorch scratchpad tensors (`torch.cuda.empty_cache()` + Python `gc.collect()`) after Stage 3 (Indexing) completes.
   - Coordinated across concurrent uploads via `_active_indexing_jobs`: cache clearing only executes when `_active_indexing_jobs == 0`, avoiding allocator lock contention while another file is mid-embedding.
2. **Adaptive Idle Worker Reaper (`worker_idle_timeout_sec: 60`)**:
   - Keeps the spawned `ProcessPoolExecutor` warm across sequential and concurrent document uploads.
   - Once the system is idle (`active_worker_jobs == 0`) for 60 seconds (configurable via `INGESTION_WORKER_IDLE_TIMEOUT_SEC`), an `asyncio` timer automatically shuts down the worker process, freeing ~1.8 GB VRAM (Docling Layout & RapidOCR models) and ~1.8 GB host RAM.
   - When a new document upload arrives, a fresh worker pool lazily re-spawns with zero downtime.
3. **Batch-Safe Document Recycling (`worker_max_tasks: 0`)**:
   - Configurable document-count limit (`INGESTION_WORKER_MAX_TASKS`). Setting this to `N > 0` recycles the worker pool once the batch is idle after completing `N` documents, preventing mid-document worker restarts between Stage 1 (Parse) and Stage 2 (Chunk).
4. **Co-hosting with Local LLMs (e.g. Ollama)**:
   - On an 8 GB VRAM GPU (such as RTX 2000 Ada or RTX 3060/4060), after ingestion finishes and the idle reaper reclaims the worker pool, VRAM usage returns to the resting baseline of **~1.2–1.4 GB** (the warm `BAAI/bge-m3` embedder).
   - This leaves **~6.5 GB VRAM free**, allowing a quantized 7B or 8B local LLM (e.g. `llama3.1:8b-instruct-q4_K_M` running via Ollama or vLLM) to coexist comfortably on the same GPU without out-of-memory (OOM) errors.
5. **On-Demand Reclamation API**:
   - Operators or automated scripts (e.g., preceding a heavy local LLM generation request) can immediately reclaim memory without waiting for the 60-second timer:
     ```bash
     curl -X POST http://127.0.0.1:8000/api/v1/system/memory/reclaim
     ```
   - If document ingestion is actively in progress, the API defers pool termination (`"deferred": true`) and automatically reclaims the worker the instant the active batch finishes.
   - Live memory telemetry is inspectable at `GET /api/v1/system/memory` and included in `GET /api/v1/system/stats`.

---

## 3. Operator Ingestion Modes & VRAM Profiles

Operators can select ingestion modes per upload batch in the web UI (`UploadDropzone` segmented control) or via the API query parameter `prefer_fast_path`:

| Mode | UI Label | Target Engine | VRAM Profile | Typical Speed | Best Used For |
|---|---|---|---|---|---|
| **Auto** | `⚡ Auto (Recommended)` | Heuristic preflight: `pymupdf4llm` on digital text; `docling` on scanned/heavy layout | ~1.8 GB if routed to GPU; 0 MB if routed to CPU | 0.4s – 25s/doc | Mixed document batches; default balanced mode. |
| **GPU Vision** | `🚀 GPU Vision` | `docling` + TableFormer + RapidOCR on CUDA | **~1.8 GB VRAM** (Docling + OCR worker) | 15s – 30s/doc | Complex tables, multi-column layouts, scanned papers, forms, diagrams. |
| **Fast CPU** | `📄 Fast CPU` | `pymupdf4llm` (CPU only) | **0 MB VRAM** (100% host CPU) | < 1s/doc | Digital text PDFs, novels, simple reports, or when preserving GPU VRAM for active local LLM inference. |

### Guidance on Mode Selection

1. **Default to Auto (`⚡ Auto`)**: For standard multi-document uploads, the 5ms PyMuPDF preflight probe checks character density ($> 120$ chars/page) and image count ($\le 2$ images/page) to select the fastest accurate engine without operator intervention.
2. **Select Fast CPU (`📄 Fast CPU`) During Heavy Local LLM Inference**: If hosting local LLMs (e.g. Ollama 8B models) close to the 8 GB VRAM limit, Fast CPU mode completely eliminates Docling GPU allocations (0 MB VRAM), avoiding CUDA out-of-memory contention.
3. **Select GPU Vision (`🚀 GPU Vision`) for Critical Layout Fidelity**: For documents with financial balance sheets, complex tabular data, or scanned invoices, GPU Vision forces deep TableFormer and OCR analysis.
4. **Post-Upload GPU Upgrades**: If a document was parsed with Fast CPU, operators can click **"Upgrade to GPU Vision"** in the Document Inspector header at any time to re-parse with Docling without losing the document ID or annotations.
5. **Auto Mode Failover Attribution**: If Docling layout parsing encounters an error or CUDA memory pressure in Auto mode, the engine gracefully falls back to PyMuPDF CPU and records the diagnostic cause into `Document.engine_fallback_reason`, visible directly in the UI and database.
6. **Concurrent Duplicate Uploads & Mode Upgrades**: Re-uploading an existing document requesting an upgrade while the file is already actively in-flight (`queued`, `parsing`, `chunking`, `indexing`) triggers an atomic database Compare-and-Swap (CAS) check, safely coalescing the request into the existing in-flight job without duplicate background tasks.

### Classification Approval Modes
Uploads also carry an `approval_mode` parameter controlling the Stage 1b classification gate (when omitted on `POST /api/v1/documents/upload`, it resolves from `WorkspaceSettings.default_approval_mode` configured in **⚙️ Settings**):
- **`🛡️ Review All`** (`review_all`, seeded default): Pauses every document at `awaiting_approval` for operator taxonomy inspection and confirmation.
- **`✨ Auto ≥ X%`** (`auto_high_confidence`): Automatically proceeds to chunking and vector indexing if `confidence >= high_confidence_threshold` (default `0.85`, configurable from $50\%\text{–}99\%$ in **⚙️ Settings**); lower-confidence documents hold at `awaiting_approval`.
- **`⚡ Auto All`** (`auto_all`): Ingests immediately using the predicted taxonomy without human approval.

---

## 4. Switching Between CPU and GPU Modes

### Sparse index compatibility note
- **On CUDA**: BGE-M3 emits native learned sparse vectors (XLM-RoBERTa token vocabulary).
- **On CPU**: FastEmbed BM25 token indices are used.
- Point payloads record `"sparse_mode"`. If switching an existing database between CPU and GPU modes, re-index the collection to prevent degraded hybrid retrieval recall:

```bash
# Re-index an existing database inside the running container
docker compose exec second-brain python -m src.ingestion.cli /app/data/raw --force
```

### Stopping and switching modes
```bash
# To switch from GPU back to CPU:
docker compose -f docker-compose.yml -f docker-compose.gpu.yml down
docker compose up -d

# To switch from CPU to GPU:
docker compose down
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

---

## 5. Verification and Troubleshooting

| Check | Command | Expected Result |
|---|---|---|
| **Service Health** | `curl -fsS http://127.0.0.1:8000/api/v1/system/health` | `{"status":"ok","database":"ok","qdrant":"ok",...}` |
| **Active Device** | `curl -fsS http://127.0.0.1:8000/api/v1/system/stats` | `"resolved_device":"cuda"` (GPU mode) or `"cpu"` (CPU mode) |
| **Qdrant Ready** | `curl -fsS http://127.0.0.1:6333/readyz` | `all shards are ready` |
| **Container Logs** | `docker compose logs -f second-brain` | Clean startup logs from Uvicorn on port 8000 |
| **CUDA inside container** | `docker compose exec second-brain python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"` | `True <Your GPU Name>` (in GPU mode) |
| **ONNX Providers** | `docker compose exec second-brain python -c "import onnxruntime as ort; print(ort.get_available_providers())"` | Includes `'CUDAExecutionProvider'` |
| **RapidOCR Detection** | `docker compose exec second-brain python -c "from rapidocr.inference_engine.onnxruntime.provider_config import ProviderConfig; pc = ProviderConfig({'use_cuda': True}); print('CUDA available:', pc.is_cuda_available())"` | `CUDA available: True` |
| **Memory Telemetry** | `curl -fsS http://127.0.0.1:8000/api/v1/system/memory` | `{"process_rss_mb":...,"cuda_available":true,"worker_pool_active":...}` |
| **Memory Reclaim** | `curl -fsS -X POST http://127.0.0.1:8000/api/v1/system/memory/reclaim` | `{"status":"ok","worker_pool_reclaimed":...,"cuda_cache_cleared":...}` (also runs SQLite & checkpointer compaction + `VACUUM` when idle) |
| **Pause Ingestion** | `curl -fsS -X POST http://127.0.0.1:8000/api/v1/system/ingestion/pause` | `{"status":"ok","ingestion_paused":true}` |
| **Resume Ingestion** | `curl -fsS -X POST http://127.0.0.1:8000/api/v1/system/ingestion/resume` | `{"status":"ok","ingestion_paused":false}` |
| **Document Retry** | `curl -fsS -X POST "http://127.0.0.1:8000/api/v1/documents/<doc_id>/retry"` | `{"id":"...","status":"queued",...}` |
| **Classifier Telemetry** | `curl -fsS http://127.0.0.1:8000/api/v1/system/stats` | `"classifier":{"model_name":"convaiinnovations/laya-multilingual",...}` |
| **Classifier Residency** | `curl -fsS http://127.0.0.1:8000/api/v1/system/memory` | Includes `"classifier_resident": true/false` |
| **Approve Classification** | `curl -fsS -X POST "http://127.0.0.1:8000/api/v1/documents/<doc_id>/approve-classification"` | `{"id":"...","status":"chunking",...}` |
| **Update Taxonomy** | `curl -fsS -X PATCH -H "Content-Type: application/json" -d '{"domain":"law","subdomain":"compliance"}' "http://127.0.0.1:8000/api/v1/documents/<doc_id>/metadata"` | Updated document metadata |
| **Batch Approve** | `curl -fsS -X POST -H "Content-Type: application/json" -d '{"document_ids":["<id1>","<id2>"]}' "http://127.0.0.1:8000/api/v1/documents/approve-batch"` | `{"approved_count":2,"failed_count":0,...}` |
| **Domain Taxonomy** | `curl -fsS http://127.0.0.1:8000/api/v1/domains` | Active domains (`engineering`, `economics`, `law`, `general`, plus any custom domains), subdomains, and document counts |
| **Workspace Settings** | `curl -fsS http://127.0.0.1:8000/api/v1/settings` | `{"id":"default","default_upload_mode":...,"default_approval_mode":...,"high_confidence_threshold":...}` |
| **Chat Stream (SSE)** | `curl.exe -fsS -N -X POST "http://127.0.0.1:8000/api/v1/chat/completions" -H "Content-Type: application/json" -d "{\"message\":\"hello\",\"domain\":\"general\"}"` | Stream of Server-Sent Events (`event: token`, `event: agent_thought`, `data: ...`). Note: `POST /api/v1/chat` returns 405 Method Not Allowed; the active route is `/api/v1/chat/completions` |
| **Legacy Vector Backfill** | `docker compose logs second-brain \| grep -i "backfill"` | Logs indicating `Backfilled legacy points in second_brain_knowledge with general domain` on boot |
| **CRAG Architecture Presentation** | Open `docs/presentations/crag_workflow_explained.html` in browser | Interactive visualization of LangGraph nodes, epistemic triad, 3-tier DBs, and system prompts |

> [!TIP]
> **Windows PowerShell CLI Invocation Rules**:
> - Always run `curl.exe` instead of `curl` (the bare `curl` alias in PowerShell invokes `Invoke-WebRequest` which fails on `-fsS` or `-d`).
> - When passing a JSON payload file with `-d`, quote the `@` prefix (`-d '@payload.json'`) to avoid PowerShell's variable splatting operator error.
> - Chain commands with `;` rather than `&&` (PowerShell 5.1 rejects `&&` as an invalid statement separator).
> - Escape nested quotes in inline JSON strings (`-d "{\"key\":\"value\"}"`) or pass through a temporary `.json` file to avoid parameter truncation.


## Exposure model

Both services publish on **`127.0.0.1` only** and the API has **no authentication**. This is deliberate for single-user local use (see [decision ledger](system-architecture.md#decision-ledger)). Do not change the port bindings to `0.0.0.0` or put the app behind a public proxy without first adding authentication — anyone who can reach port 8000 can upload and delete documents.

## Image build

The `Dockerfile` stages are commented inline; the design choices that are not obvious from it:

- Base images are pinned by digest for reproducible builds; bump them deliberately.
- Python deps are installed in two `uv sync --frozen` phases (third-party first, then the project with `--no-editable`) so source edits don't invalidate the dependency layer, and the runtime image only needs `.venv` + `src`.
- The runtime runs as non-root `appuser` (UID/GID `10001`).

## Storage

| Mount | Type | Holds | Why this type |
|---|---|---|---|
| `second-brain_qdrant_storage` (`qdrant_storage` → `/qdrant/storage`) | named volume | Vector indexes (`second_brain_knowledge`) with `on_disk=True` for vectors and HNSW | Qdrant storage on Docker Desktop bind mounts (Windows/macOS) risks corruption; on-disk mapping keeps RAM usage low |
| `./data` → `/app/data` | bind mount | `data/db/` (`second_brain.db`, `chat_history.db`), `raw/`, `parsed/`, `markdown/`, `models/` caches | SQLite databases live directly on host `./data/db` with zero volume shadowing; model caches and parsed ASTs survive rebuilds |

`backend/config.yaml` and `backend/.env` are single-file bind mounts (`:ro`); always ensure `backend/.env` exists as a regular file (`cp backend/.env.example backend/.env`) before running `docker compose up`, otherwise Docker Engine auto-creates a directory at `backend/.env` and aborts container startup (`ExitCode 127: not a directory`). While initial credentials can be seeded in `backend/.env`, LLM API keys can also be configured, masked, live-verified, or cleared directly from the Web UI under **Settings → Ingestion & AI Defaults** (`/api/v1/settings` and `/api/v1/settings/verify-key`). UI keys override `.env` live in SQLite (`./data/db/second_brain.db`) with automatic `.env` fallback and zero container restarts required. Workspace-level defaults (`default_upload_mode`, `default_approval_mode`, `high_confidence_threshold`, `default_llm_provider`, `default_llm_model`), custom domains, subdomains, and polysemy lexicon entries likewise apply live without restarts.

### WSL2/9P WAL Guidelines
When running Docker Desktop on Windows under the WSL2 backend, host Windows filesystem paths (`d:\4. Projects\second-brain\data\db\`) are mounted into the Linux container via the 9P network protocol. SQLite WAL mode uses shared memory (`-shm`) and lock protocols that do not coordinate reliably across the Windows/Linux OS boundary when both write simultaneously.
- **Read-Only Host Access**: While the `second-brain` container is running, host tools (such as DB Browser for SQLite, SQLite CLI, or python inspection scripts) must open databases in read-only mode (`?mode=ro` / URI `sqlite:///.../second_brain.db?mode=ro`) or with immutable mode.
- **Host Writes / Offline Migrations**: If running manual migrations or batch updates directly from the host Windows command line, stop the container first (`docker compose stop second-brain`) to ensure all WAL transactions are fully flushed to disk.

**Linux hosts:** if `./data` is root-owned, give it to UID `10001` (`sudo chown -R 10001:10001 ./data`) or set `user:` on the `second-brain` service.

## First run and model caches

- `BAAI/bge-m3` (~2 GB) and the FastEmbed BM25 model download into `./data/models` on first use, so the first ingestion/query is slow.
- The FlashRank reranker model is **not** downloaded unless `FLASHRANK_ALLOW_DOWNLOAD=true` is set (e.g. in `backend/.env`) or the model already exists in `./data/models/flashrank`. Without it, reranking silently falls back to a lexical heuristic — see [development.md](development.md#configuration).

## Backup and restore

[`docker-compose.yml`](../docker-compose.yml) pins `name: second-brain`, with Qdrant vector data stored in `second-brain_qdrant_storage` and SQLite databases stored directly in `./data/db/` on the host:

```bash
# 1. Back up Qdrant named volume
docker run --rm -v second-brain_qdrant_storage:/volume:ro -v "$(pwd):/backup" \
  debian:bookworm-slim tar czf /backup/qdrant_storage.tar.gz -C /volume .

# 2. Back up SQLite databases and data files directly from host ./data
tar czf data_backup.tar.gz data/db data/raw data/parsed data/markdown

# Restore — stop the stack first
docker compose stop second-brain qdrant
docker run --rm -v second-brain_qdrant_storage:/volume -v "$(pwd):/backup" \
  debian:bookworm-slim bash -c "rm -rf /volume/* && tar xzf /backup/qdrant_storage.tar.gz -C /volume"
tar xzf data_backup.tar.gz
docker compose start qdrant second-brain
```

Back up `./data` with any file-level tool. Restore SQLite and Qdrant together: SQLite rows reference Qdrant points by `doc_id`, so restoring only one side leaves documents without vectors (or orphaned vectors).
