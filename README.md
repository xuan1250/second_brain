# Second Brain — Local-First Knowledge Repository & Corrective RAG Assistant

A self-hosted **Second Brain**: upload PDF, DOCX, XLSX, and PPTX documents, have them converted to structured Markdown and indexed for hybrid (semantic + keyword) search, then ask multi-turn questions and get answers with clickable citations that open the exact source document and page.

- **Ingestion & Dynamic Classification** — User-configurable multi-domain knowledge partitioning (seeded with 4 protected system domains: `Engineering`, `Economics`, `Law`, `General` across 13 starter subdomains, extensible up to 10 active domains with custom icons, 10-color theme tokens, domain-specific chunking presets, and custom subdomains), local multilingual classification (`convaiinnovations/laya-multilingual`) with 3-state human-in-the-loop approval routing (`🛡️ Review All`, `✨ Auto High Confidence`, `⚡ Auto All`), zero worker pool starvation, and contextual chunk prefixing (24–48 token header within 256 clean token budget). User-controlled ingestion modes (`⚡ Auto`, `🚀 GPU Vision`, `📄 Fast CPU`) with 5ms adaptive fast-path preflight probe (`pymupdf4llm` for digital text vs Docling v2 + TableFormer for scanned/complex layouts), duplicate-upload auto-upgrades with atomic CAS coalescing, transparent failover attribution, self-healing subprocess recovery, stage checkpointing, and compensating Qdrant rollbacks → parent-child chunks with section breadcrumbs → Qdrant hybrid vectors (`BAAI/bge-m3` dense + learned sparse on GPU / BM25 on CPU).
- **Federated CRAG Chat** — a LangGraph Corrective RAG graph with dynamic catalog-driven domain pre-analyzer, user-editable bilingual polysemy lexicon (`DomainLexicon`) in SQLite, federated scatter-gather retrieval across domain partitions with calibrated RRF (0.5x general fallback discount), soft authority prior tie-breaking ($\alpha=0.08$), and bilingual (EN/VI) epistemic synthesis with structured conflict matrices; streamed over SSE with persistent multi-turn thread history and domain-colored interactive citations preserved across turns.
- **LLMs & Unified Workspace Settings** — OpenAI, Anthropic, DeepSeek, or any OpenAI-compatible endpoint; configurable globally in **⚙️ Settings** (`WorkspaceSettings` in SQLite) or overridden per request from the chat header.
- **Workspace** — React 19 + Vite + Tailwind CSS v4 three-pane UI: library with overflow-safe domain facet bar (`+N More ▾`), Stage 1b approval cards & review banners, segmented 3-way mode & approval selectors synced live with Workspace Settings, engine tags (`Docling GPU`, `Fast CPU`, `⚠️ Fallback`), bounded client upload queue (concurrency = 2), per-file progress, one-click failure retries, and ingestion pause/resume controls; document inspector with section-level domain overrides and 1-click "Upgrade to GPU Vision" action; assistant with sticky overflow-safe domain routing bar and domain-colored citation pills; and a unified 3-tab **⚙️ Settings Modal** (`Domains & Subdomains` with safe target-domain reassignment & optional background re-indexing, `Polysemy Lexicon`, and `Ingestion & AI Defaults`).
- **Local-first** — single user, bound to `127.0.0.1`, no auth; only LLM calls leave the machine.

## Quick start

Requires Docker with Compose v2 and at least one LLM API key.

1. **Configure credentials:**
   ```bash
   cp backend/.env.example backend/.env    # add your API key(s)
   ```

2. **Choose your runtime mode:**

   - **Mode A: CPU (Default)** — runs anywhere (Mac, Windows, Linux) without GPU requirements:
     ```bash
     docker compose up -d --build
     ```

   - **Mode B: NVIDIA GPU (CUDA-accelerated)** — for NVIDIA GPUs (RTX 2000 Ada, RTX 30/40 series) with WSL2 or NVIDIA Container Toolkit:
     ```bash
     docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
     ```

3. **Verify running status & memory telemetry:**
   - App UI: http://127.0.0.1:8000
   - API Docs: http://127.0.0.1:8000/docs
   - Check device mode:
     ```bash
     curl -fsS http://127.0.0.1:8000/api/v1/system/stats
     ```
     (Inspect `resolved_device`: `"cuda"` or `"cpu"`).
   - Check live memory & VRAM telemetry:
     ```bash
     curl -fsS http://127.0.0.1:8000/api/v1/system/memory
     ```

The first ingestion downloads ~2 GB of embedding models into `./data/models`. See [docs/deployment.md](docs/deployment.md) for detailed VRAM breakdown and hardware setup.

## Documentation

| Read | For |
|---|---|
| [docs/system-architecture.md](docs/system-architecture.md) | How the pieces fit, invariants, and why each technology was chosen |
| [docs/development.md](docs/development.md) | Running locally with `uv` + Vite, configuration and env vars, CLI batch ingestion, tests |
| [docs/deployment.md](docs/deployment.md) | Docker topology, storage volumes, exposure model, backup/restore |
| [docs/docker-beginner-guide.md](docs/docker-beginner-guide.md) | Step-by-step beginner guide: install software, prepare environment, and launch with Docker |
| [docs/presentations/crag_workflow_explained.html](docs/presentations/crag_workflow_explained.html) | Interactive visual presentation of the 7-node CRAG engine, prompt flows, and 3-tier databases |
| [docs/README.md](docs/README.md) | Full documentation index |

## Repository layout

- [`backend/`](backend/) — FastAPI app, ingestion pipeline, CRAG engine, tests (Python 3.13, managed with `uv`)
- [`frontend/`](frontend/) — React 19 + Vite + Tailwind CSS v4 SPA
- [`docs/`](docs/) — Architecture, beginner setup guide, deployment, development, and interactive CRAG presentation
- `data/` — local documents, SQLite databases (`data/db/`), parsed caches, and model caches (git-ignored)
