# Documentation Index

Start here. Each document owns one information role; source and tests own behavior.

| Document | Owns |
|---|---|
| [system-architecture.md](system-architecture.md) | Component boundaries, data flow, dual CPU/GPU compute model, and the decision ledger (why things are the way they are) |
| [development.md](development.md) | Local CPU and GPU setup with `uv` + Vite, configuration precedence, CLI ingestion, and test tiers |
| [deployment.md](deployment.md) | User guides for running with CPU vs NVIDIA GPU (CUDA), VRAM budgeting, Docker Compose topology, backup/restore, and the Windows Desktop `.exe` installer |
| [docker-beginner-guide.md](docker-beginner-guide.md) | Step-by-step guide for beginners: download software, prepare files, and run with Docker (PDF export: [user-guide.pdf](user-guide.pdf)) |
| [ai-onboarding-copilot.md](ai-onboarding-copilot.md) | Self-contained instruction prompt to upload into Claude/ChatGPT/Gemini for interactive, step-by-step setup guidance |
| [presentations/crag_workflow_explained.html](presentations/crag_workflow_explained.html) | Interactive animated presentation explaining the 7-node LangGraph Corrective RAG (CRAG) engine, activation modes, 3-tier databases, and state mutations |
| [assets/langgraph-crag-workflow.svg](assets/langgraph-crag-workflow.svg) | High-resolution SVG architecture diagram of the 7-node LangGraph CRAG workflow with all sub-modes |
| [code-review-report.md](code-review-report.md) | Full codebase comprehensive audit: findings, verification evidence, edge case evaluation, and recommendations |
| [Local engineering PDF extraction research](research/2026-11-15-local-engineering-pdf-extraction.md) | Proposed local model routes for engineering PDFs on an 8GB GPU, evidence limits, integration options, and evaluation criteria; not implemented architecture |
| [HuggingFace PDF-to-Markdown VLMs research](research/2026-11-25-huggingface-pdf-to-markdown-vlms.md) | Survey and OmniDocBench comparison of compact specialized VLMs and parallel parsing pipelines (OXR-1.0, DeepSeek-OCR 2, MinerU 2.5-Pro, HPD-Parsing) for PDF-to-Markdown conversion |


