# Beginner's Guide: Running Second Brain with Docker

Welcome! This comprehensive guide is designed for anyone—from complete beginners who have never touched Docker, Python, or Git, to experienced developers—to get **Second Brain** installed, configured, and running smoothly on your personal computer (Windows, macOS, or Linux).

---

## Table of Contents

1. [What is Second Brain?](#1-what-is-second-brain)
   - [Core Capabilities](#core-capabilities)
   - [How the System Works Under the Hood](#how-the-system-works-under-the-hood)
2. [Hardware & System Requirements](#2-hardware--system-requirements)
   - [Hardware Specifications](#hardware-specifications)
   - [Supported Operating Systems](#supported-operating-systems)
   - [CPU vs NVIDIA GPU (CUDA) Mode Comparison](#cpu-vs-nvidia-gpu-cuda-mode-comparison)
3. [Step 1: Download & Install Required Software](#step-1-download--install-required-software)
   - [1.1 Install Git](#11-install-git)
   - [1.2 Install Docker Desktop](#12-install-docker-desktop)
   - [1.3 (Optional) NVIDIA GPU Setup](#13-optional-nvidia-gpu-setup)
4. [Step 2: Obtain an AI Model API Key](#step-2-obtain-an-ai-model-api-key)
   - [Supported Cloud Providers (OpenAI, Anthropic, DeepSeek)](#supported-cloud-providers)
   - [Local Offline Models (Ollama, LM Studio, vLLM)](#local-offline-models)
5. [Step 3: Download (Clone) the Repository](#step-3-download-clone-the-repository)
6. [Step 4: Prepare the Configuration File (`.env`)](#step-4-prepare-the-configuration-file-env)
   - [4.1 Copy the Template](#41-copy-the-template)
   - [4.2 Configure Your API Keys](#42-configure-your-api-keys)
   - [4.3 File Permissions (Linux Only)](#43-file-permissions-linux-only)
7. [Step 5: Run the App with Docker Compose](#step-5-run-the-app-with-docker-compose)
   - [Mode A: CPU Mode (Default & Simplest)](#mode-a-cpu-mode-default--simplest)
   - [Mode B: NVIDIA GPU Mode (Fastest)](#mode-b-nvidia-gpu-mode-fastest)
   - [Understanding What Docker Does During First Build](#understanding-what-docker-does-during-first-build)
8. [Step 6: Verify Everything is Working](#step-6-verify-everything-is-working)
   - [6.1 Check Container Status (`docker ps`)](#61-check-container-status)
   - [6.2 View Real-Time Live Logs](#62-view-real-time-live-logs)
   - [6.3 Test System Health & Telemetry Endpoints](#63-test-system-health--telemetry-endpoints)
9. [Step 7: Complete User Guide — Three-Pane Workspace](#step-7-complete-user-guide--three-pane-workspace)
   - [7.1 Open the Web Interface](#71-open-the-web-interface)
   - [7.2 Ingestion Modes (`⚡ Auto`, `🚀 GPU Vision`, `📄 Fast CPU`)](#72-ingestion-modes)
   - [7.3 Uploading Documents & Supported Formats](#73-uploading-documents--supported-formats)
   - [7.4 Document Inspector: Markdown, Original PDF & Outline](#74-document-inspector)
   - [7.5 1-Click Upgrade to GPU Vision](#75-1-click-upgrade-to-gpu-vision)
   - [7.6 Intelligent Assistant: CRAG Visual Stepper & Citations](#76-intelligent-assistant)
   - [7.7 Scoped Chat & Multi-Turn Conversation Threads](#77-scoped-chat--multi-turn-conversation-threads)
   - [7.8 Switching AI Models on the Fly](#78-switching-ai-models-on-the-fly)
   - [7.9 Managing Custom Domains, Lexicon & Workspace Settings (`⚙️ Settings`)](#79-managing-custom-domains-lexicon--workspace-settings)
10. [Step 8: Daily Operations & Maintenance](#step-8-daily-operations--maintenance)
    - [Quick Command Reference Table](#quick-command-reference-table)
    - [Zero-Downtime Application Updates](#zero-downtime-application-updates)
    - [Memory & VRAM Management (`reclaim`)](#memory--vram-management)
    - [Data Storage & Persistence Architecture](#data-storage--persistence-architecture)
    - [Complete Backup and Restore Procedures](#complete-backup-and-restore-procedures)
    - [Clean Reset (Start Fresh)](#clean-reset-start-fresh)
11. [Step 9: Troubleshooting & FAQs](#step-9-troubleshooting--faqs)
    - [Q1: Docker daemon is not running](#q1-docker-daemon-is-not-running)
    - [Q2: Ports 8000 or 6333 are already in use](#q2-ports-8000-or-6333-are-already-in-use)
    - [Q3: Missing or Invalid API Key errors](#q3-missing-or-invalid-api-key-errors)
    - [Q4: Windows WSL 2 consumes too much RAM](#q4-windows-wsl-2-consumes-too-much-ram)
    - [Q5: Network privacy & local binding security](#q5-network-privacy--local-binding-security)
    - [Q6: Running 100% Offline with Local Ollama](#q6-running-100-offline-with-local-ollama)
    - [Q7: GPU not detected inside the container](#q7-gpu-not-detected-inside-the-container)
    - [Q8: PDF rendering / worker script issues](#q8-pdf-rendering--worker-script-issues)
    - [Q9: Managing VRAM when co-hosting with Ollama on 8 GB GPU](#q9-managing-vram-when-co-hosting-with-ollama-on-8-gb-gpu)
    - [Q10: Upgrading from an earlier version & legacy backfill](#q10-upgrading-from-an-earlier-version--legacy-backfill)

---

## 1. What is Second Brain?

### Core Capabilities

**Second Brain** is a self-hosted, private knowledge workspace that combines deep document extraction, hybrid vector search, and a conversational Corrective RAG (CRAG) assistant.

- **Private & Local-First**: Your documents, parsed markdown, database records, and vector embeddings reside 100% on your machine. Only the text queries and retrieved context excerpts sent to your chosen LLM ever leave your computer.
- **Multi-Format Ingestion**: Upload PDF, DOCX, XLSX, and PPTX files.
- **Accurate Document Parsing**: Employs Docling v2 + TableFormer + RapidOCR for scanned/complex documents and PyMuPDF for sub-second digital text processing.
- **Hybrid Semantic + Keyword Search**: Indexes your knowledge in Qdrant using dense 1024-dimensional vectors (`BAAI/bge-m3`) combined with keyword sparse vectors (BM25 on CPU or native learned sparse vectors on CUDA) fused with Reciprocal Rank Fusion (RRF).
- **Corrective RAG with Visual Reasoning**: Questions are analyzed through a multi-stage LangGraph workflow: query rewriting, hybrid retrieval, FlashRank multilingual reranking, context grading, and synthesis with verified clickable citations.
- **Interactive Source Attribution**: Every answer includes clickable citation badges (e.g. `[Doc 1, p. 4]`). Clicking a badge jumps the Document Inspector directly to the exact page, paragraph, and heading in your original document.
- **Multi-Model Support**: Switch between OpenAI (`gpt-4o`), Anthropic (`claude-sonnet-5-5`), DeepSeek (`deepseek-v4-pro`), or any local OpenAI-compatible endpoint (like Ollama) with a single click.

---

### How the System Works Under the Hood

The application runs as two orchestrated Docker services communicating over an internal container network:

```
+-----------------------------------------------------------------------------------------+
|                                      YOUR BROWSER                                       |
|                         Three-Pane React 19 + Tailwind v4 UI                            |
|                                (http://127.0.0.1:8000)                                  |
+--------------------------------------------+--------------------------------------------+
                                             |  REST / Server-Sent Events (SSE)
                                             v
+-----------------------------------------------------------------------------------------+
|                              SERVICE 1: second-brain (Port 8000)                        |
|                                                                                         |
|  +---------------------------+  +----------------------------------------------------+  |
|  |     Frontend SPA Mount    |  |               FastAPI Headless Gateway             |  |
|  |     (/app/frontend/dist)  |  |            (/api/v1/documents, /chat, /system)     |  |
|  +---------------------------+  +-------------------------+--------------------------+  |
|                                                           |                             |
|          +------------------------------------------------+--------------------+        |
|          |                                                                     |        |
|          v                                                                     v        |
|  +--------------------------------+                  +-------------------------------+  |
|  |       Ingestion Pipeline       |                  |      Corrective RAG Graph     |  |
|  |  - 5ms Fast-Path Probe         |                  |  - Query Rewriter             |  |
|  |  - Docling v2 / PyMuPDF        |                  |  - Hybrid Retriever           |  |
|  |  - Parent-Child Chunker        |                  |  - FlashRank Reranker         |  |
|  |  - BGE-M3 Dense + Sparse       |                  |  - Context Grader             |  |
|  |  - Adaptive Memory Reclaimer   |                  |  - Synthesizer / Fallback     |  |
|  +---------------+----------------+                  +---------------+---------------+  |
|                  |                                                   |                  |
+------------------|---------------------------------------------------|------------------+
                   |                                                   |
                   v                                                   v
+-------------------------------------------------+   +-----------------------------------+
|            SERVICE 2: qdrant (Port 6333)        |   |       RELATIONAL & CHECKPOINTS    |
|                                                 |   |                                   |
|  - Hybrid Dense (1024-d) + Sparse Vectors       |   |  - SQLite (second_brain.db)       |
|  - Payload Filtering by Document ID             |   |  - LangGraph (chat_history.db)    |
|  - Reciprocal Rank Fusion (RRF)                 |   |  - Stored directly in ./data/db   |
|  - Stored in Docker volume qdrant_storage       |   |                                   |
+-------------------------------------------------+   +-----------------------------------+
```

---

## 2. Hardware & System Requirements

### Hardware Specifications

| Component | Minimum | Recommended (CPU Mode) | Recommended (GPU Mode) |
|---|---|---|---|
| **CPU** | 64-bit Intel / AMD (4 cores) or Apple Silicon (M1+) | 8 cores / 16 threads | 6+ cores Intel / AMD or Apple Silicon |
| **RAM** | 8 GB host RAM | 16 GB host RAM | 16 GB–32 GB host RAM |
| **Storage** | 10 GB free disk space | 20 GB free SSD space | 25 GB free NVMe SSD space |
| **GPU** | None (runs 100% on CPU) | None required | NVIDIA GPU with **4 GB to 8 GB+ VRAM** (RTX 30/40 series, RTX 2000 Ada, A100/H100) |

---

### Supported Operating Systems

- **Windows**: Windows 10 (Build 19041+) or Windows 11 with the WSL 2 (Windows Subsystem for Linux) backend enabled.
- **macOS**: macOS Monterey 12, Ventura 13, Sonoma 14, or Sequoia 15+ (both Apple Silicon M1/M2/M3/M4 and Intel chips supported).
- **Linux**: Ubuntu 22.04 / 24.04, Debian 12, Fedora 39+, Arch Linux, or any modern distribution with Docker Engine v24+.

---

### CPU vs NVIDIA GPU (CUDA) Mode Comparison

| Feature | CPU Mode (Default) | NVIDIA GPU Mode (CUDA) |
|---|---|---|
| **Setup Complexity** | Zero configuration; runs anywhere | Requires NVIDIA Driver + WSL2 / Container Toolkit |
| **Docker Compose Command** | `docker compose up -d --build` | `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build` |
| **Docker Image** | Lightweight `Dockerfile` (~2.1 GB) | `Dockerfile.gpu` (~3.8 GB with CUDA 12.4 + Tensor Cores) |
| **Embedding Generation** | BGE-M3 on CPU + FastEmbed BM25 | Native BGE-M3 `float16` single-pass dense + learned sparse |
| **PDF Ingestion Speed** | 0.4s (digital text) to 25s (scanned) | 0.4s (digital text) to 5s–10s (complex layout / OCR) |
| **VRAM Consumption** | **0 MB VRAM** (pure host CPU) | ~1.2 GB resting; peaks at ~3.5 GB during active layout parsing |
| **Local LLM Co-Hosting** | Coexists via host RAM | **Yes!** After 60s idle, memory reclaims to leave ~6.5 GB VRAM free for Ollama |

---

## Step 1: Download & Install Required Software

You need two primary software tools installed on your computer: **Git** and **Docker Desktop**.

---

### 1.1 Install Git

Git allows you to download and receive updates for Second Brain.

- **Windows**:
  1. Download the official installer from [git-scm.com/download/win](https://git-scm.com/download/win).
  2. Run the `.exe` installer. You can safely accept all default settings by clicking **Next**.
  3. *Quick Alternative (PowerShell)*:
     ```powershell
     winget install --id Git.Git -e --source winget
     ```
- **macOS**:
  1. Open Terminal and run:
     ```bash
     xcode-select --install
     ```
  2. Click **Install** when prompted to install Apple Developer Command Line Tools.
  3. *Alternative (Homebrew)*: `brew install git`
- **Linux (Ubuntu / Debian)**:
  ```bash
  sudo apt update && sudo apt install -y git
  ```

**Verify Git Installation:**
Open your terminal (PowerShell on Windows, Terminal on Mac/Linux) and run:
```bash
git --version
```
*(Expected output: `git version 2.4x.x` or similar)*.

---

### 1.2 Install Docker Desktop

Docker packages Second Brain, Qdrant, and all dependencies into isolated containers.

#### On Windows:
1. **Enable WSL 2 (Windows Subsystem for Linux)**:
   - Open PowerShell as **Administrator** (Right-click Start menu -> *Terminal (Admin)* or *PowerShell (Admin)*).
   - Run:
     ```powershell
     wsl --install
     ```
   - If WSL is already installed, make sure it is updated:
     ```powershell
     wsl --update
     ```
   - Restart your computer if Windows prompts you to restart.
2. **Download Docker Desktop for Windows**:
   - Download the installer from [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/).
   - Run the installer. Ensure the checkbox **"Use WSL 2 instead of Hyper-V (recommended)"** is **checked**.
   - Restart your computer when prompted.
3. **Start Docker Desktop**:
   - Open Docker Desktop from the Start Menu.
   - Accept the terms of service.
   - Wait until the indicator in the bottom-left corner turns **green** with status **"Engine running"**.

#### On macOS:
1. Visit [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/).
2. Select **"Mac with Apple silicon"** (for M1/M2/M3/M4) or **"Mac with Intel chip"**.
3. Open the downloaded `.dmg` file and drag **Docker** into your **Applications** folder.
4. Launch Docker Desktop and wait for the status icon in your menu bar to turn steady green.

#### On Linux (Ubuntu / Debian):
```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Allow running docker without sudo
sudo usermod -aG docker $USER
# Log out and log back in for group changes to take effect
```

**Verify Docker Installation:**
```bash
docker --version
docker compose version
```
*(Expected output: Docker version 24+ and Docker Compose v2+)*.

---

### 1.3 (Optional) NVIDIA GPU Setup

> [!NOTE]
> If you do **not** have an NVIDIA graphics card, or prefer the simplest setup, **skip this section**. Second Brain works completely on CPU!

If your machine has an NVIDIA GPU (RTX series or Quadro/Tesla workstation GPUs) and you want hardware acceleration:

- **Windows**:
  - Install the latest NVIDIA display driver from [nvidia.com/drivers](https://www.nvidia.com/Download/index.aspx).
  - WSL 2 handles CUDA passthrough automatically; you do **not** need to install CUDA Toolkit on Windows!
  - In Docker Desktop -> **Settings (gear icon)** -> **Resources** -> **WSL Integration**, verify that your default WSL distribution is enabled.
- **Linux**:
  - Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
  - Test GPU passthrough inside Docker:
    ```bash
    docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
    ```

---

## Step 2: Obtain an AI Model API Key

Second Brain uses an LLM to rewrite user questions, evaluate context relevance, and generate Markdown answers with citations. You need at least **one** key from the following options:

### Supported Cloud Providers

1. **OpenAI** (Default, recommends `gpt-4o` or `gpt-4o-mini`):
   - Navigate to [platform.openai.com/api-keys](https://platform.openai.com/api-keys).
   - Create a secret key (begins with `sk-proj-...`).
2. **Anthropic** (recommends `claude-sonnet-5-5`):
   - Navigate to [console.anthropic.com](https://console.anthropic.com/).
   - Generate an API key (begins with `sk-ant-...`).
3. **DeepSeek** (recommends `deepseek-v4-pro` or `deepseek-chat`):
   - Navigate to [platform.deepseek.com](https://platform.deepseek.com/).
   - Generate an API key (begins with `sk-...`).

---

### Local Offline Models

If you prefer 100% free, offline AI with zero cloud API keys, you can connect Second Brain directly to **Ollama**, **LM Studio**, or **vLLM** running on your computer. See [Q6: Running 100% Offline with Local Ollama](#q6-running-100-offline-with-local-ollama) for full setup instructions.

---

## Step 3: Download (Clone) the Repository

Open your terminal or PowerShell and run:

1. Navigate to the directory where you store projects:
   ```bash
   cd ~
   ```
2. Clone the repository:
   ```bash
   git clone https://github.com/xuan1250/second_brain.git second-brain
   ```
3. Enter the project folder:
   ```bash
   cd second-brain
   ```

---

## Step 4: Prepare the Configuration File (`.env`)

Inside `second-brain/backend/`, there is a template file named `.env.example`. You must copy it to `.env` and fill in your API key.

### 4.1 Copy the Template

- **Windows PowerShell**:
  ```powershell
  Copy-Item backend\.env.example backend\.env
  ```
- **macOS / Linux**:
  ```bash
  cp backend/.env.example backend/.env
  ```

---

### 4.2 Configure Your API Keys

Open `backend/.env` in any text editor (Notepad, VS Code, nano, or TextEdit):
- **Windows**: `notepad backend\.env`
- **macOS**: `open -e backend/.env`
- **Linux**: `nano backend/.env`

The template looks like this:
```env
# Copy to .env — never commit real keys.
# Values starting with "your-" are treated as unset by src/rag/llm.py.
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
DEEPSEEK_API_KEY=your-deepseek-api-key
```

Replace the placeholder with your actual key. For example, for OpenAI:
```env
OPENAI_API_KEY=sk-proj-abc123xyz456...
ANTHROPIC_API_KEY=your-anthropic-api-key
DEEPSEEK_API_KEY=your-deepseek-api-key
```

> [!IMPORTANT]
> - Do not put quotation marks or spaces around your key.
> - Leave any providers you are not using with the `your-...` prefix (the backend safely ignores them).
> - Save the file and close your editor.

#### (Optional) Multi-Domain Classification & Approval Settings

By default, Second Brain automatically classifies uploaded documents into 4 domains (`Engineering`, `Economics`, `Law`, `General`) using a local multilingual transformer model (`convaiinnovations/laya-multilingual`) running in-process at zero token cost. You can optionally tune classification behavior in `backend/.env`:

```env
# Local multilingual classifier model (default: convaiinnovations/laya-multilingual)
LAYA_MODEL_NAME=convaiinnovations/laya-multilingual

# Whether to fall back to cloud LLM if classification confidence is low (default: true)
INGESTION_ENABLE_LLM_CLASSIFICATION_FALLBACK=true

# Confidence cutoff (0.50 to 0.99) to bypass human review under Auto (High Confidence) mode (default: 0.85)
INGESTION_HIGH_CONFIDENCE_THRESHOLD=0.85
```

---

### 4.3 File Permissions (Linux Only)

If you are running on Linux, pre-create the data storage directories and ensure write permissions for the Docker non-root user (`appuser:10001`):
```bash
mkdir -p data/db data/raw data/parsed data/markdown data/models
chmod -R 777 data
```

---

## Step 5: Run the App with Docker Compose

Choose either **Mode A (CPU Mode)** or **Mode B (NVIDIA GPU Mode)**:

---

### Mode A: CPU Mode (Default & Simplest)

Recommended for all Mac, Windows, and Linux machines without an NVIDIA GPU.

In your terminal (inside the `second-brain` folder), run:
```bash
docker compose up -d --build
```

#### Breakdown of flags:
- `docker compose`: Runs the multi-container configuration defined in `docker-compose.yml`.
- `up`: Builds, creates, and starts the services.
- `-d`: **Detached mode** — runs the containers in the background so your terminal remains free.
- `--build`: Compiles the React frontend and builds the Python backend container from source.

---

### Mode B: NVIDIA GPU Mode (Fastest)

Recommended for machines with an NVIDIA GPU (RTX 2000 Ada, RTX 30/40 series, etc.) with WSL 2 or NVIDIA Container Toolkit.

In your terminal, run:
```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

#### What happens in GPU mode:
- Applies the `docker-compose.gpu.yml` override file.
- Builds [`Dockerfile.gpu`](../Dockerfile.gpu) with CUDA 12.4 (`cu124`) PyTorch wheels and `onnxruntime-gpu==1.20.0`.
- Passes the host GPU into the container with Tensor Core acceleration.
- Embeddings and OCR run up to **5x–10x faster** on complex scanned documents.

---

### Understanding What Docker Does During First Build

The first time you execute `docker compose up -d --build`:
1. **Frontend Builder (`node:24-alpine`)**: Installs locked npm packages and compiles the React 19 + Tailwind v4 single-page app into `/app/frontend/dist`.
2. **Backend Builder (`ghcr.io/astral-sh/uv`)**: Uses Astral `uv` to resolve and install 180+ locked Python packages into an isolated virtual environment (`.venv`).
3. **Runtime Assembly (`python:3.13-slim`)**: Copies the compiled static SPA, virtual environment, and backend source into a hardened, non-root container (`appuser`).
4. **Qdrant Initialization**: Pulls the official `qdrant/qdrant:latest` image and mounts persistent vector storage.
5. **Database Seeding & Vector Backfill**: On startup, SQLite automatically seeds the taxonomy tables (`engineering`, `economics`, `law`, `general`), ensures Qdrant payload indices exist, and safely backfills any pre-existing legacy vectors into the general domain partition.

*First build takes approximately 3 to 5 minutes. Subsequent restarts take under 3 seconds!*

---

## Step 6: Verify Everything is Working

### 6.1 Check Container Status

Run:
```bash
docker compose ps
```
You should see both services running with `Up (healthy)` status:
```text
NAME                         IMAGE                       COMMAND                  SERVICE        STATUS
second-brain-qdrant-1        qdrant/qdrant:latest        "./entrypoint.sh"        qdrant         Up (healthy)
second-brain-second-brain-1  second-brain-second-brain   "uvicorn src.api.mai…"   second-brain   Up (healthy)
```

---

### 6.2 View Real-Time Live Logs

To inspect background activity or troubleshoot ingestion:
```bash
docker compose logs -f second-brain
```
*(Press `Ctrl + C` to exit the log stream at any time; the container will keep running).*

---

### 6.3 Test System Health & Telemetry Endpoints

Open your browser or run curl to test the endpoints:

- **Service Health Check**:
  ```bash
  curl.exe -fsS http://127.0.0.1:8000/api/v1/system/health
  ```
  *Expected JSON response:*
  ```json
  {"status":"ok","database":"ok","qdrant":"ok","collection":"second_brain_knowledge"}
  ```

- **Hardware & Device Stats**:
  ```bash
  curl.exe -fsS http://127.0.0.1:8000/api/v1/system/stats
  ```
  *(Check `embedding.resolved_device`: `"cuda"` or `"cpu"`, total documents, and VRAM availability).*

- **Interactive Swagger Documentation**:
  Visit [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) in your browser to inspect and interact with all REST endpoints.

---

### 6.4 Telemetry & Interactive CRAG Architecture Walkthrough

- **Live Memory & Model Telemetry**:
  ```bash
  curl.exe -fsS http://127.0.0.1:8000/api/v1/system/memory
  ```
  *(Inspects process RAM, active CUDA allocations, worker pool status, and classifier residency).*

- **Explore the Interactive CRAG Presentation**:
  Double-click or open the bundled HTML presentation in your web browser:
  👉 **[`docs/presentations/crag_workflow_explained.html`](presentations/crag_workflow_explained.html)**  
  This rich interactive application visualizes:
  - The complete 7-node LangGraph Corrective RAG engine with animated state transitions.
  - 3-tier storage architecture across SQLite relational records, Qdrant vectors, and local filesystem caches.
  - Exact system prompts used by Laya zero-shot classification and LangGraph generation nodes.

---

## Step 7: Complete User Guide — Three-Pane Workspace

### 7.1 Open the Web Interface

Open your browser and navigate to:
👉 **[http://localhost:8000](http://localhost:8000)** (or **[http://127.0.0.1:8000](http://127.0.0.1:8000)**)

The workspace is organized into three synchronized panes:

```
+------------------------+------------------------------------+--------------------------+
|  KNOWLEDGE LIBRARY     |        DOCUMENT INSPECTOR          |   INTELLIGENT ASSISTANT  |
|  (Left Pane)           |        (Center Pane)               |   (Right Pane)           |
|                        |                                    |                          |
| - Mode Selector:       | - Document Title & Metadata        | - Model Selector         |
|   Auto / GPU / Fast    | - Tabs: Markdown / Original PDF    | - Visual CRAG Stepper    |
| - Upload Dropzone      | - Outline Table of Contents        | - Chat Message Stream    |
| - Document Search      | - Citation Highlight Banner        | - Clickable Citations    |
| - Format & Status Tabs | - Zoom In / Out / Reset            | - Multi-Doc Scope Filter |
| - Engine Badges        | - "Upgrade to GPU Vision" Action   | - Thread History Drawer  |
+------------------------+------------------------------------+--------------------------+
```

---

### 7.2 Ingestion Modes

Before uploading documents, select an Ingestion Mode in the segmented control above the upload zone:

| Mode | Label | Target Engine | VRAM Cost | Speed | When to Choose |
|---|---|---|---|---|---|
| **Auto** | `⚡ Auto (Recommended)` | 5ms Preflight Probe: PyMuPDF for digital text; Docling CUDA for complex layouts | ~1.8 GB if routed to GPU; 0 MB if CPU | 0.4s – 25s/doc | **Default mode for general use.** Balances speed and accuracy automatically. |
| **GPU Vision** | `🚀 GPU Vision` | Docling v2 + TableFormer + RapidOCR on CUDA | ~1.8 GB VRAM | 15s – 30s/doc | Best for complex tables, multi-column layouts, financial reports, and scanned forms. |
| **Fast CPU** | `📄 Fast CPU` | `pymupdf4llm` (pure host CPU) | **0 MB VRAM** | < 1s/doc | Best for plain digital PDFs, books, and when preserving all GPU VRAM for local LLM inference. |

> [!TIP]
> **Transparent Failover Attribution**: If a complex document in `Auto` mode encounters a Docling failure or CUDA memory pressure, it automatically falls back to PyMuPDF CPU and displays an amber badge (`⚠️ Fallback`). Hovering over the badge shows the exact diagnostic reason.

#### Classification Approval Modes

Directly below the engine selector, choose how the human-in-the-loop approval gate operates:

| Mode | Label | Gate Behavior | Recommended For |
|---|---|---|---|
| **Review All** | `🛡️ Review All` | Pauses every uploaded document at `Awaiting Approval` for operator inspection. | **Seeded default.** Sensitive or compliance documents requiring explicit taxonomy confirmation. |
| **Auto High Confidence** | `✨ Auto ≥ 85%` | Automatically approves if prediction confidence $\ge$ threshold ($50\%\text{–}99\%$, configurable in `⚙️ Settings`); pauses at `Awaiting Approval` if below. | Fast automated ingestion while catching uncertain documents. |
| **Auto All** | `⚡ Auto All` | Automatically approves 100% of documents with the model's top prediction without pausing. | Rapid batch ingestion where manual review is not needed. |

#### Human-in-the-Loop Review Cards
When a document pauses at `Awaiting Approval`:
- An amber **"Review Classification"** action card appears directly on the document in the Knowledge Library.
- You can inspect the predicted domain, subdomain, confidence score, and extracted entities.
- Click **"Approve"** to confirm, or change the domain/subdomain with a single click before vectors are indexed!
- Subsequent files in your upload batch continue processing in the background without delay.

---

### 7.3 Uploading Documents & Supported Formats

1. Drag and drop any supported file onto the upload dropzone (or click to browse).
2. **Supported Formats**:
   - **PDF (`.pdf`)**: Digital text, scanned papers, multi-page reports.
   - **Word (`.docx`)**: Headings, lists, formatted tables, and paragraphs.
   - **Excel (`.xlsx`)**: Summarizes sheet schema and extracts 50-row bounded Markdown chunks with repeated column headers so large spreadsheets remain searchable.
   - **PowerPoint (`.pptx`)**: Slides, presentation outlines, and text boxes.
3. **First-Upload Model Download**:
   On the very first upload, the container downloads the multilingual embedding model (`BAAI/bge-m3`, ~2 GB) into `./data/models`. This takes 1–3 minutes on the first run; all subsequent uploads process immediately.
4. **Live Stage Indicators**:
   Watch the document progress through stages in real time:  
   `Queued` ➔ `Parsing` ➔ `Chunking` ➔ `Indexing` ➔ `Ready`.
5. **Client-Side Upload Queue**:
   When dropping multiple files at once, the client-side queue processes files with a concurrency limit of 2, displaying individual progress bars and error isolation without overloading the server.
6. **Ingestion Pause/Resume**:
   Click the **Pause** button in the Knowledge Library toolbar if you need to pause background ingestion temporarily to free up system resources. Click **Resume** to continue.
7. **Domain Classification & Human Approval Gate**:
   Each document is analyzed by the local multilingual classifier (`convaiinnovations/laya-multilingual`) across your active domain taxonomy (seeded with 4 protected system domains: `Engineering`, `Economics`, `Law`, `General` across 13 subdomains, plus any custom domains you create in `⚙️ Settings`). Operators can confirm or reassign domains and subdomains with 1-click in the Library card or Document Inspector review banner.
8. **Domain Facet Filtering (`+N More ▾`)**:
   Use the domain buttons in the Knowledge Library to filter documents by discipline. When you have more than 4 domains, the 3 starter specialized domains and any currently selected custom domain appear inline, while additional custom domains are accessible from the `+N More ▾` dropdown.

---

### 7.4 Document Inspector

Click any document in the Knowledge Library to open it in the Center Pane:

- **Dual-View Tabs**:
  - **Markdown Tab**: Displays the extracted structured Markdown with syntax-highlighted code blocks, formatted tables, and page break markers (`<!-- page: N -->`).
  - **Original PDF Tab**: Renders the authentic source PDF in an embedded canvas viewer with zoom controls (`Zoom In`, `Zoom Out`, `Reset`).
- **Outline TOC**: Click headings in the Outline panel to instantly jump to specific sections.
- **Citation Highlights**: When you click a citation in the Assistant, the Inspector automatically opens the relevant document, highlights the exact paragraph, and displays a citation banner with a **"Clear highlight"** button.

---

### 7.5 1-Click Upgrade to GPU Vision

If a document was parsed with `Fast CPU` mode (or qualified for the fast path under `Auto` mode) and you later realize it contains complex financial tables or charts that need deep OCR:

1. Select the document in the Knowledge Library.
2. Click the **"Upgrade to GPU Vision"** button in the Document Inspector header toolbar.
3. The backend atomically re-parses the document with Docling + TableFormer on CUDA, re-chunks, updates Qdrant vectors, and refreshes the view—retaining the original document ID and references.

---

### 7.6 Intelligent Assistant

In the Right Pane, type a question about your documents and press **Send** (or Enter):

- **Visual CRAG Reasoning Stepper**:
  A live badge bar displays each step of the Corrective RAG pipeline as it executes:
  - `Rewriting query`: Expands your prompt for better retrieval.
  - `Hybrid search`: Queries Qdrant for dense semantic and sparse keyword matches.
  - `Reranking (FlashRank)`: Uses the multilingual `ms-marco-MultiBERT-L-12` model to score and order candidate chunks.
  - `Grading context`: Evaluates whether retrieved chunks contain sufficient evidence.
  - `Synthesizing`: Generates the response with inline citation tags (e.g. `[1]`, `[2]`).
  - *(Or `Replying directly` for greetings/chitchat, or `Generating fallback notice` if no relevant documents match).*
- **Real-Time Token Streaming**: Watch the response stream word-by-word with formatted code snippets, LaTeX math formulas, and tables.
- **Interactive Citation Pills**: Click any citation pill below the answer or inline in the text to inspect the source chunk and page. Citation pills are color-coded by knowledge domain (Engineering, Economics, Law, General) so you immediately know the disciplinary origin of each quote.
- **Domain Routing Bar & Epistemic Conflict Matrix**: The sticky Domain Routing Bar at the top of the Assistant lets you lock queries to a specific domain or leave it on `Auto (All)`. When asking cross-domain questions (e.g. comparing technical vs accounting depreciation), the assistant federates retrieval and synthesizes answers with a side-by-side **Epistemic Conflict Matrix**!

---

### 7.7 Scoped Chat & Multi-Turn Conversation Threads

- **Scoped Document Retrieval**:
  By default, the assistant searches all documents in your library. Use the document selector checkboxes in the Knowledge Library or the filter dropdown in the Assistant to scope your query to specific documents.
- **Thread Management**:
  - Click **New Chat** (`+`) to start a fresh conversation.
  - Open the thread history drawer to switch between previous conversations or delete old threads.
  - All past conversation turns retain their interactive citations even when reloaded days later!

---

### 7.8 Switching AI Models on the Fly

Use the model selector dropdown in the Assistant header to switch between configured providers:
- **OpenAI**: `gpt-4o`, `gpt-4o-mini`, `o1`
- **Anthropic**: `claude-sonnet-5-5`, `claude-haiku-4-5`
- **DeepSeek**: `deepseek-v4-pro`, `deepseek-chat`
- **Local Models**: Custom OpenAI-compatible endpoints (e.g. Ollama `llama3.1`)

*Switching models is instantaneous and requires zero server restarts.*

---

### 7.9 Managing Custom Domains, Lexicon & Workspace Settings

Click the **`⚙️ Settings`** button in the top application bar to open the unified **Workspace Settings & Dynamic Taxonomy** modal:

1. **Tab 1 — Domains & Subdomains**:
   - **Starter System Domains**: The 4 starter domains (`Engineering`, `Economics`, `Law`, `General`) are marked `SYSTEM` (`is_system = true`) to protect core epistemic routing; you can still customize their icons, color tokens, descriptions, classifier hint keywords, and subdomains.
   - **Create & Edit Custom Domains**: Create up to 10 active top-level domains with custom emoji icons, a 10-color theme palette (`cyan`, `emerald`, `amber`, `rose`, `violet`, `blue`, `teal`, `orange`, `fuchsia`, `slate`), epistemic modalities, chunking strategy presets (`Standard Markdown`, `Academic Equations`, `Statutory Clauses`, `Financial Tables`), and bilingual (EN/VI) classifier keywords.
   - **Safe Delete & Reassign**: Deleting a custom domain opens a confirmation dialog to reassign all affected documents, chunks, section overrides, and Qdrant vectors to a chosen target domain (defaulting to `General`), with an optional checkbox to re-index affected documents in the background.
2. **Tab 2 — Polysemy Lexicon**:
   - Browse, search, add, and delete domain-specific definitions and contrastive warnings for polysemic terms (terms whose meaning shifts across disciplines, such as *"consideration"* in Law vs. General, or *"yield"* in Engineering vs. Finance).
3. **Tab 3 — Ingestion & AI Defaults**:
   - **Ingestion & Approval Engine**: Configure workspace-wide defaults for **Parsing Engine** (`Auto`, `GPU / Deep Vision`, `Fast Path`), **Approval Gate** (`Review All`, `Auto-Approve High Conf`, `Auto-Approve All`), and **Confidence Threshold** ($50\%\text{–}99\%$).
   - **API Keys & Live Provider Verification**: Manage and mask API keys for OpenAI, Anthropic, DeepSeek, and OpenAI-Compatible endpoints directly from the UI. Features password reveal toggles, custom Base URL for OpenAI-compatible/vLLM/Ollama proxies, 1-click **"Verify & Save"** with instant 5-second live provider ping, source badges (`UI Key` vs `.env`), and a **"Clear UI Key"** button to revert to `backend/.env`.
   - **Proactive Warnings**: Proactive alert banners immediately highlight missing or invalid keys across Settings Tab 3, the Chat Composer, and the Header Settings button with a 1-click fix link. Saving updates the workspace in real time without restarting Docker.

---

## Step 8: Daily Operations & Maintenance

### Quick Command Reference Table

| Objective | Command |
|---|---|
| **Start Stack (CPU Mode)** | `docker compose up -d` |
| **Start Stack (GPU Mode)** | `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d` |
| **Stop Stack (Pause)** | `docker compose down` |
| **View Live Container Logs** | `docker compose logs -f second-brain` |
| **Restart Application** | `docker compose restart second-brain` |
| **Update Code & Rebuild** | `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build --no-deps second-brain` |
| **Check Live Memory & VRAM** | `curl.exe -fsS http://127.0.0.1:8000/api/v1/system/memory` |
| **Trigger Immediate Memory Reclaim** | `curl.exe -X POST http://127.0.0.1:8000/api/v1/system/memory/reclaim` |
| **Pause Ongoing Ingestion** | `curl.exe -X POST http://127.0.0.1:8000/api/v1/system/ingestion/pause` |
| **Resume Ingestion** | `curl.exe -X POST http://127.0.0.1:8000/api/v1/system/ingestion/resume` |

---

### Zero-Downtime Application Updates

When updating Second Brain to a new version, rebuild only the application service while keeping the Qdrant vector store running continuously:

```bash
# 1. Pull latest code
git pull origin master

# 2. Rebuild and recreate only the second-brain container
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build --no-deps second-brain
```
*Your vector index stays completely uninterrupted and all indexed documents remain available.*

---

### Memory & VRAM Management

Second Brain includes an automated three-tiered memory reclamation engine:
1. **PyTorch CUDA Cache Flushing**: Automatically runs `torch.cuda.empty_cache()` and `gc.collect()` after each file finishes vector indexing.
2. **Adaptive Idle Worker Reaper**: After 60 seconds of inactivity (`INGESTION_WORKER_IDLE_TIMEOUT_SEC`), the spawned Docling/RapidOCR worker process automatically terminates, releasing ~1.8 GB VRAM and ~1.8 GB host RAM.
3. **Manual Reclaim API**: To immediately free memory before launching a local model in Ollama:
   ```bash
   curl.exe -X POST http://127.0.0.1:8000/api/v1/system/memory/reclaim
   ```
   *(Also triggers background compaction and `VACUUM` on `second_brain.db` and `chat_history.db` when no document ingestion is active).*

---

### Data Storage & Persistence Architecture

Your data is safely stored in persistent locations that survive container restarts and rebuilds:

1. **Host Directory (`./data`)**:
   - `data/db/`: SQLite databases (`second_brain.db` for document metadata and `chat_history.db` for LangGraph conversation state). Directly accessible on the host machine.
   - `data/raw/`: Original uploaded files (named by SHA-256 hash).
   - `data/parsed/`: Cached Docling AST representations (`{sha}.json`).
   - `data/markdown/`: Extracted Markdown mirrors (`{sha}.md`).
   - `data/models/`: Downloaded HuggingFace model weights (`bge-m3`, FlashRank).
2. **Named Volume `second-brain_qdrant_storage`**:
   - Contains Qdrant hybrid vector collections, payload indices, and point records (configured with `on_disk=True` for low RAM consumption). Uses a named volume for optimal filesystem performance on Docker Desktop.

> [!NOTE]
> **Windows WSL2 Host Inspection Rule**: If you inspect `data/db/*.db` using tools like DB Browser for SQLite or python scripts on Windows while Docker is running, always open the database files in **read-only mode** (`?mode=ro`). Windows and Linux cannot reliably coordinate SQLite WAL shared-memory (`-shm`) locks when both write simultaneously. If you need to make manual edits on the host, stop the container first (`docker compose stop second-brain`).

---

### Complete Backup and Restore Procedures

#### To create a complete backup:
```bash
# 1. Stop containers gracefully
docker compose down

# 2. Backup the ./data folder (includes SQLite databases, documents, and models)
# Windows PowerShell:
Compress-Archive -Path data -DestinationPath second_brain_data_backup.zip
# macOS / Linux:
tar -czvf second_brain_data_backup.tar.gz data/

# 3. Backup Qdrant Docker volume using a temporary container
docker run --rm -v second-brain_qdrant_storage:/qdrant -v ${PWD}:/backup alpine tar -czf /backup/qdrant_storage.tar.gz -C /qdrant .
```

#### To restore from backup:
```bash
# 1. Restore data directory
tar -xzvf second_brain_data_backup.tar.gz

# 2. Restore Qdrant volume
docker run --rm -v second-brain_qdrant_storage:/qdrant -v ${PWD}:/backup alpine tar -xzf /backup/qdrant_storage.tar.gz -C /qdrant

# 3. Start stack
docker compose up -d
```

---

### Clean Reset (Start Fresh)

If you ever want to wipe all documents, chat threads, and vector collections to start with a blank slate:

```bash
# 1. Stop containers and destroy database volumes
docker compose down -v

# 2. Delete host database and cached document files
# Windows PowerShell:
Remove-Item -Recurse -Force data\db\*, data\raw\*, data\parsed\*, data\markdown\*
# macOS / Linux:
rm -rf data/db/* data/raw/* data/parsed/* data/markdown/*
```
*(Your downloaded AI model weights in `data/models` will be preserved so you don't have to re-download 2 GB of models!)*

---

## Step 9: Troubleshooting & FAQs

### Q1: Docker daemon is not running

**Symptom**: `error during connect: This error may indicate that the docker daemon is not running.`  
**Cause**: Docker Desktop is not started.  
**Solution**: Open Docker Desktop from your Start Menu or Applications folder. Wait until the whale icon in the taskbar or menu bar shows a steady green indicator.

---

### Q2: Ports 8000 or 6333 are already in use

**Symptom**: `Bind for 0.0.0.0:8000 failed: port is already allocated.`  
**Cause**: Another application or previous container is occupying port 8000 (web) or 6333 (Qdrant).  
**Solution**:
- Stop any old containers: `docker stop $(docker ps -q)`
- On Windows, find what process is using port 8000:
  ```powershell
  Get-Process -Id (Get-NetTCPConnection -LocalPort 8000).OwningProcess
  ```
- On macOS / Linux:
  ```bash
  lsof -i :8000
  ```
- Terminate the conflicting process or change the host port mapping in `docker-compose.yml` (e.g. `"127.0.0.1:8080:8000"`).

---

### Q3: Missing or Invalid API Key errors

**Symptom**: Amber warning banner above Chat Composer (`"No valid API key configured"`), rose banner (`"Invalid API key"`), or an amber/rose indicator dot on the Header `⚙️ Settings` button.  
**Cause**: Neither `backend/.env` nor Workspace Settings has a valid API key configured for the selected LLM provider, or the provider returned 401 Unauthorized during live verification.  
**Solution (Method 1 — Recommended: UI 1-Click Fix)**:
1. Click **"Configure API Key →"** directly on the warning banner (or click `⚙️ Settings` and navigate to Tab 3: *Ingestion & AI Defaults*).
2. Enter your real key in the provider's input field.
3. Click **"Verify & Save"**; the system pings the provider within 5 seconds and updates the status to `✓ Verified`.
4. Return to chat immediately—no container restarts required!

**Solution (Method 2 — Environment File)**:
1. Open `backend/.env` on the host with your text editor.
2. Add your real key without quotation marks (e.g. `OPENAI_API_KEY=sk-proj-...`).
3. Restart the service: `docker compose restart second-brain` (or clear the UI key in Settings so the `.env` value takes effect).

---

### Q4: Windows WSL 2 consumes too much RAM

**Symptom**: Vmmem process uses 12 GB+ of RAM on Windows.  
**Cause**: WSL 2 defaults to assigning up to 50% of total host RAM.  
**Solution**:
1. Open PowerShell and edit your global WSL configuration:
   ```powershell
   notepad "$env:USERPROFILE\.wslconfig"
   ```
2. Paste the following configuration:
   ```ini
   [wsl2]
   memory=8GB
   processors=4
   ```
3. Save, close Notepad, and restart WSL:
   ```powershell
   wsl --shutdown
   ```
4. Start Docker Desktop again.

---

### Q5: Network privacy & local binding security

**Question**: Can people on my local Wi-Fi or office network see my Second Brain?  
**Answer**: **No.** In [`docker-compose.yml`](../docker-compose.yml), all ports are explicitly bound to `127.0.0.1`:
```yaml
ports:
  - "127.0.0.1:8000:8000"
  - "127.0.0.1:6333:6333"
```
This restricts access strictly to the local machine where Docker is running. The application does not listen on public interfaces (`0.0.0.0`), keeping your documents completely private.

---

### Q6: Running 100% Offline with Local Ollama

If you want a completely offline, air-gapped Second Brain using local open-source LLMs:

1. Install Ollama from [ollama.com](https://ollama.com).
2. Pull your desired model (e.g. Llama 3.1 8B):
   ```bash
   ollama run llama3.1
   ```
3. Configure Ollama to accept connections from Docker:
   - On Windows: Set environment variable `OLLAMA_HOST=0.0.0.0:11434` in System Properties or run `setx OLLAMA_HOST "0.0.0.0:11434"`.
   - On Linux: Edit `/etc/systemd/system/ollama.service.d/environment.conf` and set `Environment="OLLAMA_HOST=0.0.0.0:11434"`.
4. In `backend/.env`, configure:
   ```env
   LLM_PROVIDER=openai_compatible
   LLM_MODEL_NAME=llama3.1
   LLM_BASE_URL=http://host.docker.internal:11434/v1
   LLM_API_KEY=ollama
   ```
   *(Inside Docker containers, `host.docker.internal` securely routes to your host computer).*
5. Restart Second Brain:
   ```bash
   docker compose restart second-brain
   ```

---

### Q7: GPU not detected inside the container

**Symptom**: `curl.exe http://127.0.0.1:8000/api/v1/system/stats` shows `"resolved_device": "cpu"`.  
**Cause**: Docker was launched without the GPU override, or NVIDIA Container Toolkit is not active.  
**Solution**:
1. Make sure you launched with **both** compose files:
   ```bash
   docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
   ```
2. Verify that `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi` works on your host.
3. On Windows, check Docker Desktop -> Settings -> Resources -> WSL Integration to ensure your Linux distro is checked.

---

### Q8: PDF rendering / worker script issues

**Symptom**: PDF viewer tab displays a loading spinner indefinitely.  
**Cause**: Modern browsers require Web Worker scripts for PDF rendering.  
**Solution**: The application bundles the official PDF.js worker locally (`/assets/pdf.worker.min-*.mjs`). Ensure your browser does not block local scripts with aggressive ad-blocker extensions, or try opening [http://127.0.0.1:8000](http://127.0.0.1:8000) in an incognito/private window.

---

### Q9: Managing VRAM when co-hosting with Ollama on 8 GB GPU

**Question**: Can I run local Ollama models (e.g. `llama3.1:8b`) on an 8 GB VRAM GPU alongside Second Brain?  
**Answer**: **Yes!** Second Brain is engineered with an adaptive VRAM management architecture:
1. **Resting Footprint**: After the warm BGE-M3 embedding model initializes, resting GPU memory is only **~1.2–1.4 GB**, leaving **~6.5 GB free** for local LLMs.
2. **Idle Worker Reaper**: When heavy layout parsing (Docling + RapidOCR) finishes, an automated timer shuts down the worker process after 60s of inactivity, reclaiming ~1.8 GB VRAM.
3. **Immediate On-Demand Reclaim**: Trigger instant memory reclamation anytime before launching a heavy local LLM prompt:
   ```bash
   curl.exe -X POST http://127.0.0.1:8000/api/v1/system/memory/reclaim
   ```
4. **Fast CPU Ingestion**: When chatting with an active local LLM, select `📄 Fast CPU` mode for document uploads. This uses `pymupdf4llm` on the host CPU with **0 MB GPU VRAM**, completely avoiding GPU contention.

---

### Q10: Upgrading from an earlier version & legacy backfill

**Question**: Will updating Second Brain break my previously uploaded documents or Qdrant vectors?  
**Answer**: **No.** Second Brain includes zero-downtime, idempotent schema and vector migrations:
1. **Automatic Schema Migrations**: When the container boots, `init_db()` safely alters tables to add any new columns or relational registries without data loss.
2. **Automatic Legacy Vector Backfill**: Pre-existing vector points lacking `domain` or `is_current` metadata are automatically discovered and backfilled into the `"general"` domain partition on startup (`backfill_qdrant_legacy_payloads`).
3. **Re-Indexing Collection (Optional)**: If you switch sparse vector modes (e.g. from CPU BM25 to CUDA learned sparse) or wish to re-classify your entire library with Laya:
   ```bash
   docker compose exec second-brain python -m src.ingestion.cli /app/data/raw --force
   ```

---

### Q11: `docker compose down -v` vs `docker compose down` & Split Persistence

**Question**: Why did my vector search stop finding chunks even though all my documents still show up in the web UI after running `docker compose down -v`?  
**Cause**: **Dual-Store Architecture Lifecycle**.
- SQLite relational databases (`second_brain.db` and `chat_history.db`) are stored directly on your host machine inside `./data/db/` via a bind mount (`./data:/app/data`).
- Qdrant vector embeddings are stored inside a managed Docker named volume (`second-brain_qdrant_storage`).

When you pass the `-v` flag (`docker compose down -v`), Docker deletes the named volume (`second-brain_qdrant_storage`), but leaves host filesystem directories (`./data/`) completely untouched! As a result, SQLite retains all document metadata, while Qdrant resets to 0 vectors.

**Prevention & Recovery**:
1. **Normal shutdowns**: Always run `docker compose down` (without `-v`). This keeps both SQLite and Qdrant in perfect sync.
2. **Recovering if `-v` was run**:
   - Re-index your stored documents from raw files:
     ```bash
     docker compose exec second-brain python -m src.ingestion.cli /app/data/raw --force
     ```
   - Or restore the Qdrant volume backup using the snippet in [Step 8](#to-restore-from-backup).
3. **Host inspection rule (WSL2/9P)**: While the Docker container is active, open host SQLite databases in **read-only** mode (e.g., `sqlite3 "file:data/db/second_brain.db?mode=ro"` or read-only checkbox in DB Browser) to avoid conflicting with the Linux container's WAL locks across the WSL2 9P filesystem bridge.

---

### Q12: Windows PowerShell command syntax traps (`curl.exe`, splatting `@`, `;` vs `&&`)

**Question**: Why do API test commands fail in PowerShell with errors like `Invoke-WebRequest: parameter cannot be found`, `splatting operator '@'`, or `token '&&' is not a valid statement separator`?  
**Cause**: Windows PowerShell has alias collisions and parses symbols differently from POSIX bash shells.

**Solutions**:
1. **Use `curl.exe`, not `curl`**: In PowerShell, `curl` is an alias for `Invoke-WebRequest`. Always explicitly type `curl.exe`:
   ```powershell
   curl.exe -fsS http://127.0.0.1:8000/api/v1/system/health
   ```
2. **Quote `@` symbol for file payloads**: In PowerShell, `@` triggers variable splatting. Always enclose file payload references in quotes:
   ```powershell
   # Correct:
   curl.exe -fsS -X POST http://127.0.0.1:8000/api/v1/chat/completions -H "Content-Type: application/json" -d '@payload.json'
   ```
3. **Use `;` instead of `&&`**: PowerShell 5.1 does not recognize `&&` as a command separator. Use `;` instead:
   ```powershell
   cd backend; uv run python -m pytest --noconftest
   ```
4. **Escape inner quotes in inline JSON**:
   ```powershell
   curl.exe -fsS -X POST http://127.0.0.1:8000/api/v1/chat/completions -H "Content-Type: application/json" -d "{\"message\":\"hello\",\"domain\":\"general\"}"
   ```
5. **Node / NPM script policy**: If PowerShell blocks script execution (`running scripts is disabled on this system`), invoke npm commands using `npm.cmd` (e.g. `npm.cmd run build`).

---

*Congratulations! You now have a complete, production-grade, private Second Brain running locally with Docker.*
