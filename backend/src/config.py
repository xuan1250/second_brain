"""Typed settings loaded from config.yaml with environment variable overrides."""
from __future__ import annotations

import os
import sys
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = BACKEND_DIR / "config.yaml"


class LLMProviderConfig(BaseModel):
    model_name: str = ""
    base_url: str | None = None


class LLMConfig(BaseModel):
    provider: str = "openai"
    model_name: str = ""
    temperature: float = 0.0
    base_url: str | None = None
    providers: dict[str, LLMProviderConfig] = Field(default_factory=dict)


class EmbeddingConfig(BaseModel):
    provider: str = "local"
    model_name: str = "BAAI/bge-m3"
    dimensions: int = 1024
    device: Literal["auto", "cuda", "cpu"] = "auto"
    fp16: bool = True
    sparse_mode: Literal["auto", "bge_m3", "bm25"] = "auto"
    max_length: int = 8192


class RerankerConfig(BaseModel):
    model_name: str = "ms-marco-MultiBERT-L-12"
    top_n: int = 6
    score_threshold: float = 0.0


class ChunkingConfig(BaseModel):
    child_max_tokens: int = 256
    parent_max_tokens: int = 1024


class IngestionConfig(BaseModel):
    max_workers: int = 1
    max_upload_size_mb: int = 100
    device: Literal["auto", "cuda", "cpu"] = "auto"
    worker_idle_timeout_sec: int = 60
    worker_max_tasks: int = 0
    clear_cuda_cache_after_indexing: bool = True
    laya_model_name: str = "convaiinnovations/laya-multilingual"
    enable_llm_classification_fallback: bool = True
    high_confidence_threshold: float = 0.85


class StorageConfig(BaseModel):
    data_dir: str = "../data"
    db_dir: str = "../data/db"
    frontend_dist_dir: str = "../frontend/dist"


class QdrantConfig(BaseModel):
    url: str = "http://localhost:6333"
    collection: str = "second_brain_knowledge"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    llm: LLMConfig = LLMConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    reranker: RerankerConfig = RerankerConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    ingestion: IngestionConfig = IngestionConfig()
    storage: StorageConfig = StorageConfig()
    qdrant: QdrantConfig = QdrantConfig()

    def _resolve(self, value: str) -> Path:
        p = Path(value)
        return p if p.is_absolute() else (BACKEND_DIR / p).resolve()

    @property
    def data_dir(self) -> Path:
        return self._resolve(self.storage.data_dir)

    @property
    def db_dir(self) -> Path:
        return self._resolve(self.storage.db_dir)

    @property
    def frontend_dist_dir(self) -> Path:
        return self._resolve(self.storage.frontend_dist_dir)

    @property
    def sqlite_url(self) -> str:
        return f"sqlite+aiosqlite:///{(self.db_dir / 'second_brain.db').as_posix()}"

    @property
    def qdrant_url(self) -> str:
        return self.qdrant.url

    @property
    def qdrant_collection(self) -> str:
        return self.qdrant.collection


def load_settings(config_path: Path | str | None = None) -> Settings:
    path = Path(config_path or os.environ.get("CONFIG_PATH") or DEFAULT_CONFIG_PATH)
    raw: dict = {}
    if path.is_file():
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    settings = Settings(**raw)
    env = os.environ

    is_pytest = "PYTEST_CURRENT_TEST" in env or "pytest" in sys.modules
    if is_pytest and config_path is None:
        pytest_tmp = Path(tempfile.gettempdir()) / f"second_brain_pytest_{os.getpid()}"
        if not env.get("DATA_DIR"):
            settings.storage.data_dir = str(pytest_tmp)
        if not env.get("DB_DIR"):
            settings.storage.db_dir = str(pytest_tmp / "db")

    if env.get("QDRANT_URL"):
        settings.qdrant.url = env["QDRANT_URL"]
    if env.get("DATA_DIR"):
        settings.storage.data_dir = env["DATA_DIR"]
    if env.get("DB_DIR"):
        settings.storage.db_dir = env["DB_DIR"]
    if env.get("FRONTEND_DIST_DIR"):
        settings.storage.frontend_dist_dir = env["FRONTEND_DIST_DIR"]
    if env.get("EMBEDDING_DEVICE"):
        settings.embedding.device = env["EMBEDDING_DEVICE"].strip().lower()  # type: ignore[assignment]
    if env.get("EMBEDDING_FP16"):
        settings.embedding.fp16 = env["EMBEDDING_FP16"].strip().lower() in ("1", "true", "yes")
    if env.get("EMBEDDING_SPARSE_MODE"):
        settings.embedding.sparse_mode = env["EMBEDDING_SPARSE_MODE"].strip().lower()  # type: ignore[assignment]
    if env.get("EMBEDDING_MAX_LENGTH"):
        try:
            settings.embedding.max_length = int(env["EMBEDDING_MAX_LENGTH"])
        except ValueError:
            pass
    if env.get("INGESTION_DEVICE"):
        settings.ingestion.device = env["INGESTION_DEVICE"].strip().lower()  # type: ignore[assignment]
    if env.get("INGESTION_WORKER_IDLE_TIMEOUT_SEC"):
        try:
            settings.ingestion.worker_idle_timeout_sec = max(0, int(env["INGESTION_WORKER_IDLE_TIMEOUT_SEC"]))
        except ValueError:
            pass
    if env.get("INGESTION_WORKER_MAX_TASKS"):
        try:
            settings.ingestion.worker_max_tasks = max(0, int(env["INGESTION_WORKER_MAX_TASKS"]))
        except ValueError:
            pass
    if env.get("INGESTION_CLEAR_CUDA_CACHE"):
        settings.ingestion.clear_cuda_cache_after_indexing = env["INGESTION_CLEAR_CUDA_CACHE"].strip().lower() in (
            "1",
            "true",
            "yes",
        )
    if env.get("LAYA_MODEL_NAME"):
        settings.ingestion.laya_model_name = env["LAYA_MODEL_NAME"].strip()
    if env.get("INGESTION_ENABLE_LLM_CLASSIFICATION_FALLBACK"):
        settings.ingestion.enable_llm_classification_fallback = env[
            "INGESTION_ENABLE_LLM_CLASSIFICATION_FALLBACK"
        ].strip().lower() in ("1", "true", "yes")
    if env.get("INGESTION_HIGH_CONFIDENCE_THRESHOLD"):
        try:
            settings.ingestion.high_confidence_threshold = float(env["INGESTION_HIGH_CONFIDENCE_THRESHOLD"])
        except ValueError:
            pass
    # HF_HOME is consumed directly by huggingface libraries from the environment.
    return settings


@lru_cache
def get_settings() -> Settings:
    return load_settings()
