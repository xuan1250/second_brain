"""Relational models: documents, chunk->Qdrant point mappings, jobs, chat threads, and domain taxonomy."""
import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import Column, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlmodel import Field, Relationship, SQLModel


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return uuid.uuid4().hex


class DomainRegistry(SQLModel, table=True):
    __tablename__ = "domain_registry"

    id: str = Field(default_factory=_uuid, primary_key=True)
    slug: str = Field(index=True, unique=True)  # "engineering", "economics", "law", "general"
    name: str  # "Engineering & Technology"
    epistemic_type: str  # "empirical", "evaluative", "normative", "general"
    authority_priority: int = Field(default=1)  # 1 (Highest) to 5 (Informational)
    description: str = Field(default="")
    icon: str = Field(default="📁")
    color_token: str = Field(default="indigo")
    chunking_strategy: str = Field(default="standard_markdown")
    classifier_keywords: Optional[str] = Field(default=None)
    is_system: bool = Field(default=False)
    is_active: bool = Field(default=True)

    subdomains: list["SubdomainRegistry"] = Relationship(
        back_populates="domain",
        sa_relationship_kwargs={"cascade": "all, delete-orphan", "passive_deletes": True},
    )
    lexicon_entries: list["DomainLexicon"] = Relationship(
        back_populates="domain",
        sa_relationship_kwargs={"cascade": "all, delete-orphan", "passive_deletes": True},
    )


class SubdomainRegistry(SQLModel, table=True):
    __tablename__ = "subdomain_registry"

    id: str = Field(default_factory=_uuid, primary_key=True)
    domain_id: str = Field(
        sa_column=Column(String, ForeignKey("domain_registry.id", ondelete="CASCADE"), nullable=False, index=True)
    )
    slug: str = Field(index=True, unique=True)  # "eng.electrical", "econ.accounting", "law.statute"
    name: str  # "Electrical & Power Systems"
    standard_bodies: Optional[str] = Field(default=None)  # "IEEE, IEC, NFPA"

    domain: Optional[DomainRegistry] = Relationship(back_populates="subdomains")


class DomainLexicon(SQLModel, table=True):
    """Controlled vocabulary mapping polysemic terms to domain-specific definitions."""
    __tablename__ = "domain_lexicon"
    __table_args__ = (UniqueConstraint("term", "domain_id", name="uq_term_domain"),)

    id: str = Field(default_factory=_uuid, primary_key=True)
    term: str = Field(index=True)  # e.g. "liability", "depreciation", "current"
    domain_id: str = Field(
        sa_column=Column(String, ForeignKey("domain_registry.id", ondelete="CASCADE"), nullable=False, index=True)
    )
    specific_sense: str = Field(sa_column=Column(Text, nullable=False))
    contrastive_warning: str = Field(sa_column=Column(Text, nullable=False))

    domain: Optional[DomainRegistry] = Relationship(back_populates="lexicon_entries")


class CrossDomainEntity(SQLModel, table=True):
    """Knowledge Graph bridge linking physical equipment to financial assets and legal permits."""
    __tablename__ = "cross_domain_entities"

    id: str = Field(default_factory=_uuid, primary_key=True)
    canonical_name: str = Field(index=True, unique=True)  # e.g. "Power Transformer T-101"
    entity_type: str = Field(index=True)  # "equipment", "asset", "regulated_facility"
    description: Optional[str] = Field(default=None)

    mappings: list["EntityDomainMapping"] = Relationship(
        back_populates="entity",
        sa_relationship_kwargs={"cascade": "all, delete-orphan", "passive_deletes": True},
    )


class EntityDomainMapping(SQLModel, table=True):
    """Associates an entity with domain-specific chunks and perspectives."""
    __tablename__ = "entity_domain_mappings"

    id: str = Field(default_factory=_uuid, primary_key=True)
    entity_id: str = Field(
        sa_column=Column(String, ForeignKey("cross_domain_entities.id", ondelete="CASCADE"), nullable=False, index=True)
    )
    chunk_id: str = Field(
        sa_column=Column(String, ForeignKey("document_chunks.id", ondelete="CASCADE"), nullable=False, index=True)
    )
    domain_id: str = Field(
        sa_column=Column(String, ForeignKey("domain_registry.id", ondelete="CASCADE"), nullable=False, index=True)
    )
    role_in_domain: str  # "physical_spec", "capital_asset_entry", "environmental_permit"

    entity: Optional[CrossDomainEntity] = Relationship(back_populates="mappings")


class Document(SQLModel, table=True):
    __tablename__ = "documents"

    id: str = Field(default_factory=_uuid, primary_key=True)
    filename: str
    file_type: str
    sha256: str = Field(index=True, unique=True)
    size_bytes: int = 0
    page_count: Optional[int] = None
    chunk_count: int = 0
    status: str = "queued"  # queued | parsing | classifying | awaiting_approval | chunking | indexing | ready | failed
    error: Optional[str] = None
    parsing_engine: Optional[str] = Field(default=None)
    engine_fallback_reason: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    # Multi-domain categorization & authority metadata
    primary_domain_id: Optional[str] = Field(
        default=None,
        sa_column=Column(String, ForeignKey("domain_registry.id", ondelete="SET NULL"), nullable=True, index=True),
    )
    primary_subdomain_id: Optional[str] = Field(
        default=None,
        sa_column=Column(String, ForeignKey("subdomain_registry.id", ondelete="SET NULL"), nullable=True, index=True),
    )
    jurisdiction: Optional[str] = Field(default=None)
    authority_tier: int = Field(default=3)  # 1 (Statutes/Standards) to 5 (Informal)
    effective_year: Optional[int] = Field(default=None)
    is_current: bool = Field(default=True)
    superseded_by_doc_id: Optional[str] = Field(default=None)
    classification_result: Optional[str] = Field(default=None)  # JSON RecommendedDomainProfile

    @property
    def error_message(self) -> Optional[str]:
        return self.error

    @error_message.setter
    def error_message(self, value: Optional[str]) -> None:
        self.error = value

    chunks: list["DocumentChunk"] = Relationship(
        back_populates="document",
        sa_relationship_kwargs={"cascade": "all, delete-orphan", "passive_deletes": True},
    )
    jobs: list["IngestionJob"] = Relationship(
        back_populates="document",
        sa_relationship_kwargs={"cascade": "all, delete-orphan", "passive_deletes": True},
    )


class DocumentChunk(SQLModel, table=True):
    __tablename__ = "document_chunks"

    id: str = Field(default_factory=_uuid, primary_key=True)
    doc_id: str = Field(
        sa_column=Column(String, ForeignKey("documents.id", ondelete="CASCADE"), index=True, nullable=False)
    )
    point_id: str = Field(index=True)  # Qdrant point UUID
    parent_id: Optional[str] = None
    chunk_index: int = 0
    heading_path: str = ""
    page_numbers: Optional[str] = None  # comma-separated, optional
    token_count: int = 0
    text: str = ""
    parent_content: str = ""

    # Multi-domain partitioning & contextual retrieval
    domain_key: Optional[str] = Field(default=None, index=True)
    subdomain_key: Optional[str] = Field(default=None, index=True)
    epistemic_modality: Optional[str] = Field(default=None)  # empirical | evaluative | normative | general
    contextual_prefix: Optional[str] = Field(default=None)

    document: Optional[Document] = Relationship(back_populates="chunks")


class IngestionJob(SQLModel, table=True):
    __tablename__ = "ingestion_jobs"

    id: str = Field(default_factory=_uuid, primary_key=True)
    doc_id: str = Field(
        sa_column=Column(String, ForeignKey("documents.id", ondelete="CASCADE"), index=True, nullable=False)
    )
    status: str = "queued"  # queued | parsing | classifying | awaiting_approval | chunking | indexing | ready | failed
    stage: Optional[str] = None
    progress: float = 0.0
    duration_ms: Optional[int] = None
    stage_timings: Optional[str] = None  # JSON dict of stage -> duration_ms
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    document: Optional[Document] = Relationship(back_populates="jobs")


class ChatThread(SQLModel, table=True):
    __tablename__ = "chat_threads"

    thread_id: str = Field(default_factory=_uuid, primary_key=True)
    title: str = "New chat"
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


class WorkspaceSettings(SQLModel, table=True):
    """Singleton configuration table for workspace-wide upload, approval, and AI defaults."""
    __tablename__ = "workspace_settings"

    id: str = Field(default="default", primary_key=True)
    default_upload_mode: str = Field(default="auto")  # "auto" | "gpu" | "fast"
    default_approval_mode: str = Field(default="review_all")  # "review_all" | "auto_high_confidence" | "auto_all"
    high_confidence_threshold: float = Field(default=0.85)
    default_llm_provider: Optional[str] = Field(default=None)
    default_llm_model: Optional[str] = Field(default=None)
    llm_api_keys: Optional[str] = Field(default=None)
    llm_key_validations: Optional[str] = Field(default=None)
    llm_compat_base_url: Optional[str] = Field(default=None)
    updated_at: datetime = Field(default_factory=_now)

