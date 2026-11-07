"""LangGraph state schema and citation models for the Corrective RAG engine."""
from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class Citation(TypedDict, total=False):
    """Structured citation metadata linking an inline [N] marker to a document chunk.

    Field names remain snake_case end-to-end across LangGraph state, FastAPI JSON,
    and the React frontend.
    """

    citation_index: int
    doc_id: str
    chunk_id: str
    source_filename: str
    page_numbers: list[int]
    heading_path: str
    excerpt: str
    score: float
    domain: str
    subdomain: str
    authority_tier: int
    epistemic_type: str
    jurisdiction: str
    is_current: bool


class CitationModel(BaseModel):
    """Pydantic validation helper for Citation payloads."""

    citation_index: int = Field(..., ge=1)
    doc_id: str
    chunk_id: str
    source_filename: str
    page_numbers: list[int] = Field(default_factory=list)
    heading_path: str = ""
    excerpt: str = ""
    score: float = 0.0
    domain: str = "general"
    subdomain: str = "general"
    authority_tier: int = 3
    epistemic_type: str = "general"
    jurisdiction: str = ""
    is_current: bool = True

    def to_citation(self) -> Citation:
        return Citation(
            citation_index=int(self.citation_index),
            doc_id=str(self.doc_id),
            chunk_id=str(self.chunk_id),
            source_filename=str(self.source_filename),
            page_numbers=[int(p) for p in self.page_numbers],
            heading_path=str(self.heading_path),
            excerpt=str(self.excerpt),
            score=float(self.score),
            domain=str(self.domain or "general"),
            subdomain=str(self.subdomain or "general"),
            authority_tier=int(self.authority_tier if self.authority_tier is not None else 3),
            epistemic_type=str(self.epistemic_type or "general"),
            jurisdiction=str(self.jurisdiction or ""),
            is_current=bool(self.is_current),
        )


def format_citation_source(citation: Citation | dict[str, Any]) -> str:
    """Format a human-readable citation tag: `[Source: doc, p.X, §Heading]`.

    Omits `p.X` when `page_numbers` is empty (e.g. DOCX or XLSX documents).
    """
    source = str(citation.get("source_filename") or "document").strip()
    pages_raw = citation.get("page_numbers") or []
    pages = sorted({int(p) for p in pages_raw if isinstance(p, (int, str)) and str(p).isdigit()})
    heading = str(citation.get("heading_path") or "").strip()

    parts: list[str] = [f"Source: {source}"]
    if pages:
        if len(pages) == 1:
            parts.append(f"p.{pages[0]}")
        elif pages == list(range(pages[0], pages[-1] + 1)):
            parts.append(f"p.{pages[0]}-{pages[-1]}")
        else:
            parts.append("p." + ",".join(str(p) for p in pages))
    if heading:
        parts.append(f"§{heading}")
    return "[" + ", ".join(parts) + "]"


class AgentState(TypedDict, total=False):
    """Typed state carried across the 7-node LangGraph Corrective RAG workflow."""

    messages: Annotated[list[BaseMessage], add_messages]
    original_query: str
    rewritten_queries: list[str]
    needs_retrieval: bool
    filter_doc_ids: list[str] | None
    retrieved_chunks: list[dict[str, Any]]
    reranked_chunks: list[dict[str, Any]]
    graded_chunks: list[dict[str, Any]]
    citations: list[Citation]
    steps: list[dict[str, Any]]
    domain_filter: dict[str, Any] | None
    routing_mode: str
    target_domains: list[str]
    domain_weights: dict[str, float]
    domain_queries: dict[str, str]
    polysemy_collisions: list[dict[str, Any]]
    matched_entities: list[str]

