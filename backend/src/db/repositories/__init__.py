"""Modular SQLite and cross-store repository package."""
from __future__ import annotations

from src.db.repositories.chat import (
    delete_chat_thread_row,
    get_chat_thread_row,
    list_chat_threads,
    upsert_chat_thread,
)
from src.db.repositories.common import (
    compile_word_boundary_regex,
    is_valid_sha256_hex,
    parse_settings_json_dict,
    rewrite_classification_json_for_reassignment,
)
from src.db.repositories.documents import (
    DocumentCancellable,
    delete_document_atomic,
    get_document_by_sha,
    get_latest_job_for_document,
    get_latest_jobs_for_documents,
    list_documents_paginated,
    sync_document_metadata_across_stores,
)
from src.db.repositories.entities import (
    get_cross_domain_entity_matches,
    upsert_cross_domain_entities,
)
from src.db.repositories.lexicon import (
    add_lexicon_term,
    delete_lexicon_term,
    get_lexicon_matches,
    list_lexicon_terms,
)
from src.db.repositories.settings import (
    get_workspace_settings,
    update_workspace_settings,
)
from src.db.repositories.taxonomy import (
    create_domain,
    delete_domain_and_reassign,
    get_domain_by_slug,
    list_domains,
    resolve_domain_by_id_or_slug,
    update_domain,
)
from src.db.repositories.taxonomy_subdomains import (
    create_subdomain,
    delete_subdomain,
    list_subdomains,
    list_subdomains_by_domain_ids,
    update_subdomain,
)

__all__ = [
    "DocumentCancellable",
    "add_lexicon_term",
    "compile_word_boundary_regex",
    "create_domain",
    "create_subdomain",
    "delete_chat_thread_row",
    "delete_document_atomic",
    "delete_domain_and_reassign",
    "delete_lexicon_term",
    "delete_subdomain",
    "get_chat_thread_row",
    "get_cross_domain_entity_matches",
    "get_document_by_sha",
    "get_domain_by_slug",
    "get_latest_job_for_document",
    "get_latest_jobs_for_documents",
    "get_lexicon_matches",
    "get_workspace_settings",
    "is_valid_sha256_hex",
    "list_chat_threads",
    "list_documents_paginated",
    "list_domains",
    "list_lexicon_terms",
    "list_subdomains",
    "list_subdomains_by_domain_ids",
    "parse_settings_json_dict",
    "resolve_domain_by_id_or_slug",
    "rewrite_classification_json_for_reassignment",
    "sync_document_metadata_across_stores",
    "update_domain",
    "update_subdomain",
    "update_workspace_settings",
    "upsert_chat_thread",
    "upsert_cross_domain_entities",
]
