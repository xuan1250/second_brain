"""Data-driven SQLite column migrations, taxonomy/lexicon seeding, and legacy row backfills."""
from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import Connection, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import DomainLexicon, DomainRegistry, SubdomainRegistry
from src.db.seed_data import DEFAULT_DOMAINS, DEFAULT_SUBDOMAINS, STARTER_LEXICON
from src.db.settings_bootstrap import restore_workspace_settings_and_credentials

TABLE_MIGRATIONS: dict[str, tuple[tuple[str, str], ...]] = {
    "documents": (
        ("parsing_engine", "ALTER TABLE documents ADD COLUMN parsing_engine VARCHAR"),
        ("engine_fallback_reason", "ALTER TABLE documents ADD COLUMN engine_fallback_reason VARCHAR"),
        ("primary_domain_id", "ALTER TABLE documents ADD COLUMN primary_domain_id VARCHAR"),
        ("primary_subdomain_id", "ALTER TABLE documents ADD COLUMN primary_subdomain_id VARCHAR"),
        ("jurisdiction", "ALTER TABLE documents ADD COLUMN jurisdiction VARCHAR"),
        ("authority_tier", "ALTER TABLE documents ADD COLUMN authority_tier INTEGER DEFAULT 3"),
        ("effective_year", "ALTER TABLE documents ADD COLUMN effective_year INTEGER"),
        ("is_current", "ALTER TABLE documents ADD COLUMN is_current BOOLEAN DEFAULT 1"),
        ("superseded_by_doc_id", "ALTER TABLE documents ADD COLUMN superseded_by_doc_id VARCHAR"),
        ("classification_result", "ALTER TABLE documents ADD COLUMN classification_result TEXT"),
    ),
    "document_chunks": (
        ("domain_key", "ALTER TABLE document_chunks ADD COLUMN domain_key VARCHAR"),
        ("subdomain_key", "ALTER TABLE document_chunks ADD COLUMN subdomain_key VARCHAR"),
        ("epistemic_modality", "ALTER TABLE document_chunks ADD COLUMN epistemic_modality VARCHAR"),
        ("contextual_prefix", "ALTER TABLE document_chunks ADD COLUMN contextual_prefix TEXT"),
    ),
    "domain_registry": (
        ("icon", "ALTER TABLE domain_registry ADD COLUMN icon VARCHAR DEFAULT '📁'"),
        ("color_token", "ALTER TABLE domain_registry ADD COLUMN color_token VARCHAR DEFAULT 'indigo'"),
        ("chunking_strategy", "ALTER TABLE domain_registry ADD COLUMN chunking_strategy VARCHAR DEFAULT 'standard_markdown'"),
        ("classifier_keywords", "ALTER TABLE domain_registry ADD COLUMN classifier_keywords TEXT"),
        ("is_system", "ALTER TABLE domain_registry ADD COLUMN is_system BOOLEAN DEFAULT 0"),
        ("is_active", "ALTER TABLE domain_registry ADD COLUMN is_active BOOLEAN DEFAULT 1"),
    ),
    "workspace_settings": (
        ("llm_api_keys", "ALTER TABLE workspace_settings ADD COLUMN llm_api_keys TEXT"),
        ("llm_key_validations", "ALTER TABLE workspace_settings ADD COLUMN llm_key_validations TEXT"),
        ("llm_compat_base_url", "ALTER TABLE workspace_settings ADD COLUMN llm_compat_base_url VARCHAR"),
    ),
}


def _apply_table_migrations(
    sync_conn: Connection,
    table_name: str,
    columns: Sequence[tuple[str, str]],
) -> None:
    """Add any missing columns on `table_name` using PRAGMA table_info inspection."""
    cols_info = sync_conn.execute(text(f"PRAGMA table_info({table_name})")).fetchall()
    if not cols_info:
        return
    existing_cols = {col[1] for col in cols_info}
    for col_name, ddl in columns:
        if col_name not in existing_cols:
            sync_conn.execute(text(ddl))


def apply_all_column_migrations(sync_conn: Connection) -> None:
    """Run all additive schema column migrations across registered SQLite tables."""
    for table_name, columns in TABLE_MIGRATIONS.items():
        _apply_table_migrations(sync_conn, table_name, columns)


async def seed_taxonomy_and_lexicon(session: AsyncSession) -> None:
    """Idempotently seed default domains, subdomains, and bilingual starter lexicon entries."""
    existing_domains = (await session.execute(select(DomainRegistry))).scalars().all()
    existing_domains_map = {d.slug: d for d in existing_domains}
    for d in DEFAULT_DOMAINS:
        if d["slug"] not in existing_domains_map:
            session.add(
                DomainRegistry(
                    id=d["id"],
                    slug=d["slug"],
                    name=d["name"],
                    epistemic_type=d["epistemic_type"],
                    authority_priority=d["authority_priority"],
                    description=d["description"],
                    icon=d["icon"],
                    color_token=d["color_token"],
                    chunking_strategy=d["chunking_strategy"],
                    classifier_keywords=d["classifier_keywords"],
                    is_system=d["is_system"],
                    is_active=d["is_active"],
                )
            )
        else:
            row = existing_domains_map[d["slug"]]
            row.is_system = True
            if row.icon == "📁" or not row.icon:
                row.icon = d["icon"]
            if row.color_token == "indigo" or not row.color_token:
                row.color_token = d["color_token"]
            if row.chunking_strategy == "standard_markdown" or not row.chunking_strategy:
                row.chunking_strategy = d["chunking_strategy"]
            session.add(row)

    existing_subdomains = (await session.execute(select(SubdomainRegistry))).scalars().all()
    existing_subdomain_slugs = {s.slug for s in existing_subdomains}
    for sd in DEFAULT_SUBDOMAINS:
        if sd["slug"] not in existing_subdomain_slugs:
            session.add(
                SubdomainRegistry(
                    id=sd["id"],
                    domain_id=sd["domain_id"],
                    slug=sd["slug"],
                    name=sd["name"],
                    standard_bodies=sd["standard_bodies"],
                )
            )

    existing_lexicons = (await session.execute(select(DomainLexicon))).scalars().all()
    existing_lexicon_keys = {(lex.term.strip().lower(), lex.domain_id) for lex in existing_lexicons}
    for entry in STARTER_LEXICON:
        key = (entry["term"].strip().lower(), entry["domain_id"])
        if key not in existing_lexicon_keys:
            session.add(
                DomainLexicon(
                    term=entry["term"].strip(),
                    domain_id=entry["domain_id"],
                    specific_sense=entry["specific_sense"],
                    contrastive_warning=entry["contrastive_warning"],
                )
            )


async def backfill_legacy_records(session: AsyncSession) -> None:
    """Backfill NULL-domain legacy rows, slim duplicate chunk text, and remove orphaned thread header."""
    await session.execute(
        text(
            "UPDATE documents "
            "SET primary_domain_id = 'general', primary_subdomain_id = 'general.general' "
            "WHERE primary_domain_id IS NULL"
        )
    )
    await session.execute(
        text(
            "UPDATE document_chunks "
            "SET domain_key = 'general', subdomain_key = 'general.general', epistemic_modality = 'general' "
            "WHERE domain_key IS NULL"
        )
    )
    await session.execute(
        text(
            "UPDATE document_chunks "
            "SET text = '', parent_content = '' "
            "WHERE text != '' OR parent_content != ''"
        )
    )
    await session.execute(
        text(
            "DELETE FROM chat_threads "
            "WHERE thread_id = '1de1cd57414b458ebdeb8ea8305bd8ac'"
        )
    )


async def run_startup_migrations_and_seeds(session: AsyncSession) -> None:
    """Execute taxonomy seeding, legacy backfills, and workspace credential restoration."""
    await seed_taxonomy_and_lexicon(session)
    await backfill_legacy_records(session)
    await restore_workspace_settings_and_credentials(session)
