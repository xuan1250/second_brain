"""Exact-tag, primary-subject entity matching, and federated RRF fusion helpers."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from src.rag.qdrant_filters import _point_payload_to_candidate


def _chunk_matches_entities(
    candidate: Mapping[str, Any],
    cleaned_entities_lower: set[str],
) -> bool:
    """Return True if candidate matches extracted entities by payload tag or text substring."""
    if not cleaned_entities_lower:
        return False
    chunk_entities = {
        str(e).strip().lower()
        for e in (candidate.get("associated_entities") or [])
        if str(e).strip()
    }
    if chunk_entities & cleaned_entities_lower:
        return True
    child_lower = str(candidate.get("child_text") or "").lower()
    parent_lower = str(candidate.get("parent_content") or "").lower()
    return any(e in child_lower or e in parent_lower for e in cleaned_entities_lower)


def _is_primary_subject_match(
    candidate: Mapping[str, Any],
    cleaned_entities_lower: set[str],
) -> bool:
    """Return True when queried entity matches the primary subject header rather than an incidental body/group mention."""
    if not cleaned_entities_lower:
        return False

    record_type = str(candidate.get("record_type") or "")
    child_lines = str(candidate.get("child_text") or "").splitlines()
    heading_lower = str(candidate.get("heading_path") or "").lower()

    if record_type == "matrix_effect_col":
        subject_lines = "\n".join(
            ln for ln in child_lines[:3]
            if not ln.lstrip().lower().startswith("triggering causes")
        ).lower()
        subject_text = f"{heading_lower}\n{subject_lines}"
    elif record_type == "matrix_cause_row":
        cause_line = next(
            (ln for ln in child_lines[:3] if "cause tag:" in ln.lower()),
            child_lines[0] if child_lines else "",
        )
        subject_text = cause_line.split("|", 1)[0].lower()
    else:
        first_line = child_lines[0].lower() if child_lines else ""
        subject_text = f"{heading_lower}\n{first_line}"

    compact_subject = subject_text.replace("-", "").replace("_", "")
    return any(
        e in subject_text or e.replace("-", "").replace("_", "") in compact_subject
        for e in cleaned_entities_lower
    )


def _prioritize_scrolled_points(
    points: Sequence[Any],
    cleaned_entities_lower: set[str],
    *,
    cap: int,
) -> list[Any]:
    """Order scrolled exact-entity points so primary subject and Effect-column chunks survive culling."""
    if not points:
        return []

    def _priority_key(pt: Any) -> tuple[int, int]:
        payload = dict(getattr(pt, "payload", None) or {})
        cand = {
            "record_type": payload.get("record_type"),
            "heading_path": payload.get("heading_path") or "",
            "child_text": payload.get("child_text") or "",
        }
        is_subj = 1 if _is_primary_subject_match(cand, cleaned_entities_lower) else 0
        rtype = str(payload.get("record_type") or "")
        type_rank = 2 if rtype == "matrix_effect_col" else (1 if rtype in ("cad_logic_diagram", "schedule_row") else 0)
        return (-is_subj, -type_rank)

    return sorted(points, key=_priority_key)[:cap]


def _upsert_exact_tag_candidate(
    candidates_by_id: dict[str, dict[str, Any]],
    point: Any,
    *,
    min_score: float | None = None,
    is_subject: bool = False,
) -> str:
    """Insert or upgrade a Qdrant point as an exact_tag_match candidate and return its chunk_id."""
    payload = dict(point.payload or {})
    chunk_id = str(payload.get("chunk_id") or point.id)
    existing = candidates_by_id.get(chunk_id)
    if existing is not None:
        existing["exact_tag_match"] = True
        if is_subject:
            existing["exact_subject_match"] = True
        if min_score is not None:
            existing["score"] = max(float(existing.get("score", 0.0)), min_score)
    else:
        cand = _point_payload_to_candidate(
            point.id,
            payload,
            score=min_score if min_score is not None else 0.0,
            fallback_domain=str(payload.get("domain") or "general"),
        )
        cand["exact_tag_match"] = True
        if is_subject:
            cand["exact_subject_match"] = True
        candidates_by_id[chunk_id] = cand
    return chunk_id


def _accumulate_federated_rrf(
    results: Sequence[tuple[str, Sequence[Any]]],
    domains_to_query: Sequence[str],
    *,
    weights: Mapping[str, float],
    mode: str,
    rrf_k: int,
    cleaned_entities_lower: set[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, float], dict[str, list[dict[str, Any]]]]:
    """Accumulate Domain-Calibrated RRF scores and entity match flags across per-domain search results."""
    candidates_by_id: dict[str, dict[str, Any]] = {}
    rrf_scores_by_id: dict[str, float] = {}
    domain_chunk_lists: dict[str, list[dict[str, Any]]] = {d: [] for d in domains_to_query}

    for dom, points in results:
        dom_weight = float(weights.get(dom, 1.0))
        if dom == "general" and mode == "auto":
            dom_weight *= 0.5

        for rank_0, point in enumerate(points):
            rank = rank_0 + 1
            payload = dict(point.payload or {})
            candidate = _point_payload_to_candidate(
                point.id,
                payload,
                score=0.0,
                fallback_domain=dom,
            )
            chunk_id = candidate["chunk_id"]
            has_entity_match = _chunk_matches_entities(candidate, cleaned_entities_lower)

            rrf_contrib = dom_weight / (float(rrf_k) + float(rank))
            if has_entity_match:
                rrf_contrib += 1.0 / float(rrf_k)

            rrf_scores_by_id[chunk_id] = rrf_scores_by_id.get(chunk_id, 0.0) + rrf_contrib
            target = candidates_by_id.setdefault(chunk_id, candidate)
            if has_entity_match:
                target["exact_tag_match"] = True
                if _is_primary_subject_match(candidate, cleaned_entities_lower):
                    target["exact_subject_match"] = True
            domain_chunk_lists[dom].append(target)

    return candidates_by_id, rrf_scores_by_id, domain_chunk_lists


def _select_stratified_candidates(
    candidates_by_id: dict[str, dict[str, Any]],
    rrf_scores_by_id: Mapping[str, float],
    domain_chunk_lists: Mapping[str, Sequence[dict[str, Any]]],
    guaranteed_ids: set[str],
    *,
    top_limit: int,
    min_per_domain: int,
) -> list[dict[str, Any]]:
    """Apply final RRF scores, per-domain quota guarantees, and descending score selection."""
    for c_id, score in rrf_scores_by_id.items():
        if c_id in candidates_by_id:
            candidates_by_id[c_id]["score"] = round(score, 6)

    for c_list in domain_chunk_lists.values():
        sorted_dom = sorted(c_list, key=lambda c: float(c.get("score", 0.0)), reverse=True)
        for c in sorted_dom[:min_per_domain]:
            guaranteed_ids.add(c["chunk_id"])

    selected_candidates: list[dict[str, Any]] = [candidates_by_id[cid] for cid in guaranteed_ids]
    effective_limit = max(top_limit, len(guaranteed_ids))
    remaining = [c for cid, c in candidates_by_id.items() if cid not in guaranteed_ids]
    remaining.sort(key=lambda c: float(c.get("score", 0.0)), reverse=True)
    slots_left = max(0, effective_limit - len(selected_candidates))
    selected_candidates.extend(remaining[:slots_left])
    selected_candidates.sort(key=lambda c: float(c.get("score", 0.0)), reverse=True)
    return selected_candidates[:effective_limit]
