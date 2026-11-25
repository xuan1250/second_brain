"""Laya multilingual domain classifier, payload extractor, and process singleton lifecycle."""
from __future__ import annotations

import gc
import logging
import time
from typing import Any, Optional

from src.config import get_settings
from src.ingestion.classification.heuristics import (
    StubLayaClassifier,
    _extract_authority_and_jurisdiction,
)
from src.ingestion.classification.llm_fallback import classify_document_with_llm
from src.ingestion.classification.models import (
    BaseDomainClassifier,
    DomainClassificationResult,
    RecommendedDomainProfile,
    SectionOverride,
)
from src.ingestion.classification.taxonomy import (
    build_dynamic_taxonomy_questions,
    get_taxonomy_catalog,
)
from src.ingestion.constants import DEFAULT_DOMAIN
from src.ingestion.indexing.device import empty_torch_cuda_cache

logger = logging.getLogger(__name__)


def _extract_laya_answer(pred: Any, q_key: str) -> tuple[Optional[str], float]:
    """Safely extract choice/answer and confidence from Laya agent prediction payload.

    Supports both Laya v1/v2 schema (nested under 'answers' with 'choice' and 'answer_confidence')
    and legacy/flat dictionaries.
    """
    if not isinstance(pred, dict):
        return None, 0.0

    answers_dict = pred.get("answers")
    item = None
    if isinstance(answers_dict, dict) and q_key in answers_dict:
        item = answers_dict.get(q_key)
    elif q_key in pred:
        item = pred.get(q_key)

    if not isinstance(item, dict):
        return None, 0.0

    choice = item.get("choice")
    if choice is None:
        choice = item.get("answer")
    if choice is None:
        choice = item.get("value")
    if choice is not None:
        choice = str(choice).strip()

    raw_conf = item.get("answer_confidence")
    if raw_conf is None:
        raw_conf = item.get("confidence", 0.0)
    try:
        conf = float(raw_conf)
    except (TypeError, ValueError):
        conf = 0.0

    return choice, conf


class LayaDomainClassifier(BaseDomainClassifier):
    """Lazy singleton loading convaiinnovations/laya-multilingual with H1/H2 override scanning and LLM fallback."""

    def __init__(
        self,
        model_name: str = "convaiinnovations/laya-multilingual",
        device: Optional[str] = None,
        enable_llm_fallback: bool = True,
        high_confidence_threshold: float = 0.85,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.enable_llm_fallback = enable_llm_fallback
        self.high_confidence_threshold = high_confidence_threshold
        self._agent: Any = None
        self._load_failed: bool = False
        self._load_error: Optional[str] = None

    def _ensure_model_loaded(self) -> Any:
        """Lazily initialize the Laya agent."""
        if self._agent is not None:
            return self._agent
        if self._load_failed:
            raise RuntimeError(f"Laya model load previously failed: {self._load_error}")

        t0 = time.perf_counter()
        try:
            import laya

            if hasattr(laya, "Laya") and callable(getattr(laya, "Laya")):
                self._agent = laya.Laya(self.model_name)
            elif hasattr(laya, "load") and callable(getattr(laya, "load")):
                self._agent = laya.load(self.model_name, device=self.device)
            elif hasattr(laya, "Agent") and callable(getattr(laya, "Agent")):
                self._agent = laya.Agent(self.model_name, device=self.device)
            else:
                raise AttributeError("laya module does not expose Laya, load, or Agent entry points")

            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            logger.info("Loaded Laya classifier model '%s' in %dms", self.model_name, elapsed_ms)
            return self._agent
        except Exception as exc:
            self._load_failed = True
            self._load_error = str(exc)
            logger.warning("Failed loading Laya classifier model '%s': %s", self.model_name, exc)
            raise

    def unload(self) -> bool:
        """Unload resident model weights and release memory without calling release_cuda_cache."""
        unloaded = False
        if self._agent is not None:
            self._agent = None
            unloaded = True
        self._load_failed = False
        self._load_error = None
        empty_torch_cuda_cache()
        return unloaded

    @property
    def is_resident(self) -> bool:
        return self._agent is not None

    async def _classify_with_llm(self, text: str, filename: Optional[str]) -> Optional[RecommendedDomainProfile]:
        """Delegate opt-in LLM classification fallback when Laya confidence is low (< 0.60)."""
        return await classify_document_with_llm(text, filename)

    async def classify_document(
        self,
        text: str,
        *,
        sections: Optional[list[dict[str, str]]] = None,
        filename: Optional[str] = None,
    ) -> DomainClassificationResult:
        t0 = time.perf_counter()

        try:
            agent = self._ensure_model_loaded()
            raw_answers: dict[str, Any] = {}
            catalog = get_taxonomy_catalog()
            questions = build_dynamic_taxonomy_questions(catalog)

            state_text = f"Title / Filename: {filename or 'Unknown'}\n\n{text[:3500]}"

            domain_q = {"domain": questions["domain"]}
            domain_pred = agent.predict(state_text, domain_q)
            raw_answers["domain_prediction"] = domain_pred

            extracted_domain, extracted_conf = _extract_laya_answer(domain_pred, "domain")
            domain_ans = extracted_domain or DEFAULT_DOMAIN
            domain_conf = extracted_conf

            if domain_ans not in catalog.domains:
                domain_ans = DEFAULT_DOMAIN

            sub_q_key = f"{domain_ans}_subdomain"
            subdomain_ans = catalog.get_default_subdomain(domain_ans)
            sub_conf = domain_conf

            if sub_q_key in questions:
                sub_q = {sub_q_key: questions[sub_q_key]}
                sub_pred = agent.predict(state_text, sub_q)
                raw_answers["subdomain_prediction"] = sub_pred
                cand_sub, cand_conf = _extract_laya_answer(sub_pred, sub_q_key)
                dom_obj = catalog.get_domain(domain_ans)
                if cand_sub and dom_obj and cand_sub in dom_obj.subdomains:
                    subdomain_ans = cand_sub
                    sub_conf = cand_conf

            tier, jurisdiction, eff_year = _extract_authority_and_jurisdiction(text)

            section_overrides: list[SectionOverride] = []
            if sections:
                for sec in sections[:20]:
                    h_path = sec.get("heading_path", "")
                    sec_content = sec.get("text", "").strip()
                    if len(sec_content) < 80:
                        continue

                    sec_state = f"Heading: {h_path}\n\n{sec_content[:2000]}"
                    sec_pred = agent.predict(sec_state, domain_q)
                    sec_domain, sec_conf = _extract_laya_answer(sec_pred, "domain")

                    if sec_domain and sec_domain in catalog.domains and sec_domain != domain_ans and sec_conf >= 0.75:
                        sec_sub = catalog.get_default_subdomain(sec_domain)
                        sec_sub_key = f"{sec_domain}_subdomain"
                        if sec_sub_key in questions:
                            sec_sub_pred = agent.predict(sec_state, {sec_sub_key: questions[sec_sub_key]})
                            cand_sec_sub, _ = _extract_laya_answer(sec_sub_pred, sec_sub_key)
                            sec_dom_obj = catalog.get_domain(sec_domain)
                            if cand_sec_sub and sec_dom_obj and cand_sec_sub in sec_dom_obj.subdomains:
                                sec_sub = cand_sec_sub

                        section_overrides.append(
                            SectionOverride(
                                heading_path=h_path or "Section",
                                domain=sec_domain,
                                subdomain=sec_sub,
                                epistemic_modality=catalog.get_epistemic_modality(sec_domain),
                                confidence=sec_conf,
                                authority_tier=tier,
                                jurisdiction=jurisdiction,
                                chunking_strategy=catalog.get_chunking_strategy(sec_domain),
                            )
                        )

            if domain_conf < 0.60 and self.enable_llm_fallback:
                logger.info(
                    "Laya domain confidence %.2f < 0.60; attempting opt-in LLM classification fallback",
                    domain_conf,
                )
                llm_profile = await self._classify_with_llm(text, filename)
                if llm_profile is not None:
                    llm_profile.section_overrides = section_overrides
                    llm_profile.is_hybrid = len(section_overrides) > 0
                    elapsed_ms = max(1, int((time.perf_counter() - t0) * 1000))
                    return DomainClassificationResult(
                        recommended=llm_profile,
                        fallback_used=True,
                        fallback_reason=f"Laya low confidence ({domain_conf:.2f} < 0.60)",
                        execution_ms=elapsed_ms,
                        raw_answers=raw_answers,
                    )

            profile = RecommendedDomainProfile(
                primary_domain=domain_ans,
                primary_subdomain=subdomain_ans,
                epistemic_modality=catalog.get_epistemic_modality(domain_ans),
                confidence=domain_conf,
                confidence_subdomain=sub_conf,
                authority_tier=tier,
                jurisdiction=jurisdiction,
                effective_year=eff_year,
                is_current=True,
                is_hybrid=len(section_overrides) > 0,
                section_overrides=section_overrides,
                reasoning=f"Laya decision: domain={domain_ans} ({domain_conf:.2f}), subdomain={subdomain_ans}",
                chunking_strategy=catalog.get_chunking_strategy(domain_ans),
            )
            elapsed_ms = max(1, int((time.perf_counter() - t0) * 1000))
            return DomainClassificationResult(
                recommended=profile,
                fallback_used=False,
                execution_ms=elapsed_ms,
                raw_answers=raw_answers,
            )

        except Exception as exc:
            logger.warning("Laya classifier execution failed (%s); engaging fallback classifier", exc)
            if self.enable_llm_fallback:
                llm_profile = await self._classify_with_llm(text, filename)
                if llm_profile is not None:
                    elapsed_ms = max(1, int((time.perf_counter() - t0) * 1000))
                    return DomainClassificationResult(
                        recommended=llm_profile,
                        fallback_used=True,
                        fallback_reason=f"Laya error ({exc}); resolved via LLM",
                        execution_ms=elapsed_ms,
                    )

            stub = StubLayaClassifier()
            stub_res = await stub.classify_document(text, sections=sections, filename=filename)
            stub_res.fallback_used = True
            stub_res.fallback_reason = f"Laya error ({exc}); resolved via heuristic stub"
            stub_res.execution_ms = max(1, int((time.perf_counter() - t0) * 1000))
            return stub_res


_CLASSIFIER_INSTANCE: Optional[BaseDomainClassifier] = None
_FORCE_STUB: bool = False


def set_classifier_force_stub(force: bool) -> None:
    """Explicitly force use of StubLayaClassifier (e.g. for deterministic unit test suites)."""
    global _FORCE_STUB, _CLASSIFIER_INSTANCE
    _FORCE_STUB = force
    if force and isinstance(_CLASSIFIER_INSTANCE, LayaDomainClassifier):
        _CLASSIFIER_INSTANCE.unload()
        _CLASSIFIER_INSTANCE = None


def get_classifier(model_name: Optional[str] = None) -> BaseDomainClassifier:
    """Return the lazy singleton classifier instance."""
    global _CLASSIFIER_INSTANCE
    if _CLASSIFIER_INSTANCE is not None:
        return _CLASSIFIER_INSTANCE

    if _FORCE_STUB:
        _CLASSIFIER_INSTANCE = StubLayaClassifier()
        return _CLASSIFIER_INSTANCE

    settings = get_settings()
    m_name = model_name or settings.ingestion.laya_model_name
    _CLASSIFIER_INSTANCE = LayaDomainClassifier(
        model_name=m_name,
        device=settings.ingestion.device if settings.ingestion.device != "auto" else None,
        enable_llm_fallback=settings.ingestion.enable_llm_classification_fallback,
        high_confidence_threshold=settings.ingestion.high_confidence_threshold,
    )
    return _CLASSIFIER_INSTANCE


def unload_classifier() -> bool:
    """Unload resident classifier weights and free VRAM/RAM."""
    global _CLASSIFIER_INSTANCE
    if _CLASSIFIER_INSTANCE is not None:
        if isinstance(_CLASSIFIER_INSTANCE, LayaDomainClassifier):
            _CLASSIFIER_INSTANCE.unload()
        _CLASSIFIER_INSTANCE = None
        gc.collect()
        return True
    return False


def is_classifier_resident() -> bool:
    """Check if Laya classifier model is currently resident in memory."""
    if _CLASSIFIER_INSTANCE is not None and isinstance(_CLASSIFIER_INSTANCE, LayaDomainClassifier):
        return _CLASSIFIER_INSTANCE.is_resident
    return False


def get_classifier_model_name() -> Optional[str]:
    """Get active classifier model name if resident or configured."""
    if _CLASSIFIER_INSTANCE is not None and isinstance(_CLASSIFIER_INSTANCE, LayaDomainClassifier):
        return _CLASSIFIER_INSTANCE.model_name
    return get_settings().ingestion.laya_model_name
