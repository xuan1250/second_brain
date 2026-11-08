import React, { useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  AlertTriangle,
  Bot,
  Check,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Cpu,
  FileText,
  Filter,
  Loader2,
  MessageSquarePlus,
  Send,
  ShieldCheck,
  Sparkles,
  Square,
  Trash2,
  User,
} from "lucide-react";
import { useChatStream, useChatThreads } from "../../hooks/useChatStream";
import { useSystemStats } from "../../hooks/useDocuments";
import { useDomainsQuery, useWorkspaceSettingsQuery } from "../../hooks/useDomains";
import { resolveDomainMeta } from "../../lib/domainTheme";
import type {
  ChatMessage,
  Citation,
  CRAGStep,
} from "../../lib/types";
import { useWorkspace } from "../../lib/workspaceStore";
import { AuthorityTierBadge } from "../library/KnowledgeLibrary";
import { CitationPill, getCitationDomainColor } from "./CitationPill";
import { ErrorBoundary } from "../common/ErrorBoundary";

const CRAG_PIPELINE_STEPS: { node: string; label: string }[] = [
  { node: "rewrite_query", label: "Rewriting & Decomposing" },
  { node: "retrieve", label: "Federated search" },
  { node: "rerank", label: "Reranking (FlashRank)" },
  { node: "grade_context", label: "Grading context" },
  { node: "synthesize", label: "Synthesizing" },
];

/**
 * Replace standalone `[1]`, `[2]` markers (not followed by `(`) with safe anchor links `[1](#citation-1)`
 * so ReactMarkdown renders them via the custom `a` component without `rehype-raw`.
 * Skips code blocks and inline code spans to avoid altering code snippets.
 */
function injectCitationLinks(markdown: string): string {
  if (!markdown) return "";
  const parts = markdown.split(/(```[\s\S]*?```|`[^`\n]*`)/g);
  return parts
    .map((part, idx) => {
      if (idx % 2 === 1) return part;
      return part.replace(/\[(\d+)\](?!\()/g, "[$1](#citation-$1)");
    })
    .join("");
}

/**
 * Sticky Domain Routing Bar for per-thread domain target selection.
 */
function DomainRoutingBar() {
  const { activeThreadId, currentDomainFilter, setThreadDomainFilter } =
    useWorkspace();
  const { data: domains = [] } = useDomainsQuery();
  const [isMoreOpen, setIsMoreOpen] = useState(false);

  const isAuto =
    currentDomainFilter.mode === "auto" ||
    (!currentDomainFilter.domains.length && currentDomainFilter.mode !== "single");

  const isTierRestricted = currentDomainFilter.max_authority_tier === 2;

  const handleSelectAuto = () => {
    setThreadDomainFilter(activeThreadId, {
      ...currentDomainFilter,
      mode: "auto",
      domains: [],
    });
  };

  const handleToggleDomain = (domainKey: string, e: React.MouseEvent) => {
    e.preventDefault();
    const isShiftOrMeta = e.shiftKey || e.metaKey || e.ctrlKey;
    const currentDomains = [...currentDomainFilter.domains];
    const exists = currentDomains.includes(domainKey);

    let nextDomains: string[] = [];
    let nextMode: string = "single";

    if (isShiftOrMeta) {
      // Multi-select federated mode
      if (exists) {
        nextDomains = currentDomains.filter((d) => d !== domainKey);
      } else {
        nextDomains = [...currentDomains, domainKey];
      }
      nextMode = nextDomains.length > 1 ? "federated" : "single";
    } else {
      // Single-select toggle
      if (exists && currentDomains.length === 1 && !isAuto) {
        // Toggle back to auto
        nextDomains = [];
        nextMode = "auto";
      } else {
        nextDomains = [domainKey];
        nextMode = "single";
      }
    }

    setThreadDomainFilter(activeThreadId, {
      ...currentDomainFilter,
      mode: nextDomains.length === 0 ? "auto" : nextMode,
      domains: nextDomains,
    });
  };

  const handleToggleTierFilter = () => {
    setThreadDomainFilter(activeThreadId, {
      ...currentDomainFilter,
      max_authority_tier: isTierRestricted ? null : 2,
    });
  };

  // Facet pills inline vs overflow
  const STARTER_SLUGS = ["engineering", "economics", "law"];
  const specializedDomains = useMemo(
    () => domains.filter((d) => d.slug !== "general"),
    [domains]
  );

  const { inlinePills, overflowPills } = useMemo(() => {
    if (specializedDomains.length <= 4) {
      return {
        inlinePills: domains,
        overflowPills: [],
      };
    }

    // When total specialized domains exceed 4:
    // show the 3 starter specialized domains (engineering, economics, law) + any currently active custom domain(s)
    const selectedCustomDomains = currentDomainFilter.domains.filter(
      (d) => !STARTER_SLUGS.includes(d) && d !== "general"
    );

    const starterInline = domains.filter(
      (d) => STARTER_SLUGS.includes(d.slug) || d.slug === "general"
    );
    const customInline = domains.filter((d) =>
      selectedCustomDomains.includes(d.slug)
    );

    const inlineSet = new Set([
      ...starterInline.map((d) => d.slug),
      ...customInline.map((d) => d.slug),
    ]);
    const overflow = domains.filter((d) => !inlineSet.has(d.slug));

    return {
      inlinePills: [...starterInline, ...customInline],
      overflowPills: overflow,
    };
  }, [domains, specializedDomains.length, currentDomainFilter.domains]);

  return (
    <div className="flex flex-wrap items-center justify-between gap-1.5 rounded-xl border border-slate-800 bg-slate-900/80 px-2 py-1.5 text-xs select-none">
      <div className="flex items-center gap-1 overflow-x-auto">
        {/* Auto mode button */}
        <button
          type="button"
          onClick={handleSelectAuto}
          title="🌐 Auto: Multi-Domain Federated Synthesis with Polysemy Disambiguation"
          className={`inline-flex items-center gap-1 rounded-lg border px-2 py-1 text-[10.5px] font-medium transition ${
            isAuto
              ? "border-indigo-500 bg-indigo-600/30 text-indigo-200 font-semibold shadow-xs"
              : "border-slate-800 bg-slate-950/60 text-slate-400 hover:bg-slate-800 hover:text-slate-200"
          }`}
        >
          <span>🌐</span>
          <span>Auto</span>
        </button>

        {/* Specialized Domain Pills */}
        {inlinePills.map((dom) => {
          const isSelected =
            !isAuto && currentDomainFilter.domains.includes(dom.slug);
          const meta = resolveDomainMeta(dom.slug, domains);

          return (
            <button
              key={dom.id}
              type="button"
              onClick={(e) => handleToggleDomain(dom.slug, e)}
              title={`${meta.icon} ${meta.name}: Click to filter exclusively, Shift+Click for Federated multi-select`}
              className={`inline-flex items-center gap-1 rounded-lg border px-2 py-1 text-[10.5px] font-medium transition ${
                isSelected
                  ? `${meta.theme.activePill} font-semibold shadow-xs`
                  : "border-slate-800 bg-slate-950/60 text-slate-400 hover:bg-slate-800 hover:text-slate-200"
              }`}
            >
              <span>{meta.icon}</span>
              <span>{meta.shortLabel}</span>
            </button>
          );
        })}

        {/* Overflow Popover */}
        {overflowPills.length > 0 && (
          <div className="relative shrink-0">
            <button
              type="button"
              onClick={() => setIsMoreOpen((v) => !v)}
              className="inline-flex items-center gap-1 rounded-lg border border-slate-800 bg-slate-950/60 px-2 py-1 text-[10.5px] font-medium text-slate-400 hover:bg-slate-800 hover:text-slate-200 transition"
            >
              <span>+{overflowPills.length} More</span>
              <ChevronDown className="h-3 w-3" />
            </button>
            {isMoreOpen && (
              <div
                onClick={(e) => e.stopPropagation()}
                className="absolute left-0 top-full z-30 mt-1 w-44 rounded-xl border border-slate-700 bg-slate-950 p-1.5 shadow-xl text-xs space-y-0.5"
              >
                <div className="px-2 py-1 text-[10px] uppercase font-bold text-slate-500 border-b border-slate-800">
                  More Domains
                </div>
                {overflowPills.map((ov) => {
                  const isSelected =
                    !isAuto && currentDomainFilter.domains.includes(ov.slug);
                  const meta = resolveDomainMeta(ov.slug, domains);
                  return (
                    <button
                      key={ov.id}
                      type="button"
                      onClick={(e) => {
                        handleToggleDomain(ov.slug, e);
                        setIsMoreOpen(false);
                      }}
                      className={`flex w-full items-center justify-between rounded-lg px-2 py-1.5 text-left text-xs transition ${
                        isSelected
                          ? `${meta.theme.activePill} font-semibold`
                          : "text-slate-300 hover:bg-slate-800 hover:text-white"
                      }`}
                    >
                      <span className="flex items-center gap-1.5 truncate">
                        <span>{meta.icon}</span>
                        <span className="truncate">{meta.name}</span>
                      </span>
                      {isSelected && <Check className="h-3 w-3" />}
                    </button>
                  );
                })}
              </div>
            )}
          </div>
        )}

        {currentDomainFilter.mode === "federated" && (
          <span className="rounded-full bg-purple-500/20 border border-purple-500/30 px-1.5 py-0.5 text-[9.5px] font-mono font-medium text-purple-300">
            🔀 Fed ({currentDomainFilter.domains.length})
          </span>
        )}
      </div>

      {/* Authority Tier Filter Toggle */}
      <button
        type="button"
        onClick={handleToggleTierFilter}
        title={
          isTierRestricted
            ? "Filter active: Only authoritative documents (Tiers 1 & 2) will be retrieved"
            : "Filter inactive: All authority tiers (T1..T5) will be retrieved"
        }
        className={`inline-flex items-center gap-1 rounded-lg border px-2 py-1 text-[10.5px] font-medium transition shrink-0 ${
          isTierRestricted
            ? "border-amber-500/50 bg-amber-500/20 text-amber-200 font-semibold"
            : "border-slate-800 bg-slate-950/60 text-slate-400 hover:bg-slate-800 hover:text-slate-200"
        }`}
      >
        <ShieldCheck className="h-3 w-3 text-amber-400" />
        <span>{isTierRestricted ? "🛡️ Tier 1–2 Authoritative" : "🛡️ All Tiers"}</span>
      </button>
    </div>
  );
}

/**
 * Enhanced CRAG Stepper displaying active routing mode, polysemy alerts, and subqueries.
 */
function CRAGStepper({
  steps,
  isStreaming,
}: {
  steps: CRAGStep[];
  isStreaming?: boolean;
}) {
  const [showSubqueries, setShowSubqueries] = useState(false);

  if ((!steps || steps.length === 0) && !isStreaming) return null;

  const completedNodes = new Set(steps.map((s) => s.node));
  const hasDirectReply = completedNodes.has("direct_reply");
  const hasFallback = completedNodes.has("fallback_notice");

  const displaySteps = hasDirectReply
    ? [
        { node: "rewrite_query", label: "Rewriting & Intent" },
        { node: "direct_reply", label: "Replying directly" },
      ]
    : hasFallback
    ? [
        ...CRAG_PIPELINE_STEPS.slice(0, 4),
        { node: "fallback_notice", label: "Fallback notice" },
      ]
    : CRAG_PIPELINE_STEPS;

  // Check for polysemy collisions or routing telemetry
  const stepWithPolysemy = steps.find(
    (s) => s.polysemy_collisions && s.polysemy_collisions.length > 0
  );
  const collisions = stepWithPolysemy?.polysemy_collisions || [];

  const stepWithQueries = steps.find(
    (s) => s.domain_queries && Object.keys(s.domain_queries).length > 0
  );
  const domainQueries = stepWithQueries?.domain_queries || {};

  const activeRoutingMode =
    steps.find((s) => s.routing_mode)?.routing_mode ||
    (Object.keys(domainQueries).length > 1 ? "federated" : "single");

  return (
    <div className="mb-2.5 rounded-xl border border-slate-800/90 bg-slate-900/70 p-2.5 text-[10px] space-y-2">
      {/* Pipeline Stepper Nodes */}
      <div className="flex flex-wrap items-center gap-1.5">
        {displaySteps.map((step, idx) => {
          const done = completedNodes.has(step.node);
          const prevDone =
            idx === 0 || completedNodes.has(displaySteps[idx - 1].node);
          const isCurrent = Boolean(isStreaming && !done && prevDone);

          return (
            <React.Fragment key={step.node}>
              {idx > 0 && <span className="text-slate-600">→</span>}
              <span
                className={`inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 font-medium transition ${
                  done
                    ? "bg-emerald-500/10 text-emerald-300"
                    : isCurrent
                    ? "bg-indigo-500/20 text-indigo-300 animate-pulse"
                    : "text-slate-500"
                }`}
              >
                {done ? (
                  <CheckCircle2 className="h-2.5 w-2.5 text-emerald-400" />
                ) : isCurrent ? (
                  <Loader2 className="h-2.5 w-2.5 animate-spin text-indigo-400" />
                ) : null}
                {step.label}
              </span>
            </React.Fragment>
          );
        })}

        {/* Active routing mode pill */}
        {activeRoutingMode && (
          <span
            className={`ml-auto inline-flex items-center gap-1 rounded-full px-2 py-0.2 font-mono text-[9.5px] font-semibold border ${
              activeRoutingMode === "federated"
                ? "border-purple-500/40 bg-purple-500/15 text-purple-300"
                : "border-sky-500/40 bg-sky-500/15 text-sky-300"
            }`}
          >
            {activeRoutingMode === "federated" ? "🔀 Federated" : "🎯 Single"}
          </span>
        )}
      </div>

      {/* Polysemy Collision Warnings */}
      {collisions.length > 0 && (
        <div className="rounded-lg border border-amber-500/30 bg-amber-950/30 p-2 text-amber-200 space-y-1">
          <div className="flex items-center gap-1.5 font-semibold text-amber-300">
            <AlertTriangle className="h-3.5 w-3.5 text-amber-400 shrink-0" />
            <span>Cross-Domain Polysemic Collisions Detected:</span>
          </div>
          {collisions.map((col, cIdx) => (
            <div key={cIdx} className="text-[10px] text-amber-100/90 pl-5">
              <strong>"{col.term}"</strong> across [{Array.isArray(col.domains) ? col.domains.join(", ") : ""}]:{" "}
              <span>{col.contrastive_warning}</span>
            </div>
          ))}
        </div>
      )}

      {/* Decomposed Subqueries Accordion */}
      {Object.keys(domainQueries).length > 0 && (
        <div className="border-t border-slate-800/80 pt-1.5">
          <button
            type="button"
            onClick={() => setShowSubqueries((v) => !v)}
            className="flex items-center gap-1 text-[10px] text-indigo-300 hover:text-indigo-200 transition"
          >
            {showSubqueries ? (
              <ChevronDown className="h-3 w-3" />
            ) : (
              <ChevronRight className="h-3 w-3" />
            )}
            <span className="font-medium">
              Domain Sub-Queries ({Object.keys(domainQueries).length})
            </span>
          </button>
          {showSubqueries && (
            <div className="mt-1.5 grid gap-1 pl-4">
              {Object.entries(domainQueries).map(([dom, qText]) => (
                <div key={dom} className="text-[10px]">
                  <span className="font-semibold uppercase text-indigo-400">
                    [{dom}]:
                  </span>{" "}
                  <span className="text-slate-300 font-mono italic">
                    "{qText}"
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/**
 * Domain-Grouped Sources & Citations Drawer.
 */
function SourcesDrawer({ citations }: { citations?: Citation[] | null }) {
  const [expanded, setExpanded] = useState(true);
  const { activeCitation, setActiveCitation } = useWorkspace();
  const { data: domains = [] } = useDomainsQuery();

  const safeCitations = useMemo(
    () => (Array.isArray(citations) ? citations : []),
    [citations]
  );

  // Group citations by domain - must execute unconditionally on EVERY render
  const domainGroups = useMemo(() => {
    const groups: Record<string, Citation[]> = {};
    for (const c of safeCitations) {
      const meta = resolveDomainMeta(c.domain, domains);
      const slug = meta.slug;
      if (!groups[slug]) {
        groups[slug] = [];
      }
      groups[slug].push(c);
    }
    return groups;
  }, [safeCitations, domains]);

  const activeGroupKeys = useMemo(
    () => Object.keys(domainGroups).filter((k) => domainGroups[k].length > 0),
    [domainGroups]
  );

  if (safeCitations.length === 0) return null;

  return (
    <div className="mt-3 rounded-xl border border-slate-800/90 bg-slate-900/50 overflow-hidden">
      <button
        type="button"
        onClick={() => setExpanded((v) => !v)}
        className="flex w-full items-center justify-between px-3 py-2 text-left text-[11px] font-semibold text-slate-300 hover:bg-slate-800/50 transition"
      >
        <span className="flex items-center gap-1.5">
          <Sparkles className="h-3.5 w-3.5 text-indigo-400" />
          <span>Multi-Domain Sources & Citations ({safeCitations.length})</span>
        </span>
        {expanded ? (
          <ChevronDown className="h-3.5 w-3.5 text-slate-400" />
        ) : (
          <ChevronRight className="h-3.5 w-3.5 text-slate-400" />
        )}
      </button>

      {expanded && (
        <div className="divide-y divide-slate-800/70 border-t border-slate-800/80">
          {activeGroupKeys.map((groupKey) => {
            const list = domainGroups[groupKey];
            const meta = resolveDomainMeta(groupKey, domains);
            const groupDomain = `${meta.icon} ${meta.name}`;

            return (
              <div key={groupKey} className="p-2.5 space-y-1.5">
                <div className="flex items-center justify-between text-[10.5px] font-semibold text-slate-400">
                  <span>{groupDomain}</span>
                  <span className="font-mono text-[9.5px] text-slate-500">
                    {list.length} chunk{list.length > 1 ? "s" : ""}
                  </span>
                </div>

                <div className="space-y-1">
                  {list.map((cit) => {
                    const isSelected =
                      activeCitation?.doc_id === cit.doc_id &&
                      activeCitation?.citation_index === cit.citation_index;
                    const colorCls = getCitationDomainColor(cit.domain);

                    return (
                      <div
                        key={`${cit.citation_index}-${cit.chunk_id}`}
                        onClick={() => setActiveCitation(cit)}
                        className={`rounded-lg p-2 text-xs transition cursor-pointer border ${
                          isSelected
                            ? "bg-indigo-950/60 border-indigo-400"
                            : "border-slate-800/70 bg-slate-900/40 hover:bg-slate-800/50"
                        }`}
                      >
                        <div className="flex items-center justify-between gap-2">
                          <div className="flex items-center gap-1.5 min-w-0">
                            <span
                              className={`rounded border px-1.5 py-0.2 font-mono text-[10px] font-semibold ${colorCls.border} ${colorCls.bg} ${colorCls.text}`}
                            >
                              [{cit.citation_index}]
                            </span>
                            <FileText className="h-3.5 w-3.5 text-slate-400 shrink-0" />
                            <span
                              className="truncate font-medium text-slate-200"
                              title={cit.source_filename}
                            >
                              {cit.source_filename}
                            </span>
                          </div>

                          <div className="flex items-center gap-1.5 shrink-0 text-[10px]">
                            {cit.authority_tier && (
                              <AuthorityTierBadge tier={cit.authority_tier} />
                            )}
                            {Array.isArray(cit.page_numbers) && cit.page_numbers.length > 0 && (
                              <span className="rounded bg-slate-800 px-1.5 py-0.5 text-slate-300">
                                p.{cit.page_numbers.join(", ")}
                              </span>
                            )}
                            {typeof cit.score === "number" && (
                              <span className="font-mono text-emerald-400">
                                {cit.score.toFixed(2)}
                              </span>
                            )}
                          </div>
                        </div>

                        {cit.heading_path && (
                          <p className="mt-1 truncate text-[10px] text-indigo-300/90">
                            § {cit.heading_path}
                          </p>
                        )}

                        {cit.excerpt && (
                          <p className="mt-1 line-clamp-2 text-[11px] text-slate-400 italic leading-snug">
                            "{cit.excerpt}"
                          </p>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

function AssistantMessageContent({ message }: { message: ChatMessage }) {
  const { data: domains = [] } = useDomainsQuery();

  const citationMap = useMemo(() => {
    const map = new Map<number, Citation>();
    for (const c of message.citations || []) {
      map.set(c.citation_index, c);
    }
    return map;
  }, [message.citations]);

  const markdownWithLinks = useMemo(
    () => injectCitationLinks(message.content),
    [message.content]
  );

  const markdownComponents = useMemo<Components>(
    () => ({
      a: ({ href, children }) => {
        if (href && href.startsWith("#citation-")) {
          const idxNum = Number.parseInt(href.replace("#citation-", ""), 10);
          if (!Number.isNaN(idxNum)) {
            return (
              <CitationPill
                index={idxNum}
                citation={citationMap.get(idxNum)}
              />
            );
          }
        }
        return (
          <a
            href={href}
            target="_blank"
            rel="noreferrer"
            className="text-indigo-400 underline hover:text-indigo-300"
          >
            {children}
          </a>
        );
      },
      // Distinct Epistemic Perspective Header Bars styling
      h2: ({ children }) => {
        const text = String(children);
        const lower = text.toLowerCase();
        let barCls = "border-l-4 border-indigo-500 bg-indigo-950/20 text-indigo-200";

        if (
          lower.includes("conflict") ||
          lower.includes("trade-off") ||
          lower.includes("xung đột") ||
          lower.includes("matrix")
        ) {
          barCls = "border-l-4 border-violet-400 bg-violet-950/40 text-violet-200";
        } else {
          for (const d of domains) {
            const dName = d.name.toLowerCase();
            const dSlug = d.slug.toLowerCase();
            const dEpistemic = (d.epistemic_type || "").toLowerCase();
            if (
              lower.includes(dName) ||
              lower.includes(dSlug) ||
              (dEpistemic && lower.includes(dEpistemic))
            ) {
              const meta = resolveDomainMeta(d.slug, domains);
              barCls = meta.theme.heading.h2;
              break;
            }
          }
        }

        return (
          <h2 className={`mt-4 mb-2 rounded-r-lg px-2.5 py-1 text-sm font-bold ${barCls}`}>
            {children}
          </h2>
        );
      },
      h3: ({ children }) => {
        const text = String(children);
        const lower = text.toLowerCase();
        let barCls = "text-indigo-300";

        if (
          lower.includes("conflict") ||
          lower.includes("priority") ||
          lower.includes("matrix")
        ) {
          barCls = "text-violet-300 border-b border-violet-500/30 pb-0.5";
        } else {
          for (const d of domains) {
            const dName = d.name.toLowerCase();
            const dSlug = d.slug.toLowerCase();
            const dEpistemic = (d.epistemic_type || "").toLowerCase();
            if (
              lower.includes(dName) ||
              lower.includes(dSlug) ||
              (dEpistemic && lower.includes(dEpistemic))
            ) {
              const meta = resolveDomainMeta(d.slug, domains);
              barCls = meta.theme.heading.h3;
              break;
            }
          }
        }

        return (
          <h3 className={`mt-3 mb-1 text-xs font-semibold ${barCls}`}>
            {children}
          </h3>
        );
      },
      table: ({ children }) => (
        <div className="my-2.5 overflow-x-auto rounded-xl border border-slate-700/80 bg-slate-950/70 p-1 shadow-inner">
          <table className="w-full border-collapse text-xs">
            {children}
          </table>
        </div>
      ),
    }),
    [citationMap]
  );

  if (message.isError) {
    return (
      <div className="flex items-start gap-2 rounded-xl border border-rose-500/40 bg-rose-950/40 p-3 text-xs text-rose-200">
        <AlertTriangle className="h-4 w-4 shrink-0 text-rose-400 mt-0.5" />
        <div className="space-y-1 min-w-0">
          <p className="font-semibold text-rose-300">Assistant Error</p>
          <p className="break-words leading-relaxed">{message.content}</p>
        </div>
      </div>
    );
  }

  return (
    <div>
      {message.steps && message.steps.length > 0 && (
        <CRAGStepper steps={message.steps} isStreaming={message.isStreaming} />
      )}

      {message.content ? (
        <div className="prose prose-invert max-w-none text-xs leading-relaxed text-slate-100 space-y-2 [&_p]:my-1.5 [&_ul]:list-disc [&_ul]:pl-4 [&_ol]:list-decimal [&_ol]:pl-4 [&_li]:my-0.5 [&_th]:border [&_th]:border-slate-700 [&_th]:bg-slate-900 [&_th]:px-2.5 [&_th]:py-1.5 [&_th]:text-left [&_th]:font-semibold [&_td]:border [&_td]:border-slate-800 [&_td]:px-2.5 [&_td]:py-1.5 [&_code]:rounded [&_code]:bg-slate-800 [&_code]:px-1 [&_code]:py-0.5 [&_pre]:rounded-lg [&_pre]:bg-slate-950 [&_pre]:p-2.5 [&_pre]:overflow-x-auto">
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={markdownComponents}
          >
            {markdownWithLinks}
          </ReactMarkdown>
        </div>
      ) : message.isStreaming ? (
        <div className="flex items-center gap-2 py-1 text-xs text-slate-400">
          <Loader2 className="h-3.5 w-3.5 animate-spin text-indigo-400" />
          <span>Running Multi-Domain CRAG pipeline...</span>
        </div>
      ) : null}

      <SourcesDrawer citations={message.citations} />

      {message.model && (
        <div className="mt-2 flex items-center gap-1.5 text-[10px] text-slate-500 font-mono">
          <Cpu className="h-3 w-3 text-indigo-400/80" />
          <span>
            {message.provider ? `${message.provider} / ` : ""}
            <strong className="text-slate-400 font-semibold">{message.model}</strong>
          </span>
        </div>
      )}
    </div>
  );
}

export const ChatAssistant: React.FC = () => {
  const { data: domains = [] } = useDomainsQuery();
  const {
    activeThreadId,
    setActiveThreadId,
    selectedFilterDocIds,
    selectedProvider,
    setSelectedProvider,
    selectedModel,
    setSelectedModel,
    currentDomainFilter,
    migrateDraftDomainFilter,
    openSettingsModal,
  } = useWorkspace();

  const [inputValue, setInputValue] = useState("");
  const [showModelConfig, setShowModelConfig] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  const { data: threadsData } = useChatThreads();
  const { data: stats } = useSystemStats();
  const { data: wsSettings } = useWorkspaceSettingsQuery();

  const {
    messages,
    isStreaming,
    isLoadingHistory,
    sendMessage,
    stopStreaming,
    clearConversation,
    removeThread,
  } = useChatStream(activeThreadId, (resolvedId) => {
    setActiveThreadId(resolvedId || null);
    if (resolvedId) {
      migrateDraftDomainFilter(resolvedId);
    }
  });

  const threads = threadsData?.threads ?? threadsData?.items ?? [];
  const providers = wsSettings?.available_providers?.length
    ? wsSettings.available_providers
    : (stats?.available_providers ?? stats?.configured_providers ?? []);

  const activeProv =
    selectedProvider ||
    wsSettings?.default_llm_provider ||
    stats?.llm?.provider ||
    "openai";
  const activeProviderOption = providers.find((p) => p.provider === activeProv);
  const isMissingKey = Boolean(
    activeProviderOption && !activeProviderOption.configured
  );
  const isInvalidKey = Boolean(
    activeProviderOption && activeProviderOption.validation_status === "invalid"
  );
  const hasKeyIssue = isMissingKey || isInvalidKey;

  const hasInitializedModelRef = useRef(false);

  // Sync default provider/model from workspace settings or system stats
  useEffect(() => {
    if (wsSettings) {
      const nextProv =
        wsSettings.default_llm_provider || stats?.llm?.provider || "";
      const nextMod =
        wsSettings.default_llm_model || stats?.llm?.model_name || "";
      if (nextProv) {
        setSelectedProvider(nextProv);
      }
      if (nextMod) {
        setSelectedModel(nextMod);
      }
      hasInitializedModelRef.current = true;
      return;
    }

    if (!hasInitializedModelRef.current && stats?.llm) {
      if (!selectedProvider && stats.llm.provider) {
        setSelectedProvider(stats.llm.provider);
      }
      if (!selectedModel && stats.llm.model_name) {
        setSelectedModel(stats.llm.model_name);
      }
      if (stats.llm.provider || stats.llm.model_name) {
        hasInitializedModelRef.current = true;
      }
    }
  }, [
    wsSettings?.default_llm_provider,
    wsSettings?.default_llm_model,
    stats?.llm?.provider,
    stats?.llm?.model_name,
    setSelectedProvider,
    setSelectedModel,
  ]);

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputValue.trim() || isStreaming) return;
    const text = inputValue;
    setInputValue("");
    void sendMessage({
      message: text,
      threadId: activeThreadId,
      filterDocIds: selectedFilterDocIds,
      domainFilter: currentDomainFilter,
      provider: selectedProvider || null,
      model: selectedModel || null,
    });
  };

  const handleNewThread = () => {
    clearConversation();
    setActiveThreadId(null);
  };

  return (
    <div className="flex h-full flex-col bg-slate-950 border-l border-slate-800/80">
      {/* Header: Thread Selector + New Thread + Model Toggle */}
      <div className="border-b border-slate-800/80 px-3.5 py-2 space-y-2">
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <div className="flex h-6 w-6 items-center justify-center rounded-md bg-indigo-500/20 text-indigo-400 border border-indigo-500/30">
              <Bot className="h-3.5 w-3.5" />
            </div>
            <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-200">
              Intelligent Assistant
            </h2>
          </div>

          <div className="flex items-center gap-1.5">
            <button
              type="button"
              onClick={() => setShowModelConfig((v) => !v)}
              title={
                hasKeyIssue
                  ? isInvalidKey
                    ? `Invalid API key for ${activeProviderOption?.label || activeProv}. Click to reconfigure.`
                    : `Missing API key for ${activeProviderOption?.label || activeProv}. Click to configure.`
                  : "Configure LLM Provider & Model"
              }
              className={`inline-flex items-center gap-1.5 rounded-md border px-2 py-1 text-[11px] font-medium transition ${
                showModelConfig
                  ? "border-indigo-500 bg-indigo-500/20 text-indigo-200"
                  : hasKeyIssue
                  ? isInvalidKey
                    ? "border-rose-500/60 bg-rose-950/30 text-rose-300 hover:border-rose-400"
                    : "border-amber-500/60 bg-amber-950/30 text-amber-300 hover:border-amber-400"
                  : "border-slate-800 bg-slate-900 text-slate-300 hover:text-white"
              }`}
            >
              <Cpu className="h-3.5 w-3.5 text-indigo-400" />
              <span className="text-slate-400">{activeProv}:</span>
              <span className="text-indigo-300 font-mono font-medium max-w-[120px] truncate">
                {selectedModel || stats?.llm?.model_name || "gpt-4o"}
              </span>
              {hasKeyIssue && (
                <span
                  className={`h-2 w-2 rounded-full ${
                    isInvalidKey ? "bg-rose-500 animate-pulse" : "bg-amber-400"
                  }`}
                />
              )}
            </button>

            <button
              type="button"
              onClick={handleNewThread}
              title="Start new conversation thread"
              className="inline-flex items-center gap-1 rounded-md bg-indigo-600 px-2 py-1 text-[11px] font-medium text-white hover:bg-indigo-500 transition"
            >
              <MessageSquarePlus className="h-3 w-3" />
              <span>New</span>
            </button>
          </div>
        </div>

        {/* Thread selector bar */}
        <div className="flex items-center gap-1.5">
          <select
            aria-label="Conversation Thread"
            value={activeThreadId || ""}
            onChange={(e) => {
              const val = e.target.value;
              setActiveThreadId(val ? val : null);
            }}
            className="flex-1 rounded-lg border border-slate-800 bg-slate-900 px-2.5 py-1 text-xs text-slate-200 focus:border-indigo-500 focus:outline-none"
          >
            <option value="">New Conversation Thread</option>
            {threads.map((t) => (
              <option key={t.thread_id} value={t.thread_id}>
                {t.title || `Thread ${t.thread_id.slice(0, 8)}`}
              </option>
            ))}
          </select>

          {activeThreadId && (
            <button
              type="button"
              onClick={() => void removeThread(activeThreadId)}
              title="Delete active conversation thread"
              className="rounded-lg border border-slate-800 bg-slate-900 p-1.5 text-slate-400 hover:border-rose-500/40 hover:bg-rose-500/15 hover:text-rose-300 transition"
            >
              <Trash2 className="h-3.5 w-3.5" />
            </button>
          )}
        </div>

        {/* Sticky Domain Routing Bar */}
        <DomainRoutingBar />

        {/* Collapsible Per-Request Provider / Model Override Selector */}
        {showModelConfig && (
          <div className="rounded-xl border border-slate-800 bg-slate-900/80 p-2.5 space-y-2 text-xs">
            <div className="flex items-center justify-between text-[11px] text-slate-400">
              <span>Per-Request LLM Override</span>
              <span className="text-[10px] text-slate-500">
                Default: {stats?.llm?.provider || "openai"} /{" "}
                {stats?.llm?.model_name || "gpt-4o-mini"}
              </span>
            </div>
            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="block text-[10px] uppercase text-slate-400 mb-1">
                  Provider
                </label>
                <select
                  aria-label="LLM Provider"
                  value={selectedProvider}
                  onChange={(e) => {
                    const nextProv = e.target.value;
                    setSelectedProvider(nextProv);
                    const matched = providers.find(
                      (p) => p.provider === nextProv
                    );
                    if (matched?.default_model) {
                      setSelectedModel(matched.default_model);
                    } else {
                      if (nextProv === "deepseek") setSelectedModel("deepseek-v4-pro");
                      else if (nextProv === "anthropic") setSelectedModel("claude-sonnet-5-5");
                      else if (nextProv === "openai") setSelectedModel("gpt-4o");
                      else if (nextProv === "openai_compatible") setSelectedModel("glm-4-plus");
                    }
                  }}
                  className="w-full rounded-md border border-slate-700 bg-slate-950 px-2 py-1 text-xs text-slate-200 focus:border-indigo-500 focus:outline-none"
                >
                  {providers.map((p) => (
                    <option key={p.provider} value={p.provider}>
                      {p.label} {p.configured ? "✓" : ""}
                    </option>
                  ))}
                  {providers.length === 0 && (
                    <>
                      <option value="openai">OpenAI</option>
                      <option value="anthropic">Anthropic</option>
                      <option value="deepseek">DeepSeek</option>
                      <option value="openai_compatible">
                        OpenAI-Compatible
                      </option>
                    </>
                  )}
                </select>
              </div>
              <div>
                <label className="block text-[10px] uppercase text-slate-400 mb-1">
                  Model Name
                </label>
                <input
                  type="text"
                  list="model-suggestions"
                  aria-label="LLM Model Name"
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  placeholder={stats?.llm?.model_name || "e.g. gpt-4o-mini"}
                  className="w-full rounded-md border border-slate-700 bg-slate-950 px-2 py-1 text-xs text-slate-200 placeholder:text-slate-600 focus:border-indigo-500 focus:outline-none"
                />
                <datalist id="model-suggestions">
                  {selectedProvider === "openai" && (
                    <>
                      <option value="gpt-5" />
                      <option value="gpt-5-mini" />
                      <option value="gpt-4o" />
                      <option value="gpt-4o-mini" />
                      <option value="o3-mini" />
                      <option value="o1" />
                    </>
                  )}
                  {selectedProvider === "deepseek" && (
                    <>
                      <option value="deepseek-v4-pro" />
                      <option value="deepseek-flash" />
                      <option value="deepseek-chat" />
                      <option value="deepseek-reasoner" />
                    </>
                  )}
                  {selectedProvider === "anthropic" && (
                    <>
                      <option value="claude-sonnet-5-5" />
                      <option value="claude-opus-5-5" />
                      <option value="claude-opus-4-8" />
                      <option value="claude-sonnet-4-6" />
                      <option value="claude-sonnet-4-5-20250929" />
                      <option value="claude-haiku-4-5-20251001" />
                    </>
                  )}
                  {selectedProvider === "openai_compatible" && (
                    <>
                      <option value="glm-4-plus" />
                      <option value="qwen-plus" />
                      <option value="llama-3.3-70b" />
                    </>
                  )}
                </datalist>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-3.5 space-y-4">
        {isLoadingHistory ? (
          <div className="flex flex-col items-center justify-center py-12 text-slate-400 gap-2">
            <Loader2 className="h-5 w-5 animate-spin text-indigo-400" />
            <span className="text-xs">Loading conversation history...</span>
          </div>
        ) : messages.length === 0 ? (
          <div className="flex h-full flex-col items-center justify-center text-center px-4">
            <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 mb-3">
              <Sparkles className="h-6 w-6" />
            </div>
            <h3 className="text-xs font-semibold text-slate-200">
              Multi-Domain Corrective RAG Assistant
            </h3>
            <p className="mt-1 max-w-xs text-[11px] leading-relaxed text-slate-400">
              Ask questions across Engineering, Economics, and Law documents.
              Polysemic terms are automatically disambiguated with domain-colored
              citations <code className="text-cyan-300">[1]</code>.
            </p>
            <div className="mt-4 grid w-full max-w-xs gap-1.5">
              {[
                "Evaluate the technical failure risks and statutory environmental liabilities of leaking PCB transformers.",
                "Compare physical asset degradation with MACRS balance sheet depreciation rules.",
              ].map((prompt) => (
                <button
                  key={prompt}
                  type="button"
                  onClick={() => setInputValue(prompt)}
                  className="rounded-xl border border-slate-800/90 bg-slate-900/60 px-3 py-2 text-left text-[11px] text-slate-300 hover:border-indigo-500/50 hover:bg-slate-900 transition"
                >
                  "{prompt}"
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((msg, idx) => {
            const isUser = msg.role === "user";

            // Format turn domain scope badge
            let scopeBadge = null;
            if (isUser && msg.domain_filter) {
              const df = msg.domain_filter;
              const hasDomains = Array.isArray(df.domains) && df.domains.length > 0;
              const isT2 = df.max_authority_tier === 2;

              if (hasDomains) {
                const domNames = df.domains
                  .map((d) => resolveDomainMeta(d, domains).shortLabel)
                  .join(", ");
                scopeBadge = (
                  <span className="inline-flex items-center gap-1 rounded-full border border-indigo-500/40 bg-indigo-950/60 px-2 py-0.2 text-[9.5px] font-mono text-indigo-300">
                    <span>{df.mode === "federated" ? "🔀" : "🎯"}</span>
                    <span>{domNames}</span>
                    {isT2 && <span className="text-amber-300">(T1–2)</span>}
                  </span>
                );
              } else if (isT2) {
                scopeBadge = (
                  <span className="inline-flex items-center gap-1 rounded-full border border-amber-500/40 bg-amber-950/60 px-2 py-0.2 text-[9.5px] font-mono text-amber-300">
                    <span>🛡️ T1–2 Only</span>
                  </span>
                );
              }
            }

            return (
              <div
                key={msg.id || idx}
                className={`flex gap-2.5 ${
                  isUser ? "justify-end" : "justify-start"
                }`}
              >
                {!isUser && (
                  <div className="flex h-6 w-6 shrink-0 items-center justify-center rounded-lg bg-indigo-600/20 border border-indigo-500/30 text-indigo-400 mt-0.5">
                    <Bot className="h-3.5 w-3.5" />
                  </div>
                )}

                <div
                  className={`max-w-[88%] rounded-2xl px-3.5 py-2.5 ${
                    isUser
                      ? "bg-indigo-600 text-white text-xs leading-relaxed"
                      : "bg-slate-900/90 border border-slate-800/90 text-slate-100 w-full"
                  }`}
                >
                  {isUser ? (
                    <div>
                      {scopeBadge && <div className="mb-1.5">{scopeBadge}</div>}
                      <p className="whitespace-pre-wrap break-words">
                        {msg.content}
                      </p>
                    </div>
                  ) : (
                    <ErrorBoundary fallbackTitle="Response Render Error">
                      <AssistantMessageContent message={msg} />
                    </ErrorBoundary>
                  )}
                </div>

                {isUser && (
                  <div className="flex h-6 w-6 shrink-0 items-center justify-center rounded-lg bg-slate-800 text-slate-300 mt-0.5">
                    <User className="h-3.5 w-3.5" />
                  </div>
                )}
              </div>
            );
          })
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Composer Footer */}
      <form
        onSubmit={handleSubmit}
        className="border-t border-slate-800/80 bg-slate-950 p-3 space-y-2"
      >
        {selectedFilterDocIds.length > 0 && (
          <div className="flex items-center gap-1.5 text-[11px] text-indigo-300">
            <Filter className="h-3 w-3 text-indigo-400" />
            <span>
              Query scoped to{" "}
              <strong>{selectedFilterDocIds.length}</strong> selected document
              {selectedFilterDocIds.length > 1 ? "s" : ""}
            </span>
          </div>
        )}

        {/* Surface 2: Proactive Chat Composer Alert Banner */}
        {hasKeyIssue && (
          <div
            className={`flex items-center justify-between gap-2.5 rounded-xl border p-2.5 text-xs ${
              isInvalidKey
                ? "border-rose-500/40 bg-rose-950/40 text-rose-200"
                : "border-amber-500/40 bg-amber-950/40 text-amber-200"
            }`}
          >
            <div className="flex items-start gap-2 min-w-0">
              <AlertTriangle
                className={`h-4 w-4 shrink-0 mt-0.5 ${
                  isInvalidKey ? "text-rose-400" : "text-amber-400"
                }`}
              />
              <div className="space-y-0.5 min-w-0">
                <p className="font-semibold text-[11.5px] leading-tight truncate">
                  {isInvalidKey
                    ? activeProviderOption?.validation_error ||
                      `API key verification failed for ${activeProviderOption?.label || activeProv}.`
                    : `No API key configured for ${activeProviderOption?.label || activeProv}.`}
                </p>
                <p className="text-[10.5px] opacity-80 leading-tight">
                  {isInvalidKey
                    ? "Upstream rejected this key. Update it in Workspace Settings."
                    : "No valid key found in Workspace Settings or backend/.env."}
                </p>
              </div>
            </div>
            <button
              type="button"
              onClick={() => openSettingsModal("ingestion")}
              className={`shrink-0 inline-flex items-center gap-1 rounded-lg px-2.5 py-1 text-[11px] font-semibold transition cursor-pointer shadow-xs ${
                isInvalidKey
                  ? "bg-rose-600 text-white hover:bg-rose-500"
                  : "bg-amber-600 text-white hover:bg-amber-500"
              }`}
            >
              <span>Configure API Key</span>
              <span>&rarr;</span>
            </button>
          </div>
        )}

        <div className="flex items-end gap-2">
          <textarea
            rows={2}
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                handleSubmit(e);
              }
            }}
            placeholder="Ask your Second Brain... (Enter to send, Shift+Enter for newline)"
            disabled={isStreaming}
            className="flex-1 resize-none rounded-xl border border-slate-800 bg-slate-900/90 px-3 py-2 text-xs text-slate-100 placeholder:text-slate-500 focus:border-indigo-500 focus:outline-none disabled:opacity-60"
          />

          {isStreaming ? (
            <button
              type="button"
              onClick={stopStreaming}
              title="Stop generation"
              className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-rose-600 text-white hover:bg-rose-500 transition"
            >
              <Square className="h-3.5 w-3.5 fill-current" />
            </button>
          ) : (
            <button
              type="submit"
              disabled={!inputValue.trim()}
              title="Send message"
              className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-indigo-600 text-white hover:bg-indigo-500 disabled:opacity-40 disabled:cursor-not-allowed transition"
            >
              <Send className="h-3.5 w-3.5" />
            </button>
          )}
        </div>
      </form>
    </div>
  );
};
