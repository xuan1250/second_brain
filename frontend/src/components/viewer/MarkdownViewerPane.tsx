import React, { useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  AlertCircle,
  Check,
  Hash,
  Loader2,
  Sparkles,
} from "lucide-react";
import { useDocumentMarkdown } from "../../hooks/useDocuments";
import type {
  ActiveCitation,
  DocumentItem,
  HeadingOutlineItem,
  RecommendedDomainProfile,
  SectionOverride,
} from "../../lib/types";
import { useDomainsQuery } from "../../hooks/useDomains";
import { resolveDomainMeta } from "../../lib/domainTheme";
import { getDomainKey } from "../library/KnowledgeLibrary";

export interface MarkdownViewerPaneProps {
  docId: string;
  activeDoc?: DocumentItem | null;
  activeCitation: ActiveCitation | null;
  scale: number;
  onUpdateSectionOverride?: (override: SectionOverride) => void;
  sectionOverrides?: SectionOverride[];
}

function normalizeForMatch(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, " ")
    .replace(/\s+/g, " ")
    .trim();
}

export const MarkdownViewerPane: React.FC<MarkdownViewerPaneProps> = ({
  docId,
  activeDoc,
  activeCitation,
  scale,
  onUpdateSectionOverride,
  sectionOverrides,
}) => {
  const { data: domains = [] } = useDomainsQuery();
  const { data, isLoading, isError, error } = useDocumentMarkdown(docId, true);
  const contentContainerRef = useRef<HTMLDivElement | null>(null);

  const [activeOverrideHeading, setActiveOverrideHeading] = useState<string | null>(null);

  const isCitedDoc = activeCitation?.doc_id === docId;

  // Parse section overrides from activeDoc and local overrides prop
  const overridesMap = useMemo(() => {
    const map = new Map<string, SectionOverride>();
    if (activeDoc?.classification_result) {
      try {
        const parsed = JSON.parse(
          activeDoc.classification_result
        ) as RecommendedDomainProfile;
        if (Array.isArray(parsed.section_overrides)) {
          for (const ov of parsed.section_overrides) {
            if (ov.heading_path) {
              map.set(ov.heading_path.toLowerCase(), ov);
            }
          }
        }
      } catch {
        // ignore
      }
    }
    if (Array.isArray(sectionOverrides)) {
      for (const ov of sectionOverrides) {
        if (ov.heading_path) {
          map.set(ov.heading_path.toLowerCase(), ov);
        }
      }
    }
    return map;
  }, [activeDoc?.classification_result, sectionOverrides]);

  const docDomain = activeDoc ? getDomainKey(activeDoc) : "general";

  const targetSnippet = useMemo(() => {
    if (!isCitedDoc || !activeCitation) return "";
    const rawExcerpt = (activeCitation.excerpt || "").trim();
    if (rawExcerpt.length >= 12) {
      return normalizeForMatch(rawExcerpt.slice(0, 90));
    }
    return "";
  }, [isCitedDoc, activeCitation]);

  const targetHeadingLeaf = useMemo(() => {
    if (!isCitedDoc || !activeCitation?.heading_path) return "";
    const parts = activeCitation.heading_path
      .split(">")
      .map((s) => s.trim())
      .filter(Boolean);
    return parts.length > 0 ? normalizeForMatch(parts[parts.length - 1]) : "";
  }, [isCitedDoc, activeCitation?.heading_path]);

  // Scroll to cited heading or excerpt and apply pulse highlight
  useEffect(() => {
    const container = contentContainerRef.current;
    if (!container || !data?.markdown || !isCitedDoc || !activeCitation) {
      return;
    }

    // Clear previous highlights
    container
      .querySelectorAll(".citation-pulse-highlight")
      .forEach((el) => el.classList.remove("citation-pulse-highlight"));

    const timer = window.setTimeout(() => {
      let matchedElement: HTMLElement | null = null;

      // 1. Try matching excerpt text across paragraphs, list items, table cells, or blockquotes
      if (targetSnippet) {
        const snippetWords = targetSnippet.split(" ").slice(0, 7).join(" ");
        const blocks = Array.from(
          container.querySelectorAll<HTMLElement>(
            "p, li, tr, td, blockquote, pre, h1, h2, h3, h4, h5, h6"
          )
        );
        for (const el of blocks) {
          const normText = normalizeForMatch(el.textContent || "");
          if (
            normText &&
            snippetWords.length >= 6 &&
            normText.includes(snippetWords)
          ) {
            matchedElement = el;
            break;
          }
        }
      }

      // 2. Fallback to matching heading_path leaf across h1..h6
      if (!matchedElement && targetHeadingLeaf) {
        const headings = Array.from(
          container.querySelectorAll<HTMLElement>("h1, h2, h3, h4, h5, h6")
        );
        for (const h of headings) {
          const normHeading = normalizeForMatch(h.textContent || "");
          if (
            normHeading &&
            (normHeading.includes(targetHeadingLeaf) ||
              targetHeadingLeaf.includes(normHeading))
          ) {
            matchedElement = h;
            break;
          }
        }
      }

      if (matchedElement) {
        matchedElement.classList.add("citation-pulse-highlight");
        matchedElement.scrollIntoView({
          behavior: "smooth",
          block: "center",
        });
      }
    }, 120);

    return () => window.clearTimeout(timer);
  }, [
    data?.markdown,
    isCitedDoc,
    activeCitation,
    targetSnippet,
    targetHeadingLeaf,
  ]);

  const scrollToHeading = (heading: HeadingOutlineItem) => {
    const container = contentContainerRef.current;
    if (!container) return;
    const normTarget = normalizeForMatch(heading.text);
    const headingEls = Array.from(
      container.querySelectorAll<HTMLElement>("h1, h2, h3, h4, h5, h6")
    );
    const found = headingEls.find(
      (el) => normalizeForMatch(el.textContent || "") === normTarget
    );
    if (found) {
      container
        .querySelectorAll(".citation-pulse-highlight")
        .forEach((el) => el.classList.remove("citation-pulse-highlight"));
      found.classList.add("citation-pulse-highlight");
      found.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  };

  const getHeadingDomain = (h: HeadingOutlineItem): { domain: string; isOverride: boolean } => {
    const pathKey = (h.heading_path || h.text).toLowerCase();
    const ov = overridesMap.get(pathKey);
    if (ov) {
      return { domain: ov.domain, isOverride: true };
    }
    return { domain: docDomain, isOverride: false };
  };

  const handleSelectDomainOverride = (
    h: HeadingOutlineItem,
    newDomain: string
  ) => {
    setActiveOverrideHeading(null);
    if (onUpdateSectionOverride) {
      const meta = resolveDomainMeta(newDomain, domains);
      onUpdateSectionOverride({
        heading_path: h.heading_path || h.text,
        domain: meta.slug,
        subdomain: meta.subdomains?.[0]?.slug || meta.slug,
        epistemic_modality: meta.epistemicType,
        confidence: 1.0,
        authority_tier: activeDoc?.authority_tier ?? 3,
      });
    }
  };

  if (isLoading) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-2 text-slate-400">
        <Loader2 className="h-6 w-6 animate-spin text-indigo-400" />
        <span className="text-xs">Loading structured Markdown mirror...</span>
      </div>
    );
  }

  if (isError || !data) {
    return (
      <div className="flex h-full flex-col items-center justify-center p-6 text-center">
        <div className="max-w-md rounded-xl border border-amber-500/30 bg-amber-950/20 p-4 text-xs text-amber-200">
          <div className="flex items-center justify-center gap-1.5 font-semibold text-amber-300 mb-1">
            <AlertCircle className="h-4 w-4" />
            <span>Markdown Mirror Not Ready</span>
          </div>
          <p className="text-slate-300">
            {error instanceof Error
              ? error.message
              : "Document is still being converted or Markdown output is unavailable."}
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-full overflow-hidden bg-slate-950">
      {/* Section Outline Sidebar with Interactive Mini Domain Badges */}
      {data.headings && data.headings.length > 0 && (
        <aside className="hidden xl:flex w-56 shrink-0 flex-col border-r border-slate-800/80 bg-slate-900/30 overflow-y-auto p-3">
          <div className="flex items-center justify-between text-[11px] font-semibold uppercase tracking-wider text-slate-400 mb-2">
            <div className="flex items-center gap-1.5">
              <Hash className="h-3 w-3 text-indigo-400" />
              <span>Section Outline</span>
            </div>
            <span className="text-[10px] text-slate-500 font-mono">
              {data.headings.length}
            </span>
          </div>
          <nav className="space-y-1">
            {data.headings.map((h, idx) => {
              const { domain, isOverride } = getHeadingDomain(h);
              const meta = resolveDomainMeta(domain, domains);
              const isPopoverOpen = activeOverrideHeading === h.heading_path;

              return (
                <div
                  key={`${h.line}-${idx}`}
                  className="relative group/heading flex items-center justify-between gap-1 rounded-md px-1.5 py-1 text-[11px] hover:bg-slate-800/70 transition"
                  style={{ paddingLeft: `${Math.max(4, (h.level - 1) * 8 + 4)}px` }}
                >
                  <button
                    type="button"
                    onClick={() => scrollToHeading(h)}
                    className="flex-1 text-left truncate text-slate-400 hover:text-indigo-300 transition"
                    title={h.heading_path}
                  >
                    {h.text}
                  </button>

                  {/* Interactive Domain Badge */}
                  <div className="relative shrink-0">
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        setActiveOverrideHeading(
                          isPopoverOpen ? null : h.heading_path
                        );
                      }}
                      title={`${meta.name}${isOverride ? " (Section Override)" : " (Inherited)"} - Click to reassign domain`}
                      className={`inline-flex items-center justify-center rounded px-1 py-0.5 text-[11px] transition cursor-pointer ${
                        isOverride
                          ? "border border-purple-500/50 bg-purple-950/40"
                          : "hover:bg-slate-800"
                      }`}
                    >
                      <span>{meta.icon}</span>
                    </button>

                    {/* Popover to reassign section domain */}
                    {isPopoverOpen && (
                      <div
                        onClick={(e) => e.stopPropagation()}
                        className="absolute right-0 top-full z-30 mt-1 w-40 rounded-lg border border-slate-700 bg-slate-900 p-1 shadow-xl text-[11px] space-y-0.5"
                      >
                        <div className="px-1.5 py-0.5 text-[9.5px] uppercase font-semibold text-slate-400 border-b border-slate-800">
                          Reassign Section
                        </div>
                        {domains.map((dom) => {
                          const domMeta = resolveDomainMeta(dom.slug, domains);
                          const isCurrent = domain === dom.slug;
                          return (
                            <button
                              key={dom.slug}
                              type="button"
                              onClick={() => handleSelectDomainOverride(h, dom.slug)}
                              className={`flex w-full items-center justify-between rounded px-2 py-1 text-left hover:bg-slate-800 transition ${
                                isCurrent ? "font-semibold text-white bg-slate-800/60" : "text-slate-300"
                              }`}
                            >
                              <span className="flex items-center gap-1.5 truncate">
                                <span>{domMeta.icon}</span>
                                <span className="truncate">{domMeta.name}</span>
                              </span>
                              {isCurrent && <Check className="h-3 w-3 text-emerald-400" />}
                            </button>
                          );
                        })}
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </nav>
        </aside>
      )}

      {/* Markdown Body */}
      <div
        ref={contentContainerRef}
        className="flex-1 overflow-y-auto px-6 py-5 space-y-4"
        style={{ fontSize: `${Math.round(scale * 100)}%` }}
      >
        {/* Active Citation Target Card */}
        {isCitedDoc && activeCitation && (
          <div className="rounded-xl border border-indigo-500/40 bg-indigo-950/30 p-3 text-xs text-indigo-100">
            <div className="flex items-center gap-1.5 font-semibold text-indigo-300">
              <Sparkles className="h-3.5 w-3.5 text-indigo-400" />
              <span>
                Highlighted Citation
                {activeCitation.heading_path
                  ? ` · §${activeCitation.heading_path}`
                  : ""}
                {activeCitation.page_numbers?.length
                  ? ` · p.${activeCitation.page_numbers.join(", ")}`
                  : ""}
              </span>
            </div>
            {activeCitation.excerpt && (
              <p className="mt-1.5 text-slate-200 italic leading-relaxed">
                "{activeCitation.excerpt}"
              </p>
            )}
          </div>
        )}

        <article className="prose prose-invert max-w-none text-slate-200 text-sm leading-relaxed space-y-3 [&_h1]:text-xl [&_h1]:font-bold [&_h1]:text-white [&_h1]:mt-4 [&_h1]:mb-2 [&_h2]:text-lg [&_h2]:font-semibold [&_h2]:text-slate-100 [&_h2]:mt-4 [&_h2]:mb-2 [&_h3]:text-base [&_h3]:font-semibold [&_h3]:text-indigo-200 [&_h3]:mt-3 [&_p]:my-2 [&_ul]:list-disc [&_ul]:pl-5 [&_ol]:list-decimal [&_ol]:pl-5 [&_li]:my-0.5 [&_table]:w-full [&_table]:border-collapse [&_table]:my-3 [&_table]:text-xs [&_th]:border [&_th]:border-slate-700 [&_th]:bg-slate-900 [&_th]:px-2.5 [&_th]:py-1.5 [&_th]:text-left [&_th]:font-semibold [&_td]:border [&_td]:border-slate-800 [&_td]:px-2.5 [&_td]:py-1.5 [&_pre]:rounded-lg [&_pre]:bg-slate-900 [&_pre]:p-3 [&_pre]:overflow-x-auto [&_code]:rounded [&_code]:bg-slate-800 [&_code]:px-1 [&_code]:py-0.5 [&_code]:text-indigo-200">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>
            {data.markdown}
          </ReactMarkdown>
        </article>
      </div>
    </div>
  );
};
