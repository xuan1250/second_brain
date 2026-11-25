import React, { useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { useDocumentMarkdown } from "../../hooks/useDocuments";
import { useDomainsQuery } from "../../hooks/useDomains";
import { useMarkdownCitationHighlight } from "../../hooks/useMarkdownCitationHighlight";
import {
  getDomainKey,
  parseClassificationResult,
} from "../../lib/documentMeta";
import { resolveDomainMeta } from "../../lib/domainTheme";
import type {
  ActiveCitation,
  DocumentItem,
  HeadingOutlineItem,
  SectionOverride,
} from "../../lib/types";
import { AlertBanner } from "../common/AlertBanner";
import { CitationExcerptCard } from "../common/CitationExcerptCard";
import { LoadingState } from "../common/LoadingState";
import { MarkdownOutlineSidebar } from "./MarkdownOutlineSidebar";

export interface MarkdownViewerPaneProps {
  docId: string;
  activeDoc?: DocumentItem | null;
  activeCitation: ActiveCitation | null;
  scale: number;
  onUpdateSectionOverride?: (override: SectionOverride) => void;
  sectionOverrides?: SectionOverride[];
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
  const [activeOverrideHeading, setActiveOverrideHeading] = useState<
    string | null
  >(null);

  const isCitedDoc = activeCitation?.doc_id === docId;

  const { scrollToHeading } = useMarkdownCitationHighlight({
    containerRef: contentContainerRef,
    markdown: data?.markdown,
    isCitedDoc,
    activeCitation,
  });

  const overridesMap = useMemo(() => {
    const map = new Map<string, SectionOverride>();
    const parsed = activeDoc ? parseClassificationResult(activeDoc) : null;
    for (const ov of parsed?.section_overrides ?? []) {
      if (ov.heading_path) map.set(ov.heading_path.toLowerCase(), ov);
    }
    for (const ov of sectionOverrides ?? []) {
      if (ov.heading_path) map.set(ov.heading_path.toLowerCase(), ov);
    }
    return map;
  }, [activeDoc, sectionOverrides]);

  const docDomain = activeDoc ? getDomainKey(activeDoc) : "general";

  const getHeadingDomain = (
    h: HeadingOutlineItem
  ): { domain: string; isOverride: boolean } => {
    const ov = overridesMap.get((h.heading_path || h.text).toLowerCase());
    return ov
      ? { domain: ov.domain, isOverride: true }
      : { domain: docDomain, isOverride: false };
  };

  const handleSelectDomainOverride = (
    h: HeadingOutlineItem,
    newDomain: string
  ) => {
    setActiveOverrideHeading(null);
    if (!onUpdateSectionOverride) return;
    const meta = resolveDomainMeta(newDomain, domains);
    onUpdateSectionOverride({
      heading_path: h.heading_path || h.text,
      domain: meta.slug,
      subdomain: meta.subdomains?.[0]?.slug || meta.slug,
      epistemic_modality: meta.epistemicType,
      confidence: 1.0,
      authority_tier: activeDoc?.authority_tier ?? 3,
    });
  };

  if (isLoading) {
    return (
      <LoadingState
        label="Loading structured Markdown mirror..."
        className="h-full"
      />
    );
  }

  if (isError || !data) {
    return (
      <div className="flex h-full flex-col items-center justify-center p-6">
        <AlertBanner
          variant="warning"
          title="Markdown Mirror Not Ready"
          className="max-w-md"
          message={
            error instanceof Error
              ? error.message
              : "Document is still being converted or Markdown output is unavailable."
          }
        />
      </div>
    );
  }

  return (
    <div className="flex h-full overflow-hidden bg-slate-950">
      {data.headings && data.headings.length > 0 && (
        <MarkdownOutlineSidebar
          headings={data.headings}
          domains={domains}
          activeOverrideHeading={activeOverrideHeading}
          onActiveOverrideHeadingChange={setActiveOverrideHeading}
          getHeadingDomain={getHeadingDomain}
          onScrollToHeading={scrollToHeading}
          onSelectDomainOverride={handleSelectDomainOverride}
        />
      )}

      <div
        ref={contentContainerRef}
        className="flex-1 overflow-y-auto px-6 py-5 space-y-4"
        style={{ fontSize: `${Math.round(scale * 100)}%` }}
      >
        {isCitedDoc && activeCitation && (
          <CitationExcerptCard
            showSparkles
            title={
              <>
                Highlighted Citation
                {activeCitation.heading_path
                  ? ` · §${activeCitation.heading_path}`
                  : ""}
                {activeCitation.page_numbers?.length
                  ? ` · p.${activeCitation.page_numbers.join(", ")}`
                  : ""}
              </>
            }
            excerpt={activeCitation.excerpt}
          />
        )}

        <article className="prose prose-invert markdown-prose-shared max-w-none text-slate-200 text-sm leading-relaxed space-y-3 [&_h1]:text-xl [&_h1]:font-bold [&_h1]:text-white [&_h1]:mt-4 [&_h1]:mb-2 [&_h2]:text-lg [&_h2]:font-semibold [&_h2]:text-slate-100 [&_h2]:mt-4 [&_h2]:mb-2 [&_h3]:text-base [&_h3]:font-semibold [&_h3]:text-indigo-200 [&_h3]:mt-3 [&_p]:my-2 [&_ul]:pl-5 [&_ol]:pl-5 [&_table]:my-3 [&_pre]:bg-slate-900 [&_pre]:p-3 [&_code]:text-indigo-200">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>
            {data.markdown}
          </ReactMarkdown>
        </article>
      </div>
    </div>
  );
};
