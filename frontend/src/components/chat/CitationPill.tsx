import React from "react";
import type { Citation } from "../../lib/types";
import { useWorkspace } from "../../lib/workspaceStore";
import { useDomainsQuery } from "../../hooks/useDomains";
import { resolveDomainMeta } from "../../lib/domainTheme";

export interface CitationPillProps {
  index: number;
  citation?: Citation;
}

export function getCitationDomainColor(domain?: string | null) {
  return resolveDomainMeta(domain).theme.citation;
}

export const CitationPill: React.FC<CitationPillProps> = ({
  index,
  citation,
}) => {
  const { data: domains = [] } = useDomainsQuery();
  const { activeCitation, setActiveCitation } = useWorkspace();

  const isActive =
    Boolean(citation) &&
    activeCitation?.doc_id === citation?.doc_id &&
    activeCitation?.citation_index === citation?.citation_index;

  const meta = resolveDomainMeta(citation?.domain, domains);
  const colorCls = meta.theme.citation;

  const domainLabel = meta.name;
  const subdomainLabel = citation?.subdomain
    ? ` › ${citation.subdomain.split(".").pop()}`
    : "";
  const tierLabel = citation?.authority_tier
    ? ` · T${citation.authority_tier}`
    : "";
  const jurisLabel = citation?.jurisdiction ? ` · ${citation.jurisdiction}` : "";
  const pageLabel =
    Array.isArray(citation?.page_numbers) && citation.page_numbers.length > 0
      ? ` · p.${citation.page_numbers.join(",")}`
      : typeof citation?.heading_path === "string" && citation.heading_path
      ? ` · §${citation.heading_path.split(">").pop()?.trim() || ""}`
      : "";

  const tooltip = citation
    ? `[${index}] ${citation.source_filename || ""} (${domainLabel}${subdomainLabel}${tierLabel}${jurisLabel}${pageLabel})${
        typeof citation.excerpt === "string" && citation.excerpt ? ` — "${citation.excerpt.slice(0, 140)}"` : ""
      }`
    : `Citation [${index}]`;

  return (
    <button
      type="button"
      onClick={(e) => {
        e.preventDefault();
        e.stopPropagation();
        if (citation) {
          setActiveCitation(citation);
        }
      }}
      title={tooltip}
      className={`mx-0.5 inline-flex items-center gap-0.5 rounded px-1.5 py-0.2 text-[11px] font-mono font-semibold align-baseline border transition cursor-pointer ${
        isActive
          ? `${colorCls.activeBorder} ${colorCls.activeBg} ${colorCls.activeText} shadow-xs ring-1 ring-white/20`
          : citation
          ? `${colorCls.border} ${colorCls.bg} ${colorCls.text} hover:scale-105 hover:brightness-125`
          : "border-slate-700 bg-slate-800 text-slate-400"
      }`}
    >
      <span>[{index}]</span>
    </button>
  );
};
