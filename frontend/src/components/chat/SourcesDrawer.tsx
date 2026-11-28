import React, { useMemo, useState } from "react";
import { ChevronDown, ChevronRight, FileText, Sparkles } from "lucide-react";
import { useDomainsQuery } from "../../hooks/useDomains";
import { resolveDomainMeta } from "../../lib/domainTheme";
import type { Citation } from "../../lib/types";
import { useWorkspace } from "../../lib/workspaceStore";
import { AuthorityTierBadge } from "../common/DocumentBadges";

export const SourcesDrawer: React.FC<{
  citations?: Citation[] | null;
}> = ({ citations }) => {
  const [expanded, setExpanded] = useState(true);
  const { activeCitation, setActiveCitation } = useWorkspace();
  const { data: domains = [] } = useDomainsQuery();

  const safeCitations = useMemo(
    () => (Array.isArray(citations) ? citations : []),
    [citations]
  );

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
                    const colorCls = resolveDomainMeta(
                      cit.domain,
                      domains
                    ).theme.citation;

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
                            {Array.isArray(cit.page_numbers) &&
                              cit.page_numbers.length > 0 && (
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
};
