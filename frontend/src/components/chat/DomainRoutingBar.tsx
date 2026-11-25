import React, { useState } from "react";
import { Check, ChevronDown, ShieldCheck } from "lucide-react";
import { useDomainFacetPartition } from "../../hooks/useDomainFacetPartition";
import { useDomainsQuery } from "../../hooks/useDomains";
import { resolveDomainMeta } from "../../lib/domainTheme";
import { useWorkspace } from "../../lib/workspaceStore";
import { PopoverMenu } from "../common/PopoverMenu";

export const DomainRoutingBar: React.FC = () => {
  const { activeThreadId, currentDomainFilter, setThreadDomainFilter } =
    useWorkspace();
  const { data: domains = [] } = useDomainsQuery();
  const [isMoreOpen, setIsMoreOpen] = useState(false);

  const isAuto =
    currentDomainFilter.mode === "auto" ||
    (!currentDomainFilter.domains.length &&
      currentDomainFilter.mode !== "single");

  const isTierRestricted = currentDomainFilter.max_authority_tier === 2;

  const { inlineFacets: inlinePills, overflowFacets: overflowPills } =
    useDomainFacetPartition(domains, currentDomainFilter.domains);

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
      if (exists) {
        nextDomains = currentDomains.filter((d) => d !== domainKey);
      } else {
        nextDomains = [...currentDomains, domainKey];
      }
      nextMode = nextDomains.length > 1 ? "federated" : "single";
    } else if (exists && currentDomains.length === 1 && !isAuto) {
      nextDomains = [];
      nextMode = "auto";
    } else {
      nextDomains = [domainKey];
      nextMode = "single";
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

  return (
    <div className="flex flex-wrap items-center justify-between gap-1.5 rounded-xl border border-slate-800 bg-slate-900/80 px-2 py-1.5 text-xs select-none">
      <div className="flex items-center gap-1 overflow-x-auto">
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

        {overflowPills.length > 0 && (
          <PopoverMenu
            open={isMoreOpen}
            onOpenChange={setIsMoreOpen}
            headerLabel="More Domains"
            trigger={
              <button
                type="button"
                onClick={() => setIsMoreOpen((v) => !v)}
                className="inline-flex items-center gap-1 rounded-lg border border-slate-800 bg-slate-950/60 px-2 py-1 text-[10.5px] font-medium text-slate-400 hover:bg-slate-800 hover:text-slate-200 transition"
              >
                <span>+{overflowPills.length} More</span>
                <ChevronDown className="h-3 w-3" />
              </button>
            }
          >
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
          </PopoverMenu>
        )}

        {currentDomainFilter.mode === "federated" && (
          <span className="rounded-full bg-purple-500/20 border border-purple-500/30 px-1.5 py-0.5 text-[9.5px] font-mono font-medium text-purple-300">
            🔀 Fed ({currentDomainFilter.domains.length})
          </span>
        )}
      </div>

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
        <span>
          {isTierRestricted ? "🛡️ Tier 1–2 Authoritative" : "🛡️ All Tiers"}
        </span>
      </button>
    </div>
  );
};
