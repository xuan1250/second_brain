import React, { useState } from "react";
import { Check, ChevronDown, Filter } from "lucide-react";
import { useDomainFacetPartition } from "../../hooks/useDomainFacetPartition";
import { resolveDomainMeta } from "../../lib/domainTheme";
import type { DomainItem } from "../../lib/types";
import { PopoverMenu } from "../common/PopoverMenu";
import { SearchInput } from "../common/SearchInput";

const FILE_TYPE_FILTERS = [
  { id: "all", label: "All" },
  { id: "pdf", label: "PDF" },
  { id: "docx", label: "DOCX" },
  { id: "xlsx", label: "XLSX" },
  { id: "pptx", label: "PPTX" },
] as const;

export interface LibraryFilterBarProps {
  searchText: string;
  onSearchTextChange: (value: string) => void;
  domainFacet: string;
  onDomainFacetChange: (slug: string) => void;
  fileTypeFilter: string;
  onFileTypeFilterChange: (fileType: string) => void;
  domains: DomainItem[];
  awaitingCount: number;
  selectedFilterDocIdsCount: number;
  onClearFilterDocIds: () => void;
}

export const LibraryFilterBar: React.FC<LibraryFilterBarProps> = ({
  searchText,
  onSearchTextChange,
  domainFacet,
  onDomainFacetChange,
  fileTypeFilter,
  onFileTypeFilterChange,
  domains,
  awaitingCount,
  selectedFilterDocIdsCount,
  onClearFilterDocIds,
}) => {
  const [isMoreOpen, setIsMoreOpen] = useState(false);

  const { inlineFacets, overflowFacets } = useDomainFacetPartition(domains, [
    domainFacet,
  ]);

  return (
    <div className="space-y-2 p-3 border-b border-slate-800/60">
      <SearchInput
        value={searchText}
        onChange={onSearchTextChange}
        placeholder="Search documents..."
      />

      <div className="flex items-center gap-1 overflow-x-auto pb-0.5">
        <button
          type="button"
          onClick={() => onDomainFacetChange("all")}
          className={`inline-flex items-center gap-1 rounded-md px-2 py-0.5 text-[11px] font-medium transition shrink-0 ${
            domainFacet === "all"
              ? "bg-indigo-600 text-white shadow-xs font-semibold"
              : "bg-slate-900 text-slate-400 hover:bg-slate-800 hover:text-slate-200"
          }`}
        >
          <span>🌐</span>
          <span>All</span>
        </button>

        {inlineFacets.map((dom) => {
          const active = domainFacet === dom.slug;
          const meta = resolveDomainMeta(dom.slug, domains);
          return (
            <button
              key={dom.id}
              type="button"
              onClick={() => onDomainFacetChange(dom.slug)}
              className={`inline-flex items-center gap-1 rounded-md px-2 py-0.5 text-[11px] font-medium transition shrink-0 ${
                active
                  ? "bg-indigo-600 text-white shadow-xs font-semibold"
                  : "bg-slate-900 text-slate-400 hover:bg-slate-800 hover:text-slate-200"
              }`}
            >
              <span>{meta.icon}</span>
              <span>{meta.shortLabel}</span>
            </button>
          );
        })}

        {overflowFacets.length > 0 && (
          <PopoverMenu
            open={isMoreOpen}
            onOpenChange={setIsMoreOpen}
            headerLabel="More Domains"
            trigger={
              <button
                type="button"
                onClick={() => setIsMoreOpen((v) => !v)}
                className="inline-flex items-center gap-1 rounded-md bg-slate-900 px-2 py-0.5 text-[11px] font-medium text-slate-400 hover:bg-slate-800 hover:text-slate-200 transition"
              >
                <span>+{overflowFacets.length} More</span>
                <ChevronDown className="h-3 w-3" />
              </button>
            }
          >
            {overflowFacets.map((ov) => {
              const active = domainFacet === ov.slug;
              const meta = resolveDomainMeta(ov.slug, domains);
              return (
                <button
                  key={ov.id}
                  type="button"
                  onClick={() => {
                    onDomainFacetChange(ov.slug);
                    setIsMoreOpen(false);
                  }}
                  className={`flex w-full items-center justify-between rounded-lg px-2 py-1.5 text-left text-xs transition ${
                    active
                      ? "bg-indigo-600 text-white font-semibold"
                      : "text-slate-300 hover:bg-slate-800 hover:text-white"
                  }`}
                >
                  <span className="flex items-center gap-1.5 truncate">
                    <span>{meta.icon}</span>
                    <span className="truncate">{meta.name}</span>
                  </span>
                  {active && <Check className="h-3 w-3" />}
                </button>
              );
            })}
          </PopoverMenu>
        )}

        <button
          type="button"
          onClick={() => onDomainFacetChange("review")}
          className={`inline-flex items-center gap-1 rounded-md px-2 py-0.5 text-[11px] font-medium transition shrink-0 ${
            domainFacet === "review"
              ? "bg-amber-600 text-white shadow-xs font-semibold"
              : "bg-slate-900 text-slate-400 hover:bg-slate-800 hover:text-slate-200"
          }`}
        >
          <span>⏳</span>
          <span>Review</span>
          {awaitingCount > 0 && (
            <span className="rounded-full bg-amber-500/30 text-amber-200 px-1.5 text-[9.5px] font-bold">
              {awaitingCount}
            </span>
          )}
        </button>
      </div>

      <div className="flex flex-wrap items-center gap-1">
        {FILE_TYPE_FILTERS.map((chip) => {
          const active = fileTypeFilter === chip.id;
          return (
            <button
              key={chip.id}
              type="button"
              onClick={() => onFileTypeFilterChange(chip.id)}
              className={`rounded-md px-2 py-0.5 text-[10.5px] font-medium transition ${
                active
                  ? "bg-slate-700 text-white shadow-xs"
                  : "bg-slate-900 text-slate-500 hover:bg-slate-800 hover:text-slate-300"
              }`}
            >
              {chip.label}
            </button>
          );
        })}
      </div>

      {selectedFilterDocIdsCount > 0 && (
        <div className="flex items-center justify-between rounded-lg bg-indigo-950/50 border border-indigo-500/30 px-2.5 py-1 text-[11px] text-indigo-200">
          <span className="flex items-center gap-1.5">
            <Filter className="h-3 w-3 text-indigo-400" />
            Scoped to {selectedFilterDocIdsCount} doc
            {selectedFilterDocIdsCount > 1 ? "s" : ""}
          </span>
          <button
            type="button"
            onClick={onClearFilterDocIds}
            className="text-indigo-300 hover:text-white underline text-[10px]"
          >
            Reset
          </button>
        </div>
      )}
    </div>
  );
};
