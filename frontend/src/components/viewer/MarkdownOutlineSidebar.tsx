import React from "react";
import { Check, Hash } from "lucide-react";
import { resolveDomainMeta } from "../../lib/domainTheme";
import type { DomainItem, HeadingOutlineItem } from "../../lib/types";
import { PopoverMenu } from "../common/PopoverMenu";

export interface MarkdownOutlineSidebarProps {
  headings: HeadingOutlineItem[];
  domains: DomainItem[];
  activeOverrideHeading: string | null;
  onActiveOverrideHeadingChange: (headingPath: string | null) => void;
  getHeadingDomain: (h: HeadingOutlineItem) => {
    domain: string;
    isOverride: boolean;
  };
  onScrollToHeading: (h: HeadingOutlineItem) => void;
  onSelectDomainOverride: (h: HeadingOutlineItem, newDomain: string) => void;
}

export const MarkdownOutlineSidebar: React.FC<MarkdownOutlineSidebarProps> = ({
  headings,
  domains,
  activeOverrideHeading,
  onActiveOverrideHeadingChange,
  getHeadingDomain,
  onScrollToHeading,
  onSelectDomainOverride,
}) => {
  return (
    <aside className="hidden xl:flex w-56 shrink-0 flex-col border-r border-slate-800/80 bg-slate-900/30 overflow-y-auto p-3">
      <div className="flex items-center justify-between text-[11px] font-semibold uppercase tracking-wider text-slate-400 mb-2">
        <div className="flex items-center gap-1.5">
          <Hash className="h-3 w-3 text-indigo-400" />
          <span>Section Outline</span>
        </div>
        <span className="text-[10px] text-slate-500 font-mono">
          {headings.length}
        </span>
      </div>
      <nav className="space-y-1">
        {headings.map((h, idx) => {
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
                onClick={() => onScrollToHeading(h)}
                className="flex-1 text-left truncate text-slate-400 hover:text-indigo-300 transition"
                title={h.heading_path}
              >
                {h.text}
              </button>

              <PopoverMenu
                open={isPopoverOpen}
                onOpenChange={(nextOpen) =>
                  onActiveOverrideHeadingChange(
                    nextOpen ? h.heading_path : null
                  )
                }
                align="right"
                widthClass="w-40"
                headerLabel="Reassign Section"
                trigger={
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      onActiveOverrideHeadingChange(
                        isPopoverOpen ? null : h.heading_path
                      );
                    }}
                    title={`${meta.name}${
                      isOverride ? " (Section Override)" : " (Inherited)"
                    } - Click to reassign domain`}
                    className={`inline-flex items-center justify-center rounded px-1 py-0.5 text-[11px] transition cursor-pointer ${
                      isOverride
                        ? "border border-purple-500/50 bg-purple-950/40"
                        : "hover:bg-slate-800"
                    }`}
                  >
                    <span>{meta.icon}</span>
                  </button>
                }
              >
                {domains.map((dom) => {
                  const domMeta = resolveDomainMeta(dom.slug, domains);
                  const isCurrent = domain === dom.slug;
                  return (
                    <button
                      key={dom.slug}
                      type="button"
                      onClick={() => onSelectDomainOverride(h, dom.slug)}
                      className={`flex w-full items-center justify-between rounded px-2 py-1 text-left hover:bg-slate-800 transition ${
                        isCurrent
                          ? "font-semibold text-white bg-slate-800/60"
                          : "text-slate-300"
                      }`}
                    >
                      <span className="flex items-center gap-1.5 truncate">
                        <span>{domMeta.icon}</span>
                        <span className="truncate">{domMeta.name}</span>
                      </span>
                      {isCurrent && (
                        <Check className="h-3 w-3 text-emerald-400" />
                      )}
                    </button>
                  );
                })}
              </PopoverMenu>
            </div>
          );
        })}
      </nav>
    </aside>
  );
};
