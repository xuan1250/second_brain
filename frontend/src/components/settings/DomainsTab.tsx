import React, { useMemo, useState } from "react";
import { ChevronDown, ChevronRight, Lock, Plus, Trash2 } from "lucide-react";
import { resolveDomainMeta } from "../../lib/domainTheme";
import type { DomainItem, SubdomainItem } from "../../lib/types";
import { LoadingState } from "../common/LoadingState";
import { DeleteDomainDialog } from "./DeleteDomainDialog";
import { DeleteSubdomainDialog } from "./DeleteSubdomainDialog";
import { DomainFormPanel } from "./DomainFormPanel";
import { SubdomainManagerPanel } from "./SubdomainManagerPanel";

export interface DomainsTabProps {
  domains: DomainItem[];
  isLoading: boolean;
}

export const DomainsTab: React.FC<DomainsTabProps> = ({
  domains,
  isLoading,
}) => {
  const [editingDomain, setEditingDomain] = useState<DomainItem | null>(null);
  const [isCreatingDomain, setIsCreatingDomain] = useState<boolean>(false);
  const [expandedDomainId, setExpandedDomainId] = useState<string | null>(null);
  const [subdomainDomainSlug, setSubdomainDomainSlug] = useState<string | null>(
    null
  );
  const [deletingDomain, setDeletingDomain] = useState<DomainItem | null>(null);
  const [deletingSubdomain, setDeletingSubdomain] = useState<{
    domain: DomainItem;
    subdomain: SubdomainItem;
  } | null>(null);

  const activeDomainsCount = useMemo(
    () => domains.filter((d) => d.is_active).length,
    [domains]
  );

  return (
    <>
      {(isCreatingDomain || editingDomain) && (
        <DomainFormPanel
          key={editingDomain?.id ?? "new"}
          editingDomain={editingDomain}
          onClose={() => {
            setIsCreatingDomain(false);
            setEditingDomain(null);
          }}
        />
      )}

      {subdomainDomainSlug && (
        <SubdomainManagerPanel
          domainSlug={subdomainDomainSlug}
          domains={domains}
          onClose={() => setSubdomainDomainSlug(null)}
          onRequestDeleteSubdomain={(domain, subdomain) =>
            setDeletingSubdomain({ domain, subdomain })
          }
        />
      )}

      <div className="flex items-center justify-between">
        <span className="text-xs text-slate-400">
          Active Domains:{" "}
          <strong className="text-white">{activeDomainsCount} / 10</strong>
        </span>
        <button
          type="button"
          disabled={activeDomainsCount >= 10 || isCreatingDomain}
          onClick={() => {
            setEditingDomain(null);
            setIsCreatingDomain(true);
          }}
          title={
            activeDomainsCount >= 10
              ? "Maximum limit of 10 active top-level domains reached"
              : "Create a new custom domain"
          }
          className="inline-flex items-center gap-1.5 rounded-lg bg-indigo-600 px-3 py-1.5 text-xs font-semibold text-white hover:bg-indigo-500 disabled:opacity-40 transition"
        >
          <Plus className="h-3.5 w-3.5" />
          <span>New Domain</span>
        </button>
      </div>

      {isLoading ? (
        <LoadingState label="Loading domains..." className="p-8" />
      ) : (
        <div className="space-y-2">
          {domains.map((dom) => {
            const meta = resolveDomainMeta(dom.slug, domains);
            const isExpanded = expandedDomainId === dom.id;

            return (
              <div
                key={dom.id}
                className="rounded-xl border border-slate-800/90 bg-slate-900/60 p-3 hover:border-slate-700 transition"
              >
                <div className="flex items-center justify-between gap-3">
                  <div className="flex items-center gap-2.5">
                    <span className="text-xl select-none">{dom.icon}</span>
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="font-bold text-white text-sm">
                          {dom.name}
                        </span>
                        <span className="font-mono text-[10.5px] text-slate-400">
                          {dom.slug}
                        </span>
                        {dom.is_system && (
                          <span className="inline-flex items-center gap-1 rounded-full border border-amber-500/30 bg-amber-500/10 px-2 py-0.2 text-[9.5px] font-semibold text-amber-300">
                            <Lock className="h-2.5 w-2.5" />
                            <span>SYSTEM</span>
                          </span>
                        )}
                        <span
                          className={`rounded-full border px-2 py-0.2 text-[9.5px] font-medium ${meta.theme.badge}`}
                        >
                          {meta.epistemicType}
                        </span>
                      </div>
                      <p className="text-xs text-slate-400 line-clamp-1 mt-0.5">
                        {dom.description || "No description provided."}
                      </p>
                    </div>
                  </div>

                  <div className="flex items-center gap-2 shrink-0">
                    <span className="text-xs text-slate-400 font-mono">
                      {dom.document_count} doc
                      {dom.document_count === 1 ? "" : "s"}
                    </span>

                    <button
                      type="button"
                      onClick={() =>
                        setExpandedDomainId(isExpanded ? null : dom.id)
                      }
                      className="rounded-lg border border-slate-800 bg-slate-900 px-2.5 py-1 text-xs text-slate-300 hover:bg-slate-800 flex items-center gap-1"
                    >
                      <span>{dom.subdomains.length} Subdomains</span>
                      {isExpanded ? (
                        <ChevronDown className="h-3 w-3" />
                      ) : (
                        <ChevronRight className="h-3 w-3" />
                      )}
                    </button>

                    <button
                      type="button"
                      onClick={() => {
                        setIsCreatingDomain(false);
                        setEditingDomain(dom);
                      }}
                      className="rounded-lg border border-slate-800 bg-slate-900 px-2.5 py-1 text-xs text-slate-300 hover:bg-slate-800"
                    >
                      Edit
                    </button>

                    <button
                      type="button"
                      disabled={dom.is_system}
                      onClick={() => setDeletingDomain(dom)}
                      title={
                        dom.is_system
                          ? "System starter domain cannot be deleted"
                          : "Delete domain & reassign documents"
                      }
                      className="rounded-lg border border-slate-800 bg-slate-900 p-1 text-slate-400 hover:text-rose-400 hover:bg-slate-800 disabled:opacity-30 disabled:hover:text-slate-400 transition"
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </button>
                  </div>
                </div>

                {isExpanded && (
                  <div className="mt-3 border-t border-slate-800/80 pt-2.5 space-y-2">
                    <div className="flex items-center justify-between text-[11px] text-slate-400">
                      <span className="font-semibold uppercase text-slate-500">
                        Registered Subdomains
                      </span>
                      <button
                        type="button"
                        onClick={() => setSubdomainDomainSlug(dom.slug)}
                        className="text-indigo-400 hover:text-indigo-300 flex items-center gap-1"
                      >
                        <Plus className="h-3 w-3" />
                        <span>Manage Subdomains</span>
                      </button>
                    </div>
                    <div className="flex flex-wrap gap-1.5">
                      {dom.subdomains.map((sub) => (
                        <span
                          key={sub.id}
                          className="rounded-lg border border-slate-800 bg-slate-950/80 px-2.5 py-1 text-xs text-slate-300"
                        >
                          <strong>{sub.name}</strong>{" "}
                          <span className="font-mono text-[10.5px] text-slate-500">
                            ({sub.slug})
                          </span>
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      <DeleteDomainDialog
        deletingDomain={deletingDomain}
        domains={domains}
        onClose={() => setDeletingDomain(null)}
      />

      <DeleteSubdomainDialog
        target={deletingSubdomain}
        onClose={() => setDeletingSubdomain(null)}
      />
    </>
  );
};
