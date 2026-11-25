import React, { useState } from "react";
import { Plus, Trash2, X } from "lucide-react";
import { useCreateSubdomainMutation } from "../../hooks/useDomains";
import type {
  DomainItem,
  SubdomainCreatePayload,
  SubdomainItem,
} from "../../lib/types";
import { AlertBanner } from "../common/AlertBanner";

export interface SubdomainManagerPanelProps {
  domainSlug: string;
  domains: DomainItem[];
  onClose: () => void;
  onRequestDeleteSubdomain: (
    domain: DomainItem,
    subdomain: SubdomainItem
  ) => void;
}

export const SubdomainManagerPanel: React.FC<SubdomainManagerPanelProps> = ({
  domainSlug,
  domains,
  onClose,
  onRequestDeleteSubdomain,
}) => {
  // Derive live domain from latest query data so additions/deletions reflect immediately
  const domain = domains.find((d) => d.slug === domainSlug) ?? null;
  const createSubdomainMutation = useCreateSubdomainMutation();

  const [subdomainSlugInput, setSubdomainSlugInput] = useState("");
  const [subdomainNameInput, setSubdomainNameInput] = useState("");
  const [subdomainStandardBodies, setSubdomainStandardBodies] = useState("");
  const [subdomainFormError, setSubdomainFormError] = useState<string | null>(
    null
  );

  if (!domain) return null;

  const handleCreateSubdomain = async (e: React.FormEvent) => {
    e.preventDefault();
    setSubdomainFormError(null);

    if (!subdomainNameInput.trim()) {
      setSubdomainFormError("Subdomain name is required.");
      return;
    }
    if (!subdomainSlugInput.trim()) {
      setSubdomainFormError("Subdomain slug is required.");
      return;
    }

    try {
      const payload: SubdomainCreatePayload = {
        slug: subdomainSlugInput.trim().toLowerCase(),
        name: subdomainNameInput.trim(),
        standard_bodies: subdomainStandardBodies.trim() || null,
      };
      await createSubdomainMutation.mutateAsync({
        domainSlug: domain.slug,
        payload,
      });
      setSubdomainSlugInput("");
      setSubdomainNameInput("");
      setSubdomainStandardBodies("");
    } catch (err: unknown) {
      setSubdomainFormError(
        err instanceof Error ? err.message : "Failed to create subdomain."
      );
    }
  };

  return (
    <div className="rounded-xl border border-slate-700 bg-slate-900/90 p-4 space-y-3">
      <div className="flex items-center justify-between border-b border-slate-800 pb-2">
        <h4 className="text-xs font-bold text-white flex items-center gap-1.5">
          <span>
            Subdomains for {domain.icon} {domain.name}
          </span>
        </h4>
        <button
          type="button"
          onClick={onClose}
          className="text-slate-400 hover:text-white"
        >
          <X className="h-4 w-4" />
        </button>
      </div>

      {subdomainFormError && (
        <AlertBanner variant="error" message={subdomainFormError} />
      )}

      <form
        onSubmit={handleCreateSubdomain}
        className="grid grid-cols-1 md:grid-cols-4 gap-2 text-xs"
      >
        <input
          type="text"
          placeholder="Subdomain Name *"
          value={subdomainNameInput}
          onChange={(e) => setSubdomainNameInput(e.target.value)}
          className="w-full rounded border border-slate-700 bg-slate-950 px-2 py-1 text-slate-200"
        />
        <input
          type="text"
          placeholder="Slug (e.g. cardiology) *"
          value={subdomainSlugInput}
          onChange={(e) => setSubdomainSlugInput(e.target.value)}
          className="w-full rounded border border-slate-700 bg-slate-950 px-2 py-1 text-slate-200 font-mono text-[11px]"
        />
        <input
          type="text"
          placeholder="Standard Bodies (optional)"
          value={subdomainStandardBodies}
          onChange={(e) => setSubdomainStandardBodies(e.target.value)}
          className="w-full rounded border border-slate-700 bg-slate-950 px-2 py-1 text-slate-200"
        />
        <button
          type="submit"
          disabled={createSubdomainMutation.isPending}
          className="inline-flex items-center justify-center gap-1 w-full rounded bg-indigo-600 px-3 py-1 font-semibold text-white hover:bg-indigo-500 disabled:opacity-50"
        >
          <Plus className="h-3.5 w-3.5" />
          <span>Add Subdomain</span>
        </button>
      </form>

      <div className="divide-y divide-slate-800 border border-slate-800 rounded-lg overflow-hidden">
        {domain.subdomains.map((sub) => {
          const isSingle = domain.subdomains.length <= 1;
          const isRootGeneral = sub.slug === "general.general";
          const cannotDelete = isSingle || isRootGeneral;

          return (
            <div
              key={sub.id}
              className="flex items-center justify-between p-2 text-xs hover:bg-slate-800/40"
            >
              <div>
                <span className="font-semibold text-slate-200">{sub.name}</span>
                <span className="ml-2 font-mono text-[10px] text-slate-400">
                  {sub.slug}
                </span>
                {sub.standard_bodies && (
                  <span className="ml-2 text-[10.5px] text-slate-500">
                    ({sub.standard_bodies})
                  </span>
                )}
              </div>
              <button
                type="button"
                disabled={cannotDelete}
                onClick={() => onRequestDeleteSubdomain(domain, sub)}
                title={
                  isRootGeneral
                    ? "Root general.general cannot be deleted"
                    : isSingle
                    ? "Cannot delete the only remaining subdomain for this domain"
                    : "Delete subdomain and reassign records"
                }
                className="text-slate-400 hover:text-rose-400 disabled:opacity-30 disabled:hover:text-slate-400 p-1 rounded"
              >
                <Trash2 className="h-3.5 w-3.5" />
              </button>
            </div>
          );
        })}
      </div>
    </div>
  );
};
