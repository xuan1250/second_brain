import React, { useEffect, useMemo, useState } from "react";
import { Trash2 } from "lucide-react";
import { useDeleteDomainMutation } from "../../hooks/useDomains";
import type { DomainItem } from "../../lib/types";
import { useWorkspace } from "../../lib/workspaceStore";
import { ConfirmDialog } from "../common/ConfirmDialog";

export interface DeleteDomainDialogProps {
  deletingDomain: DomainItem | null;
  domains: DomainItem[];
  onClose: () => void;
}

export const DeleteDomainDialog: React.FC<DeleteDomainDialogProps> = ({
  deletingDomain,
  domains,
  onClose,
}) => {
  const { pruneDeletedDomain } = useWorkspace();
  const deleteDomainMutation = useDeleteDomainMutation();

  const [targetDomainSlug, setTargetDomainSlug] = useState("general");
  const [targetSubdomainSlug, setTargetSubdomainSlug] = useState<string | null>(
    null
  );
  const [reindexOnDelete, setReindexOnDelete] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  useEffect(() => {
    if (!deletingDomain) return;
    setTargetDomainSlug(
      deletingDomain.slug === "general" ? "engineering" : "general"
    );
    setTargetSubdomainSlug(null);
    setReindexOnDelete(false);
    setDeleteError(null);
  }, [deletingDomain]);

  const availableTargetDomains = useMemo(() => {
    if (!deletingDomain) return domains;
    return domains.filter((d) => d.slug !== deletingDomain.slug);
  }, [domains, deletingDomain]);

  const targetDomainObj = useMemo(() => {
    return (
      domains.find((d) => d.slug === targetDomainSlug) ||
      domains.find((d) => d.slug === "general") ||
      domains[0]
    );
  }, [domains, targetDomainSlug]);

  const handleConfirm = async () => {
    if (!deletingDomain) return;
    setDeleteError(null);

    try {
      await deleteDomainMutation.mutateAsync({
        slug: deletingDomain.slug,
        params: {
          target_domain: targetDomainSlug,
          target_subdomain: targetSubdomainSlug || undefined,
          re_index: reindexOnDelete,
        },
      });
      pruneDeletedDomain(deletingDomain.slug);
      onClose();
    } catch (err: unknown) {
      setDeleteError(
        err instanceof Error ? err.message : "Failed to delete domain."
      );
    }
  };

  return (
    <ConfirmDialog
      open={Boolean(deletingDomain)}
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
      zIndexClass="z-60"
      maxWidthClass="max-w-lg"
      title={`Delete Custom Domain: ${deletingDomain?.name ?? ""}`}
      description={
        <>
          Deleting domain{" "}
          <strong className="text-white">{deletingDomain?.name}</strong> will
          permanently remove the domain. All existing{" "}
          <strong className="text-rose-300">
            {deletingDomain?.document_count ?? 0} documents
          </strong>{" "}
          and their vectors will be atomically reassigned to the target domain
          below.
        </>
      }
      confirmLabel="Delete & Reassign"
      confirmIcon={<Trash2 className="h-3.5 w-3.5" />}
      isPending={deleteDomainMutation.isPending}
      error={deleteError}
      onConfirm={handleConfirm}
    >
      <div className="space-y-3 text-xs">
        <div>
          <label className="block text-[11px] font-semibold text-slate-300 mb-1">
            Reassign Documents To Target Domain *
          </label>
          <select
            value={targetDomainSlug}
            onChange={(e) => {
              setTargetDomainSlug(e.target.value);
              setTargetSubdomainSlug(null);
            }}
            className="w-full rounded-lg border border-slate-700 bg-slate-900 px-3 py-1.5 text-slate-200"
          >
            {availableTargetDomains.map((d) => (
              <option key={d.slug} value={d.slug}>
                {d.icon} {d.name} ({d.slug})
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="block text-[11px] font-semibold text-slate-300 mb-1">
            Target Subdomain (Optional)
          </label>
          <select
            value={targetSubdomainSlug || ""}
            onChange={(e) => setTargetSubdomainSlug(e.target.value || null)}
            className="w-full rounded-lg border border-slate-700 bg-slate-900 px-3 py-1.5 text-slate-200"
          >
            <option value="">Default Subdomain</option>
            {targetDomainObj?.subdomains.map((s) => (
              <option key={s.slug} value={s.slug}>
                {s.name} ({s.slug})
              </option>
            ))}
          </select>
        </div>

        <div className="rounded-lg border border-slate-800 bg-slate-900/60 p-3">
          <label className="flex items-center gap-2 cursor-pointer text-slate-300">
            <input
              type="checkbox"
              checked={reindexOnDelete}
              onChange={(e) => setReindexOnDelete(e.target.checked)}
              className="rounded border-slate-700 bg-slate-900 text-indigo-600"
            />
            <span>
              Re-index affected documents in background to regenerate chunk
              contextual headers
            </span>
          </label>
        </div>
      </div>
    </ConfirmDialog>
  );
};
