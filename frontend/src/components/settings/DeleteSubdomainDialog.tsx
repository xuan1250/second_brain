import React, { useEffect, useState } from "react";
import { useDeleteSubdomainMutation } from "../../hooks/useDomains";
import type { DomainItem, SubdomainItem } from "../../lib/types";
import { ConfirmDialog } from "../common/ConfirmDialog";

export interface DeleteSubdomainDialogProps {
  target: { domain: DomainItem; subdomain: SubdomainItem } | null;
  onClose: () => void;
}

export const DeleteSubdomainDialog: React.FC<DeleteSubdomainDialogProps> = ({
  target,
  onClose,
}) => {
  const deleteSubdomainMutation = useDeleteSubdomainMutation();
  const [targetSubdomain, setTargetSubdomain] = useState("");
  const [deleteError, setDeleteError] = useState<string | null>(null);

  useEffect(() => {
    if (!target) return;
    const remaining = target.domain.subdomains.filter(
      (s) => s.slug !== target.subdomain.slug
    );
    setTargetSubdomain(remaining[0]?.slug || "");
    setDeleteError(null);
  }, [target]);

  const handleConfirm = async () => {
    if (!target) return;
    setDeleteError(null);
    try {
      await deleteSubdomainMutation.mutateAsync({
        domainSlug: target.domain.slug,
        subSlug: target.subdomain.slug,
        targetSubdomain: targetSubdomain || undefined,
      });
      onClose();
    } catch (err: unknown) {
      setDeleteError(
        err instanceof Error ? err.message : "Failed to delete subdomain."
      );
    }
  };

  return (
    <ConfirmDialog
      open={Boolean(target)}
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
      zIndexClass="z-60"
      maxWidthClass="max-w-md"
      title={`Delete Subdomain: ${target?.subdomain.name ?? ""}`}
      description={
        <>
          Reassign affected records in domain{" "}
          <strong>{target?.domain.name}</strong> to another subdomain:
        </>
      }
      confirmLabel="Delete Subdomain"
      isPending={deleteSubdomainMutation.isPending}
      error={deleteError}
      onConfirm={handleConfirm}
    >
      <div className="space-y-3 text-xs">
        <div>
          <label className="block text-[11px] font-semibold text-slate-300 mb-1">
            Target Subdomain *
          </label>
          <select
            value={targetSubdomain}
            onChange={(e) => setTargetSubdomain(e.target.value)}
            className="w-full rounded-lg border border-slate-700 bg-slate-900 px-3 py-1.5 text-slate-200"
          >
            {target?.domain.subdomains
              .filter((s) => s.slug !== target.subdomain.slug)
              .map((s) => (
                <option key={s.slug} value={s.slug}>
                  {s.name} ({s.slug})
                </option>
              ))}
          </select>
        </div>
      </div>
    </ConfirmDialog>
  );
};
