import React from "react";
import { Check, Loader2, ShieldCheck } from "lucide-react";
import type { DomainItem } from "../../lib/types";
import { DocumentMetadataFields } from "../common/DocumentMetadataFields";
import { ModalShell } from "../common/ModalShell";

export interface EditMetadataDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  filename?: string;
  domains: DomainItem[];
  editDomain: string;
  onEditDomainChange: (value: string) => void;
  editSubdomain: string;
  onEditSubdomainChange: (value: string) => void;
  editTier: number;
  onEditTierChange: (value: number) => void;
  editJurisdiction: string;
  onEditJurisdictionChange: (value: string) => void;
  editYear?: number;
  onEditYearChange: (value: number | undefined) => void;
  isPending: boolean;
  onSave: () => void;
}

export const EditMetadataDialog: React.FC<EditMetadataDialogProps> = ({
  open,
  onOpenChange,
  filename,
  domains,
  editDomain,
  onEditDomainChange,
  editSubdomain,
  onEditSubdomainChange,
  editTier,
  onEditTierChange,
  editJurisdiction,
  onEditJurisdictionChange,
  editYear,
  onEditYearChange,
  isPending,
  onSave,
}) => {
  return (
    <ModalShell
      open={open}
      onOpenChange={onOpenChange}
      title="Edit Document Domain & Metadata"
      icon={<ShieldCheck className="h-4 w-4 text-indigo-400" />}
      description={
        <>
          Update domain classification, authority tier, and jurisdiction for{" "}
          <strong className="text-slate-200">{filename}</strong>.
        </>
      }
      footer={
        <>
          <button
            type="button"
            onClick={() => onOpenChange(false)}
            className="rounded-lg border border-slate-700 bg-slate-800 px-3 py-1.5 text-xs font-medium text-slate-200 hover:bg-slate-700"
          >
            Cancel
          </button>
          <button
            type="button"
            disabled={isPending}
            onClick={onSave}
            className="inline-flex items-center gap-1.5 rounded-lg bg-indigo-600 px-3 py-1.5 text-xs font-semibold text-white hover:bg-indigo-500 disabled:opacity-50"
          >
            {isPending ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            ) : (
              <Check className="h-3.5 w-3.5" />
            )}
            <span>Save Metadata</span>
          </button>
        </>
      }
    >
      <div className="mt-4">
        <DocumentMetadataFields
          domains={domains}
          domain={editDomain}
          onDomainChange={onEditDomainChange}
          subdomain={editSubdomain}
          onSubdomainChange={onEditSubdomainChange}
          authorityTier={editTier}
          onAuthorityTierChange={onEditTierChange}
          jurisdiction={editJurisdiction}
          onJurisdictionChange={onEditJurisdictionChange}
          effectiveYear={editYear}
          onEffectiveYearChange={onEditYearChange}
          showEffectiveYear
          layout="stacked"
        />
      </div>
    </ModalShell>
  );
};
