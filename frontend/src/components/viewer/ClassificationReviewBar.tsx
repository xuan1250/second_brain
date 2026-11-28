import React from "react";
import { CheckCircle2, Edit, Loader2, ShieldAlert } from "lucide-react";
import { formatPercent } from "../../lib/format";
import type { DomainItem, RecommendedDomainProfile } from "../../lib/types";
import { AuthorityTierBadge, DomainBadge } from "../common/DocumentBadges";
import { DocumentMetadataFields } from "../common/DocumentMetadataFields";

export interface ClassificationReviewBarProps {
  domains: DomainItem[];
  profile: RecommendedDomainProfile | null;
  editDomain: string;
  onEditDomainChange: (value: string) => void;
  editSubdomain: string;
  onEditSubdomainChange: (value: string) => void;
  editTier: number;
  onEditTierChange: (value: number) => void;
  editJurisdiction: string;
  onEditJurisdictionChange: (value: string) => void;
  editYear?: number;
  isAdjusting: boolean;
  onToggleAdjusting: () => void;
  isApproving: boolean;
  onApprove: () => void;
}

export const ClassificationReviewBar: React.FC<
  ClassificationReviewBarProps
> = ({
  domains,
  profile,
  editDomain,
  onEditDomainChange,
  editSubdomain,
  onEditSubdomainChange,
  editTier,
  onEditTierChange,
  editJurisdiction,
  onEditJurisdictionChange,
  editYear,
  isAdjusting,
  onToggleAdjusting,
  isApproving,
  onApprove,
}) => {
  return (
    <div className="border-b border-amber-500/40 bg-gradient-to-r from-amber-950/80 via-amber-900/60 to-slate-950 px-4 py-3 text-xs text-amber-100">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <div className="flex h-6 w-6 items-center justify-center rounded-md bg-amber-500/20 text-amber-300 border border-amber-500/30">
            <ShieldAlert className="h-4 w-4" />
          </div>
          <div>
            <span className="font-semibold text-amber-200">
              Classification & Authority Review Required
            </span>
            <span className="ml-2 text-[10.5px] text-amber-300/80">
              Confidence: {formatPercent(profile?.confidence, 0.8)}
            </span>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={onToggleAdjusting}
            className="inline-flex items-center gap-1 rounded-lg border border-amber-500/40 bg-amber-950/60 px-2.5 py-1 text-[11px] font-medium text-amber-200 hover:bg-amber-900/60 transition"
          >
            <Edit className="h-3 w-3" />
            <span>{isAdjusting ? "Hide Editor" : "Adjust Metadata"}</span>
          </button>

          <button
            type="button"
            disabled={isApproving}
            onClick={onApprove}
            className="inline-flex items-center gap-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white px-3 py-1 text-[11px] font-semibold transition shadow-xs disabled:opacity-50"
          >
            {isApproving ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            ) : (
              <CheckCircle2 className="h-3.5 w-3.5" />
            )}
            <span>Approve Classification</span>
          </button>
        </div>
      </div>

      <div className="mt-2 flex flex-wrap items-center gap-2 text-[11px]">
        <span className="text-slate-400">Recommended:</span>
        <DomainBadge
          domainKey={editDomain}
          subdomain={editSubdomain}
          domains={domains}
        />
        <AuthorityTierBadge tier={editTier} />
        {editJurisdiction && (
          <span className="rounded bg-slate-900/80 border border-slate-800 px-1.5 py-0.5 text-[10px] text-slate-300 font-mono">
            Jurisdiction: {editJurisdiction}
          </span>
        )}
        {editYear && (
          <span className="rounded bg-slate-900/80 border border-slate-800 px-1.5 py-0.5 text-[10px] text-slate-300 font-mono">
            Year: {editYear}
          </span>
        )}
        {profile?.is_hybrid && (
          <span className="rounded bg-purple-950/60 border border-purple-500/30 px-2 py-0.5 text-[10px] text-purple-300">
            Hybrid: {profile.section_overrides?.length || 0} section overrides
          </span>
        )}
      </div>

      {isAdjusting && (
        <div className="mt-3 rounded-xl border border-amber-500/30 bg-slate-950/80 p-3 space-y-2.5">
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
            layout="four-col"
          />
        </div>
      )}
    </div>
  );
};
