import React, { useMemo, useState } from "react";
import { Check, Edit2, Loader2, Sparkles } from "lucide-react";
import { useDomainsQuery } from "../../hooks/useDomains";
import {
  getDomainKey,
  parseClassificationResult,
} from "../../lib/documentMeta";
import { formatPercent } from "../../lib/format";
import type { ApprovedClassificationRequest, DocumentItem } from "../../lib/types";
import { AuthorityTierBadge, DomainBadge } from "../common/DocumentBadges";
import { DocumentMetadataFields } from "../common/DocumentMetadataFields";

export interface AwaitingApprovalCardProps {
  doc: DocumentItem;
  onApprove: (docId: string, payload?: ApprovedClassificationRequest) => void;
  isApproving: boolean;
}

export const AwaitingApprovalCard: React.FC<AwaitingApprovalCardProps> = ({
  doc,
  onApprove,
  isApproving,
}) => {
  const { data: domains = [] } = useDomainsQuery();
  const profile = useMemo(() => parseClassificationResult(doc), [doc]);
  const [isEditing, setIsEditing] = useState(false);

  const defaultDomain = profile?.primary_domain || getDomainKey(doc);
  const defaultSubdomain =
    profile?.primary_subdomain || doc.primary_subdomain_id || "";
  const defaultTier = profile?.authority_tier ?? doc.authority_tier ?? 3;
  const defaultJurisdiction = profile?.jurisdiction ?? doc.jurisdiction ?? "";

  const [domain, setDomain] = useState<string>(defaultDomain);
  const [subdomain, setSubdomain] = useState<string>(defaultSubdomain);
  const [tier, setTier] = useState<number>(defaultTier);
  const [jurisdiction, setJurisdiction] = useState<string>(defaultJurisdiction);

  const isLLMFallback = Boolean(
    profile?.reasoning?.toLowerCase().includes("llm fallback")
  );

  const handleApproveRecommended = (e: React.MouseEvent) => {
    e.stopPropagation();
    onApprove(doc.id);
  };

  const handleSaveAndApprove = (e: React.MouseEvent) => {
    e.stopPropagation();
    onApprove(doc.id, {
      domain,
      subdomain: subdomain || undefined,
      authority_tier: tier,
      jurisdiction: jurisdiction || undefined,
    });
    setIsEditing(false);
  };

  return (
    <div
      onClick={(e) => e.stopPropagation()}
      className="mt-2 rounded-xl border border-amber-500/40 bg-amber-950/20 p-2.5 text-xs text-amber-100 space-y-2"
    >
      <div className="flex items-center justify-between gap-2 border-b border-amber-500/20 pb-1.5">
        <div className="flex items-center gap-1.5 font-semibold text-amber-300">
          <Sparkles className="h-3.5 w-3.5 text-amber-400" />
          <span>
            {isLLMFallback
              ? "LLM Fallback Recommendation"
              : "Laya Classification Recommendation"}
          </span>
        </div>
        <span className="rounded-full bg-amber-500/20 px-2 py-0.5 text-[10px] font-mono font-medium text-amber-200">
          {formatPercent(profile?.confidence, 0.8)} conf
        </span>
      </div>

      {!isEditing ? (
        <div className="space-y-1.5">
          <div className="flex flex-wrap items-center gap-1.5">
            <DomainBadge
              domainKey={profile?.primary_domain || defaultDomain}
              subdomain={profile?.primary_subdomain || defaultSubdomain}
              domains={domains}
            />
            <AuthorityTierBadge tier={defaultTier} />
            {profile?.jurisdiction && (
              <span className="rounded bg-slate-800/80 px-1.5 py-0.5 text-[10px] text-slate-300 font-mono">
                {profile.jurisdiction}
              </span>
            )}
            {profile?.effective_year && (
              <span className="rounded bg-slate-800/80 px-1.5 py-0.5 text-[10px] text-slate-300 font-mono">
                {profile.effective_year}
              </span>
            )}
            {profile?.is_hybrid && (
              <span className="rounded bg-purple-950/60 border border-purple-500/30 px-1.5 py-0.5 text-[10px] text-purple-300">
                Hybrid: {profile.section_overrides?.length || 0} overrides
              </span>
            )}
          </div>

          {profile?.reasoning && (
            <p className="text-[10.5px] text-slate-300 italic line-clamp-2">
              "{profile.reasoning}"
            </p>
          )}

          <div className="flex items-center justify-end gap-1.5 pt-1">
            <button
              type="button"
              onClick={() => setIsEditing(true)}
              className="inline-flex items-center gap-1 rounded-lg border border-slate-700 bg-slate-800 px-2 py-1 text-[10.5px] font-medium text-slate-300 hover:bg-slate-700 hover:text-white transition"
            >
              <Edit2 className="h-3 w-3" />
              <span>Edit</span>
            </button>
            <button
              type="button"
              disabled={isApproving}
              onClick={handleApproveRecommended}
              className="inline-flex items-center gap-1 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white px-2.5 py-1 text-[10.5px] font-semibold transition shadow-xs disabled:opacity-50"
            >
              {isApproving ? (
                <Loader2 className="h-3 w-3 animate-spin" />
              ) : (
                <Check className="h-3 w-3" />
              )}
              <span>Approve Recommended</span>
            </button>
          </div>
        </div>
      ) : (
        <div className="space-y-2 pt-1">
          <DocumentMetadataFields
            domains={domains}
            domain={domain}
            onDomainChange={setDomain}
            subdomain={subdomain}
            onSubdomainChange={setSubdomain}
            authorityTier={tier}
            onAuthorityTierChange={setTier}
            jurisdiction={jurisdiction}
            onJurisdictionChange={setJurisdiction}
            layout="two-col"
          />

          <div className="flex items-center justify-end gap-1.5 pt-1">
            <button
              type="button"
              onClick={() => setIsEditing(false)}
              className="rounded-lg border border-slate-700 bg-slate-800 px-2 py-1 text-[10.5px] font-medium text-slate-400 hover:text-white"
            >
              Cancel
            </button>
            <button
              type="button"
              disabled={isApproving}
              onClick={handleSaveAndApprove}
              className="inline-flex items-center gap-1 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white px-2.5 py-1 text-[10.5px] font-semibold transition disabled:opacity-50"
            >
              {isApproving ? (
                <Loader2 className="h-3 w-3 animate-spin" />
              ) : (
                <Check className="h-3 w-3" />
              )}
              <span>Save & Approve</span>
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
