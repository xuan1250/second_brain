import React, { useMemo } from "react";
import { AUTHORITY_TIERS } from "../../lib/documentMeta";
import type { DomainItem } from "../../lib/types";

export interface DocumentMetadataFieldsProps {
  domains: DomainItem[];
  domain: string;
  onDomainChange: (domainSlug: string) => void;
  subdomain: string;
  onSubdomainChange: (subdomainSlug: string) => void;
  authorityTier: number;
  onAuthorityTierChange: (tier: number) => void;
  jurisdiction: string;
  onJurisdictionChange: (jurisdiction: string) => void;
  effectiveYear?: number | undefined;
  onEffectiveYearChange?: (year: number | undefined) => void;
  showEffectiveYear?: boolean;
  layout?: "two-col" | "four-col" | "stacked";
}

export const DocumentMetadataFields: React.FC<DocumentMetadataFieldsProps> = ({
  domains,
  domain,
  onDomainChange,
  subdomain,
  onSubdomainChange,
  authorityTier,
  onAuthorityTierChange,
  jurisdiction,
  onJurisdictionChange,
  effectiveYear,
  onEffectiveYearChange,
  showEffectiveYear = false,
  layout = "two-col",
}) => {
  const selectedDomainObj = useMemo(
    () => domains.find((d) => d.slug === domain),
    [domains, domain]
  );

  const handleDomainSelect = (nextDomain: string) => {
    onDomainChange(nextDomain);
    const found = domains.find((d) => d.slug === nextDomain);
    if (found?.subdomains?.length) {
      onSubdomainChange(found.subdomains[0].slug);
    }
  };

  const inputCls =
    layout === "stacked"
      ? "w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-1.5 text-slate-200 focus:border-indigo-500 focus:outline-none"
      : "w-full rounded border border-slate-700 bg-slate-900 px-2 py-1 text-slate-200 focus:outline-none";

  const labelCls =
    layout === "stacked"
      ? "block text-[11px] font-medium text-slate-300 mb-1"
      : "block text-[10px] font-semibold uppercase text-slate-400 mb-1";

  const gridCls =
    layout === "stacked"
      ? "space-y-3 text-xs"
      : layout === "four-col"
      ? "grid grid-cols-2 md:grid-cols-4 gap-2 text-[11px]"
      : "grid grid-cols-2 gap-2 text-[11px]";

  return (
    <div className={gridCls}>
      <div>
        <label className={labelCls}>Primary Domain</label>
        <select
          value={domain}
          onChange={(e) => handleDomainSelect(e.target.value)}
          className={inputCls}
        >
          {domains.map((d) => (
            <option key={d.slug} value={d.slug}>
              {d.icon} {d.name}
              {layout === "stacked" ? ` (${d.epistemic_type})` : ""}
            </option>
          ))}
        </select>
      </div>

      <div>
        <label className={labelCls}>Subdomain</label>
        {selectedDomainObj && selectedDomainObj.subdomains.length > 0 ? (
          <select
            value={subdomain}
            onChange={(e) => onSubdomainChange(e.target.value)}
            className={inputCls}
          >
            {selectedDomainObj.subdomains.map((s) => (
              <option key={s.slug} value={s.slug}>
                {s.name} ({s.slug})
              </option>
            ))}
          </select>
        ) : (
          <input
            type="text"
            value={subdomain}
            onChange={(e) => onSubdomainChange(e.target.value)}
            placeholder="e.g. electrical, finance, regulation"
            className={inputCls}
          />
        )}
      </div>

      <div className={layout === "stacked" ? "grid grid-cols-2 gap-3" : "contents"}>
        <div>
          <label className={labelCls}>Authority Tier</label>
          <select
            value={authorityTier}
            onChange={(e) => onAuthorityTierChange(Number(e.target.value))}
            className={inputCls}
          >
            {AUTHORITY_TIERS.map((tier) => (
              <option key={tier.tier} value={tier.tier}>
                {tier.label}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className={labelCls}>
            {layout === "stacked" ? "Jurisdiction / Standard Body" : "Jurisdiction"}
          </label>
          <input
            type="text"
            value={jurisdiction}
            onChange={(e) => onJurisdictionChange(e.target.value)}
            placeholder="e.g. US, VN, IEEE, EPA"
            className={inputCls}
          />
        </div>
      </div>

      {showEffectiveYear && onEffectiveYearChange && (
        <div>
          <label className={labelCls}>Effective Year</label>
          <input
            type="number"
            value={effectiveYear ?? ""}
            onChange={(e) =>
              onEffectiveYearChange(
                e.target.value ? Number(e.target.value) : undefined
              )
            }
            placeholder="e.g. 2024"
            className={inputCls}
          />
        </div>
      )}
    </div>
  );
};
