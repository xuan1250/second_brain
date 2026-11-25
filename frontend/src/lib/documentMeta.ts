import type { DocumentItem, RecommendedDomainProfile } from "./types";

export interface AuthorityTierMeta {
  tier: number;
  shortLabel: string;
  label: string;
  title: string;
  badgeClass: string;
}

export const AUTHORITY_TIERS: AuthorityTierMeta[] = [
  {
    tier: 1,
    shortLabel: "T1",
    label: "T1 - Binding Statutes / Primary Standards",
    title: "Tier 1: Binding Statutes / Primary Standards (IEEE, IEC, US Code)",
    badgeClass: "bg-rose-500/15 border-rose-500/30 text-rose-300",
  },
  {
    tier: 2,
    shortLabel: "T2",
    label: "T2 - Regulatory Directives / Standard Bodies",
    title: "Tier 2: Regulatory Directives / Administrative Guidance (CFR, FASB)",
    badgeClass: "bg-purple-500/15 border-purple-500/30 text-purple-300",
  },
  {
    tier: 3,
    shortLabel: "T3",
    label: "T3 - Standard Internal / Technical",
    title: "Tier 3: Industry Practice / Standard Technical Reports",
    badgeClass: "bg-cyan-500/15 border-cyan-500/30 text-cyan-300",
  },
  {
    tier: 4,
    shortLabel: "T4",
    label: "T4 - Drafts / Working Papers",
    title: "Tier 4: Working Papers / Draft Guidelines",
    badgeClass: "bg-emerald-500/15 border-emerald-500/30 text-emerald-300",
  },
  {
    tier: 5,
    shortLabel: "T5",
    label: "T5 - Informal / Unverified",
    title: "Tier 5: Informal Notes / Unverified Documents",
    badgeClass: "bg-slate-500/15 border-slate-500/30 text-slate-400",
  },
];

export function getAuthorityTierMeta(tier?: number | null): AuthorityTierMeta {
  const t = tier ?? 3;
  return AUTHORITY_TIERS.find((item) => item.tier === t) ?? AUTHORITY_TIERS[2];
}

export function parseClassificationResult(
  doc: DocumentItem
): RecommendedDomainProfile | null {
  if (!doc.classification_result) return null;
  try {
    return JSON.parse(doc.classification_result) as RecommendedDomainProfile;
  } catch {
    return null;
  }
}

export function getDomainKey(doc: DocumentItem): string {
  if (doc.primary_domain_id) {
    return doc.primary_domain_id.toLowerCase();
  }
  const parsed = parseClassificationResult(doc);
  if (parsed?.primary_domain) {
    return parsed.primary_domain.toLowerCase();
  }
  return "general";
}

export function isPdfFilenameOrType(
  filenameOrType: string | undefined | null
): boolean {
  if (!filenameOrType) return false;
  const lower = filenameOrType.trim().toLowerCase();
  return lower === "pdf" || lower.endsWith(".pdf");
}
