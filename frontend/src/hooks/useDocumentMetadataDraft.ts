import { useCallback, useEffect, useState } from "react";
import { getDomainKey } from "../lib/documentMeta";
import type {
  DocumentItem,
  RecommendedDomainProfile,
  SectionOverride,
} from "../lib/types";

export function useDocumentMetadataDraft(
  activeDoc: DocumentItem | null | undefined,
  profile: RecommendedDomainProfile | null
) {
  const initialDomain =
    profile?.primary_domain || (activeDoc ? getDomainKey(activeDoc) : "general");
  const initialSubdomain =
    profile?.primary_subdomain || activeDoc?.primary_subdomain_id || "";
  const initialTier = profile?.authority_tier ?? activeDoc?.authority_tier ?? 3;
  const initialJurisdiction =
    profile?.jurisdiction ?? activeDoc?.jurisdiction ?? "";
  const initialYear =
    profile?.effective_year ?? activeDoc?.effective_year ?? undefined;

  const [editDomain, setEditDomain] = useState<string>(initialDomain);
  const [editSubdomain, setEditSubdomain] = useState<string>(initialSubdomain);
  const [editTier, setEditTier] = useState<number>(initialTier);
  const [editJurisdiction, setEditJurisdiction] =
    useState<string>(initialJurisdiction);
  const [editYear, setEditYear] = useState<number | undefined>(initialYear);
  const [localOverrides, setLocalOverrides] = useState<SectionOverride[]>(
    profile?.section_overrides || []
  );

  useEffect(() => {
    const initDomain =
      profile?.primary_domain || (activeDoc ? getDomainKey(activeDoc) : "general");
    const initSubdomain =
      profile?.primary_subdomain || activeDoc?.primary_subdomain_id || "";
    const initTier = profile?.authority_tier ?? activeDoc?.authority_tier ?? 3;
    const initJurisdiction =
      profile?.jurisdiction ?? activeDoc?.jurisdiction ?? "";
    const initYear =
      profile?.effective_year ?? activeDoc?.effective_year ?? undefined;
    const initOverrides = profile?.section_overrides || [];

    setEditDomain(initDomain);
    setEditSubdomain(initSubdomain);
    setEditTier(initTier);
    setEditJurisdiction(initJurisdiction);
    setEditYear(initYear);
    setLocalOverrides(initOverrides);
  }, [activeDoc?.id, activeDoc?.classification_result, profile]);

  const resetFromDocument = useCallback(() => {
    if (!activeDoc) return;
    setEditDomain(getDomainKey(activeDoc));
    setEditSubdomain(activeDoc.primary_subdomain_id || "");
    setEditTier(activeDoc.authority_tier || 3);
    setEditJurisdiction(activeDoc.jurisdiction || "");
    setEditYear(activeDoc.effective_year || undefined);
  }, [activeDoc]);

  const handleSectionOverride = useCallback((override: SectionOverride) => {
    setLocalOverrides((prev) => {
      const filtered = prev.filter(
        (o) =>
          o.heading_path.toLowerCase() !== override.heading_path.toLowerCase()
      );
      return [...filtered, override];
    });
  }, []);

  return {
    editDomain,
    setEditDomain,
    editSubdomain,
    setEditSubdomain,
    editTier,
    setEditTier,
    editJurisdiction,
    setEditJurisdiction,
    editYear,
    setEditYear,
    localOverrides,
    resetFromDocument,
    handleSectionOverride,
  };
}
