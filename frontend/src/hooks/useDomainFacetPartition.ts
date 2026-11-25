import { useMemo } from "react";
import type { DomainItem } from "../lib/types";

const STARTER_SLUGS = ["engineering", "economics", "law"];

export function useDomainFacetPartition(
  domains: DomainItem[],
  activeSlugs: string[]
): { inlineFacets: DomainItem[]; overflowFacets: DomainItem[] } {
  const specializedDomains = useMemo(
    () => domains.filter((d) => d.slug !== "general"),
    [domains]
  );

  return useMemo(() => {
    if (specializedDomains.length <= 4) {
      return {
        inlineFacets: domains,
        overflowFacets: [],
      };
    }

    const activeCustomSlugs = activeSlugs.filter(
      (slug) =>
        slug !== "all" &&
        slug !== "review" &&
        !STARTER_SLUGS.includes(slug) &&
        slug !== "general"
    );

    const starterInline = domains.filter(
      (d) => STARTER_SLUGS.includes(d.slug) || d.slug === "general"
    );
    const customInline =
      activeCustomSlugs.length > 0
        ? domains.filter((d) => activeCustomSlugs.includes(d.slug))
        : [];

    const inlineSet = new Set([
      ...starterInline.map((d) => d.slug),
      ...customInline.map((d) => d.slug),
    ]);
    const overflow = domains.filter((d) => !inlineSet.has(d.slug));

    return {
      inlineFacets: [...starterInline, ...customInline],
      overflowFacets: overflow,
    };
  }, [domains, specializedDomains.length, activeSlugs]);
}
