import { STARTER_DOMAINS_FALLBACK } from "./domainCatalog";
import {
  DOMAIN_COLOR_PALETTE,
  type DomainColorTheme,
} from "./domainPalette";
import type { DomainColorToken, DomainItem, SubdomainItem } from "./types";

export { ALLOWED_DOMAIN_COLORS, DOMAIN_COLOR_PALETTE } from "./domainPalette";
export type { DomainColorTheme } from "./domainPalette";
export {
  CHUNKING_STRATEGIES,
  EPISTEMIC_TYPES,
  STARTER_DOMAINS_FALLBACK,
} from "./domainCatalog";

export interface ResolvedDomainMeta {
  id?: string;
  slug: string;
  name: string;
  shortLabel: string;
  icon: string;
  colorToken: DomainColorToken;
  epistemicType: string;
  authorityPriority: number;
  chunkingStrategy: string;
  isSystem: boolean;
  subdomains: SubdomainItem[];
  theme: DomainColorTheme;
}

/**
 * Resolve rich metadata and static Tailwind theme tokens for a domain slug or ID.
 */
export function resolveDomainMeta(
  slugOrId?: string | null,
  domains?: DomainItem[] | null
): ResolvedDomainMeta {
  const norm = (slugOrId || "").trim().toLowerCase();

  if (domains && domains.length > 0) {
    const found = domains.find(
      (d) =>
        d.slug.toLowerCase() === norm ||
        d.id.toLowerCase() === norm ||
        d.name.toLowerCase() === norm
    );
    if (found) {
      const colorToken = (
        found.color_token in DOMAIN_COLOR_PALETTE ? found.color_token : "slate"
      ) as DomainColorToken;
      const theme =
        DOMAIN_COLOR_PALETTE[colorToken] || DOMAIN_COLOR_PALETTE.slate;
      const shortLabel =
        found.name.length <= 6
          ? found.name
          : found.slug.length <= 6
          ? found.slug.toUpperCase()
          : found.name.slice(0, 5);

      return {
        id: found.id,
        slug: found.slug,
        name: found.name,
        shortLabel,
        icon: found.icon || "📁",
        colorToken,
        epistemicType: found.epistemic_type || "general",
        authorityPriority: found.authority_priority ?? 3,
        chunkingStrategy: found.chunking_strategy || "standard_markdown",
        isSystem: found.is_system,
        subdomains: found.subdomains || [],
        theme,
      };
    }
  }

  const fallback = STARTER_DOMAINS_FALLBACK.find(
    (d) =>
      d.slug === norm ||
      d.id === norm ||
      norm.includes(d.slug) ||
      d.name.toLowerCase() === norm
  );

  if (fallback) {
    const colorToken = fallback.color_token as DomainColorToken;
    const theme = DOMAIN_COLOR_PALETTE[colorToken] || DOMAIN_COLOR_PALETTE.slate;
    const shortLabel =
      fallback.slug === "engineering"
        ? "Eng"
        : fallback.slug === "economics"
        ? "Econ"
        : fallback.slug === "law"
        ? "Law"
        : "Gen";

    return {
      id: fallback.id,
      slug: fallback.slug,
      name: fallback.name,
      shortLabel,
      icon: fallback.icon,
      colorToken,
      epistemicType: fallback.epistemic_type,
      authorityPriority: fallback.authority_priority,
      chunkingStrategy: fallback.chunking_strategy,
      isSystem: fallback.is_system,
      subdomains: fallback.subdomains,
      theme,
    };
  }

  const cleanSlug = norm || "general";
  const capitalized =
    cleanSlug.charAt(0).toUpperCase() +
    cleanSlug.slice(1).replace(/[-_]/g, " ");
  const shortLabel = cleanSlug.slice(0, 5);
  const theme = DOMAIN_COLOR_PALETTE.slate;

  return {
    slug: cleanSlug,
    name: capitalized,
    shortLabel,
    icon: "📁",
    colorToken: "slate",
    epistemicType: "general",
    authorityPriority: 3,
    chunkingStrategy: "standard_markdown",
    isSystem: false,
    subdomains: [],
    theme,
  };
}
