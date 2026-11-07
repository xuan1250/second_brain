import type {
  ChunkingStrategy,
  DomainColorToken,
  DomainItem,
  EpistemicType,
  SubdomainItem,
} from "./types";

export interface DomainColorTheme {
  token: DomainColorToken;
  label: string;
  swatchBg: string;
  badge: string;
  activePill: string;
  citation: {
    border: string;
    bg: string;
    text: string;
    activeBorder: string;
    activeBg: string;
    activeText: string;
  };
  heading: {
    h2: string;
    h3: string;
  };
}

/**
 * Static Tailwind v4 palette lookup table.
 * NO dynamic string interpolation (bg-${color}-500) to ensure static scanner catches all classes.
 */
export const DOMAIN_COLOR_PALETTE: Record<DomainColorToken, DomainColorTheme> = {
  cyan: {
    token: "cyan",
    label: "Cyan",
    swatchBg: "bg-cyan-500",
    badge: "bg-cyan-500/15 border-cyan-500/30 text-cyan-300",
    activePill: "bg-cyan-500/25 border-cyan-400 text-cyan-200",
    citation: {
      border: "border-cyan-500/40",
      bg: "bg-cyan-500/15",
      text: "text-cyan-300",
      activeBorder: "border-cyan-400",
      activeBg: "bg-cyan-500/30",
      activeText: "text-cyan-100",
    },
    heading: {
      h2: "border-l-4 border-cyan-400 bg-cyan-950/30 text-cyan-200",
      h3: "text-cyan-300 border-b border-cyan-500/30 pb-0.5",
    },
  },
  emerald: {
    token: "emerald",
    label: "Emerald",
    swatchBg: "bg-emerald-500",
    badge: "bg-emerald-500/15 border-emerald-500/30 text-emerald-300",
    activePill: "bg-emerald-500/25 border-emerald-400 text-emerald-200",
    citation: {
      border: "border-emerald-500/40",
      bg: "bg-emerald-500/15",
      text: "text-emerald-300",
      activeBorder: "border-emerald-400",
      activeBg: "bg-emerald-500/30",
      activeText: "text-emerald-100",
    },
    heading: {
      h2: "border-l-4 border-emerald-400 bg-emerald-950/30 text-emerald-200",
      h3: "text-emerald-300 border-b border-emerald-500/30 pb-0.5",
    },
  },
  amber: {
    token: "amber",
    label: "Amber",
    swatchBg: "bg-amber-500",
    badge: "bg-amber-500/15 border-amber-500/30 text-amber-300",
    activePill: "bg-amber-500/25 border-amber-400 text-amber-200",
    citation: {
      border: "border-amber-500/40",
      bg: "bg-amber-500/15",
      text: "text-amber-300",
      activeBorder: "border-amber-400",
      activeBg: "bg-amber-500/30",
      activeText: "text-amber-100",
    },
    heading: {
      h2: "border-l-4 border-amber-400 bg-amber-950/30 text-amber-200",
      h3: "text-amber-300 border-b border-amber-500/30 pb-0.5",
    },
  },
  rose: {
    token: "rose",
    label: "Rose",
    swatchBg: "bg-rose-500",
    badge: "bg-rose-500/15 border-rose-500/30 text-rose-300",
    activePill: "bg-rose-500/25 border-rose-400 text-rose-200",
    citation: {
      border: "border-rose-500/40",
      bg: "bg-rose-500/15",
      text: "text-rose-300",
      activeBorder: "border-rose-400",
      activeBg: "bg-rose-500/30",
      activeText: "text-rose-100",
    },
    heading: {
      h2: "border-l-4 border-rose-400 bg-rose-950/30 text-rose-200",
      h3: "text-rose-300 border-b border-rose-500/30 pb-0.5",
    },
  },
  violet: {
    token: "violet",
    label: "Violet",
    swatchBg: "bg-violet-500",
    badge: "bg-violet-500/15 border-violet-500/30 text-violet-300",
    activePill: "bg-violet-500/25 border-violet-400 text-violet-200",
    citation: {
      border: "border-violet-500/40",
      bg: "bg-violet-500/15",
      text: "text-violet-300",
      activeBorder: "border-violet-400",
      activeBg: "bg-violet-500/30",
      activeText: "text-violet-100",
    },
    heading: {
      h2: "border-l-4 border-violet-400 bg-violet-950/30 text-violet-200",
      h3: "text-violet-300 border-b border-violet-500/30 pb-0.5",
    },
  },
  blue: {
    token: "blue",
    label: "Blue",
    swatchBg: "bg-blue-500",
    badge: "bg-blue-500/15 border-blue-500/30 text-blue-300",
    activePill: "bg-blue-500/25 border-blue-400 text-blue-200",
    citation: {
      border: "border-blue-500/40",
      bg: "bg-blue-500/15",
      text: "text-blue-300",
      activeBorder: "border-blue-400",
      activeBg: "bg-blue-500/30",
      activeText: "text-blue-100",
    },
    heading: {
      h2: "border-l-4 border-blue-400 bg-blue-950/30 text-blue-200",
      h3: "text-blue-300 border-b border-blue-500/30 pb-0.5",
    },
  },
  teal: {
    token: "teal",
    label: "Teal",
    swatchBg: "bg-teal-500",
    badge: "bg-teal-500/15 border-teal-500/30 text-teal-300",
    activePill: "bg-teal-500/25 border-teal-400 text-teal-200",
    citation: {
      border: "border-teal-500/40",
      bg: "bg-teal-500/15",
      text: "text-teal-300",
      activeBorder: "border-teal-400",
      activeBg: "bg-teal-500/30",
      activeText: "text-teal-100",
    },
    heading: {
      h2: "border-l-4 border-teal-400 bg-teal-950/30 text-teal-200",
      h3: "text-teal-300 border-b border-teal-500/30 pb-0.5",
    },
  },
  orange: {
    token: "orange",
    label: "Orange",
    swatchBg: "bg-orange-500",
    badge: "bg-orange-500/15 border-orange-500/30 text-orange-300",
    activePill: "bg-orange-500/25 border-orange-400 text-orange-200",
    citation: {
      border: "border-orange-500/40",
      bg: "bg-orange-500/15",
      text: "text-orange-300",
      activeBorder: "border-orange-400",
      activeBg: "bg-orange-500/30",
      activeText: "text-orange-100",
    },
    heading: {
      h2: "border-l-4 border-orange-400 bg-orange-950/30 text-orange-200",
      h3: "text-orange-300 border-b border-orange-500/30 pb-0.5",
    },
  },
  fuchsia: {
    token: "fuchsia",
    label: "Fuchsia",
    swatchBg: "bg-fuchsia-500",
    badge: "bg-fuchsia-500/15 border-fuchsia-500/30 text-fuchsia-300",
    activePill: "bg-fuchsia-500/25 border-fuchsia-400 text-fuchsia-200",
    citation: {
      border: "border-fuchsia-500/40",
      bg: "bg-fuchsia-500/15",
      text: "text-fuchsia-300",
      activeBorder: "border-fuchsia-400",
      activeBg: "bg-fuchsia-500/30",
      activeText: "text-fuchsia-100",
    },
    heading: {
      h2: "border-l-4 border-fuchsia-400 bg-fuchsia-950/30 text-fuchsia-200",
      h3: "text-fuchsia-300 border-b border-fuchsia-500/30 pb-0.5",
    },
  },
  slate: {
    token: "slate",
    label: "Slate",
    swatchBg: "bg-slate-500",
    badge: "bg-slate-500/15 border-slate-500/30 text-slate-300",
    activePill: "bg-slate-500/25 border-slate-400 text-slate-200",
    citation: {
      border: "border-slate-500/40",
      bg: "bg-slate-500/15",
      text: "text-slate-300",
      activeBorder: "border-slate-400",
      activeBg: "bg-slate-500/30",
      activeText: "text-slate-100",
    },
    heading: {
      h2: "border-l-4 border-slate-400 bg-slate-950/30 text-slate-200",
      h3: "text-slate-300 border-b border-slate-500/30 pb-0.5",
    },
  },
  indigo: {
    token: "indigo",
    label: "Indigo",
    swatchBg: "bg-indigo-500",
    badge: "bg-indigo-500/15 border-indigo-500/30 text-indigo-300",
    activePill: "bg-indigo-500/25 border-indigo-400 text-indigo-200",
    citation: {
      border: "border-indigo-500/40",
      bg: "bg-indigo-500/15",
      text: "text-indigo-300",
      activeBorder: "border-indigo-400",
      activeBg: "bg-indigo-500/30",
      activeText: "text-indigo-100",
    },
    heading: {
      h2: "border-l-4 border-indigo-400 bg-indigo-950/30 text-indigo-200",
      h3: "text-indigo-300 border-b border-indigo-500/30 pb-0.5",
    },
  },
};

export const ALLOWED_DOMAIN_COLORS: DomainColorToken[] = [
  "cyan",
  "emerald",
  "amber",
  "rose",
  "violet",
  "blue",
  "teal",
  "orange",
  "fuchsia",
  "slate",
];

export const CHUNKING_STRATEGIES: {
  value: ChunkingStrategy;
  label: string;
  description: string;
}[] = [
  {
    value: "standard_markdown",
    label: "Standard Markdown",
    description: "Hierarchical heading chunking for general documents.",
  },
  {
    value: "academic_equations",
    label: "Academic Equations & LaTeX",
    description:
      "Keeps LaTeX equations, proofs, and definitions contiguous without mid-formula splitting.",
  },
  {
    value: "statutory_clauses",
    label: "Statutory Clauses (Điều / Khoản)",
    description:
      "Preserves Vietnamese & civil law statutory boundaries (Điều, Khoản, Điểm) as atomic units.",
  },
  {
    value: "financial_tables",
    label: "Financial Tables & Banners",
    description:
      "Prevents breaking across balance sheets, income statements, and Markdown data tables.",
  },
];

export const EPISTEMIC_TYPES: {
  value: EpistemicType;
  label: string;
  description: string;
}[] = [
  {
    value: "empirical",
    label: "Empirical / Technical",
    description:
      "Data-driven, factual, engineering and natural science claims.",
  },
  {
    value: "evaluative",
    label: "Evaluative / Financial",
    description:
      "Value, risk, economics, accounting, and market projections.",
  },
  {
    value: "normative",
    label: "Normative / Statutory",
    description:
      "Mandates, legal rules, regulatory compliance, and policies.",
  },
  {
    value: "descriptive",
    label: "Descriptive",
    description: "Narrative, historical, or background explanations.",
  },
  {
    value: "general",
    label: "General",
    description: "Multi-domain or unspecialized documents.",
  },
];

/**
 * Starter fallback domains for instant rendering during initial load.
 */
export const STARTER_DOMAINS_FALLBACK: DomainItem[] = [
  {
    id: "system-engineering",
    slug: "engineering",
    name: "Engineering",
    epistemic_type: "empirical",
    authority_priority: 1,
    description: "Empirical, technical, mechanical, and software documentation",
    icon: "⚙️",
    color_token: "cyan",
    chunking_strategy: "academic_equations",
    classifier_keywords: "engineering, technical, code, software, electrical, mechanical",
    is_system: true,
    is_active: true,
    document_count: 0,
    subdomains: [
      { id: "sub-eng-1", domain_id: "system-engineering", slug: "eng.electrical", name: "Electrical" },
      { id: "sub-eng-2", domain_id: "system-engineering", slug: "eng.control", name: "Control Systems" },
      { id: "sub-eng-3", domain_id: "system-engineering", slug: "eng.mechanical", name: "Mechanical" },
      { id: "sub-eng-4", domain_id: "system-engineering", slug: "eng.technology", name: "Technology" },
    ],
  },
  {
    id: "system-economics",
    slug: "economics",
    name: "Economics",
    epistemic_type: "evaluative",
    authority_priority: 2,
    description: "Evaluative, financial, economic, accounting, and market analysis",
    icon: "📈",
    color_token: "emerald",
    chunking_strategy: "financial_tables",
    classifier_keywords: "economics, finance, accounting, investment, market, tax",
    is_system: true,
    is_active: true,
    document_count: 0,
    subdomains: [
      { id: "sub-econ-1", domain_id: "system-economics", slug: "econ.finance", name: "Finance" },
      { id: "sub-econ-2", domain_id: "system-economics", slug: "econ.accounting", name: "Accounting" },
      { id: "sub-econ-3", domain_id: "system-economics", slug: "econ.investment", name: "Investment" },
      { id: "sub-econ-4", domain_id: "system-economics", slug: "econ.taxation", name: "Taxation" },
    ],
  },
  {
    id: "system-law",
    slug: "law",
    name: "Law",
    epistemic_type: "normative",
    authority_priority: 3,
    description: "Normative, statutory, legal, regulatory, and contractual documents",
    icon: "⚖️",
    color_token: "amber",
    chunking_strategy: "statutory_clauses",
    classifier_keywords: "law, statute, regulation, compliance, contract, legal",
    is_system: true,
    is_active: true,
    document_count: 0,
    subdomains: [
      { id: "sub-law-1", domain_id: "system-law", slug: "law.statute", name: "Statute" },
      { id: "sub-law-2", domain_id: "system-law", slug: "law.regulation", name: "Regulation" },
      { id: "sub-law-3", domain_id: "system-law", slug: "law.case_law", name: "Case Law" },
      { id: "sub-law-4", domain_id: "system-law", slug: "law.contract", name: "Contract" },
    ],
  },
  {
    id: "system-general",
    slug: "general",
    name: "General",
    epistemic_type: "general",
    authority_priority: 4,
    description: "Descriptive and unspecialized multi-domain reference documents",
    icon: "📄",
    color_token: "slate",
    chunking_strategy: "standard_markdown",
    classifier_keywords: "general, overview, reference, miscellaneous",
    is_system: true,
    is_active: true,
    document_count: 0,
    subdomains: [
      { id: "sub-gen-1", domain_id: "system-general", slug: "general.general", name: "General" },
    ],
  },
];

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
 * Replaces old buggy string coercions (like getDomainKey defaulting to "general").
 */
export function resolveDomainMeta(
  slugOrId?: string | null,
  domains?: DomainItem[] | null
): ResolvedDomainMeta {
  const norm = (slugOrId || "").trim().toLowerCase();

  // Search provided active domains first
  if (domains && domains.length > 0) {
    const found = domains.find(
      (d) =>
        d.slug.toLowerCase() === norm ||
        d.id.toLowerCase() === norm ||
        d.name.toLowerCase() === norm
    );
    if (found) {
      const colorToken = (found.color_token in DOMAIN_COLOR_PALETTE
        ? found.color_token
        : "slate") as DomainColorToken;
      const theme = DOMAIN_COLOR_PALETTE[colorToken] || DOMAIN_COLOR_PALETTE.slate;
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

  // Fallback to starter domains
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

  // Graceful fallback for custom domain slug when registry is loading or not yet cached
  const cleanSlug = norm || "general";
  const capitalized =
    cleanSlug.charAt(0).toUpperCase() + cleanSlug.slice(1).replace(/[-_]/g, " ");
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
