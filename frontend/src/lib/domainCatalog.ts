import type { ChunkingStrategy, DomainItem, EpistemicType } from "./types";

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
