export type DomainColorToken =
  | "cyan"
  | "emerald"
  | "amber"
  | "rose"
  | "violet"
  | "blue"
  | "teal"
  | "orange"
  | "fuchsia"
  | "slate"
  | "indigo";

export type ChunkingStrategy =
  | "standard_markdown"
  | "academic_equations"
  | "statutory_clauses"
  | "financial_tables";

export type EpistemicType =
  | "empirical"
  | "evaluative"
  | "normative"
  | "descriptive"
  | "general";

export interface SubdomainItem {
  id: string;
  domain_id: string;
  slug: string;
  name: string;
  standard_bodies?: string | null;
}

export type SubdomainItemResponse = SubdomainItem;

export interface DomainItem {
  id: string;
  slug: string;
  name: string;
  epistemic_type: EpistemicType | string;
  authority_priority: number;
  description: string;
  icon: string;
  color_token: DomainColorToken | string;
  chunking_strategy: ChunkingStrategy | string;
  classifier_keywords?: string | null;
  is_system: boolean;
  is_active: boolean;
  document_count: number;
  subdomains: SubdomainItem[];
}

export type DomainItemResponse = DomainItem;

export interface DomainCreatePayload {
  slug: string;
  name: string;
  description: string;
  classifier_keywords?: string | null;
  color_token?: DomainColorToken | string;
  chunking_strategy?: ChunkingStrategy | string;
  epistemic_type?: EpistemicType | string;
  authority_priority?: number;
  icon?: string;
  is_active?: boolean;
}

export interface DomainUpdatePayload {
  name?: string;
  epistemic_type?: EpistemicType | string;
  authority_priority?: number;
  description?: string;
  icon?: string;
  color_token?: DomainColorToken | string;
  chunking_strategy?: ChunkingStrategy | string;
  classifier_keywords?: string | null;
  is_active?: boolean;
  re_index?: boolean;
}

export interface DomainDeleteParams {
  target_domain?: string;
  target_subdomain?: string | null;
  re_index?: boolean;
}

export interface DomainDeleteResponse {
  deleted_slug: string;
  target_domain: string;
  target_subdomain: string;
  reassigned_documents: number;
  reassigned_chunks: number;
  re_index_queued: number;
}

export interface SubdomainCreatePayload {
  slug: string;
  name: string;
  standard_bodies?: string | null;
}

export interface LexiconItem {
  id: string;
  term: string;
  domain_id: string;
  specific_sense: string;
  contrastive_warning: string;
  created_at?: string | null;
}

export type LexiconItemResponse = LexiconItem;

export interface LexiconCreatePayload {
  term: string;
  domain_id: string;
  specific_sense: string;
  contrastive_warning: string;
}
