export type DocumentStatus =
  | "queued"
  | "parsing"
  | "classifying"
  | "awaiting_approval"
  | "chunking"
  | "indexing"
  | "processing"
  | "ready"
  | "failed";

export type SupportedFileType = "pdf" | "docx" | "xlsx" | "pptx" | string;

export interface SectionOverride {
  heading_path: string;
  domain: "engineering" | "economics" | "law" | "general" | string;
  subdomain: string;
  epistemic_modality:
    | "empirical"
    | "evaluative"
    | "normative"
    | "descriptive"
    | "general"
    | string;
  confidence: number;
  authority_tier: number;
  jurisdiction?: string | null;
}

export interface RecommendedDomainProfile {
  primary_domain: "engineering" | "economics" | "law" | "general" | string;
  primary_subdomain: string;
  epistemic_modality:
    | "empirical"
    | "evaluative"
    | "normative"
    | "descriptive"
    | "general"
    | string;
  confidence: number;
  confidence_subdomain?: number;
  authority_tier: number;
  jurisdiction?: string | null;
  effective_year?: number | null;
  is_current: boolean;
  is_hybrid: boolean;
  section_overrides: SectionOverride[];
  reasoning?: string | null;
}

export interface DomainFilterConfig {
  mode: "auto" | "single" | "federated" | string;
  domains: string[];
  subdomains: string[];
  max_authority_tier?: number | null;
  only_current: boolean;
}

export interface PolysemyCollision {
  term: string;
  domains: string[];
  senses: Record<string, string>;
  contrastive_warning: string;
}

export interface DocumentItem {
  id: string;
  filename: string;
  file_type: SupportedFileType;
  sha256: string;
  size_bytes: number;
  page_count: number | null;
  chunk_count: number;
  status: DocumentStatus | string;
  error: string | null;
  error_message: string | null;
  job_id: string | null;
  stage: string | null;
  progress: number;
  duration_ms: number | null;
  stage_timings: Record<string, number>;
  duplicate: boolean;
  parsing_engine?: "docling" | "pymupdf4llm" | string | null;
  engine_fallback_reason?: string | null;
  primary_domain_id?: string | null;
  primary_subdomain_id?: string | null;
  jurisdiction?: string | null;
  authority_tier?: number;
  effective_year?: number | null;
  is_current?: boolean;
  classification_result?: string | null;
  created_at: string;
  updated_at: string;
  documents?: DocumentItem[] | null;
}

export type DocumentResponse = DocumentItem;

export interface DocumentListResponse {
  items: DocumentItem[];
  documents: DocumentItem[];
  total: number;
  limit: number;
  offset: number;
}

export interface HeadingOutlineItem {
  level: number;
  text: string;
  heading_path: string;
  line: number;
}

export interface DocumentMarkdownResponse {
  id: string;
  doc_id: string;
  filename: string;
  file_type: string;
  page_count: number | null;
  markdown: string;
  headings: HeadingOutlineItem[];
}

export interface DocumentDeleteResponse {
  id: string;
  deleted: boolean;
  status: string;
}

export interface ApprovedClassificationRequest {
  domain?: string | null;
  subdomain?: string | null;
  authority_tier?: number | null;
  jurisdiction?: string | null;
  effective_year?: number | null;
  is_current?: boolean | null;
  section_overrides?: SectionOverride[] | null;
}

export interface ApproveBatchRequest {
  doc_ids: string[];
}

export interface ApproveBatchResponse {
  approved_count: number;
  documents: DocumentItem[];
}

export interface UpdateDocumentMetadataRequest {
  primary_domain_id?: string | null;
  primary_subdomain_id?: string | null;
  domain?: string | null;
  subdomain?: string | null;
  authority_tier?: number | null;
  jurisdiction?: string | null;
  effective_year?: number | null;
  is_current?: boolean | null;
  section_overrides?: SectionOverride[] | null;
}

export interface Citation {
  citation_index: number;
  doc_id: string;
  chunk_id: string;
  source_filename: string;
  page_numbers: number[];
  heading_path: string;
  excerpt: string;
  score: number;
  domain?: string | null;
  subdomain?: string | null;
  authority_tier?: number | null;
  epistemic_type?: string | null;
  jurisdiction?: string | null;
  is_current?: boolean | null;
}

export interface ActiveCitation {
  citation_index?: number;
  doc_id: string;
  chunk_id?: string;
  source_filename: string;
  page_numbers: number[];
  heading_path: string;
  excerpt: string;
  score?: number;
  domain?: string | null;
  subdomain?: string | null;
  authority_tier?: number | null;
  epistemic_type?: string | null;
  jurisdiction?: string | null;
  is_current?: boolean | null;
  triggered_at?: number;
}

export interface CRAGStep {
  node: string;
  label: string;
  status: "running" | "completed" | string;
  routing_mode?: "single" | "federated" | string;
  target_domains?: string[];
  polysemy_collisions?: PolysemyCollision[];
  domain_queries?: Record<string, string>;
}

export interface ChatRequest {
  message: string;
  thread_id?: string | null;
  filter_doc_ids?: string[] | null;
  domain_filter?: DomainFilterConfig | null;
  provider?: string | null;
  model?: string | null;
}

export interface ChatMessage {
  id?: string;
  role: "user" | "assistant" | "system" | string;
  content: string;
  citations: Citation[];
  steps?: CRAGStep[];
  domain_filter?: DomainFilterConfig | null;
  provider?: string | null;
  model?: string | null;
  isStreaming?: boolean;
  isError?: boolean;
}

export interface ThreadSummary {
  thread_id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

export type ChatThread = ThreadSummary;

export interface ThreadListResponse {
  threads: ThreadSummary[];
  items: ThreadSummary[];
  total: number;
  limit?: number;
  offset?: number;
}

export interface ThreadDetailResponse {
  thread_id: string;
  title: string;
  created_at: string | null;
  updated_at: string | null;
  messages: ChatMessage[];
  citations: Citation[];
}

export interface ThreadDeleteResponse {
  thread_id: string;
  deleted: boolean;
  status: string;
}

export interface ProviderOption {
  provider: string;
  label: string;
  configured: boolean;
  default_model: string;
}

export interface SystemHealth {
  status: string;
  database: string;
  qdrant: string;
  collection: string;
}

export interface MemoryStatsResponse {
  process_rss_mb: number;
  cuda_allocated_mb: number;
  cuda_reserved_mb: number;
  cuda_device_name?: string | null;
  worker_idle_timeout_sec: number;
  worker_max_tasks: number;
  classifier_resident?: boolean;
  classifier_model_name?: string | null;
}

export interface SystemStats {
  total_documents: number;
  ready_documents: number;
  total_chunks: number;
  total_threads: number;
  qdrant_vectors_count: number;
  vector_count: number;
  ingestion_paused?: boolean;
  documents_by_status: Record<string, number>;
  llm: {
    provider?: string;
    model_name?: string;
    temperature?: number;
    base_url?: string | null;
  };
  embedding: {
    provider?: string;
    model_name?: string;
    dimensions?: number;
  };
  reranker: {
    model_name?: string;
    top_n?: number;
    score_threshold?: number;
  };
  classifier?: {
    resident: boolean;
    model_name?: string;
    mode?: string;
    fallback_enabled?: boolean;
    high_confidence_threshold?: number;
  };
  memory?: MemoryStatsResponse | null;
  configured_providers: ProviderOption[];
  available_providers: ProviderOption[];
}

export type SSEEvent =
  | {
      event: "step";
      data: {
        node: string;
        label: string;
        status: "completed" | string;
        routing_mode?: "single" | "federated" | string;
        target_domains?: string[];
        polysemy_collisions?: PolysemyCollision[];
        domain_queries?: Record<string, string>;
      };
    }
  | {
      event: "citations";
      data: {
        citations: Citation[];
      };
    }
  | {
      event: "token";
      data: {
        content: string;
        token: string;
      };
    }
  | {
      event: "done";
      data: {
        thread_id: string;
        message: string;
        citations: Citation[];
        provider?: string;
        model?: string;
      };
    }
  | {
      event: "error";
      data: {
        thread_id: string;
        message: string;
        error: string;
      };
    };

// ============================================================================
// Multi-Domain Taxonomy, Lexicon & Workspace Settings Types
// ============================================================================

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

export interface SubdomainUpdatePayload {
  name?: string;
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

export interface LexiconUpdatePayload {
  specific_sense?: string;
  contrastive_warning?: string;
}

export interface WorkspaceSettings {
  id: string;
  default_upload_mode: "auto" | "gpu" | "fast" | string;
  default_approval_mode:
    | "review_all"
    | "auto_high_confidence"
    | "auto_all"
    | string;
  high_confidence_threshold: number;
  default_llm_provider: string | null;
  default_llm_model: string | null;
  available_providers: ProviderOption[];
  updated_at?: string | null;
}

export type WorkspaceSettingsResponse = WorkspaceSettings;

export interface WorkspaceSettingsUpdatePayload {
  default_upload_mode?: "auto" | "gpu" | "fast";
  default_approval_mode?:
    | "review_all"
    | "auto_high_confidence"
    | "auto_all";
  high_confidence_threshold?: number;
  default_llm_provider?: string | null;
  default_llm_model?: string | null;
}
