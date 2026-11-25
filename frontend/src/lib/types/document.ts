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
  parsing_engine?: "docling" | "pymupdf4llm" | "engineering_cad" | string | null;
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
