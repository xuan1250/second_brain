export interface ProviderOption {
  provider: string;
  label: string;
  configured: boolean;
  default_model: string;
  key_source?: "ui" | "env" | "none";
  masked_key?: string | null;
  validation_status?: "valid" | "invalid" | "untested" | "missing";
  validation_error?: string | null;
  base_url?: string | null;
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
  llm_compat_base_url?: string | null;
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
  api_keys?: Record<string, string | null>;
  llm_compat_base_url?: string | null;
}

export interface VerifyKeyPayload {
  provider: "openai" | "anthropic" | "deepseek" | "openai_compatible" | string;
  api_key?: string | null;
  model?: string | null;
  base_url?: string | null;
  save_if_valid?: boolean;
}

export interface VerifyKeyResponse {
  provider: string;
  valid: boolean;
  validation_status: "valid" | "invalid" | "untested" | "missing";
  validation_error?: string | null;
  key_source: "ui" | "env" | "none";
  masked_key?: string | null;
  tested_model: string;
  provider_option: ProviderOption;
}
