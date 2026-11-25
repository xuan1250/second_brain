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
