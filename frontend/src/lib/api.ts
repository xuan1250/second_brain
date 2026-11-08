import type {
  ApprovedClassificationRequest,
  ApproveBatchResponse,
  DocumentDeleteResponse,
  DocumentItem,
  DocumentListResponse,
  DocumentMarkdownResponse,
  DomainCreatePayload,
  DomainDeleteParams,
  DomainDeleteResponse,
  DomainItem,
  DomainUpdatePayload,
  LexiconCreatePayload,
  LexiconItem,
  SubdomainCreatePayload,
  SubdomainItem,
  SubdomainUpdatePayload,
  SystemHealth,
  SystemStats,
  ThreadDeleteResponse,
  ThreadDetailResponse,
  ThreadListResponse,
  UpdateDocumentMetadataRequest,
  VerifyKeyPayload,
  VerifyKeyResponse,
  WorkspaceSettings,
  WorkspaceSettingsUpdatePayload,
} from "./types";

export const API_BASE = "/api/v1";

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let errorDetail = `HTTP ${response.status}: ${response.statusText}`;
    try {
      const payload = (await response.json()) as { detail?: string; message?: string };
      if (payload?.detail) {
        errorDetail =
          typeof payload.detail === "string"
            ? payload.detail
            : JSON.stringify(payload.detail);
      } else if (payload?.message) {
        errorDetail = payload.message;
      }
    } catch {
      // Fallback to statusText
    }
    throw new Error(errorDetail);
  }
  return (await response.json()) as T;
}

export interface ListDocumentsParams {
  status?: string;
  file_type?: string;
  domain?: string;
  search?: string;
  limit?: number;
  offset?: number;
}

export async function fetchDocuments(
  params: ListDocumentsParams = {}
): Promise<DocumentListResponse> {
  const query = new URLSearchParams();
  if (params.status) query.set("status", params.status);
  if (params.file_type && params.file_type !== "all") {
    query.set("file_type", params.file_type);
  }
  if (params.domain && params.domain !== "all") {
    query.set("domain", params.domain);
  }
  if (params.search && params.search.trim()) {
    query.set("search", params.search.trim());
  }
  query.set("limit", String(params.limit ?? 100));
  query.set("offset", String(params.offset ?? 0));

  const res = await fetch(`${API_BASE}/documents?${query.toString()}`);
  return handleResponse<DocumentListResponse>(res);
}

export async function fetchDocument(docId: string): Promise<DocumentItem> {
  const res = await fetch(`${API_BASE}/documents/${encodeURIComponent(docId)}`);
  return handleResponse<DocumentItem>(res);
}

export async function fetchDocumentStatus(docId: string): Promise<DocumentItem> {
  const res = await fetch(
    `${API_BASE}/documents/${encodeURIComponent(docId)}/status`
  );
  return handleResponse<DocumentItem>(res);
}

export async function uploadDocuments(
  files: File[],
  preferFastPath?: boolean | null,
  approvalMode?: "review_all" | "auto_high_confidence" | "auto_all"
): Promise<DocumentItem> {
  const formData = new FormData();
  if (files.length === 1) {
    formData.append("file", files[0]);
  } else {
    for (const file of files) {
      formData.append("files", file);
    }
  }
  const query = new URLSearchParams();
  if (preferFastPath !== undefined && preferFastPath !== null) {
    query.set("prefer_fast_path", String(preferFastPath));
  }
  if (approvalMode) {
    query.set("approval_mode", approvalMode);
  }
  const qs = query.toString();
  const url = `${API_BASE}/documents/upload${qs ? `?${qs}` : ""}`;
  const res = await fetch(url, {
    method: "POST",
    body: formData,
  });
  return handleResponse<DocumentItem>(res);
}

export async function approveClassification(
  docId: string,
  payload?: ApprovedClassificationRequest
): Promise<DocumentItem> {
  const res = await fetch(
    `${API_BASE}/documents/${encodeURIComponent(docId)}/approve-classification`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload ?? {}),
    }
  );
  return handleResponse<DocumentItem>(res);
}

export async function approveBatchClassifications(
  docIds: string[]
): Promise<ApproveBatchResponse> {
  const res = await fetch(`${API_BASE}/documents/approve-batch`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ doc_ids: docIds }),
  });
  return handleResponse<ApproveBatchResponse>(res);
}

export async function updateDocumentMetadata(
  docId: string,
  payload: UpdateDocumentMetadataRequest
): Promise<DocumentItem> {
  const res = await fetch(
    `${API_BASE}/documents/${encodeURIComponent(docId)}/metadata`,
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }
  );
  return handleResponse<DocumentItem>(res);
}

export async function deleteDocument(
  docId: string
): Promise<DocumentDeleteResponse> {
  const res = await fetch(`${API_BASE}/documents/${encodeURIComponent(docId)}`, {
    method: "DELETE",
  });
  return handleResponse<DocumentDeleteResponse>(res);
}

export async function fetchDocumentMarkdown(
  docId: string
): Promise<DocumentMarkdownResponse> {
  const res = await fetch(
    `${API_BASE}/documents/${encodeURIComponent(docId)}/markdown`
  );
  return handleResponse<DocumentMarkdownResponse>(res);
}

export function getDocumentRawUrl(docId: string): string {
  return `${API_BASE}/documents/${encodeURIComponent(docId)}/raw`;
}

export async function fetchChatThreads(params?: {
  limit?: number;
  offset?: number;
}): Promise<ThreadListResponse> {
  const query = new URLSearchParams();
  if (params?.limit !== undefined) query.set("limit", String(params.limit));
  if (params?.offset !== undefined) query.set("offset", String(params.offset));
  const qs = query.toString() ? `?${query.toString()}` : "";
  const res = await fetch(`${API_BASE}/chat/threads${qs}`);
  return handleResponse<ThreadListResponse>(res);
}

export async function fetchChatThread(
  threadId: string
): Promise<ThreadDetailResponse> {
  const res = await fetch(
    `${API_BASE}/chat/threads/${encodeURIComponent(threadId)}`
  );
  return handleResponse<ThreadDetailResponse>(res);
}

export async function deleteChatThread(
  threadId: string
): Promise<ThreadDeleteResponse> {
  const res = await fetch(
    `${API_BASE}/chat/threads/${encodeURIComponent(threadId)}`,
    {
      method: "DELETE",
    }
  );
  return handleResponse<ThreadDeleteResponse>(res);
}

export async function fetchSystemHealth(): Promise<SystemHealth> {
  const res = await fetch(`${API_BASE}/system/health`);
  return handleResponse<SystemHealth>(res);
}

export async function fetchSystemStats(): Promise<SystemStats> {
  const res = await fetch(`${API_BASE}/system/stats`);
  return handleResponse<SystemStats>(res);
}

export async function retryDocument(
  docId: string,
  options?: { forceReparse?: boolean; preferFastPath?: boolean }
): Promise<DocumentItem> {
  const query = new URLSearchParams();
  if (options?.forceReparse !== undefined) {
    query.set("force_reparse", String(options.forceReparse));
  }
  if (options?.preferFastPath !== undefined) {
    query.set("prefer_fast_path", String(options.preferFastPath));
  }
  const qs = query.toString();
  const url = `${API_BASE}/documents/${encodeURIComponent(docId)}/retry${qs ? `?${qs}` : ""}`;
  const res = await fetch(url, {
    method: "POST",
  });
  return handleResponse<DocumentItem>(res);
}

export async function pauseIngestion(): Promise<{
  status: string;
  ingestion_paused: boolean;
}> {
  const res = await fetch(`${API_BASE}/system/ingestion/pause`, {
    method: "POST",
  });
  return handleResponse<{ status: string; ingestion_paused: boolean }>(res);
}

export async function resumeIngestion(): Promise<{
  status: string;
  ingestion_paused: boolean;
}> {
  const res = await fetch(`${API_BASE}/system/ingestion/resume`, {
    method: "POST",
  });
  return handleResponse<{ status: string; ingestion_paused: boolean }>(res);
}

// ============================================================================
// Multi-Domain Taxonomy, Lexicon & Workspace Settings API
// ============================================================================

export async function fetchDomains(
  includeInactive = false
): Promise<DomainItem[]> {
  const qs = includeInactive ? "?include_inactive=true" : "";
  const res = await fetch(`${API_BASE}/domains${qs}`);
  return handleResponse<DomainItem[]>(res);
}

export async function createDomain(
  payload: DomainCreatePayload
): Promise<DomainItem> {
  const res = await fetch(`${API_BASE}/domains`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return handleResponse<DomainItem>(res);
}

export async function updateDomain(
  slug: string,
  payload: DomainUpdatePayload
): Promise<DomainItem> {
  const res = await fetch(`${API_BASE}/domains/${encodeURIComponent(slug)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return handleResponse<DomainItem>(res);
}

export async function deleteDomain(
  slug: string,
  params: DomainDeleteParams = {}
): Promise<DomainDeleteResponse> {
  const query = new URLSearchParams();
  if (params.target_domain) query.set("target_domain", params.target_domain);
  if (params.target_subdomain)
    query.set("target_subdomain", params.target_subdomain);
  if (params.re_index) query.set("re_index", String(params.re_index));
  const qs = query.toString() ? `?${query.toString()}` : "";

  const res = await fetch(
    `${API_BASE}/domains/${encodeURIComponent(slug)}${qs}`,
    {
      method: "DELETE",
    }
  );
  return handleResponse<DomainDeleteResponse>(res);
}

export async function createSubdomain(
  domainSlug: string,
  payload: SubdomainCreatePayload
): Promise<SubdomainItem> {
  const res = await fetch(
    `${API_BASE}/domains/${encodeURIComponent(domainSlug)}/subdomains`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }
  );
  return handleResponse<SubdomainItem>(res);
}

export async function updateSubdomain(
  domainSlug: string,
  subSlug: string,
  payload: SubdomainUpdatePayload
): Promise<SubdomainItem> {
  const res = await fetch(
    `${API_BASE}/domains/${encodeURIComponent(domainSlug)}/subdomains/${encodeURIComponent(subSlug)}`,
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }
  );
  return handleResponse<SubdomainItem>(res);
}

export async function deleteSubdomain(
  domainSlug: string,
  subSlug: string,
  targetSubdomain?: string | null
): Promise<{ status: string; deleted_slug: string; target_subdomain?: string }> {
  const query = new URLSearchParams();
  if (targetSubdomain) query.set("target_subdomain", targetSubdomain);
  const qs = query.toString() ? `?${query.toString()}` : "";

  const res = await fetch(
    `${API_BASE}/domains/${encodeURIComponent(domainSlug)}/subdomains/${encodeURIComponent(subSlug)}${qs}`,
    {
      method: "DELETE",
    }
  );
  return handleResponse<{
    status: string;
    deleted_slug: string;
    target_subdomain?: string;
  }>(res);
}

export async function fetchLexicon(params?: {
  domain_id?: string;
  search?: string;
}): Promise<LexiconItem[]> {
  const query = new URLSearchParams();
  if (params?.domain_id) query.set("domain_id", params.domain_id);
  if (params?.search && params.search.trim())
    query.set("search", params.search.trim());
  const qs = query.toString() ? `?${query.toString()}` : "";

  const res = await fetch(`${API_BASE}/lexicon${qs}`);
  return handleResponse<LexiconItem[]>(res);
}

export async function upsertLexiconTerm(
  payload: LexiconCreatePayload
): Promise<LexiconItem> {
  const res = await fetch(`${API_BASE}/lexicon`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return handleResponse<LexiconItem>(res);
}

export async function deleteLexiconTerm(
  id: string
): Promise<{ deleted: boolean; id: string }> {
  const res = await fetch(`${API_BASE}/lexicon/${encodeURIComponent(id)}`, {
    method: "DELETE",
  });
  return handleResponse<{ deleted: boolean; id: string }>(res);
}

export async function fetchWorkspaceSettings(): Promise<WorkspaceSettings> {
  const res = await fetch(`${API_BASE}/settings`);
  return handleResponse<WorkspaceSettings>(res);
}

export async function updateWorkspaceSettings(
  payload: WorkspaceSettingsUpdatePayload
): Promise<WorkspaceSettings> {
  const res = await fetch(`${API_BASE}/settings`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return handleResponse<WorkspaceSettings>(res);
}

export async function verifyLlmKey(
  payload: VerifyKeyPayload
): Promise<VerifyKeyResponse> {
  const res = await fetch(`${API_BASE}/settings/verify-key`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return handleResponse<VerifyKeyResponse>(res);
}
