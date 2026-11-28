import type {
  ApprovedClassificationRequest,
  ApproveBatchResponse,
  DocumentDeleteResponse,
  DocumentItem,
  DocumentListResponse,
  DocumentMarkdownResponse,
  UpdateDocumentMetadataRequest,
} from "../types";
import { API_BASE, jsonInit, requestJson } from "./client";

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
  return requestJson<DocumentListResponse>(`/documents?${query.toString()}`);
}

export async function fetchDocument(docId: string): Promise<DocumentItem> {
  return requestJson<DocumentItem>(`/documents/${encodeURIComponent(docId)}`);
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
  return requestJson<DocumentItem>(
    `/documents/upload${qs ? `?${qs}` : ""}`,
    {
      method: "POST",
      body: formData,
    }
  );
}

export async function approveClassification(
  docId: string,
  payload?: ApprovedClassificationRequest
): Promise<DocumentItem> {
  return requestJson<DocumentItem>(
    `/documents/${encodeURIComponent(docId)}/approve-classification`,
    jsonInit("POST", payload ?? {})
  );
}

export async function approveBatchClassifications(
  docIds: string[]
): Promise<ApproveBatchResponse> {
  return requestJson<ApproveBatchResponse>(
    "/documents/approve-batch",
    jsonInit("POST", { doc_ids: docIds })
  );
}

export async function updateDocumentMetadata(
  docId: string,
  payload: UpdateDocumentMetadataRequest
): Promise<DocumentItem> {
  return requestJson<DocumentItem>(
    `/documents/${encodeURIComponent(docId)}/metadata`,
    jsonInit("PATCH", payload)
  );
}

export async function deleteDocument(
  docId: string
): Promise<DocumentDeleteResponse> {
  return requestJson<DocumentDeleteResponse>(
    `/documents/${encodeURIComponent(docId)}`,
    { method: "DELETE" }
  );
}

export async function fetchDocumentMarkdown(
  docId: string
): Promise<DocumentMarkdownResponse> {
  return requestJson<DocumentMarkdownResponse>(
    `/documents/${encodeURIComponent(docId)}/markdown`
  );
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
  return requestJson<DocumentItem>(
    `/documents/${encodeURIComponent(docId)}/retry${qs ? `?${qs}` : ""}`,
    { method: "POST" }
  );
}

export function getDocumentRawUrl(docId: string): string {
  return `${API_BASE}/documents/${encodeURIComponent(docId)}/raw`;
}
