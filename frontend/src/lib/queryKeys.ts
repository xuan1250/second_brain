import type { QueryClient } from "@tanstack/react-query";
import type { ListDocumentsParams } from "./api";

export const queryKeys = {
  documents: (params?: ListDocumentsParams) =>
    params ? (["documents", params] as const) : (["documents"] as const),
  document: (docId: string | null) => ["document", docId] as const,
  documentMarkdown: (docId: string | null) =>
    ["document-markdown", docId] as const,
  systemStats: () => ["system-stats"] as const,
  domains: (includeInactive?: boolean) =>
    includeInactive === undefined
      ? (["domains"] as const)
      : (["domains", { includeInactive }] as const),
  lexicon: (params?: { domain_id?: string; search?: string }) =>
    params ? (["lexicon", params] as const) : (["lexicon"] as const),
  workspaceSettings: () => ["workspaceSettings"] as const,
  chatThreads: () => ["chat-threads"] as const,
};

export function invalidateDocumentQueries(
  qc: QueryClient,
  docId?: string | null
): void {
  void qc.invalidateQueries({ queryKey: queryKeys.documents() });
  if (docId) {
    void qc.invalidateQueries({ queryKey: queryKeys.document(docId) });
    void qc.invalidateQueries({ queryKey: queryKeys.documentMarkdown(docId) });
  }
  void qc.invalidateQueries({ queryKey: queryKeys.systemStats() });
}
