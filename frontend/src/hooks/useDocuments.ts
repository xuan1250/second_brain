import { useEffect, useRef } from "react";
import {
  useMutation,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query";
import {
  approveBatchClassifications,
  approveClassification,
  deleteDocument,
  fetchDocument,
  fetchDocumentMarkdown,
  fetchDocuments,
  fetchSystemStats,
  type ListDocumentsParams,
  pauseIngestion,
  resumeIngestion,
  retryDocument,
  updateDocumentMetadata,
} from "../lib/api";
import { invalidateDocumentQueries, queryKeys } from "../lib/queryKeys";
import type {
  ApprovedClassificationRequest,
  DocumentItem,
  UpdateDocumentMetadataRequest,
} from "../lib/types";

const IN_PROGRESS_STATUSES = new Set([
  "queued",
  "parsing",
  "classifying",
  "chunking",
  "indexing",
  "processing",
]);

export function isDocumentProcessing(
  doc: DocumentItem | null | undefined
): boolean {
  if (!doc) return false;
  const status = (doc.status || "").toLowerCase();
  const stage = (doc.stage || "").toLowerCase();
  return IN_PROGRESS_STATUSES.has(status) || IN_PROGRESS_STATUSES.has(stage);
}

export function useDocuments(params: ListDocumentsParams = {}) {
  return useQuery({
    queryKey: queryKeys.documents(params),
    queryFn: () => fetchDocuments(params),
    refetchInterval: (query) => {
      const data = query.state.data;
      const list = data?.items ?? data?.documents ?? [];
      const anyProcessing = list.some((doc) => isDocumentProcessing(doc));
      return anyProcessing ? 2000 : false;
    },
  });
}

export function useDocument(docId: string | null) {
  const queryClient = useQueryClient();
  const prevStatusRef = useRef<string | null>(null);

  const query = useQuery({
    queryKey: queryKeys.document(docId),
    queryFn: () => {
      if (!docId) throw new Error("No document ID");
      return fetchDocument(docId);
    },
    enabled: Boolean(docId),
    refetchInterval: (q) => {
      const doc = q.state.data;
      return isDocumentProcessing(doc) ? 2000 : false;
    },
  });

  const currentStatus = query.data?.status;
  useEffect(() => {
    if (!docId || !currentStatus) return;
    const prev = prevStatusRef.current;
    if (
      prev &&
      IN_PROGRESS_STATUSES.has(prev.toLowerCase()) &&
      currentStatus.toLowerCase() === "ready"
    ) {
      void queryClient.invalidateQueries({
        queryKey: queryKeys.documentMarkdown(docId),
      });
    }
    prevStatusRef.current = currentStatus;
  }, [currentStatus, docId, queryClient]);

  return query;
}

export function useDocumentMarkdown(docId: string | null, enabled = true) {
  return useQuery({
    queryKey: queryKeys.documentMarkdown(docId),
    queryFn: () => {
      if (!docId) throw new Error("No document ID");
      return fetchDocumentMarkdown(docId);
    },
    enabled: Boolean(docId) && enabled,
    retry: 1,
  });
}

export function useApproveClassificationMutation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({
      docId,
      payload,
    }: {
      docId: string;
      payload?: ApprovedClassificationRequest;
    }) => approveClassification(docId, payload),
    onSuccess: (_data, variables) => {
      invalidateDocumentQueries(queryClient, variables.docId);
    },
  });
}

export function useApproveBatchMutation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (docIds: string[]) => approveBatchClassifications(docIds),
    onSuccess: () => {
      invalidateDocumentQueries(queryClient);
    },
  });
}

export function useUpdateDocumentMetadataMutation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({
      docId,
      payload,
    }: {
      docId: string;
      payload: UpdateDocumentMetadataRequest;
    }) => updateDocumentMetadata(docId, payload),
    onSuccess: (_data, variables) => {
      invalidateDocumentQueries(queryClient, variables.docId);
    },
  });
}

export function useDeleteDocumentMutation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (docId: string) => deleteDocument(docId),
    onSuccess: (_data, docId) => {
      invalidateDocumentQueries(queryClient, docId);
    },
  });
}

export const DEFAULT_STATS_POLL_MS = 10_000;
export const FAST_STATS_POLL_MS = 2_000;

export interface UseSystemStatsOptions {
  refetchInterval?: number | false;
  fastPoll?: boolean;
}

export function useSystemStats(options?: UseSystemStatsOptions) {
  const interval =
    options?.refetchInterval !== undefined
      ? options.refetchInterval
      : options?.fastPoll
      ? FAST_STATS_POLL_MS
      : DEFAULT_STATS_POLL_MS;

  return useQuery({
    queryKey: queryKeys.systemStats(),
    queryFn: () => fetchSystemStats(),
    staleTime: 5000,
    refetchInterval: interval,
  });
}

export function useRetryDocumentMutation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({
      docId,
      options,
    }: {
      docId: string;
      options?: { forceReparse?: boolean; preferFastPath?: boolean };
    }) => retryDocument(docId, options),
    onSuccess: (_data, variables) => {
      invalidateDocumentQueries(queryClient, variables.docId);
    },
  });
}

export function usePauseIngestionMutation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: () => pauseIngestion(),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: queryKeys.systemStats() });
    },
  });
}

export function useResumeIngestionMutation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: () => resumeIngestion(),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: queryKeys.systemStats() });
    },
  });
}
