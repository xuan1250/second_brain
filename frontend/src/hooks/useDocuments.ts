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
  uploadDocuments,
} from "../lib/api";
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

export function isDocumentProcessing(doc: DocumentItem | null | undefined): boolean {
  if (!doc) return false;
  const status = (doc.status || "").toLowerCase();
  const stage = (doc.stage || "").toLowerCase();
  return IN_PROGRESS_STATUSES.has(status) || IN_PROGRESS_STATUSES.has(stage);
}

export function useDocuments(params: ListDocumentsParams = {}) {
  return useQuery({
    queryKey: ["documents", params],
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
    queryKey: ["document", docId],
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
        queryKey: ["document-markdown", docId],
      });
    }
    prevStatusRef.current = currentStatus;
  }, [currentStatus, docId, queryClient]);

  return query;
}

export function useDocumentMarkdown(docId: string | null, enabled = true) {
  return useQuery({
    queryKey: ["document-markdown", docId],
    queryFn: () => {
      if (!docId) throw new Error("No document ID");
      return fetchDocumentMarkdown(docId);
    },
    enabled: Boolean(docId) && enabled,
    retry: 1,
  });
}

export type UploadDocumentsVariables =
  | File[]
  | {
      files: File[];
      preferFastPath?: boolean | null;
      approvalMode?: "review_all" | "auto_high_confidence" | "auto_all";
    };

export function useUploadDocumentsMutation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (variables: UploadDocumentsVariables) => {
      if (Array.isArray(variables)) {
        return uploadDocuments(variables);
      }
      return uploadDocuments(
        variables.files,
        variables.preferFastPath,
        variables.approvalMode
      );
    },
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["documents"] });
      void queryClient.invalidateQueries({ queryKey: ["system-stats"] });
    },
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
      void queryClient.invalidateQueries({ queryKey: ["documents"] });
      void queryClient.invalidateQueries({
        queryKey: ["document", variables.docId],
      });
      void queryClient.invalidateQueries({
        queryKey: ["document-markdown", variables.docId],
      });
      void queryClient.invalidateQueries({ queryKey: ["system-stats"] });
    },
  });
}

export function useApproveBatchMutation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (docIds: string[]) => approveBatchClassifications(docIds),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["documents"] });
      void queryClient.invalidateQueries({ queryKey: ["system-stats"] });
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
      void queryClient.invalidateQueries({ queryKey: ["documents"] });
      void queryClient.invalidateQueries({
        queryKey: ["document", variables.docId],
      });
      void queryClient.invalidateQueries({
        queryKey: ["document-markdown", variables.docId],
      });
      void queryClient.invalidateQueries({ queryKey: ["system-stats"] });
    },
  });
}

export function useDeleteDocumentMutation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (docId: string) => deleteDocument(docId),
    onSuccess: (_data, docId) => {
      void queryClient.invalidateQueries({ queryKey: ["documents"] });
      void queryClient.invalidateQueries({ queryKey: ["document", docId] });
      void queryClient.invalidateQueries({
        queryKey: ["document-markdown", docId],
      });
      void queryClient.invalidateQueries({ queryKey: ["system-stats"] });
    },
  });
}

export interface UseSystemStatsOptions {
  refetchInterval?: number | false;
}

export function useSystemStats(options?: UseSystemStatsOptions) {
  return useQuery({
    queryKey: ["system-stats"],
    queryFn: () => fetchSystemStats(),
    staleTime: 5000,
    refetchInterval: options?.refetchInterval ?? 10000,
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
      void queryClient.invalidateQueries({ queryKey: ["documents"] });
      void queryClient.invalidateQueries({ queryKey: ["document", variables.docId] });
      void queryClient.invalidateQueries({
        queryKey: ["document-markdown", variables.docId],
      });
      void queryClient.invalidateQueries({ queryKey: ["system-stats"] });
    },
  });
}

export function usePauseIngestionMutation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: () => pauseIngestion(),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["system-stats"] });
    },
  });
}

export function useResumeIngestionMutation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: () => resumeIngestion(),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["system-stats"] });
    },
  });
}
