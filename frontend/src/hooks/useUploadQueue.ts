import { useCallback, useEffect, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { uploadDocuments } from "../lib/api";
import type { ApprovalMode, IngestionMode } from "../lib/llmCatalog";
import { invalidateDocumentQueries } from "../lib/queryKeys";
import type { DocumentItem } from "../lib/types";

const ALLOWED_EXTENSIONS = [".pdf", ".docx", ".xlsx", ".pptx"];
const MAX_CONCURRENT_UPLOADS = 2;

function hasAllowedExtension(filename: string): boolean {
  const lower = filename.toLowerCase();
  return ALLOWED_EXTENSIONS.some((ext) => lower.endsWith(ext));
}

export interface UploadItem {
  id: string;
  file: File;
  status: "queued" | "uploading" | "success" | "error";
  error?: string;
  preferFastPath?: boolean | null;
  approvalMode?: ApprovalMode;
}

export function useUploadQueue({
  mode,
  approvalMode,
  onUploaded,
}: {
  mode: IngestionMode;
  approvalMode: ApprovalMode;
  onUploaded: (doc: DocumentItem) => void;
}) {
  const [queue, setQueue] = useState<UploadItem[]>([]);
  const activeIdsRef = useRef<Set<string>>(new Set());
  const debounceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const queryClient = useQueryClient();

  const triggerDebouncedInvalidation = useCallback(() => {
    if (debounceTimerRef.current) {
      clearTimeout(debounceTimerRef.current);
    }
    debounceTimerRef.current = setTimeout(() => {
      invalidateDocumentQueries(queryClient);
    }, 500);
  }, [queryClient]);

  useEffect(() => {
    return () => {
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current);
      }
    };
  }, []);

  const enqueueFiles = useCallback(
    (fileList: FileList | File[]) => {
      const files = Array.from(fileList);
      if (files.length === 0) return;

      const calculatedPreferFastPath =
        mode === "gpu" ? false : mode === "fast" ? true : undefined;

      const newItems: UploadItem[] = files.map((file) => {
        const id = `${file.name}-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
        if (!hasAllowedExtension(file.name)) {
          return {
            id,
            file,
            status: "error",
            error: "Unsupported format. Allowed: PDF, DOCX, XLSX, PPTX.",
            preferFastPath: calculatedPreferFastPath,
            approvalMode,
          };
        }
        return {
          id,
          file,
          status: "queued",
          preferFastPath: calculatedPreferFastPath,
          approvalMode,
        };
      });

      setQueue((prev) => [...prev, ...newItems]);
    },
    [mode, approvalMode]
  );

  useEffect(() => {
    const availableSlots = MAX_CONCURRENT_UPLOADS - activeIdsRef.current.size;
    if (availableSlots <= 0) return;

    const queuedCandidates = queue.filter(
      (item) => item.status === "queued" && !activeIdsRef.current.has(item.id)
    );
    if (queuedCandidates.length === 0) return;

    const toProcess = queuedCandidates.slice(0, availableSlots);
    for (const item of toProcess) {
      activeIdsRef.current.add(item.id);
    }

    setQueue((prev) =>
      prev.map((item) =>
        toProcess.some((p) => p.id === item.id)
          ? { ...item, status: "uploading" }
          : item
      )
    );

    for (const item of toProcess) {
      void (async () => {
        try {
          const preferFastPath =
            item.preferFastPath !== undefined
              ? item.preferFastPath
              : mode === "gpu"
              ? false
              : mode === "fast"
              ? true
              : undefined;
          const currentApproval = item.approvalMode || approvalMode;
          const result = await uploadDocuments(
            [item.file],
            preferFastPath,
            currentApproval
          );
          const uploadedDocs = result.documents?.length
            ? result.documents
            : [result];
          if (uploadedDocs[0]) {
            onUploaded(uploadedDocs[0]);
          }
          activeIdsRef.current.delete(item.id);
          setQueue((prev) =>
            prev.map((q) => (q.id === item.id ? { ...q, status: "success" } : q))
          );
          triggerDebouncedInvalidation();
        } catch (err: unknown) {
          const msg =
            err instanceof Error ? err.message : "Document upload failed.";
          activeIdsRef.current.delete(item.id);
          setQueue((prev) =>
            prev.map((q) =>
              q.id === item.id ? { ...q, status: "error", error: msg } : q
            )
          );
        }
      })();
    }
  }, [queue, mode, approvalMode, onUploaded, triggerDebouncedInvalidation]);

  const retryItem = useCallback((id: string) => {
    activeIdsRef.current.delete(id);
    setQueue((prev) =>
      prev.map((item) =>
        item.id === id ? { ...item, status: "queued", error: undefined } : item
      )
    );
  }, []);

  const removeItem = useCallback((id: string) => {
    activeIdsRef.current.delete(id);
    setQueue((prev) => prev.filter((item) => item.id !== id));
  }, []);

  const clearCompleted = useCallback(() => {
    setQueue((prev) => prev.filter((item) => item.status !== "success"));
  }, []);

  return {
    queue,
    enqueueFiles,
    retryItem,
    removeItem,
    clearCompleted,
  };
}
