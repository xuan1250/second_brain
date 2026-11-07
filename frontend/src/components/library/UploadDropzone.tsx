import React, { useCallback, useEffect, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import {
  AlertCircle,
  CheckCircle2,
  Clock,
  Loader2,
  RotateCcw,
  ShieldCheck,
  UploadCloud,
  X,
} from "lucide-react";
import { uploadDocuments } from "../../lib/api";
import { useWorkspace } from "../../lib/workspaceStore";
import { useWorkspaceSettingsQuery } from "../../hooks/useDomains";

const ALLOWED_EXTENSIONS = [".pdf", ".docx", ".xlsx", ".pptx"];
const MAX_CONCURRENT_UPLOADS = 2;

function hasAllowedExtension(filename: string): boolean {
  const lower = filename.toLowerCase();
  return ALLOWED_EXTENSIONS.some((ext) => lower.endsWith(ext));
}

function formatBytes(bytes: number): string {
  if (!bytes || bytes <= 0) return "0 B";
  const kb = bytes / 1024;
  if (kb < 1024) return `${kb.toFixed(1)} KB`;
  return `${(kb / 1024).toFixed(2)} MB`;
}

export type IngestionMode = "auto" | "gpu" | "fast";
export type ApprovalMode = "review_all" | "auto_high_confidence" | "auto_all";

export interface UploadItem {
  id: string;
  file: File;
  status: "queued" | "uploading" | "success" | "error";
  error?: string;
  preferFastPath?: boolean | null;
  approvalMode?: ApprovalMode;
}

export const UploadDropzone: React.FC = () => {
  const { data: wsSettings } = useWorkspaceSettingsQuery();
  const [mode, setMode] = useState<IngestionMode>("auto");
  const [approvalMode, setApprovalMode] = useState<ApprovalMode>("review_all");

  useEffect(() => {
    if (!wsSettings) return;
    if (wsSettings.default_upload_mode) {
      setMode(wsSettings.default_upload_mode as IngestionMode);
    }
    if (wsSettings.default_approval_mode) {
      setApprovalMode(wsSettings.default_approval_mode as ApprovalMode);
    }
  }, [
    wsSettings?.default_upload_mode,
    wsSettings?.default_approval_mode,
    wsSettings?.updated_at,
  ]);

  const thresholdPct = Math.round(
    (wsSettings?.high_confidence_threshold ?? 0.85) * 100
  );

  const [isDragging, setIsDragging] = useState(false);
  const [queue, setQueue] = useState<UploadItem[]>([]);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const activeIdsRef = useRef<Set<string>>(new Set());
  const debounceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const queryClient = useQueryClient();
  const { selectDocument } = useWorkspace();

  const triggerDebouncedInvalidation = useCallback(() => {
    if (debounceTimerRef.current) {
      clearTimeout(debounceTimerRef.current);
    }
    debounceTimerRef.current = setTimeout(() => {
      void queryClient.invalidateQueries({ queryKey: ["documents"] });
      void queryClient.invalidateQueries({ queryKey: ["system-stats"] });
    }, 500);
  }, [queryClient]);

  useEffect(() => {
    return () => {
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current);
      }
    };
  }, []);

  const handleFiles = useCallback(
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

  // Worker loop for maintaining up to MAX_CONCURRENT_UPLOADS active uploads
  useEffect(() => {
    const availableSlots = MAX_CONCURRENT_UPLOADS - activeIdsRef.current.size;
    if (availableSlots <= 0) return;

    const queuedCandidates = queue.filter(
      (item) => item.status === "queued" && !activeIdsRef.current.has(item.id)
    );
    if (queuedCandidates.length === 0) return;

    const toProcess = queuedCandidates.slice(0, availableSlots);

    // Mark active immediately
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
          const result = await uploadDocuments([item.file], preferFastPath, currentApproval);
          const uploadedDocs = result.documents?.length
            ? result.documents
            : [result];
          if (uploadedDocs[0]) {
            selectDocument(uploadedDocs[0]);
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
  }, [queue, mode, approvalMode, selectDocument, triggerDebouncedInvalidation]);

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

  const onDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const onDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const onDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFiles(e.dataTransfer.files);
    }
  };

  const onFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFiles(e.target.files);
      e.target.value = "";
    }
  };

  const activeCount = queue.filter(
    (q) => q.status === "uploading" || q.status === "queued"
  ).length;
  const hasCompleted = queue.some((q) => q.status === "success");

  return (
    <div className="space-y-2.5">
      {/* 3-way Segmented Parser Engine Selector */}
      <div
        role="group"
        aria-label="Ingestion processing mode"
        className="grid grid-cols-3 gap-1 rounded-xl border border-slate-800 bg-slate-900/80 p-1 text-xs select-none"
      >
        <button
          type="button"
          onClick={() => setMode("auto")}
          title="⚡ Auto (Recommended): Balanced heuristic routing"
          className={`flex items-center justify-center gap-1 rounded-lg px-1.5 py-1 text-[10.5px] font-medium transition text-center leading-tight ${
            mode === "auto"
              ? "bg-indigo-600 text-white shadow-xs font-semibold"
              : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/60"
          }`}
        >
          ⚡ Auto Engine
        </button>
        <button
          type="button"
          onClick={() => setMode("gpu")}
          title="🚀 GPU Vision: Docling OCR + layout analysis"
          className={`flex items-center justify-center gap-1 rounded-lg px-1.5 py-1 text-[10.5px] font-medium transition text-center leading-tight ${
            mode === "gpu"
              ? "bg-purple-600 text-white shadow-xs font-semibold"
              : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/60"
          }`}
        >
          🚀 GPU Vision
        </button>
        <button
          type="button"
          onClick={() => setMode("fast")}
          title="📄 Fast CPU: PyMuPDF high-speed extraction"
          className={`flex items-center justify-center gap-1 rounded-lg px-1.5 py-1 text-[10.5px] font-medium transition text-center leading-tight ${
            mode === "fast"
              ? "bg-sky-600 text-white shadow-xs font-semibold"
              : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/60"
          }`}
        >
          📄 Fast CPU
        </button>
      </div>

      {/* 3-state Domain Approval Mode Selector */}
      <div
        role="group"
        aria-label="Domain classification approval mode"
        className="grid grid-cols-3 gap-1 rounded-xl border border-slate-800/90 bg-slate-900/90 p-1 text-xs select-none"
      >
        <button
          type="button"
          onClick={() => setApprovalMode("review_all")}
          title="🛡️ Review All: Pauses at awaiting_approval for manual verification"
          className={`flex items-center justify-center gap-1 rounded-lg px-1 py-1 text-[10px] font-medium transition text-center leading-tight ${
            approvalMode === "review_all"
              ? "bg-amber-600/90 text-white shadow-xs font-semibold"
              : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/60"
          }`}
        >
          <ShieldCheck className="h-3 w-3 shrink-0" />
          <span>
            Review All
            {(wsSettings?.default_approval_mode ?? "review_all") === "review_all"
              ? " (Default)"
              : ""}
          </span>
        </button>
        <button
          type="button"
          onClick={() => setApprovalMode("auto_high_confidence")}
          title={`✨ Auto-Approve High Confidence: Automatically proceeds if Laya confidence ≥ ${thresholdPct}%`}
          className={`flex items-center justify-center gap-1 rounded-lg px-1 py-1 text-[10px] font-medium transition text-center leading-tight ${
            approvalMode === "auto_high_confidence"
              ? "bg-emerald-600/90 text-white shadow-xs font-semibold"
              : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/60"
          }`}
        >
          <span>✨ Auto ≥ {thresholdPct}%</span>
        </button>
        <button
          type="button"
          onClick={() => setApprovalMode("auto_all")}
          title="⚡ Auto-Approve All: Automatically accept all model classifications without pausing"
          className={`flex items-center justify-center gap-1 rounded-lg px-1 py-1 text-[10px] font-medium transition text-center leading-tight ${
            approvalMode === "auto_all"
              ? "bg-indigo-600 text-white shadow-xs font-semibold"
              : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/60"
          }`}
        >
          <span>⚡ Auto All</span>
        </button>
      </div>

      {/* Dropzone Area */}
      <div
        role="button"
        tabIndex={0}
        onClick={() => fileInputRef.current?.click()}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            fileInputRef.current?.click();
          }
        }}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
        className={`group relative flex flex-col items-center justify-center rounded-xl border border-dashed px-3 py-3 text-center transition cursor-pointer ${
          isDragging
            ? "border-indigo-400 bg-indigo-500/15"
            : "border-slate-700/80 bg-slate-900/60 hover:border-indigo-500/60 hover:bg-slate-900"
        }`}
      >
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept=".pdf,.docx,.xlsx,.pptx"
          onChange={onFileInputChange}
          className="hidden"
          data-testid="file-upload-input"
        />

        {activeCount > 0 ? (
          <div className="flex items-center gap-2 text-xs text-indigo-300 py-1">
            <Loader2 className="h-4 w-4 animate-spin text-indigo-400" />
            <span>
              Uploading queue ({activeCount} remaining, max 2 concurrent)...
            </span>
          </div>
        ) : (
          <>
            <div className="flex items-center gap-2 text-xs font-medium text-slate-200">
              <UploadCloud className="h-4 w-4 text-indigo-400 group-hover:scale-110 transition-transform" />
              <span>Drop files or click to ingest</span>
            </div>
            <p className="mt-1 text-[11px] text-slate-400">
              Supports <span className="text-slate-300">PDF, DOCX, XLSX, PPTX</span>
            </p>
          </>
        )}
      </div>

      {queue.length > 0 && (
        <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-2 text-xs space-y-1.5">
          <div className="flex items-center justify-between px-1 text-[11px] text-slate-400 border-b border-slate-800/60 pb-1.5">
            <span>
              Upload Queue ({queue.filter((i) => i.status === "uploading").length}{" "}
              active, {queue.filter((i) => i.status === "queued").length} queued)
            </span>
            {hasCompleted && (
              <button
                type="button"
                onClick={clearCompleted}
                className="text-[10px] text-indigo-400 hover:text-indigo-300 transition"
              >
                Clear completed
              </button>
            )}
          </div>

          <div className="max-h-40 overflow-y-auto space-y-1">
            {queue.map((item) => (
              <div
                key={item.id}
                className={`flex items-center justify-between gap-2 rounded-lg px-2 py-1 text-[11px] ${
                  item.status === "error"
                    ? "bg-rose-950/30 border border-rose-500/20 text-rose-200"
                    : item.status === "success"
                    ? "bg-emerald-950/20 text-emerald-300"
                    : item.status === "uploading"
                    ? "bg-indigo-950/30 text-indigo-200"
                    : "bg-slate-800/40 text-slate-400"
                }`}
              >
                <div className="flex items-center gap-1.5 min-w-0 flex-1">
                  {item.status === "uploading" && (
                    <Loader2 className="h-3 w-3 shrink-0 animate-spin text-indigo-400" />
                  )}
                  {item.status === "queued" && (
                    <Clock className="h-3 w-3 shrink-0 text-slate-500" />
                  )}
                  {item.status === "success" && (
                    <CheckCircle2 className="h-3 w-3 shrink-0 text-emerald-400" />
                  )}
                  {item.status === "error" && (
                    <AlertCircle className="h-3 w-3 shrink-0 text-rose-400" />
                  )}

                  <span className="truncate font-medium" title={item.file.name}>
                    {item.file.name}
                  </span>
                  <span className="shrink-0 text-[10px] opacity-70">
                    ({formatBytes(item.file.size)})
                  </span>
                </div>

                <div className="flex items-center gap-1 shrink-0">
                  {item.status === "error" && (
                    <>
                      <button
                        type="button"
                        onClick={() => retryItem(item.id)}
                        title="Retry upload"
                        className="inline-flex items-center gap-0.5 rounded px-1.5 py-0.5 text-[10px] bg-rose-500/20 hover:bg-rose-500/30 text-rose-300 transition"
                      >
                        <RotateCcw className="h-2.5 w-2.5" />
                        Retry
                      </button>
                      <button
                        type="button"
                        onClick={() => removeItem(item.id)}
                        title="Dismiss"
                        className="rounded p-0.5 text-slate-400 hover:text-slate-200"
                      >
                        <X className="h-3 w-3" />
                      </button>
                    </>
                  )}
                  {item.status === "success" && (
                    <button
                      type="button"
                      onClick={() => removeItem(item.id)}
                      title="Remove"
                      className="rounded p-0.5 text-slate-500 hover:text-slate-300"
                    >
                      <X className="h-3 w-3" />
                    </button>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
