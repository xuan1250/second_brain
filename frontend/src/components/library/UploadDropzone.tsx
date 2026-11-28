import React, { useEffect, useRef, useState } from "react";
import { Loader2, ShieldCheck, UploadCloud } from "lucide-react";
import { useUploadQueue } from "../../hooks/useUploadQueue";
import { useWorkspaceSettingsQuery } from "../../hooks/useWorkspaceSettings";
import { formatPercent } from "../../lib/format";
import {
  type ApprovalMode,
  INGESTION_MODE_OPTIONS,
  type IngestionMode,
} from "../../lib/llmCatalog";
import { useWorkspace } from "../../lib/workspaceStore";
import { UploadQueueList } from "./UploadQueueList";

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

  const thresholdLabel = formatPercent(
    wsSettings?.high_confidence_threshold,
    0.85
  );

  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const { selectDocument } = useWorkspace();

  const { queue, enqueueFiles, retryItem, removeItem, clearCompleted } =
    useUploadQueue({
      mode,
      approvalMode,
      onUploaded: selectDocument,
    });

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
      enqueueFiles(e.dataTransfer.files);
    }
  };

  const onFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      enqueueFiles(e.target.files);
      e.target.value = "";
    }
  };

  const activeCount = queue.filter(
    (q) => q.status === "uploading" || q.status === "queued"
  ).length;

  return (
    <div className="space-y-2.5">
      <div
        role="group"
        aria-label="Ingestion processing mode"
        className="grid grid-cols-3 gap-1 rounded-xl border border-slate-800 bg-slate-900/80 p-1 text-xs select-none"
      >
        {INGESTION_MODE_OPTIONS.map((opt) => (
          <button
            key={opt.id}
            type="button"
            onClick={() => setMode(opt.id)}
            title={opt.compactTooltip}
            className={`flex items-center justify-center gap-1 rounded-lg px-1.5 py-1 text-[10.5px] font-medium transition text-center leading-tight ${
              mode === opt.id
                ? opt.compactActiveClass
                : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/60"
            }`}
          >
            {opt.compactLabel}
          </button>
        ))}
      </div>

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
          title={`✨ Auto-Approve High Confidence: Automatically proceeds if Laya confidence ≥ ${thresholdLabel}`}
          className={`flex items-center justify-center gap-1 rounded-lg px-1 py-1 text-[10px] font-medium transition text-center leading-tight ${
            approvalMode === "auto_high_confidence"
              ? "bg-emerald-600/90 text-white shadow-xs font-semibold"
              : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/60"
          }`}
        >
          <span>✨ Auto ≥ {thresholdLabel}</span>
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
              Supports{" "}
              <span className="text-slate-300">PDF, DOCX, XLSX, PPTX</span>
            </p>
          </>
        )}
      </div>

      <UploadQueueList
        queue={queue}
        onRetry={retryItem}
        onRemove={removeItem}
        onClearCompleted={clearCompleted}
      />
    </div>
  );
};
