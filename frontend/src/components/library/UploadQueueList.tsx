import React from "react";
import {
  AlertCircle,
  CheckCircle2,
  Clock,
  Loader2,
  RotateCcw,
  X,
} from "lucide-react";
import type { UploadItem } from "../../hooks/useUploadQueue";
import { formatBytes } from "../../lib/format";

export interface UploadQueueListProps {
  queue: UploadItem[];
  onRetry: (id: string) => void;
  onRemove: (id: string) => void;
  onClearCompleted: () => void;
}

export const UploadQueueList: React.FC<UploadQueueListProps> = ({
  queue,
  onRetry,
  onRemove,
  onClearCompleted,
}) => {
  if (queue.length === 0) return null;

  const hasCompleted = queue.some((q) => q.status === "success");
  const uploadingCount = queue.filter((i) => i.status === "uploading").length;
  const queuedCount = queue.filter((i) => i.status === "queued").length;

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-2 text-xs space-y-1.5">
      <div className="flex items-center justify-between px-1 text-[11px] text-slate-400 border-b border-slate-800/60 pb-1.5">
        <span>
          Upload Queue ({uploadingCount} active, {queuedCount} queued)
        </span>
        {hasCompleted && (
          <button
            type="button"
            onClick={onClearCompleted}
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
                    onClick={() => onRetry(item.id)}
                    title="Retry upload"
                    className="inline-flex items-center gap-0.5 rounded px-1.5 py-0.5 text-[10px] bg-rose-500/20 hover:bg-rose-500/30 text-rose-300 transition"
                  >
                    <RotateCcw className="h-2.5 w-2.5" />
                    Retry
                  </button>
                  <button
                    type="button"
                    onClick={() => onRemove(item.id)}
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
                  onClick={() => onRemove(item.id)}
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
  );
};
