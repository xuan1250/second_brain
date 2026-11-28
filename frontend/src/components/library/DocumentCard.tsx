import React from "react";
import { CheckSquare, RotateCcw, Square, Trash2 } from "lucide-react";
import { getDomainKey } from "../../lib/documentMeta";
import { formatBytes } from "../../lib/format";
import type {
  ApprovedClassificationRequest,
  DocumentItem,
  DomainItem,
} from "../../lib/types";
import {
  AuthorityTierBadge,
  DomainBadge,
  EngineBadge,
  FileTypeIcon,
  StatusBadge,
} from "../common/DocumentBadges";
import { AwaitingApprovalCard } from "./AwaitingApprovalCard";

export interface DocumentCardProps {
  doc: DocumentItem;
  domains: DomainItem[];
  isSelected: boolean;
  isScoped: boolean;
  isCitedDoc: boolean;
  isApproving: boolean;
  isRetrying: boolean;
  onSelect: () => void;
  onToggleScope: () => void;
  onApprove: (docId: string, payload?: ApprovedClassificationRequest) => void;
  onRetry: (options?: { forceReparse?: boolean; preferFastPath?: boolean }) => void;
  onRequestDelete: () => void;
}

export const DocumentCard: React.FC<DocumentCardProps> = ({
  doc,
  domains,
  isSelected,
  isScoped,
  isCitedDoc,
  isApproving,
  isRetrying,
  onSelect,
  onToggleScope,
  onApprove,
  onRetry,
  onRequestDelete,
}) => {
  const isAwaiting = (doc.status || "").toLowerCase() === "awaiting_approval";
  const domainKey = getDomainKey(doc);

  return (
    <div
      onClick={onSelect}
      className={`group relative flex flex-col rounded-xl border p-2.5 transition cursor-pointer ${
        isAwaiting
          ? "border-amber-500/50 bg-amber-950/15 hover:border-amber-400 hover:bg-amber-950/25"
          : isSelected
          ? "border-indigo-500/60 bg-indigo-950/30"
          : isCitedDoc
          ? "border-amber-500/50 bg-amber-950/20"
          : "border-slate-800/60 bg-slate-900/40 hover:border-slate-700 hover:bg-slate-900/80"
      }`}
    >
      <div className="flex items-start gap-2.5">
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            onToggleScope();
          }}
          title={
            isScoped
              ? "Remove from RAG query scope"
              : "Include in scoped RAG query"
          }
          className="mt-0.5 text-slate-400 hover:text-indigo-400 transition"
        >
          {isScoped ? (
            <CheckSquare className="h-4 w-4 text-indigo-400" />
          ) : (
            <Square className="h-4 w-4" />
          )}
        </button>

        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-1.5">
            <FileTypeIcon fileType={doc.file_type} />
            <span
              className="truncate text-xs font-medium text-slate-100"
              title={doc.filename}
            >
              {doc.filename}
            </span>
          </div>

          <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
            <StatusBadge doc={doc} />
            {!isAwaiting && (
              <>
                <DomainBadge
                  domainKey={domainKey}
                  subdomain={doc.primary_subdomain_id}
                  domains={domains}
                />
                <AuthorityTierBadge tier={doc.authority_tier} />
              </>
            )}
            <EngineBadge doc={doc} />
            <span className="text-[10px] uppercase text-slate-500 font-mono">
              {doc.file_type}
            </span>
            <span className="text-[10px] text-slate-500">
              · {formatBytes(doc.size_bytes)}
            </span>
            {doc.page_count ? (
              <span className="text-[10px] text-slate-500">
                · {doc.page_count}p
              </span>
            ) : null}
          </div>

          {isAwaiting && (
            <AwaitingApprovalCard
              doc={doc}
              onApprove={onApprove}
              isApproving={isApproving}
            />
          )}

          {doc.status === "failed" && (
            <div
              className="mt-2 space-y-1.5"
              onClick={(e) => e.stopPropagation()}
            >
              {(doc.error_message || doc.error) && (
                <div className="rounded-md bg-rose-950/40 border border-rose-500/20 px-2 py-1 text-[10px] text-rose-300">
                  <span className="font-semibold text-rose-200">
                    Failed at stage [{doc.stage || "unknown"}]:{" "}
                  </span>
                  <span className="break-words">
                    {doc.error_message || doc.error}
                  </span>
                </div>
              )}
              <div className="flex flex-wrap items-center gap-1.5 pt-0.5">
                <button
                  type="button"
                  disabled={isRetrying}
                  onClick={() => onRetry()}
                  title="Retry with Stage 1 checkpoint if cached"
                  className="inline-flex items-center gap-1 rounded bg-indigo-600 hover:bg-indigo-500 text-white px-2 py-0.5 text-[10px] font-medium transition disabled:opacity-50"
                >
                  <RotateCcw className="h-2.5 w-2.5" />
                  Retry
                </button>

                <button
                  type="button"
                  disabled={isRetrying}
                  onClick={() =>
                    onRetry({ preferFastPath: true, forceReparse: true })
                  }
                  title="Bypass Docling OCR, parse digital text via fast CPU (pymupdf4llm)"
                  className="rounded border border-slate-700 bg-slate-800 hover:bg-slate-700 text-slate-300 px-1.5 py-0.5 text-[10px] font-medium transition disabled:opacity-50"
                >
                  Fast-Text Retry
                </button>

                <button
                  type="button"
                  disabled={isRetrying}
                  onClick={() =>
                    onRetry({ preferFastPath: false, forceReparse: true })
                  }
                  title="Force Docling + TableFormer full vision re-parsing"
                  className="rounded border border-slate-700 bg-slate-800 hover:bg-slate-700 text-slate-300 px-1.5 py-0.5 text-[10px] font-medium transition disabled:opacity-50"
                >
                  Full Vision Retry
                </button>
              </div>
            </div>
          )}
        </div>

        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            onRequestDelete();
          }}
          title={`Delete ${doc.filename}`}
          className="opacity-70 group-hover:opacity-100 rounded-md p-1 text-slate-500 hover:bg-rose-500/15 hover:text-rose-400 transition"
        >
          <Trash2 className="h-3.5 w-3.5" />
        </button>
      </div>
    </div>
  );
};
