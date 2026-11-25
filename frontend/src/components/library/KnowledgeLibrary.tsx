import React, { useEffect, useMemo, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import {
  CheckCircle2,
  FileText,
  Loader2,
  ShieldCheck,
  Trash2,
} from "lucide-react";
import {
  isDocumentProcessing,
  useApproveBatchMutation,
  useApproveClassificationMutation,
  useDeleteDocumentMutation,
  useDocuments,
  usePauseIngestionMutation,
  useResumeIngestionMutation,
  useRetryDocumentMutation,
  useSystemStats,
} from "../../hooks/useDocuments";
import { useDomainsQuery } from "../../hooks/useDomains";
import { getDomainKey } from "../../lib/documentMeta";
import { queryKeys } from "../../lib/queryKeys";
import type { ApprovedClassificationRequest, DocumentItem } from "../../lib/types";
import { useWorkspace } from "../../lib/workspaceStore";
import { AlertBanner } from "../common/AlertBanner";
import { ConfirmDialog } from "../common/ConfirmDialog";
import { EmptyState } from "../common/EmptyState";
import { LoadingState } from "../common/LoadingState";
import { DocumentCard } from "./DocumentCard";
import { LibraryFilterBar } from "./LibraryFilterBar";
import { LibraryHeader } from "./LibraryHeader";
import { LibraryStatsFooter } from "./LibraryStatsFooter";
import { UploadDropzone } from "./UploadDropzone";

export const KnowledgeLibrary: React.FC = () => {
  const [searchText, setSearchText] = useState("");
  const [fileTypeFilter, setFileTypeFilter] = useState<string>("all");
  const [domainFacet, setDomainFacet] = useState<string>("all");
  const [docPendingDelete, setDocPendingDelete] =
    useState<DocumentItem | null>(null);

  const { data: domains = [] } = useDomainsQuery();

  useEffect(() => {
    if (
      domainFacet !== "all" &&
      domainFacet !== "review" &&
      domains.length > 0 &&
      !domains.some((d) => d.slug === domainFacet)
    ) {
      setDomainFacet("all");
    }
  }, [domains, domainFacet]);

  const {
    selectedDocumentId,
    selectDocument,
    selectedFilterDocIds,
    toggleFilterDocId,
    clearFilterDocIds,
    setSelectedFilterDocIds,
    activeCitation,
    clearActiveCitation,
  } = useWorkspace();

  const queryClient = useQueryClient();
  const { data, isLoading, isError, error } = useDocuments({
    file_type: fileTypeFilter === "all" ? undefined : fileTypeFilter,
    search: searchText || undefined,
  });

  const rawDocuments = data?.items ?? data?.documents ?? [];
  const anyProcessing = rawDocuments.some((doc) => isDocumentProcessing(doc));
  const { data: stats } = useSystemStats({ fastPoll: anyProcessing });

  const prevProcessingRef = useRef(false);
  useEffect(() => {
    if (prevProcessingRef.current && !anyProcessing) {
      void queryClient.invalidateQueries({ queryKey: queryKeys.systemStats() });
    }
    prevProcessingRef.current = anyProcessing;
  }, [anyProcessing, queryClient]);

  const deleteMutation = useDeleteDocumentMutation();
  const retryMutation = useRetryDocumentMutation();
  const pauseMutation = usePauseIngestionMutation();
  const resumeMutation = useResumeIngestionMutation();
  const approveClassificationMutation = useApproveClassificationMutation();
  const approveBatchMutation = useApproveBatchMutation();

  const awaitingApprovalDocs = useMemo(
    () =>
      rawDocuments.filter(
        (doc) => (doc.status || "").toLowerCase() === "awaiting_approval"
      ),
    [rawDocuments]
  );

  const filteredDocuments = useMemo(() => {
    if (domainFacet === "all") return rawDocuments;
    if (domainFacet === "review") {
      return rawDocuments.filter((doc) => {
        const s = (doc.status || "").toLowerCase();
        return s === "awaiting_approval" || s === "classifying";
      });
    }
    return rawDocuments.filter((doc) => getDomainKey(doc) === domainFacet);
  }, [rawDocuments, domainFacet]);

  const handleConfirmDelete = async () => {
    if (!docPendingDelete) return;
    const targetId = docPendingDelete.id;
    try {
      await deleteMutation.mutateAsync(targetId);
      setSelectedFilterDocIds((prev) => prev.filter((id) => id !== targetId));
      if (selectedDocumentId === targetId) selectDocument(null);
      if (activeCitation?.doc_id === targetId) clearActiveCitation();
    } finally {
      setDocPendingDelete(null);
    }
  };

  const handleApproveDoc = (
    docId: string,
    payload?: ApprovedClassificationRequest
  ) => {
    approveClassificationMutation.mutate({ docId, payload });
  };

  const handleApproveAllRecommended = () => {
    const ids = awaitingApprovalDocs.map((d) => d.id);
    if (ids.length > 0) {
      approveBatchMutation.mutate(ids);
    }
  };

  const isUnfiltered =
    fileTypeFilter === "all" && !searchText && domainFacet === "all";

  return (
    <div className="flex h-full flex-col bg-slate-950 border-r border-slate-800/80 select-none">
      <LibraryHeader
        totalDocs={data?.total ?? rawDocuments.length}
        ingestionPaused={stats?.ingestion_paused}
        isTogglePending={pauseMutation.isPending || resumeMutation.isPending}
        onToggleIngestion={() => {
          if (stats?.ingestion_paused) void resumeMutation.mutate();
          else void pauseMutation.mutate();
        }}
      />

      <div className="p-3 border-b border-slate-800/60">
        <UploadDropzone />
      </div>

      <LibraryFilterBar
        searchText={searchText}
        onSearchTextChange={setSearchText}
        domainFacet={domainFacet}
        onDomainFacetChange={setDomainFacet}
        fileTypeFilter={fileTypeFilter}
        onFileTypeFilterChange={setFileTypeFilter}
        domains={domains}
        awaitingCount={awaitingApprovalDocs.length}
        selectedFilterDocIdsCount={selectedFilterDocIds.length}
        onClearFilterDocIds={clearFilterDocIds}
      />

      {awaitingApprovalDocs.length > 0 && (
        <div className="sticky top-0 z-10 flex items-center justify-between gap-2 bg-gradient-to-r from-amber-950/90 to-amber-900/80 border-b border-amber-500/40 px-3 py-2 text-xs text-amber-100 backdrop-blur-xs">
          <div className="flex items-center gap-1.5 min-w-0">
            <ShieldCheck className="h-4 w-4 text-amber-400 shrink-0" />
            <span className="truncate font-medium">
              <strong>{awaitingApprovalDocs.length}</strong> document
              {awaitingApprovalDocs.length > 1 ? "s" : ""} awaiting domain approval
            </span>
          </div>
          <button
            type="button"
            disabled={approveBatchMutation.isPending}
            onClick={handleApproveAllRecommended}
            className="inline-flex items-center gap-1 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white px-2.5 py-1 text-[11px] font-semibold transition shrink-0 shadow-xs disabled:opacity-50"
          >
            {approveBatchMutation.isPending ? (
              <Loader2 className="h-3 w-3 animate-spin" />
            ) : (
              <CheckCircle2 className="h-3 w-3" />
            )}
            <span>Approve All Recommended</span>
          </button>
        </div>
      )}

      <div className="flex-1 overflow-y-auto p-2 space-y-1.5">
        {isLoading ? (
          <LoadingState label="Loading knowledge base..." />
        ) : isError ? (
          <AlertBanner
            variant="error"
            message={`Failed to load documents: ${
              error instanceof Error ? error.message : "Unknown error"
            }`}
          />
        ) : filteredDocuments.length === 0 ? (
          <EmptyState
            icon={<FileText className="h-8 w-8 text-slate-700" />}
            title="No documents found"
            description={
              domainFacet !== "all"
                ? `No documents match facet "${domainFacet}".`
                : "Upload PDF, Word, Excel, or PowerPoint files above to build your Second Brain."
            }
          />
        ) : (
          filteredDocuments.map((doc) => (
            <DocumentCard
              key={doc.id}
              doc={doc}
              domains={domains}
              isSelected={selectedDocumentId === doc.id}
              isScoped={selectedFilterDocIds.includes(doc.id)}
              isCitedDoc={activeCitation?.doc_id === doc.id}
              isApproving={approveClassificationMutation.isPending}
              isRetrying={retryMutation.isPending}
              onSelect={() => selectDocument(doc)}
              onToggleScope={() => toggleFilterDocId(doc.id)}
              onApprove={handleApproveDoc}
              onRetry={(options) =>
                void retryMutation.mutate({ docId: doc.id, options })
              }
              onRequestDelete={() => setDocPendingDelete(doc)}
            />
          ))
        )}
      </div>

      {stats && (
        <LibraryStatsFooter
          stats={stats}
          rawDocuments={rawDocuments}
          totalFromList={data?.total}
          isUnfiltered={isUnfiltered}
        />
      )}

      <ConfirmDialog
        open={Boolean(docPendingDelete)}
        onOpenChange={(open) => {
          if (!open) setDocPendingDelete(null);
        }}
        title="Delete Document Atomically?"
        description={
          <>
            This will permanently remove{" "}
            <span className="font-medium text-slate-200">
              {docPendingDelete?.filename}
            </span>{" "}
            from Qdrant hybrid vectors, SQLite metadata tables, and cached raw /
            Markdown files on disk.
          </>
        }
        confirmLabel="Delete Document"
        confirmIcon={<Trash2 className="h-3.5 w-3.5" />}
        isPending={deleteMutation.isPending}
        onConfirm={handleConfirmDelete}
      />
    </div>
  );
};
