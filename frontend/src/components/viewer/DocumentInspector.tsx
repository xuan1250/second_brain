import React, { useMemo, useState } from "react";
import * as Tabs from "@radix-ui/react-tabs";
import { BookOpen } from "lucide-react";
import { useDocumentMetadataDraft } from "../../hooks/useDocumentMetadataDraft";
import {
  isDocumentProcessing,
  useApproveClassificationMutation,
  useDocument,
  useRetryDocumentMutation,
  useUpdateDocumentMetadataMutation,
} from "../../hooks/useDocuments";
import { useDomainsQuery } from "../../hooks/useDomains";
import {
  isPdfFilenameOrType,
  parseClassificationResult,
} from "../../lib/documentMeta";
import type {
  ApprovedClassificationRequest,
  UpdateDocumentMetadataRequest,
} from "../../lib/types";
import { type InspectorTab, useWorkspace } from "../../lib/workspaceStore";
import { EmptyState } from "../common/EmptyState";
import { CitationBridgeBanner } from "./CitationBridgeBanner";
import { ClassificationReviewBar } from "./ClassificationReviewBar";
import { EditMetadataDialog } from "./EditMetadataDialog";
import { InspectorToolbar } from "./InspectorToolbar";
import { MarkdownViewerPane } from "./MarkdownViewerPane";
import { PdfViewerPane } from "./PdfViewerPane";

export const DocumentInspector: React.FC = () => {
  const { data: domains = [] } = useDomainsQuery();
  const {
    selectedDocumentId,
    selectedDocument,
    inspectorTab,
    setInspectorTab,
    activeCitation,
    clearActiveCitation,
  } = useWorkspace();

  const [scale, setScale] = useState<number>(1.0);
  const [showMetadataDialog, setShowMetadataDialog] = useState(false);
  const [isAdjustingApproval, setIsAdjustingApproval] = useState(false);

  const { data: fetchedDoc } = useDocument(selectedDocumentId);
  const retryMutation = useRetryDocumentMutation();
  const approveMutation = useApproveClassificationMutation();
  const updateMetaMutation = useUpdateDocumentMetadataMutation();

  const activeDoc = fetchedDoc ?? selectedDocument;
  const isPdf =
    isPdfFilenameOrType(activeDoc?.file_type) ||
    isPdfFilenameOrType(activeDoc?.filename) ||
    isPdfFilenameOrType(activeCitation?.source_filename);

  const isProcessing = isDocumentProcessing(activeDoc);
  const isUpgrading =
    retryMutation.isPending ||
    (isProcessing && activeDoc?.parsing_engine === "pymupdf4llm");
  const canUpgrade =
    activeDoc?.file_type?.toLowerCase() === "pdf" &&
    activeDoc?.parsing_engine === "pymupdf4llm" &&
    (activeDoc?.status === "ready" || isUpgrading);

  const isAwaitingApproval =
    (activeDoc?.status || "").toLowerCase() === "awaiting_approval";

  const profile = useMemo(
    () => (activeDoc ? parseClassificationResult(activeDoc) : null),
    [activeDoc]
  );

  const draft = useDocumentMetadataDraft(activeDoc, profile);

  const zoomOut = () =>
    setScale((s) => Math.max(0.6, Math.round((s - 0.15) * 100) / 100));
  const zoomIn = () =>
    setScale((s) => Math.min(2.0, Math.round((s + 0.15) * 100) / 100));
  const resetZoom = () => setScale(1.0);

  const buildMetadataPayload = (): UpdateDocumentMetadataRequest => ({
    domain: draft.editDomain,
    subdomain: draft.editSubdomain || undefined,
    authority_tier: draft.editTier,
    jurisdiction: draft.editJurisdiction || undefined,
    effective_year: draft.editYear,
    section_overrides:
      draft.localOverrides.length > 0 ? draft.localOverrides : undefined,
  });

  const handleApprove = (payload?: ApprovedClassificationRequest) => {
    if (!activeDoc) return;
    approveMutation.mutate({
      docId: activeDoc.id,
      payload: payload ?? buildMetadataPayload(),
    });
    setIsAdjustingApproval(false);
  };

  const handleSaveMetadata = () => {
    if (!activeDoc) return;
    updateMetaMutation.mutate({
      docId: activeDoc.id,
      payload: buildMetadataPayload(),
    });
    setShowMetadataDialog(false);
  };

  if (!selectedDocumentId) {
    return (
      <EmptyState
        className="h-full bg-slate-950 p-8 select-none"
        icon={
          <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-slate-900 border border-slate-800 text-indigo-400 shadow-inner">
            <BookOpen className="h-7 w-7" />
          </div>
        }
        title="Document Inspector"
        description={
          <>
            Select a document in the <strong>Knowledge Library</strong> on the
            left, or click any inline citation pill{" "}
            <code className="text-indigo-300">[1]</code> in the{" "}
            <strong>Intelligent Assistant</strong> to inspect the exact source
            page or structured Markdown section.
          </>
        }
      />
    );
  }

  const displayFilename =
    activeDoc?.filename ||
    activeCitation?.source_filename ||
    `Document ${selectedDocumentId.slice(0, 8)}`;

  return (
    <Tabs.Root
      value={inspectorTab}
      onValueChange={(val) => setInspectorTab(val as InspectorTab)}
      className="flex h-full flex-col bg-slate-950 overflow-hidden"
    >
      {isAwaitingApproval && activeDoc && (
        <ClassificationReviewBar
          domains={domains}
          profile={profile}
          editDomain={draft.editDomain}
          onEditDomainChange={draft.setEditDomain}
          editSubdomain={draft.editSubdomain}
          onEditSubdomainChange={draft.setEditSubdomain}
          editTier={draft.editTier}
          onEditTierChange={draft.setEditTier}
          editJurisdiction={draft.editJurisdiction}
          onEditJurisdictionChange={draft.setEditJurisdiction}
          editYear={draft.editYear}
          isAdjusting={isAdjustingApproval}
          onToggleAdjusting={() => setIsAdjustingApproval((v) => !v)}
          isApproving={approveMutation.isPending}
          onApprove={() => handleApprove()}
        />
      )}

      <InspectorToolbar
        activeDoc={activeDoc}
        displayFilename={displayFilename}
        domains={domains}
        isPdf={isPdf}
        canUpgrade={canUpgrade}
        isUpgrading={isUpgrading}
        scale={scale}
        onEditMetadata={() => {
          draft.resetFromDocument();
          setShowMetadataDialog(true);
        }}
        onUpgradeToGpu={() => {
          if (activeDoc) {
            retryMutation.mutate({
              docId: activeDoc.id,
              options: { forceReparse: true, preferFastPath: false },
            });
          }
        }}
        onZoomOut={zoomOut}
        onResetZoom={resetZoom}
        onZoomIn={zoomIn}
      />

      {activeCitation && activeCitation.doc_id === selectedDocumentId && (
        <CitationBridgeBanner
          citation={activeCitation}
          onDismiss={clearActiveCitation}
        />
      )}

      <Tabs.Content value="pdf" className="flex-1 overflow-hidden outline-none">
        {isPdf ? (
          <PdfViewerPane
            docId={selectedDocumentId}
            filename={displayFilename}
            scale={scale}
            activeCitation={activeCitation}
          />
        ) : (
          <MarkdownViewerPane
            docId={selectedDocumentId}
            activeDoc={activeDoc}
            activeCitation={activeCitation}
            scale={scale}
            onUpdateSectionOverride={draft.handleSectionOverride}
            sectionOverrides={draft.localOverrides}
          />
        )}
      </Tabs.Content>

      <Tabs.Content
        value="markdown"
        className="flex-1 overflow-hidden outline-none"
      >
        <MarkdownViewerPane
          docId={selectedDocumentId}
          activeDoc={activeDoc}
          activeCitation={activeCitation}
          scale={scale}
          onUpdateSectionOverride={draft.handleSectionOverride}
          sectionOverrides={draft.localOverrides}
        />
      </Tabs.Content>

      <EditMetadataDialog
        open={showMetadataDialog}
        onOpenChange={setShowMetadataDialog}
        filename={activeDoc?.filename}
        domains={domains}
        editDomain={draft.editDomain}
        onEditDomainChange={draft.setEditDomain}
        editSubdomain={draft.editSubdomain}
        onEditSubdomainChange={draft.setEditSubdomain}
        editTier={draft.editTier}
        onEditTierChange={draft.setEditTier}
        editJurisdiction={draft.editJurisdiction}
        onEditJurisdictionChange={draft.setEditJurisdiction}
        editYear={draft.editYear}
        onEditYearChange={draft.setEditYear}
        isPending={updateMetaMutation.isPending}
        onSave={handleSaveMetadata}
      />
    </Tabs.Root>
  );
};
