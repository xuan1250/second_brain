import React, { createContext, useCallback, useContext, useMemo, useState } from "react";
import { isPdfFilenameOrType } from "../documentMeta";
import type { ActiveCitation, Citation, DocumentItem } from "../types";

export type InspectorTab = "pdf" | "markdown";

export interface ViewerContextValue {
  selectedDocumentId: string | null;
  selectedDocument: DocumentItem | null;
  inspectorTab: InspectorTab;
  activeCitation: ActiveCitation | null;
  selectDocument: (
    doc: DocumentItem | string | null,
    preferredTab?: InspectorTab
  ) => void;
  setSelectedDocumentId: (docId: string | null) => void;
  setInspectorTab: (tab: InspectorTab) => void;
  setActiveCitation: (citation: Citation | ActiveCitation | null) => void;
  clearActiveCitation: () => void;
}

const ViewerContext = createContext<ViewerContextValue | null>(null);

export const ViewerProvider: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  const [selectedDocumentId, setSelectedDocumentIdState] = useState<
    string | null
  >(null);
  const [selectedDocument, setSelectedDocument] = useState<DocumentItem | null>(
    null
  );
  const [inspectorTab, setInspectorTab] = useState<InspectorTab>("pdf");
  const [activeCitation, setActiveCitationState] =
    useState<ActiveCitation | null>(null);

  const selectDocument = useCallback(
    (doc: DocumentItem | string | null, preferredTab?: InspectorTab) => {
      if (!doc) {
        setSelectedDocumentIdState(null);
        setSelectedDocument(null);
        return;
      }
      if (typeof doc === "string") {
        setSelectedDocumentIdState(doc);
        setSelectedDocument((prev) => (prev?.id === doc ? prev : null));
        if (preferredTab) {
          setInspectorTab(preferredTab);
        }
        return;
      }
      setSelectedDocumentIdState(doc.id);
      setSelectedDocument(doc);
      if (preferredTab) {
        setInspectorTab(preferredTab);
      } else {
        const isPdf =
          isPdfFilenameOrType(doc.file_type) ||
          isPdfFilenameOrType(doc.filename);
        setInspectorTab(isPdf ? "pdf" : "markdown");
      }
    },
    []
  );

  const setSelectedDocumentId = useCallback((docId: string | null) => {
    setSelectedDocumentIdState(docId);
    if (!docId) {
      setSelectedDocument(null);
    } else {
      setSelectedDocument((prev) => (prev?.id === docId ? prev : null));
    }
  }, []);

  const setActiveCitation = useCallback(
    (citation: Citation | ActiveCitation | null) => {
      if (!citation) {
        setActiveCitationState(null);
        return;
      }
      const nextCitation: ActiveCitation = {
        ...citation,
        page_numbers: Array.isArray(citation.page_numbers)
          ? citation.page_numbers
          : [],
        heading_path: citation.heading_path ?? "",
        excerpt: citation.excerpt ?? "",
        triggered_at: Date.now(),
      };
      setActiveCitationState(nextCitation);
      if (nextCitation.doc_id) {
        setSelectedDocumentIdState(nextCitation.doc_id);
        setSelectedDocument((prev) =>
          prev?.id === nextCitation.doc_id ? prev : null
        );
      }

      const isPdfSource = isPdfFilenameOrType(nextCitation.source_filename);
      const hasPages = nextCitation.page_numbers.length > 0;
      setInspectorTab(isPdfSource && hasPages ? "pdf" : "markdown");
    },
    []
  );

  const clearActiveCitation = useCallback(() => {
    setActiveCitationState(null);
  }, []);

  const value = useMemo<ViewerContextValue>(
    () => ({
      selectedDocumentId,
      selectedDocument,
      inspectorTab,
      activeCitation,
      selectDocument,
      setSelectedDocumentId,
      setInspectorTab,
      setActiveCitation,
      clearActiveCitation,
    }),
    [
      selectedDocumentId,
      selectedDocument,
      inspectorTab,
      activeCitation,
      selectDocument,
      setSelectedDocumentId,
      setActiveCitation,
      clearActiveCitation,
    ]
  );

  return (
    <ViewerContext.Provider value={value}>{children}</ViewerContext.Provider>
  );
};

export function useViewer(): ViewerContextValue {
  const ctx = useContext(ViewerContext);
  if (!ctx) {
    throw new Error("useViewer must be used inside <WorkspaceProvider>");
  }
  return ctx;
}
