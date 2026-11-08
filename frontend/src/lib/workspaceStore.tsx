import React, {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
} from "react";
import type {
  ActiveCitation,
  Citation,
  DocumentItem,
  DomainFilterConfig,
} from "./types";

export type InspectorTab = "pdf" | "markdown";

export const DEFAULT_DOMAIN_FILTER: DomainFilterConfig = {
  mode: "auto",
  domains: [],
  subdomains: [],
  max_authority_tier: null,
  only_current: true,
};

const STORAGE_KEY_THREAD_FILTERS = "second_brain_thread_domain_filters";

function loadThreadFilters(): Record<string, DomainFilterConfig> {
  try {
    const raw = localStorage.getItem(STORAGE_KEY_THREAD_FILTERS);
    if (raw) {
      return JSON.parse(raw);
    }
  } catch {
    // ignore
  }
  return {};
}

function saveThreadFilters(map: Record<string, DomainFilterConfig>) {
  try {
    localStorage.setItem(STORAGE_KEY_THREAD_FILTERS, JSON.stringify(map));
  } catch {
    // ignore
  }
}

export interface WorkspaceStateContextValue {
  selectedDocumentId: string | null;
  selectedDocument: DocumentItem | null;
  inspectorTab: InspectorTab;
  activeCitation: ActiveCitation | null;
  selectedFilterDocIds: string[];
  activeThreadId: string | null;
  selectedProvider: string;
  selectedModel: string;
  currentDomainFilter: DomainFilterConfig;
  setThreadDomainFilter: (
    threadId: string | null,
    filter: DomainFilterConfig
  ) => void;
  migrateDraftDomainFilter: (resolvedThreadId: string) => void;
  pruneDeletedDomain: (deletedSlug: string) => void;
  selectDocument: (
    doc: DocumentItem | string | null,
    preferredTab?: InspectorTab
  ) => void;
  setSelectedDocumentId: (docId: string | null) => void;
  setInspectorTab: (tab: InspectorTab) => void;
  setActiveCitation: (citation: Citation | ActiveCitation | null) => void;
  clearActiveCitation: () => void;
  toggleFilterDocId: (docId: string) => void;
  setSelectedFilterDocIds: React.Dispatch<React.SetStateAction<string[]>>;
  clearFilterDocIds: () => void;
  setActiveThreadId: (threadId: string | null) => void;
  setSelectedProvider: (provider: string) => void;
  setSelectedModel: (model: string) => void;
  isSettingsModalOpen: boolean;
  settingsModalTab: "domains" | "lexicon" | "ingestion";
  openSettingsModal: (tab?: "domains" | "lexicon" | "ingestion") => void;
  setSettingsModalOpen: (open: boolean) => void;
}

const WorkspaceStateContext = createContext<WorkspaceStateContextValue | null>(
  null
);

function isPdfFilenameOrType(filenameOrType: string | undefined | null): boolean {
  if (!filenameOrType) return false;
  const lower = filenameOrType.trim().toLowerCase();
  return lower === "pdf" || lower.endsWith(".pdf");
}

export const WorkspaceProvider: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  const [selectedDocumentId, setSelectedDocumentIdState] = useState<string | null>(
    null
  );
  const [selectedDocument, setSelectedDocument] = useState<DocumentItem | null>(
    null
  );
  const [inspectorTab, setInspectorTab] = useState<InspectorTab>("pdf");
  const [activeCitation, setActiveCitationState] =
    useState<ActiveCitation | null>(null);
  const [selectedFilterDocIds, setSelectedFilterDocIds] = useState<string[]>([]);
  const [activeThreadId, setActiveThreadId] = useState<string | null>(null);

  const [threadFilters, setThreadFilters] = useState<
    Record<string, DomainFilterConfig>
  >(() => loadThreadFilters());

  const [selectedProvider, setSelectedProviderState] = useState<string>(() => {
    try {
      return localStorage.getItem("second_brain_provider") || "";
    } catch {
      return "";
    }
  });
  const [selectedModel, setSelectedModelState] = useState<string>(() => {
    try {
      return localStorage.getItem("second_brain_model") || "";
    } catch {
      return "";
    }
  });

  const setSelectedProvider = useCallback((provider: string) => {
    setSelectedProviderState(provider);
    try {
      if (provider) localStorage.setItem("second_brain_provider", provider);
      else localStorage.removeItem("second_brain_provider");
    } catch {
      // ignore storage errors
    }
  }, []);

  const setSelectedModel = useCallback((model: string) => {
    setSelectedModelState(model);
    try {
      if (model) localStorage.setItem("second_brain_model", model);
      else localStorage.removeItem("second_brain_model");
    } catch {
      // ignore storage errors
    }
  }, []);

  const [isSettingsModalOpen, setIsSettingsModalOpen] = useState<boolean>(false);
  const [settingsModalTab, setSettingsModalTab] = useState<
    "domains" | "lexicon" | "ingestion"
  >("domains");

  const openSettingsModal = useCallback(
    (tab?: "domains" | "lexicon" | "ingestion") => {
      if (tab) {
        setSettingsModalTab(tab);
      }
      setIsSettingsModalOpen(true);
    },
    []
  );

  const setSettingsModalOpen = useCallback((open: boolean) => {
    setIsSettingsModalOpen(open);
  }, []);

  const currentDomainFilter = useMemo<DomainFilterConfig>(() => {
    const key = activeThreadId || "__draft__";
    return threadFilters[key] || DEFAULT_DOMAIN_FILTER;
  }, [activeThreadId, threadFilters]);

  const setThreadDomainFilter = useCallback(
    (threadId: string | null, filter: DomainFilterConfig) => {
      const key = threadId || "__draft__";
      setThreadFilters((prev) => {
        const next = { ...prev, [key]: filter };
        saveThreadFilters(next);
        return next;
      });
    },
    []
  );

  const migrateDraftDomainFilter = useCallback((resolvedThreadId: string) => {
    setThreadFilters((prev) => {
      const draft = prev["__draft__"];
      if (!draft) return prev;
      const next = { ...prev, [resolvedThreadId]: draft };
      delete next["__draft__"];
      saveThreadFilters(next);
      return next;
    });
  }, []);

  const pruneDeletedDomain = useCallback((deletedSlug: string) => {
    setThreadFilters((prev) => {
      let changed = false;
      const next: Record<string, DomainFilterConfig> = {};
      for (const [key, filter] of Object.entries(prev)) {
        if (filter.domains && filter.domains.includes(deletedSlug)) {
          changed = true;
          const nextDomains = filter.domains.filter((d) => d !== deletedSlug);
          next[key] = {
            ...filter,
            domains: nextDomains,
            mode: nextDomains.length === 0 ? "auto" : filter.mode,
          };
        } else {
          next[key] = filter;
        }
      }
      if (changed) {
        saveThreadFilters(next);
        return next;
      }
      return prev;
    });
  }, []);

  const selectDocument = useCallback(
    (doc: DocumentItem | string | null, preferredTab?: InspectorTab) => {
      if (!doc) {
        setSelectedDocumentIdState(null);
        setSelectedDocument(null);
        return;
      }
      if (typeof doc === "string") {
        setSelectedDocumentIdState(doc);
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
          isPdfFilenameOrType(doc.file_type) || isPdfFilenameOrType(doc.filename);
        setInspectorTab(isPdf ? "pdf" : "markdown");
      }
    },
    []
  );

  const setSelectedDocumentId = useCallback((docId: string | null) => {
    setSelectedDocumentIdState(docId);
    if (!docId) {
      setSelectedDocument(null);
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
      }

      const isPdfSource = isPdfFilenameOrType(nextCitation.source_filename);
      const hasPages = nextCitation.page_numbers.length > 0;
      if (isPdfSource && hasPages) {
        setInspectorTab("pdf");
      } else {
        setInspectorTab("markdown");
      }
    },
    []
  );

  const clearActiveCitation = useCallback(() => {
    setActiveCitationState(null);
  }, []);

  const toggleFilterDocId = useCallback((docId: string) => {
    setSelectedFilterDocIds((prev) =>
      prev.includes(docId) ? prev.filter((id) => id !== docId) : [...prev, docId]
    );
  }, []);

  const clearFilterDocIds = useCallback(() => {
    setSelectedFilterDocIds([]);
  }, []);

  const value = useMemo<WorkspaceStateContextValue>(
    () => ({
      selectedDocumentId,
      selectedDocument,
      inspectorTab,
      activeCitation,
      selectedFilterDocIds,
      activeThreadId,
      selectedProvider,
      selectedModel,
      currentDomainFilter,
      setThreadDomainFilter,
      migrateDraftDomainFilter,
      pruneDeletedDomain,
      selectDocument,
      setSelectedDocumentId,
      setInspectorTab,
      setActiveCitation,
      clearActiveCitation,
      toggleFilterDocId,
      setSelectedFilterDocIds,
      clearFilterDocIds,
      setActiveThreadId,
      setSelectedProvider,
      setSelectedModel,
      isSettingsModalOpen,
      settingsModalTab,
      openSettingsModal,
      setSettingsModalOpen,
    }),
    [
      selectedDocumentId,
      selectedDocument,
      inspectorTab,
      activeCitation,
      selectedFilterDocIds,
      activeThreadId,
      selectedProvider,
      selectedModel,
      currentDomainFilter,
      setThreadDomainFilter,
      migrateDraftDomainFilter,
      pruneDeletedDomain,
      selectDocument,
      setSelectedDocumentId,
      setActiveCitation,
      clearActiveCitation,
      toggleFilterDocId,
      clearFilterDocIds,
      isSettingsModalOpen,
      settingsModalTab,
      openSettingsModal,
      setSettingsModalOpen,
    ]
  );

  return (
    <WorkspaceStateContext.Provider value={value}>
      {children}
    </WorkspaceStateContext.Provider>
  );
};

export function useWorkspace(): WorkspaceStateContextValue {
  const ctx = useContext(WorkspaceStateContext);
  if (!ctx) {
    throw new Error("useWorkspace must be used inside a <WorkspaceProvider>");
  }
  return ctx;
}
