import React, { createContext, useCallback, useContext, useMemo, useState } from "react";
import type { DomainFilterConfig } from "../types";

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
      return JSON.parse(raw) as Record<string, DomainFilterConfig>;
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

export interface ChatFilterContextValue {
  selectedFilterDocIds: string[];
  activeThreadId: string | null;
  currentDomainFilter: DomainFilterConfig;
  setThreadDomainFilter: (
    threadId: string | null,
    filter: DomainFilterConfig
  ) => void;
  migrateDraftDomainFilter: (resolvedThreadId: string) => void;
  pruneDeletedDomain: (deletedSlug: string) => void;
  toggleFilterDocId: (docId: string) => void;
  setSelectedFilterDocIds: React.Dispatch<React.SetStateAction<string[]>>;
  clearFilterDocIds: () => void;
  setActiveThreadId: (threadId: string | null) => void;
}

const ChatFilterContext = createContext<ChatFilterContextValue | null>(null);

export const ChatFilterProvider: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  const [selectedFilterDocIds, setSelectedFilterDocIds] = useState<string[]>(
    []
  );
  const [activeThreadId, setActiveThreadId] = useState<string | null>(null);
  const [threadFilters, setThreadFilters] = useState<
    Record<string, DomainFilterConfig>
  >(() => loadThreadFilters());

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

  const toggleFilterDocId = useCallback((docId: string) => {
    setSelectedFilterDocIds((prev) =>
      prev.includes(docId) ? prev.filter((id) => id !== docId) : [...prev, docId]
    );
  }, []);

  const clearFilterDocIds = useCallback(() => {
    setSelectedFilterDocIds([]);
  }, []);

  const value = useMemo<ChatFilterContextValue>(
    () => ({
      selectedFilterDocIds,
      activeThreadId,
      currentDomainFilter,
      setThreadDomainFilter,
      migrateDraftDomainFilter,
      pruneDeletedDomain,
      toggleFilterDocId,
      setSelectedFilterDocIds,
      clearFilterDocIds,
      setActiveThreadId,
    }),
    [
      selectedFilterDocIds,
      activeThreadId,
      currentDomainFilter,
      setThreadDomainFilter,
      migrateDraftDomainFilter,
      pruneDeletedDomain,
      toggleFilterDocId,
      clearFilterDocIds,
    ]
  );

  return (
    <ChatFilterContext.Provider value={value}>
      {children}
    </ChatFilterContext.Provider>
  );
};

export function useChatFilter(): ChatFilterContextValue {
  const ctx = useContext(ChatFilterContext);
  if (!ctx) {
    throw new Error("useChatFilter must be used inside <WorkspaceProvider>");
  }
  return ctx;
}
