import React, { createContext, useCallback, useContext, useMemo, useState } from "react";

export type SettingsTabId = "domains" | "lexicon" | "ingestion";

export interface SettingsModalContextValue {
  isSettingsModalOpen: boolean;
  settingsModalTab: SettingsTabId;
  openSettingsModal: (tab?: SettingsTabId) => void;
  setSettingsModalOpen: (open: boolean) => void;
}

const SettingsModalContext = createContext<SettingsModalContextValue | null>(
  null
);

export const SettingsModalProvider: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  const [isSettingsModalOpen, setIsSettingsModalOpen] =
    useState<boolean>(false);
  const [settingsModalTab, setSettingsModalTab] =
    useState<SettingsTabId>("domains");

  const openSettingsModal = useCallback((tab?: SettingsTabId) => {
    if (tab) {
      setSettingsModalTab(tab);
    }
    setIsSettingsModalOpen(true);
  }, []);

  const setSettingsModalOpen = useCallback((open: boolean) => {
    setIsSettingsModalOpen(open);
  }, []);

  const value = useMemo<SettingsModalContextValue>(
    () => ({
      isSettingsModalOpen,
      settingsModalTab,
      openSettingsModal,
      setSettingsModalOpen,
    }),
    [
      isSettingsModalOpen,
      settingsModalTab,
      openSettingsModal,
      setSettingsModalOpen,
    ]
  );

  return (
    <SettingsModalContext.Provider value={value}>
      {children}
    </SettingsModalContext.Provider>
  );
};

export function useSettingsModal(): SettingsModalContextValue {
  const ctx = useContext(SettingsModalContext);
  if (!ctx) {
    throw new Error("useSettingsModal must be used inside <WorkspaceProvider>");
  }
  return ctx;
}
