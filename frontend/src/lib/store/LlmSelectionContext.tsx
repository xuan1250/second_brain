import React, { createContext, useCallback, useContext, useMemo, useState } from "react";

export interface LlmSelectionContextValue {
  selectedProvider: string;
  selectedModel: string;
  setSelectedProvider: (provider: string) => void;
  setSelectedModel: (model: string) => void;
}

const LlmSelectionContext = createContext<LlmSelectionContextValue | null>(
  null
);

export const LlmSelectionProvider: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
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

  const value = useMemo<LlmSelectionContextValue>(
    () => ({
      selectedProvider,
      selectedModel,
      setSelectedProvider,
      setSelectedModel,
    }),
    [selectedProvider, selectedModel, setSelectedProvider, setSelectedModel]
  );

  return (
    <LlmSelectionContext.Provider value={value}>
      {children}
    </LlmSelectionContext.Provider>
  );
};

export function useLlmSelection(): LlmSelectionContextValue {
  const ctx = useContext(LlmSelectionContext);
  if (!ctx) {
    throw new Error("useLlmSelection must be used inside <WorkspaceProvider>");
  }
  return ctx;
}
