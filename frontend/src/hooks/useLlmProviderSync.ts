import { useCallback, useEffect, useRef } from "react";
import { useUpdateWorkspaceSettingsMutation } from "./useWorkspaceSettings";
import { LLM_PROVIDERS } from "../lib/llmCatalog";
import type { ProviderOption, SystemStats, WorkspaceSettings } from "../lib/types";

export function useLlmProviderSync({
  wsSettings,
  stats,
  providers,
  selectedProvider,
  selectedModel,
  setSelectedProvider,
  setSelectedModel,
}: {
  wsSettings?: WorkspaceSettings;
  stats?: SystemStats;
  providers: ProviderOption[];
  selectedProvider: string;
  selectedModel: string;
  setSelectedProvider: (provider: string) => void;
  setSelectedModel: (model: string) => void;
}) {
  const updateWorkspaceSettingsMutation = useUpdateWorkspaceSettingsMutation();

  const hasInitializedModelRef = useRef(false);
  const selectedProviderRef = useRef(selectedProvider);
  selectedProviderRef.current = selectedProvider;
  const selectedModelRef = useRef(selectedModel);
  selectedModelRef.current = selectedModel;
  const lastSubmittedOverrideRef = useRef<string | null>(null);

  useEffect(() => {
    if (wsSettings) {
      lastSubmittedOverrideRef.current = null;
      if (wsSettings.default_llm_provider) {
        const matchedOption = providers.find(
          (p) => p.provider === wsSettings.default_llm_provider
        );
        setSelectedProvider(wsSettings.default_llm_provider);
        setSelectedModel(
          wsSettings.default_llm_model ||
            matchedOption?.default_model ||
            stats?.llm?.model_name ||
            ""
        );
        hasInitializedModelRef.current = true;
        return;
      }

      const currentProvider = selectedProviderRef.current;
      const currentOption = currentProvider
        ? providers.find((p) => p.provider === currentProvider)
        : undefined;

      if (currentProvider && currentOption?.configured) {
        if (!selectedModelRef.current && currentOption.default_model) {
          setSelectedModel(currentOption.default_model);
        }
        hasInitializedModelRef.current = true;
        return;
      }

      const configuredProviders = providers.filter((p) => p.configured);
      const preferredConfigured =
        configuredProviders.find((p) => p.provider === stats?.llm?.provider) ||
        configuredProviders[0];

      if (preferredConfigured) {
        setSelectedProvider(preferredConfigured.provider);
        setSelectedModel(
          preferredConfigured.default_model ||
            (preferredConfigured.provider === stats?.llm?.provider
              ? stats?.llm?.model_name
              : "") ||
            ""
        );
      } else if (!currentProvider && stats?.llm?.provider) {
        setSelectedProvider(stats.llm.provider);
        if (!selectedModelRef.current && stats.llm.model_name) {
          setSelectedModel(stats.llm.model_name);
        }
      }
      hasInitializedModelRef.current = true;
      return;
    }

    if (!hasInitializedModelRef.current && stats?.llm) {
      if (!selectedProviderRef.current && stats.llm.provider) {
        setSelectedProvider(stats.llm.provider);
      }
      if (
        !selectedModelRef.current &&
        stats.llm.model_name &&
        (!selectedProviderRef.current ||
          selectedProviderRef.current === stats.llm.provider)
      ) {
        setSelectedModel(stats.llm.model_name);
      }
      if (stats.llm.provider || stats.llm.model_name) {
        hasInitializedModelRef.current = true;
      }
    }
  }, [
    wsSettings,
    providers,
    stats?.llm?.provider,
    stats?.llm?.model_name,
    setSelectedProvider,
    setSelectedModel,
  ]);

  const commitModelOverride = useCallback(
    (rawValue: string) => {
      const trimmedModel = rawValue.trim();
      const effectiveProvider =
        selectedProvider ||
        wsSettings?.default_llm_provider ||
        stats?.llm?.provider ||
        "";
      if (!effectiveProvider) return;
      if (trimmedModel !== selectedModel) {
        setSelectedModel(trimmedModel);
      }
      const commitKey = `${effectiveProvider}::${trimmedModel}`;
      if (
        lastSubmittedOverrideRef.current === commitKey ||
        (effectiveProvider === (wsSettings?.default_llm_provider || "") &&
          trimmedModel === (wsSettings?.default_llm_model || ""))
      ) {
        return;
      }
      lastSubmittedOverrideRef.current = commitKey;
      updateWorkspaceSettingsMutation.mutate(
        {
          default_llm_provider: effectiveProvider,
          default_llm_model: trimmedModel || null,
        },
        {
          onError: () => {
            lastSubmittedOverrideRef.current = null;
          },
        }
      );
    },
    [
      selectedProvider,
      selectedModel,
      wsSettings?.default_llm_provider,
      wsSettings?.default_llm_model,
      stats?.llm?.provider,
      setSelectedModel,
      updateWorkspaceSettingsMutation,
    ]
  );

  const selectProviderAndPersist = useCallback(
    (nextProv: string) => {
      setSelectedProvider(nextProv);
      const matched = providers.find((p) => p.provider === nextProv);
      const catalogEntry = LLM_PROVIDERS.find((p) => p.id === nextProv);
      const nextModel =
        matched?.default_model || catalogEntry?.defaultModelFallback || "";

      if (nextModel) {
        setSelectedModel(nextModel);
      }
      lastSubmittedOverrideRef.current = `${nextProv}::${nextModel}`;
      updateWorkspaceSettingsMutation.mutate(
        {
          default_llm_provider: nextProv || null,
          default_llm_model: nextModel || null,
        },
        {
          onError: () => {
            lastSubmittedOverrideRef.current = null;
          },
        }
      );
    },
    [
      providers,
      setSelectedProvider,
      setSelectedModel,
      updateWorkspaceSettingsMutation,
    ]
  );

  return {
    commitModelOverride,
    selectProviderAndPersist,
  };
}
