import { useMemo } from "react";
import type { ProviderOption, SystemStats, WorkspaceSettings } from "../lib/types";

export interface ActiveProviderStatusResult {
  providers: ProviderOption[];
  activeProv: string;
  activeOption: ProviderOption | undefined;
  isMissingKey: boolean;
  isInvalidKey: boolean;
  hasKeyIssue: boolean;
  defaultBadgeProvider: string;
  defaultBadgeModel: string;
}

export function useActiveProviderStatus({
  selectedProvider,
  selectedModel,
  wsSettings,
  stats,
}: {
  selectedProvider?: string;
  selectedModel?: string;
  wsSettings?: WorkspaceSettings;
  stats?: SystemStats;
}): ActiveProviderStatusResult {
  const providers = useMemo(
    () =>
      wsSettings?.available_providers?.length
        ? wsSettings.available_providers
        : (stats?.available_providers ?? stats?.configured_providers ?? []),
    [
      wsSettings?.available_providers,
      stats?.available_providers,
      stats?.configured_providers,
    ]
  );

  const activeProv =
    selectedProvider ||
    wsSettings?.default_llm_provider ||
    stats?.llm?.provider ||
    "openai";

  const activeOption = useMemo(
    () => providers.find((p) => p.provider === activeProv),
    [providers, activeProv]
  );

  const isMissingKey = Boolean(activeOption && !activeOption.configured);
  const isInvalidKey = Boolean(
    activeOption && activeOption.validation_status === "invalid"
  );
  const hasKeyIssue = isMissingKey || isInvalidKey;

  const defaultBadgeProvider =
    wsSettings?.default_llm_provider ||
    selectedProvider ||
    stats?.llm?.provider ||
    "openai";

  const defaultBadgeOption = useMemo(
    () => providers.find((p) => p.provider === defaultBadgeProvider),
    [providers, defaultBadgeProvider]
  );

  const defaultBadgeModel =
    wsSettings?.default_llm_model ||
    defaultBadgeOption?.default_model ||
    selectedModel ||
    stats?.llm?.model_name ||
    "gpt-4o-mini";

  return {
    providers,
    activeProv,
    activeOption,
    isMissingKey,
    isInvalidKey,
    hasKeyIssue,
    defaultBadgeProvider,
    defaultBadgeModel,
  };
}
