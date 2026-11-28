import React, { useState } from "react";
import { Key } from "lucide-react";
import {
  useUpdateWorkspaceSettingsMutation,
  useVerifyLlmKeyMutation,
} from "../../hooks/useWorkspaceSettings";
import { LLM_PROVIDERS } from "../../lib/llmCatalog";
import type { WorkspaceSettings } from "../../lib/types";
import { useWorkspace } from "../../lib/workspaceStore";
import { ProviderApiKeyCard } from "./ProviderApiKeyCard";

export interface ProviderApiKeysSectionProps {
  wsSettings?: WorkspaceSettings;
  llmProvider: string;
  llmModel: string;
  compatBaseUrl: string;
  apiKeyDrafts: Record<string, string>;
  onCompatBaseUrlChange: (value: string) => void;
  onApiKeyDraftsChange: React.Dispatch<
    React.SetStateAction<Record<string, string>>
  >;
  onDefaultApplied: (provider: string, model: string) => void;
}

export const ProviderApiKeysSection: React.FC<ProviderApiKeysSectionProps> = ({
  wsSettings,
  llmProvider,
  llmModel,
  compatBaseUrl,
  apiKeyDrafts,
  onCompatBaseUrlChange,
  onApiKeyDraftsChange,
  onDefaultApplied,
}) => {
  const { setSelectedProvider, setSelectedModel } = useWorkspace();
  const updateSettingsMutation = useUpdateWorkspaceSettingsMutation();
  const verifyKeyMutation = useVerifyLlmKeyMutation();

  const [revealedKeys, setRevealedKeys] = useState<Record<string, boolean>>({});
  const [verifyingProvider, setVerifyingProvider] = useState<string | null>(
    null
  );
  const [verifyFeedback, setVerifyFeedback] = useState<
    Record<string, { valid: boolean; message: string }>
  >({});

  const handleVerifyKey = async (provId: string) => {
    setVerifyingProvider(provId);
    setVerifyFeedback((prev) => {
      const next = { ...prev };
      delete next[provId];
      return next;
    });
    try {
      const draftKey = apiKeyDrafts[provId]?.trim() || null;
      const isDefaultProv =
        provId === (llmProvider || wsSettings?.default_llm_provider || "openai");
      const modelParam = isDefaultProv
        ? llmModel.trim() || undefined
        : undefined;
      const baseUrlParam =
        provId === "openai_compatible" ? compatBaseUrl.trim() || null : null;

      const res = await verifyKeyMutation.mutateAsync({
        provider: provId,
        api_key: draftKey,
        model: modelParam,
        base_url: baseUrlParam,
        save_if_valid: true,
      });

      if (res.valid) {
        const availableProviders = wsSettings?.available_providers ?? [];
        const currentDefaultProv =
          llmProvider || wsSettings?.default_llm_provider || "openai";
        const matchingActiveOption = availableProviders.find(
          (p) => p.provider === currentDefaultProv
        );
        const shouldPersistDefault =
          llmProvider === provId ||
          !wsSettings?.default_llm_provider ||
          !matchingActiveOption?.configured;

        if (shouldPersistDefault) {
          const matchingOpt = availableProviders.find(
            (p) => p.provider === provId
          );
          const effectiveModel =
            llmProvider === provId && llmModel.trim()
              ? llmModel.trim()
              : res.tested_model ||
                matchingOpt?.default_model ||
                res.provider_option?.default_model ||
                "";

          await updateSettingsMutation.mutateAsync({
            default_llm_provider: provId,
            default_llm_model: effectiveModel || null,
            ...(provId === "openai_compatible"
              ? { llm_compat_base_url: compatBaseUrl.trim() || null }
              : {}),
          });
          onDefaultApplied(provId, effectiveModel);
          setSelectedProvider(provId);
          setSelectedModel(effectiveModel);
        }

        setVerifyFeedback((prev) => ({
          ...prev,
          [provId]: {
            valid: true,
            message: "✓ Key verified & saved successfully",
          },
        }));
        onApiKeyDraftsChange((prev) => {
          const next = { ...prev };
          delete next[provId];
          return next;
        });
      } else {
        setVerifyFeedback((prev) => ({
          ...prev,
          [provId]: {
            valid: false,
            message: res.validation_error || "Verification failed",
          },
        }));
      }
    } catch (err: unknown) {
      setVerifyFeedback((prev) => ({
        ...prev,
        [provId]: {
          valid: false,
          message:
            err instanceof Error ? err.message : "Verification error",
        },
      }));
    } finally {
      setVerifyingProvider(null);
    }
  };

  const handleClearUiKey = async (provId: string) => {
    try {
      await updateSettingsMutation.mutateAsync({
        api_keys: { [provId]: "" },
      });
      onApiKeyDraftsChange((prev) => {
        const next = { ...prev };
        delete next[provId];
        return next;
      });
      setVerifyFeedback((prev) => ({
        ...prev,
        [provId]: {
          valid: true,
          message: "UI key cleared; reverted to .env fallback.",
        },
      }));
    } catch (err: unknown) {
      setVerifyFeedback((prev) => ({
        ...prev,
        [provId]: {
          valid: false,
          message:
            err instanceof Error ? err.message : "Failed to clear UI key",
        },
      }));
    }
  };

  const activeDefault =
    llmProvider || wsSettings?.default_llm_provider || "openai";

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4 space-y-3">
      <div>
        <label className="text-xs font-bold text-white flex items-center gap-1.5">
          <Key className="h-3.5 w-3.5 text-indigo-400" />
          <span>API Keys & Provider Live Verification</span>
        </label>
        <p className="text-[11.5px] text-slate-400 mt-0.5">
          Configure and live-test credentials for each model provider. Keys
          saved here override{" "}
          <code className="bg-slate-800 px-1 py-0.5 rounded text-slate-300">
            backend/.env
          </code>
          .
        </p>
      </div>

      <div className="grid grid-cols-1 gap-3 pt-1">
        {LLM_PROVIDERS.map((item) => {
          const opt = (wsSettings?.available_providers ?? []).find(
            (p) => p.provider === item.id
          );
          return (
            <ProviderApiKeyCard
              key={item.id}
              providerId={item.id}
              providerLabel={item.label}
              option={opt}
              isDefault={item.id === activeDefault}
              draftKey={apiKeyDrafts[item.id] ?? ""}
              isRevealed={Boolean(revealedKeys[item.id])}
              isVerifying={verifyingProvider === item.id}
              isVerifyBusy={verifyKeyMutation.isPending}
              isClearBusy={updateSettingsMutation.isPending}
              compatBaseUrl={compatBaseUrl}
              feedback={verifyFeedback[item.id]}
              onDraftChange={(value) =>
                onApiKeyDraftsChange((prev) => ({
                  ...prev,
                  [item.id]: value,
                }))
              }
              onToggleReveal={() =>
                setRevealedKeys((prev) => ({
                  ...prev,
                  [item.id]: !prev[item.id],
                }))
              }
              onCompatBaseUrlChange={onCompatBaseUrlChange}
              onVerifyKey={() => void handleVerifyKey(item.id)}
              onClearUiKey={() => void handleClearUiKey(item.id)}
            />
          );
        })}
      </div>
    </div>
  );
};
