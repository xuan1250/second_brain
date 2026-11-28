import React, { useEffect, useRef, useState } from "react";
import { Check, Loader2 } from "lucide-react";
import { useActiveProviderStatus } from "../../hooks/useActiveProviderStatus";
import { useUpdateWorkspaceSettingsMutation } from "../../hooks/useWorkspaceSettings";
import type { ApprovalMode, IngestionMode } from "../../lib/llmCatalog";
import type { WorkspaceSettings } from "../../lib/types";
import { useWorkspace } from "../../lib/workspaceStore";
import { AlertBanner } from "../common/AlertBanner";
import { IngestionEngineCards } from "./IngestionEngineCards";
import { ProviderApiKeysSection } from "./ProviderApiKeysSection";

export interface IngestionDefaultsTabProps {
  open: boolean;
  wsSettings?: WorkspaceSettings;
}

export const IngestionDefaultsTab: React.FC<IngestionDefaultsTabProps> = ({
  open,
  wsSettings,
}) => {
  const { setSelectedProvider, setSelectedModel } = useWorkspace();
  const updateSettingsMutation = useUpdateWorkspaceSettingsMutation();

  const [uploadMode, setUploadMode] = useState<IngestionMode>("fast");
  const [approvalMode, setApprovalMode] = useState<ApprovalMode>("review_all");
  const [confidenceThreshold, setConfidenceThreshold] = useState<number>(0.85);
  const [llmProvider, setLlmProvider] = useState<string>("");
  const [llmModel, setLlmModel] = useState<string>("");
  const [compatBaseUrl, setCompatBaseUrl] = useState<string>("");
  const [apiKeyDrafts, setApiKeyDrafts] = useState<Record<string, string>>({});
  const [settingsSuccessMsg, setSettingsSuccessMsg] = useState<string | null>(
    null
  );
  const [settingsSaveError, setSettingsSaveError] = useState<string | null>(
    null
  );

  const prevOpenRef = useRef(false);
  const hasHydratedRef = useRef(false);
  const successTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    return () => {
      if (successTimerRef.current) {
        clearTimeout(successTimerRef.current);
      }
    };
  }, []);

  useEffect(() => {
    const justOpened = open && !prevOpenRef.current;
    prevOpenRef.current = open;

    if (!open) {
      hasHydratedRef.current = false;
      return;
    }

    if (wsSettings && (justOpened || !hasHydratedRef.current)) {
      setUploadMode((wsSettings.default_upload_mode as IngestionMode) || "fast");
      setApprovalMode(
        (wsSettings.default_approval_mode as ApprovalMode) || "review_all"
      );
      setConfidenceThreshold(wsSettings.high_confidence_threshold ?? 0.85);
      setLlmProvider(wsSettings.default_llm_provider || "");
      setLlmModel(wsSettings.default_llm_model || "");
      setCompatBaseUrl(wsSettings.llm_compat_base_url || "");
      hasHydratedRef.current = true;
    }
  }, [open, wsSettings]);

  const {
    activeProv: activeDefaultProv,
    activeOption: matchingOption,
    isMissingKey: isMissingActiveKey,
    isInvalidKey: isInvalidActiveKey,
  } = useActiveProviderStatus({
    selectedProvider: llmProvider,
    wsSettings,
  });

  const handleSaveSettings = async (e: React.FormEvent) => {
    e.preventDefault();
    setSettingsSuccessMsg(null);
    setSettingsSaveError(null);
    try {
      const apiKeysPayload: Record<string, string | null> = {};
      for (const [p, k] of Object.entries(apiKeyDrafts)) {
        if (k && k.trim()) {
          apiKeysPayload[p] = k.trim();
        }
      }

      const updated = await updateSettingsMutation.mutateAsync({
        default_upload_mode: uploadMode,
        default_approval_mode: approvalMode,
        high_confidence_threshold: confidenceThreshold,
        default_llm_provider: llmProvider || null,
        default_llm_model: llmModel || null,
        llm_compat_base_url: compatBaseUrl.trim() || null,
        ...(Object.keys(apiKeysPayload).length > 0
          ? { api_keys: apiKeysPayload }
          : {}),
      });
      setSelectedProvider(updated.default_llm_provider || "");
      setSelectedModel(updated.default_llm_model || "");
      setApiKeyDrafts({});
      setSettingsSuccessMsg("Workspace defaults saved successfully.");
      if (successTimerRef.current) {
        clearTimeout(successTimerRef.current);
      }
      successTimerRef.current = setTimeout(
        () => setSettingsSuccessMsg(null),
        3000
      );
    } catch (err: unknown) {
      setSettingsSaveError(
        err instanceof Error
          ? err.message
          : "Failed to update workspace settings."
      );
    }
  };

  return (
    <>
      {settingsSuccessMsg && (
        <AlertBanner variant="success" message={settingsSuccessMsg} />
      )}
      {settingsSaveError && (
        <AlertBanner
          variant="error"
          message={settingsSaveError}
          onDismiss={() => setSettingsSaveError(null)}
        />
      )}

      {isMissingActiveKey && (
        <AlertBanner
          variant="warning"
          title={`Missing API Key for Default Provider (${
            matchingOption?.label || activeDefaultProv
          })`}
          message={
            <>
              No API key was detected in Workspace Settings or{" "}
              <code className="bg-amber-950/80 px-1 py-0.5 rounded text-amber-200">
                backend/.env
              </code>
              . Chat inference will fail until a valid key is provided and
              verified below.
            </>
          }
        />
      )}

      {isInvalidActiveKey && (
        <AlertBanner
          variant="error"
          title={`Invalid API Key for Default Provider (${
            matchingOption?.label || activeDefaultProv
          })`}
          message={
            matchingOption?.validation_error ||
            "Upstream rejected the configured key. Please check your credentials or enter a replacement key below."
          }
        />
      )}

      <form onSubmit={handleSaveSettings} className="space-y-4 text-xs">
        <IngestionEngineCards
          wsSettings={wsSettings}
          uploadMode={uploadMode}
          onUploadModeChange={setUploadMode}
          approvalMode={approvalMode}
          onApprovalModeChange={setApprovalMode}
          confidenceThreshold={confidenceThreshold}
          onConfidenceThresholdChange={setConfidenceThreshold}
          llmProvider={llmProvider}
          onLlmProviderChange={setLlmProvider}
          llmModel={llmModel}
          onLlmModelChange={setLlmModel}
        />

        <ProviderApiKeysSection
          wsSettings={wsSettings}
          llmProvider={llmProvider}
          llmModel={llmModel}
          compatBaseUrl={compatBaseUrl}
          apiKeyDrafts={apiKeyDrafts}
          onCompatBaseUrlChange={setCompatBaseUrl}
          onApiKeyDraftsChange={setApiKeyDrafts}
          onDefaultApplied={(prov, mod) => {
            setLlmProvider(prov);
            setLlmModel(mod);
          }}
        />

        <div className="flex items-center justify-end gap-2 pt-2">
          <button
            type="submit"
            disabled={updateSettingsMutation.isPending}
            className="inline-flex items-center gap-1.5 rounded-lg bg-indigo-600 px-4 py-2 text-xs font-semibold text-white hover:bg-indigo-500 disabled:opacity-50 shadow-xs"
          >
            {updateSettingsMutation.isPending ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Check className="h-4 w-4" />
            )}
            <span>Save Workspace Settings</span>
          </button>
        </div>
      </form>
    </>
  );
};
