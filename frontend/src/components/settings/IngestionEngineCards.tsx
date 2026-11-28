import React from "react";
import {
  APPROVAL_MODE_OPTIONS,
  type ApprovalMode,
  INGESTION_MODE_OPTIONS,
  type IngestionMode,
  LLM_PROVIDERS,
} from "../../lib/llmCatalog";
import type { WorkspaceSettings } from "../../lib/types";

export interface IngestionEngineCardsProps {
  wsSettings?: WorkspaceSettings;
  uploadMode: IngestionMode;
  onUploadModeChange: (mode: IngestionMode) => void;
  approvalMode: ApprovalMode;
  onApprovalModeChange: (mode: ApprovalMode) => void;
  confidenceThreshold: number;
  onConfidenceThresholdChange: (value: number) => void;
  llmProvider: string;
  onLlmProviderChange: (provider: string) => void;
  llmModel: string;
  onLlmModelChange: (model: string) => void;
}

export const IngestionEngineCards: React.FC<IngestionEngineCardsProps> = ({
  wsSettings,
  uploadMode,
  onUploadModeChange,
  approvalMode,
  onApprovalModeChange,
  confidenceThreshold,
  onConfidenceThresholdChange,
  llmProvider,
  onLlmProviderChange,
  llmModel,
  onLlmModelChange,
}) => {
  const handleProviderSelect = (nextProv: string) => {
    onLlmProviderChange(nextProv);
    if (!nextProv) {
      onLlmModelChange("");
      return;
    }
    const available = wsSettings?.available_providers ?? [];
    const knownDefaults = new Set(
      available.map((p) => p.default_model).filter(Boolean)
    );
    const matched = available.find((p) => p.provider === nextProv);
    if (
      matched?.default_model &&
      (!llmModel.trim() || knownDefaults.has(llmModel.trim()))
    ) {
      onLlmModelChange(matched.default_model);
    }
  };

  return (
    <>
      <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4 space-y-2">
        <label className="block text-xs font-bold text-white">
          Default Parsing & Ingestion Engine
        </label>
        <p className="text-[11.5px] text-slate-400">
          Determines document parsing pipeline when uploaded through drag & drop
          or API.
        </p>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-2 pt-1">
          {INGESTION_MODE_OPTIONS.map((opt) => (
            <button
              key={opt.id}
              type="button"
              onClick={() => onUploadModeChange(opt.id)}
              className={`rounded-lg border p-3 text-left transition ${
                uploadMode === opt.id
                  ? "border-indigo-500 bg-indigo-950/40 text-white"
                  : "border-slate-800 bg-slate-950/60 text-slate-400 hover:border-slate-700 hover:text-slate-200"
              }`}
            >
              <div className="font-semibold text-xs text-white">
                {opt.settingsTitle}
              </div>
              <div className="text-[10.5px] text-slate-400 mt-0.5">
                {opt.settingsDesc}
              </div>
            </button>
          ))}
        </div>
      </div>

      <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4 space-y-2">
        <label className="block text-xs font-bold text-white">
          Default Human-in-the-Loop Approval Gate
        </label>
        <p className="text-[11.5px] text-slate-400">
          Control when new documents require manual approval before vector
          indexing.
        </p>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-2 pt-1">
          {APPROVAL_MODE_OPTIONS.map((opt) => (
            <button
              key={opt.id}
              type="button"
              onClick={() => onApprovalModeChange(opt.id)}
              className={`rounded-lg border p-3 text-left transition ${
                approvalMode === opt.id
                  ? "border-emerald-500 bg-emerald-950/40 text-white"
                  : "border-slate-800 bg-slate-950/60 text-slate-400 hover:border-slate-700 hover:text-slate-200"
              }`}
            >
              <div className="font-semibold text-xs text-white">
                {opt.settingsTitle}
              </div>
              <div className="text-[10.5px] text-slate-400 mt-0.5">
                {opt.settingsDesc}
              </div>
            </button>
          ))}
        </div>
      </div>

      <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4 space-y-2">
        <div className="flex items-center justify-between">
          <div>
            <label className="block text-xs font-bold text-white">
              Auto-Approve Confidence Threshold
            </label>
            <p className="text-[11.5px] text-slate-400">
              Minimum zero-shot classifier confidence score required for
              auto-approval.
            </p>
          </div>
          <span className="font-mono text-sm font-bold text-emerald-400">
            {Math.round(confidenceThreshold * 100)}%
          </span>
        </div>
        <input
          type="range"
          min={0.5}
          max={0.99}
          step={0.01}
          value={confidenceThreshold}
          onChange={(e) => onConfidenceThresholdChange(Number(e.target.value))}
          className="w-full accent-emerald-500 cursor-pointer"
        />
        <div className="flex justify-between text-[10px] text-slate-500 font-mono">
          <span>50% (Permissive)</span>
          <span>85% (Balanced)</span>
          <span>99% (Strict)</span>
        </div>
      </div>

      <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4 space-y-3">
        <label className="block text-xs font-bold text-white">
          Default Workspace LLM Provider & Model
        </label>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <div>
            <label className="block text-[11px] font-semibold text-slate-300 mb-1">
              LLM Provider
            </label>
            <select
              value={llmProvider}
              onChange={(e) => handleProviderSelect(e.target.value)}
              className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-1.5 text-slate-200"
            >
              <option value="">Default (From Backend Config)</option>
              {LLM_PROVIDERS.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.label}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-[11px] font-semibold text-slate-300 mb-1">
              Model Name
            </label>
            <input
              type="text"
              value={llmModel}
              onChange={(e) => onLlmModelChange(e.target.value)}
              placeholder="e.g. gpt-4o, claude-3-5-sonnet-20241022"
              className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-1.5 text-slate-200"
            />
          </div>
        </div>
      </div>
    </>
  );
};
