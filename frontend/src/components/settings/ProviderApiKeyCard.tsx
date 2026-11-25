import React from "react";
import {
  AlertCircle,
  Check,
  Eye,
  EyeOff,
  Loader2,
  ShieldCheck,
} from "lucide-react";
import type { ProviderOption } from "../../lib/types";

export interface ProviderApiKeyCardProps {
  providerId: string;
  providerLabel: string;
  option?: ProviderOption;
  isDefault: boolean;
  draftKey: string;
  isRevealed: boolean;
  isVerifying: boolean;
  isVerifyBusy: boolean;
  isClearBusy: boolean;
  compatBaseUrl?: string;
  feedback?: { valid: boolean; message: string };
  onDraftChange: (value: string) => void;
  onToggleReveal: () => void;
  onCompatBaseUrlChange?: (value: string) => void;
  onVerifyKey: () => void;
  onClearUiKey: () => void;
}

export const ProviderApiKeyCard: React.FC<ProviderApiKeyCardProps> = ({
  providerId,
  providerLabel,
  option,
  isDefault,
  draftKey,
  isRevealed,
  isVerifying,
  isVerifyBusy,
  isClearBusy,
  compatBaseUrl = "",
  feedback,
  onDraftChange,
  onToggleReveal,
  onCompatBaseUrlChange,
  onVerifyKey,
  onClearUiKey,
}) => {
  const keySource = option?.key_source || "none";
  const maskedKey = option?.masked_key;
  const valStatus = option?.validation_status || "untested";
  const valError = option?.validation_error;

  return (
    <div className="rounded-lg border border-slate-800/80 bg-slate-950/70 p-3.5 space-y-2.5">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className="font-semibold text-xs text-white">
            {option?.label || providerLabel}
          </span>
          {isDefault && (
            <span className="rounded-full bg-indigo-950 border border-indigo-500/30 px-2 py-0.5 text-[10px] font-medium text-indigo-300">
              Default
            </span>
          )}
        </div>

        <div className="flex items-center gap-1.5">
          {keySource === "ui" ? (
            <span className="inline-flex items-center gap-1 rounded-md bg-emerald-950/80 border border-emerald-500/40 px-2 py-0.5 text-[10px] font-mono text-emerald-300">
              UI Key ({maskedKey || "saved"})
            </span>
          ) : keySource === "env" ? (
            <span className="inline-flex items-center gap-1 rounded-md bg-sky-950/80 border border-sky-500/40 px-2 py-0.5 text-[10px] font-mono text-sky-300">
              .env ({maskedKey || "configured"})
            </span>
          ) : (
            <span className="inline-flex items-center gap-1 rounded-md bg-amber-950/80 border border-amber-500/40 px-2 py-0.5 text-[10px] font-semibold text-amber-300">
              Missing Key
            </span>
          )}

          {valStatus === "valid" ? (
            <span className="inline-flex items-center gap-1 rounded-md bg-emerald-900/40 border border-emerald-600/40 px-2 py-0.5 text-[10px] font-medium text-emerald-300">
              ✓ Verified
            </span>
          ) : valStatus === "invalid" ? (
            <span className="inline-flex items-center gap-1 rounded-md bg-rose-900/40 border border-rose-600/40 px-2 py-0.5 text-[10px] font-medium text-rose-300">
              ✗ Invalid Key
            </span>
          ) : valStatus === "missing" ? (
            <span className="inline-flex items-center gap-1 rounded-md bg-amber-900/40 border border-amber-600/40 px-2 py-0.5 text-[10px] font-medium text-amber-300">
              Missing
            </span>
          ) : (
            <span className="inline-flex items-center gap-1 rounded-md bg-slate-800 border border-slate-700 px-2 py-0.5 text-[10px] font-medium text-slate-400">
              Untested
            </span>
          )}
        </div>
      </div>

      <div className="space-y-2">
        <div className="relative flex items-center">
          <input
            type={isRevealed ? "text" : "password"}
            autoComplete="off"
            spellCheck={false}
            value={draftKey}
            onChange={(e) => onDraftChange(e.target.value)}
            placeholder={
              maskedKey
                ? `Leave blank to keep ${maskedKey}`
                : `Enter ${providerLabel} API key...`
            }
            className="w-full rounded-lg border border-slate-700 bg-slate-900 px-3 py-1.5 pr-9 font-mono text-xs text-slate-200 placeholder:text-slate-500 placeholder:font-sans focus:border-indigo-500 focus:outline-none"
          />
          <button
            type="button"
            onClick={onToggleReveal}
            title={isRevealed ? "Hide API key" : "Show API key"}
            className="absolute right-2 text-slate-400 hover:text-slate-200"
          >
            {isRevealed ? (
              <EyeOff className="h-4 w-4" />
            ) : (
              <Eye className="h-4 w-4" />
            )}
          </button>
        </div>

        {providerId === "openai_compatible" && onCompatBaseUrlChange && (
          <div>
            <label className="block text-[11px] font-semibold text-slate-400 mb-1">
              Base URL (vLLM / Ollama / Local / Custom)
            </label>
            <input
              type="text"
              autoComplete="off"
              spellCheck={false}
              value={compatBaseUrl}
              onChange={(e) => onCompatBaseUrlChange(e.target.value)}
              placeholder="http://localhost:11434/v1"
              className="w-full rounded-lg border border-slate-700 bg-slate-900 px-3 py-1.5 font-mono text-xs text-slate-200 placeholder:text-slate-500 placeholder:font-sans focus:border-indigo-500 focus:outline-none"
            />
          </div>
        )}
      </div>

      <div className="flex flex-wrap items-center justify-between gap-2 pt-1">
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={onVerifyKey}
            disabled={isVerifying || isVerifyBusy}
            className="inline-flex items-center gap-1.5 rounded-lg border border-indigo-500/50 bg-indigo-600/20 px-3 py-1 text-xs font-semibold text-indigo-300 hover:bg-indigo-600/30 disabled:opacity-50 transition"
          >
            {isVerifying ? (
              <>
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
                <span>Verifying...</span>
              </>
            ) : (
              <>
                <ShieldCheck className="h-3.5 w-3.5 text-indigo-400" />
                <span>Verify & Save</span>
              </>
            )}
          </button>

          {keySource === "ui" && (
            <button
              type="button"
              onClick={onClearUiKey}
              disabled={isClearBusy}
              title="Remove UI override and revert to backend/.env"
              className="inline-flex items-center gap-1 rounded-lg border border-slate-700 bg-slate-900 px-2.5 py-1 text-xs text-slate-400 hover:border-rose-500/50 hover:text-rose-300 transition"
            >
              <span>Clear UI Key</span>
            </button>
          )}
        </div>

        {feedback && (
          <div
            className={`rounded-md px-2.5 py-1 text-[11px] flex items-center gap-1.5 ${
              feedback.valid
                ? "bg-emerald-950/60 text-emerald-300 border border-emerald-500/40"
                : "bg-rose-950/60 text-rose-300 border border-rose-500/40"
            }`}
          >
            {feedback.valid ? (
              <Check className="h-3.5 w-3.5 shrink-0 text-emerald-400" />
            ) : (
              <AlertCircle className="h-3.5 w-3.5 shrink-0 text-rose-400" />
            )}
            <span>{feedback.message}</span>
          </div>
        )}

        {!feedback && valStatus === "invalid" && valError && (
          <div className="rounded-md px-2.5 py-1 text-[11px] bg-rose-950/60 text-rose-300 border border-rose-500/40 flex items-center gap-1.5">
            <AlertCircle className="h-3.5 w-3.5 shrink-0 text-rose-400" />
            <span>{valError}</span>
          </div>
        )}
      </div>
    </div>
  );
};
