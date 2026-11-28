import React from "react";
import { LLM_PROVIDERS, MODEL_SUGGESTIONS } from "../../lib/llmCatalog";
import type { ProviderOption } from "../../lib/types";

export interface ModelOverridePanelProps {
  providers: ProviderOption[];
  selectedProvider: string;
  selectedModel: string;
  defaultBadgeProvider: string;
  defaultBadgeModel: string;
  fallbackModelPlaceholder?: string;
  onSelectProvider: (provider: string) => void;
  onModelChange: (model: string) => void;
  onCommitModel: (model: string) => void;
}

export const ModelOverridePanel: React.FC<ModelOverridePanelProps> = ({
  providers,
  selectedProvider,
  selectedModel,
  defaultBadgeProvider,
  defaultBadgeModel,
  fallbackModelPlaceholder,
  onSelectProvider,
  onModelChange,
  onCommitModel,
}) => {
  const suggestions = MODEL_SUGGESTIONS[selectedProvider] ?? [];

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/80 p-2.5 space-y-2 text-xs">
      <div className="flex items-center justify-between text-[11px] text-slate-400">
        <span>Per-Request LLM Override</span>
        <span className="text-[10px] text-slate-500">
          Default: {defaultBadgeProvider} / {defaultBadgeModel}
        </span>
      </div>
      <div className="grid grid-cols-2 gap-2">
        <div>
          <label className="block text-[10px] uppercase text-slate-400 mb-1">
            Provider
          </label>
          <select
            aria-label="LLM Provider"
            value={selectedProvider}
            onChange={(e) => onSelectProvider(e.target.value)}
            className="w-full rounded-md border border-slate-700 bg-slate-950 px-2 py-1 text-xs text-slate-200 focus:border-indigo-500 focus:outline-none"
          >
            {providers.map((p) => (
              <option key={p.provider} value={p.provider}>
                {p.label} {p.configured ? "✓" : ""}
              </option>
            ))}
            {providers.length === 0 &&
              LLM_PROVIDERS.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.shortLabel}
                </option>
              ))}
          </select>
        </div>

        <div>
          <label className="block text-[10px] uppercase text-slate-400 mb-1">
            Model Name
          </label>
          <input
            type="text"
            list="model-suggestions"
            aria-label="LLM Model Name"
            value={selectedModel}
            onChange={(e) => onModelChange(e.target.value)}
            onBlur={(e) => onCommitModel(e.currentTarget.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                onCommitModel(e.currentTarget.value);
                e.currentTarget.blur();
              }
            }}
            placeholder={fallbackModelPlaceholder || "e.g. gpt-4o-mini"}
            className="w-full rounded-md border border-slate-700 bg-slate-950 px-2 py-1 text-xs text-slate-200 placeholder:text-slate-600 focus:border-indigo-500 focus:outline-none"
          />
          <datalist id="model-suggestions">
            {suggestions.map((modelName) => (
              <option key={modelName} value={modelName} />
            ))}
          </datalist>
        </div>
      </div>
    </div>
  );
};
