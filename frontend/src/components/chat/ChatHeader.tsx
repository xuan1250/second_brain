import React from "react";
import { Bot, Cpu, MessageSquarePlus, Trash2 } from "lucide-react";
import type { ProviderOption, ThreadSummary } from "../../lib/types";
import { StatusDot } from "../common/StatusDot";
import { DomainRoutingBar } from "./DomainRoutingBar";
import { ModelOverridePanel } from "./ModelOverridePanel";

export interface ChatHeaderProps {
  threads: ThreadSummary[];
  activeThreadId: string | null;
  onSelectThread: (threadId: string | null) => void;
  onNewThread: () => void;
  onRequestDeleteThread: () => void;
  showModelConfig: boolean;
  onToggleModelConfig: () => void;
  providers: ProviderOption[];
  activeProv: string;
  activeOption?: ProviderOption;
  selectedProvider: string;
  selectedModel: string;
  fallbackModelName?: string;
  defaultBadgeProvider: string;
  defaultBadgeModel: string;
  isInvalidKey: boolean;
  hasKeyIssue: boolean;
  onSelectProvider: (provider: string) => void;
  onModelChange: (model: string) => void;
  onCommitModel: (model: string) => void;
}

export const ChatHeader: React.FC<ChatHeaderProps> = ({
  threads,
  activeThreadId,
  onSelectThread,
  onNewThread,
  onRequestDeleteThread,
  showModelConfig,
  onToggleModelConfig,
  providers,
  activeProv,
  activeOption,
  selectedProvider,
  selectedModel,
  fallbackModelName,
  defaultBadgeProvider,
  defaultBadgeModel,
  isInvalidKey,
  hasKeyIssue,
  onSelectProvider,
  onModelChange,
  onCommitModel,
}) => {
  return (
    <div className="border-b border-slate-800/80 px-3.5 py-2 space-y-2">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <div className="flex h-6 w-6 items-center justify-center rounded-md bg-indigo-500/20 text-indigo-400 border border-indigo-500/30">
            <Bot className="h-3.5 w-3.5" />
          </div>
          <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-200">
            Intelligent Assistant
          </h2>
        </div>

        <div className="flex items-center gap-1.5">
          <button
            type="button"
            onClick={onToggleModelConfig}
            title={
              hasKeyIssue
                ? isInvalidKey
                  ? `Invalid API key for ${
                      activeOption?.label || activeProv
                    }. Click to reconfigure.`
                  : `Missing API key for ${
                      activeOption?.label || activeProv
                    }. Click to configure.`
                : "Configure LLM Provider & Model"
            }
            className={`inline-flex items-center gap-1.5 rounded-md border px-2 py-1 text-[11px] font-medium transition ${
              showModelConfig
                ? "border-indigo-500 bg-indigo-500/20 text-indigo-200"
                : hasKeyIssue
                ? isInvalidKey
                  ? "border-rose-500/60 bg-rose-950/30 text-rose-300 hover:border-rose-400"
                  : "border-amber-500/60 bg-amber-950/30 text-amber-300 hover:border-amber-400"
                : "border-slate-800 bg-slate-900 text-slate-300 hover:text-white"
            }`}
          >
            <Cpu className="h-3.5 w-3.5 text-indigo-400" />
            <span className="text-slate-400">{activeProv}:</span>
            <span className="text-indigo-300 font-mono font-medium max-w-[120px] truncate">
              {selectedModel || fallbackModelName || "gpt-4o"}
            </span>
            {hasKeyIssue && (
              <StatusDot
                tone={isInvalidKey ? "error" : "warning"}
                pulse={isInvalidKey}
              />
            )}
          </button>

          <button
            type="button"
            onClick={onNewThread}
            title="Start new conversation thread"
            className="inline-flex items-center gap-1 rounded-md bg-indigo-600 px-2 py-1 text-[11px] font-medium text-white hover:bg-indigo-500 transition"
          >
            <MessageSquarePlus className="h-3 w-3" />
            <span>New</span>
          </button>
        </div>
      </div>

      <div className="flex items-center gap-1.5">
        <select
          aria-label="Conversation Thread"
          value={activeThreadId || ""}
          onChange={(e) => {
            const val = e.target.value;
            onSelectThread(val ? val : null);
          }}
          className="flex-1 rounded-lg border border-slate-800 bg-slate-900 px-2.5 py-1 text-xs text-slate-200 focus:border-indigo-500 focus:outline-none"
        >
          <option value="">New Conversation Thread</option>
          {threads.map((t) => (
            <option key={t.thread_id} value={t.thread_id}>
              {t.title || `Thread ${t.thread_id.slice(0, 8)}`}
            </option>
          ))}
        </select>

        {activeThreadId && (
          <button
            type="button"
            onClick={onRequestDeleteThread}
            title="Delete active conversation thread"
            className="rounded-lg border border-slate-800 bg-slate-900 p-1.5 text-slate-400 hover:border-rose-500/40 hover:bg-rose-500/15 hover:text-rose-300 transition"
          >
            <Trash2 className="h-3.5 w-3.5" />
          </button>
        )}
      </div>

      <DomainRoutingBar />

      {showModelConfig && (
        <ModelOverridePanel
          providers={providers}
          selectedProvider={selectedProvider}
          selectedModel={selectedModel}
          defaultBadgeProvider={defaultBadgeProvider}
          defaultBadgeModel={defaultBadgeModel}
          fallbackModelPlaceholder={fallbackModelName}
          onSelectProvider={onSelectProvider}
          onModelChange={onModelChange}
          onCommitModel={onCommitModel}
        />
      )}
    </div>
  );
};
