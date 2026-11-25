import React from "react";
import { Filter, Send, Square } from "lucide-react";
import type { ProviderOption } from "../../lib/types";
import { AlertBanner } from "../common/AlertBanner";

export interface ChatComposerProps {
  inputValue: string;
  onInputValueChange: (value: string) => void;
  isStreaming: boolean;
  selectedFilterDocIdsCount: number;
  hasKeyIssue: boolean;
  isInvalidKey: boolean;
  activeProv: string;
  activeOption?: ProviderOption;
  onOpenApiSettings: () => void;
  onStopStreaming: () => void;
  onSubmit: (e: React.FormEvent) => void;
}

export const ChatComposer: React.FC<ChatComposerProps> = ({
  inputValue,
  onInputValueChange,
  isStreaming,
  selectedFilterDocIdsCount,
  hasKeyIssue,
  isInvalidKey,
  activeProv,
  activeOption,
  onOpenApiSettings,
  onStopStreaming,
  onSubmit,
}) => {
  return (
    <form
      onSubmit={onSubmit}
      className="border-t border-slate-800/80 bg-slate-950 p-3 space-y-2"
    >
      {selectedFilterDocIdsCount > 0 && (
        <div className="flex items-center gap-1.5 text-[11px] text-indigo-300">
          <Filter className="h-3 w-3 text-indigo-400" />
          <span>
            Query scoped to <strong>{selectedFilterDocIdsCount}</strong> selected
            document{selectedFilterDocIdsCount > 1 ? "s" : ""}
          </span>
        </div>
      )}

      {hasKeyIssue && (
        <AlertBanner
          variant={isInvalidKey ? "error" : "warning"}
          className="p-2.5"
          title={
            isInvalidKey
              ? activeOption?.validation_error ||
                `API key verification failed for ${
                  activeOption?.label || activeProv
                }.`
              : `No API key configured for ${activeOption?.label || activeProv}.`
          }
          message={
            <span className="text-[10.5px] opacity-80">
              {isInvalidKey
                ? "Upstream rejected this key. Update it in Workspace Settings."
                : "No valid key found in Workspace Settings or backend/.env."}
            </span>
          }
          action={
            <button
              type="button"
              onClick={onOpenApiSettings}
              className={`shrink-0 inline-flex items-center gap-1 rounded-lg px-2.5 py-1 text-[11px] font-semibold text-white transition cursor-pointer shadow-xs ${
                isInvalidKey
                  ? "bg-rose-600 hover:bg-rose-500"
                  : "bg-amber-600 hover:bg-amber-500"
              }`}
            >
              <span>Configure API Key</span>
              <span>&rarr;</span>
            </button>
          }
        />
      )}

      <div className="flex items-end gap-2">
        <textarea
          rows={2}
          value={inputValue}
          onChange={(e) => onInputValueChange(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              onSubmit(e);
            }
          }}
          placeholder="Ask your Second Brain... (Enter to send, Shift+Enter for newline)"
          disabled={isStreaming}
          className="flex-1 resize-none rounded-xl border border-slate-800 bg-slate-900/90 px-3 py-2 text-xs text-slate-100 placeholder:text-slate-500 focus:border-indigo-500 focus:outline-none disabled:opacity-60"
        />

        {isStreaming ? (
          <button
            type="button"
            onClick={onStopStreaming}
            title="Stop generation"
            className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-rose-600 text-white hover:bg-rose-500 transition"
          >
            <Square className="h-3.5 w-3.5 fill-current" />
          </button>
        ) : (
          <button
            type="submit"
            disabled={!inputValue.trim()}
            title="Send message"
            className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-indigo-600 text-white hover:bg-indigo-500 disabled:opacity-40 disabled:cursor-not-allowed transition"
          >
            <Send className="h-3.5 w-3.5" />
          </button>
        )}
      </div>
    </form>
  );
};
