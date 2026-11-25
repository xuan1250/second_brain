import React, { useEffect, useRef, useState } from "react";
import { Sparkles, Trash2 } from "lucide-react";
import { useActiveProviderStatus } from "../../hooks/useActiveProviderStatus";
import { useChatStream, useChatThreads } from "../../hooks/useChatStream";
import { useSystemStats } from "../../hooks/useDocuments";
import { useDomainsQuery } from "../../hooks/useDomains";
import { useLlmProviderSync } from "../../hooks/useLlmProviderSync";
import { useWorkspaceSettingsQuery } from "../../hooks/useWorkspaceSettings";
import { useWorkspace } from "../../lib/workspaceStore";
import { ConfirmDialog } from "../common/ConfirmDialog";
import { EmptyState } from "../common/EmptyState";
import { LoadingState } from "../common/LoadingState";
import { ChatComposer } from "./ChatComposer";
import { ChatHeader } from "./ChatHeader";
import { ChatMessageItem } from "./ChatMessageItem";

const SUGGESTED_PROMPTS = [
  "Evaluate the technical failure risks and statutory environmental liabilities of leaking PCB transformers.",
  "Compare physical asset degradation with MACRS balance sheet depreciation rules.",
];

export const ChatAssistant: React.FC = () => {
  const { data: domains = [] } = useDomainsQuery();
  const {
    activeThreadId,
    setActiveThreadId,
    selectedFilterDocIds,
    selectedProvider,
    setSelectedProvider,
    selectedModel,
    setSelectedModel,
    currentDomainFilter,
    migrateDraftDomainFilter,
    openSettingsModal,
  } = useWorkspace();

  const [inputValue, setInputValue] = useState("");
  const [showModelConfig, setShowModelConfig] = useState(false);
  const [confirmingThreadDelete, setConfirmingThreadDelete] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  const { data: threadsData } = useChatThreads();
  const { data: stats } = useSystemStats();
  const { data: wsSettings } = useWorkspaceSettingsQuery();

  const {
    messages,
    isStreaming,
    isLoadingHistory,
    sendMessage,
    stopStreaming,
    clearConversation,
    removeThread,
  } = useChatStream(activeThreadId, (resolvedId) => {
    setActiveThreadId(resolvedId || null);
    if (resolvedId) {
      migrateDraftDomainFilter(resolvedId);
    }
  });

  const threads = threadsData?.threads ?? threadsData?.items ?? [];
  const {
    providers,
    activeProv,
    activeOption,
    isInvalidKey,
    hasKeyIssue,
    defaultBadgeProvider,
    defaultBadgeModel,
  } = useActiveProviderStatus({
    selectedProvider,
    selectedModel,
    wsSettings,
    stats,
  });

  const { commitModelOverride, selectProviderAndPersist } = useLlmProviderSync({
    wsSettings,
    stats,
    providers,
    selectedProvider,
    selectedModel,
    setSelectedProvider,
    setSelectedModel,
  });

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputValue.trim() || isStreaming) return;
    const text = inputValue;
    setInputValue("");
    void sendMessage({
      message: text,
      threadId: activeThreadId,
      filterDocIds: selectedFilterDocIds,
      domainFilter: currentDomainFilter,
      provider: selectedProvider || null,
      model: selectedModel || null,
    });
  };

  const handleNewThread = () => {
    clearConversation();
    setActiveThreadId(null);
  };

  return (
    <div className="flex h-full flex-col bg-slate-950 border-l border-slate-800/80">
      <ChatHeader
        threads={threads}
        activeThreadId={activeThreadId}
        onSelectThread={setActiveThreadId}
        onNewThread={handleNewThread}
        onRequestDeleteThread={() => setConfirmingThreadDelete(true)}
        showModelConfig={showModelConfig}
        onToggleModelConfig={() => setShowModelConfig((v) => !v)}
        providers={providers}
        activeProv={activeProv}
        activeOption={activeOption}
        selectedProvider={selectedProvider}
        selectedModel={selectedModel}
        fallbackModelName={stats?.llm?.model_name}
        defaultBadgeProvider={defaultBadgeProvider}
        defaultBadgeModel={defaultBadgeModel}
        isInvalidKey={isInvalidKey}
        hasKeyIssue={hasKeyIssue}
        onSelectProvider={selectProviderAndPersist}
        onModelChange={setSelectedModel}
        onCommitModel={commitModelOverride}
      />

      <div className="flex-1 overflow-y-auto p-3.5 space-y-4">
        {isLoadingHistory ? (
          <LoadingState label="Loading conversation history..." />
        ) : messages.length === 0 ? (
          <EmptyState
            className="h-full px-4"
            icon={
              <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-indigo-500/10 border border-indigo-500/20 text-indigo-400">
                <Sparkles className="h-6 w-6" />
              </div>
            }
            title="Multi-Domain Corrective RAG Assistant"
            description={
              <>
                Ask questions across Engineering, Economics, and Law documents.
                Polysemic terms are automatically disambiguated with
                domain-colored citations{" "}
                <code className="text-cyan-300">[1]</code>.
              </>
            }
          >
            <div className="mt-4 grid w-full max-w-xs gap-1.5">
              {SUGGESTED_PROMPTS.map((prompt) => (
                <button
                  key={prompt}
                  type="button"
                  onClick={() => setInputValue(prompt)}
                  className="rounded-xl border border-slate-800/90 bg-slate-900/60 px-3 py-2 text-left text-[11px] text-slate-300 hover:border-indigo-500/50 hover:bg-slate-900 transition"
                >
                  "{prompt}"
                </button>
              ))}
            </div>
          </EmptyState>
        ) : (
          messages.map((msg, idx) => (
            <ChatMessageItem
              key={msg.id || idx}
              message={msg}
              domains={domains}
            />
          ))
        )}
        <div ref={messagesEndRef} />
      </div>

      <ChatComposer
        inputValue={inputValue}
        onInputValueChange={setInputValue}
        isStreaming={isStreaming}
        selectedFilterDocIdsCount={selectedFilterDocIds.length}
        hasKeyIssue={hasKeyIssue}
        isInvalidKey={isInvalidKey}
        activeProv={activeProv}
        activeOption={activeOption}
        onOpenApiSettings={() => openSettingsModal("ingestion")}
        onStopStreaming={stopStreaming}
        onSubmit={handleSubmit}
      />

      <ConfirmDialog
        open={confirmingThreadDelete}
        onOpenChange={setConfirmingThreadDelete}
        title="Delete Conversation Thread?"
        description="This will permanently delete the active conversation thread and its message history."
        confirmLabel="Delete Thread"
        confirmIcon={<Trash2 className="h-3.5 w-3.5" />}
        onConfirm={async () => {
          if (!activeThreadId) return;
          await removeThread(activeThreadId);
          setConfirmingThreadDelete(false);
        }}
      />
    </div>
  );
};
