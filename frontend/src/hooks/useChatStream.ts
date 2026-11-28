import { useCallback, useEffect, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  API_BASE,
  deleteChatThread,
  fetchChatThread,
  fetchChatThreads,
} from "../lib/api";
import { queryKeys } from "../lib/queryKeys";
import { readSSEStream } from "../lib/sse";
import type {
  ChatMessage,
  ChatRequest,
  Citation,
  CRAGStep,
  DomainFilterConfig,
} from "../lib/types";

export interface SendChatOptions {
  message: string;
  threadId?: string | null;
  filterDocIds?: string[];
  domainFilter?: DomainFilterConfig | null;
  provider?: string | null;
  model?: string | null;
}

export function useChatThreads() {
  return useQuery({
    queryKey: queryKeys.chatThreads(),
    queryFn: () => fetchChatThreads(),
  });
}

export function useChatStream(
  activeThreadId: string | null,
  onThreadIdResolved?: (threadId: string) => void
) {
  const queryClient = useQueryClient();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);

  const abortControllerRef = useRef<AbortController | null>(null);
  const skipNextThreadFetchRef = useRef<string | null>(null);

  useEffect(() => {
    if (!activeThreadId) {
      setMessages([]);
      return;
    }
    if (skipNextThreadFetchRef.current === activeThreadId) {
      skipNextThreadFetchRef.current = null;
      return;
    }

    let cancelled = false;
    setIsLoadingHistory(true);
    fetchChatThread(activeThreadId)
      .then((detail) => {
        if (cancelled) return;
        setMessages(
          (detail.messages || []).map((m, idx) => ({
            id: `${activeThreadId}-${idx}`,
            role: m.role,
            content: m.content,
            citations: m.citations || [],
            domain_filter: m.domain_filter ?? null,
            steps: m.steps ?? [],
            provider: m.provider ?? null,
            model: m.model ?? null,
          }))
        );
      })
      .catch(() => {})
      .finally(() => {
        if (!cancelled) setIsLoadingHistory(false);
      });

    return () => {
      cancelled = true;
    };
  }, [activeThreadId]);

  const stopStreaming = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setIsStreaming(false);
  }, []);

  const clearConversation = useCallback(() => {
    stopStreaming();
    setMessages([]);
  }, [stopStreaming]);

  const removeThread = useCallback(
    async (threadId: string) => {
      await deleteChatThread(threadId);
      await queryClient.invalidateQueries({ queryKey: queryKeys.chatThreads() });
      await queryClient.invalidateQueries({ queryKey: queryKeys.systemStats() });
      if (activeThreadId === threadId) {
        clearConversation();
        onThreadIdResolved?.("");
      }
    },
    [activeThreadId, clearConversation, onThreadIdResolved, queryClient]
  );

  const sendMessage = useCallback(
    async ({
      message,
      threadId,
      filterDocIds,
      domainFilter,
      provider,
      model,
    }: SendChatOptions) => {
      const trimmed = message.trim();
      if (!trimmed || isStreaming) return;

      stopStreaming();
      const userTurnId = `user-${Date.now()}`;
      const assistantTurnId = `assistant-${Date.now()}`;

      setMessages((prev) => [
        ...prev,
        {
          id: userTurnId,
          role: "user",
          content: trimmed,
          citations: [],
          domain_filter: domainFilter ?? null,
        },
        {
          id: assistantTurnId,
          role: "assistant",
          content: "",
          citations: [],
          steps: [],
          provider: provider || null,
          model: model || null,
          isStreaming: true,
          isError: false,
        },
      ]);
      setIsStreaming(true);

      const controller = new AbortController();
      abortControllerRef.current = controller;
      const body: ChatRequest = {
        message: trimmed,
        thread_id: threadId || activeThreadId || null,
        filter_doc_ids: filterDocIds?.length ? filterDocIds : null,
        domain_filter: domainFilter ?? null,
        provider: provider || null,
        model: model || null,
      };
      let accumulatedContent = "";
      let latestCitations: Citation[] = [];
      let collectedSteps: CRAGStep[] = [];
      const updateAssistant = (updater: (m: ChatMessage) => ChatMessage) =>
        setMessages((prev) => prev.map((m) => (m.id === assistantTurnId ? updater(m) : m)));
      const syncResolvedThread = (resolvedId?: string) => {
        if (resolvedId && resolvedId !== activeThreadId) {
          skipNextThreadFetchRef.current = resolvedId;
          onThreadIdResolved?.(resolvedId);
        }
      };

      try {
        await readSSEStream(
          `${API_BASE}/chat/completions`,
          body,
          controller.signal,
          {
            onStep: (nextStep) => {
              collectedSteps = [
                ...collectedSteps.filter((s) => s.node !== nextStep.node),
                nextStep,
              ];
              updateAssistant((m) => ({ ...m, steps: [...collectedSteps] }));
            },
            onCitations: (citations) => {
              latestCitations = citations;
              updateAssistant((m) => ({ ...m, citations: latestCitations }));
            },
            onToken: (piece) => {
              accumulatedContent += piece;
              updateAssistant((m) => ({ ...m, content: accumulatedContent }));
            },
            onDone: (doneData) => {
              if (doneData.message && !accumulatedContent) {
                accumulatedContent = doneData.message;
              }
              if (Array.isArray(doneData.citations)) {
                latestCitations = doneData.citations;
              }
              updateAssistant((m) => ({
                ...m,
                content: doneData.message || accumulatedContent,
                citations: latestCitations,
                steps: [...collectedSteps],
                provider: doneData.provider || m.provider || null,
                model: doneData.model || m.model || null,
                isStreaming: false,
              }));
              syncResolvedThread(doneData.thread_id);
            },
            onErrorEvent: (errData) => {
              const errMsg =
                errData.error ||
                errData.message ||
                "An unexpected error occurred during RAG execution.";
              updateAssistant((m) => ({
                ...m,
                content: errMsg,
                isStreaming: false,
                isError: true,
              }));
              syncResolvedThread(errData.thread_id);
            },
          }
        );
      } catch (err: unknown) {
        if (err instanceof DOMException && err.name === "AbortError") {
          updateAssistant((m) => ({
            ...m,
            content: accumulatedContent || "(Generation cancelled by user.)",
            isStreaming: false,
          }));
        } else {
          const errMsg =
            err instanceof Error ? err.message : "Failed to stream response.";
          updateAssistant((m) => ({
            ...m,
            content: errMsg,
            isStreaming: false,
            isError: true,
          }));
        }
      } finally {
        abortControllerRef.current = null;
        setIsStreaming(false);
        updateAssistant((m) => ({ ...m, isStreaming: false }));
        void queryClient.invalidateQueries({ queryKey: queryKeys.chatThreads() });
        void queryClient.invalidateQueries({ queryKey: queryKeys.systemStats() });
        void queryClient.invalidateQueries({
          queryKey: queryKeys.workspaceSettings(),
        });
      }
    },
    [activeThreadId, isStreaming, onThreadIdResolved, queryClient, stopStreaming]
  );

  return {
    messages,
    isStreaming,
    isLoadingHistory,
    sendMessage,
    stopStreaming,
    clearConversation,
    removeThread,
  };
}
