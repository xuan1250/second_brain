import { useCallback, useEffect, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  API_BASE,
  deleteChatThread,
  fetchChatThread,
  fetchChatThreads,
} from "../lib/api";
import type {
  ChatMessage,
  ChatRequest,
  Citation,
  CRAGStep,
  DomainFilterConfig,
  PolysemyCollision,
} from "../lib/types";

interface ParsedSSEFrame {
  event: string;
  data: unknown;
}

function parseSSEBlock(rawBlock: string): ParsedSSEFrame | null {
  const lines = rawBlock.split(/\r?\n/);
  let eventName = "message";
  const dataLines: string[] = [];

  for (const line of lines) {
    if (line.startsWith("event:")) {
      eventName = line.slice("event:".length).trim();
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice("data:".length).trim());
    }
  }

  if (dataLines.length === 0) return null;
  const rawData = dataLines.join("\n");
  try {
    return {
      event: eventName,
      data: JSON.parse(rawData),
    };
  } catch {
    return {
      event: eventName,
      data: { message: rawData },
    };
  }
}

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
    queryKey: ["chat-threads"],
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
  const [activeSteps, setActiveSteps] = useState<CRAGStep[]>([]);
  const [streamError, setStreamError] = useState<string | null>(null);

  const abortControllerRef = useRef<AbortController | null>(null);
  const skipNextThreadFetchRef = useRef<string | null>(null);

  // Load thread history when activeThreadId changes
  useEffect(() => {
    if (!activeThreadId) {
      setMessages([]);
      setActiveSteps([]);
      setStreamError(null);
      return;
    }

    if (skipNextThreadFetchRef.current === activeThreadId) {
      skipNextThreadFetchRef.current = null;
      return;
    }

    let cancelled = false;
    setIsLoadingHistory(true);
    setStreamError(null);

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
      .catch((err: unknown) => {
        if (cancelled) return;
        const msg =
          err instanceof Error ? err.message : "Failed to load conversation";
        setStreamError(msg);
      })
      .finally(() => {
        if (!cancelled) {
          setIsLoadingHistory(false);
        }
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
    setActiveSteps([]);
    setStreamError(null);
  }, [stopStreaming]);

  const removeThread = useCallback(
    async (threadId: string) => {
      await deleteChatThread(threadId);
      await queryClient.invalidateQueries({ queryKey: ["chat-threads"] });
      await queryClient.invalidateQueries({ queryKey: ["system-stats"] });
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
      setStreamError(null);
      setActiveSteps([]);

      const userTurnId = `user-${Date.now()}`;
      const assistantTurnId = `assistant-${Date.now()}`;

      const userMessage: ChatMessage = {
        id: userTurnId,
        role: "user",
        content: trimmed,
        citations: [],
        domain_filter: domainFilter ?? null,
      };

      const initialAssistantMessage: ChatMessage = {
        id: assistantTurnId,
        role: "assistant",
        content: "",
        citations: [],
        steps: [],
        provider: provider || null,
        model: model || null,
        isStreaming: true,
        isError: false,
      };

      setMessages((prev) => [...prev, userMessage, initialAssistantMessage]);
      setIsStreaming(true);

      const controller = new AbortController();
      abortControllerRef.current = controller;

      const body: ChatRequest = {
        message: trimmed,
        thread_id: threadId || activeThreadId || null,
        filter_doc_ids:
          filterDocIds && filterDocIds.length > 0 ? filterDocIds : null,
        domain_filter: domainFilter ?? null,
        provider: provider ? provider : null,
        model: model ? model : null,
      };

      let accumulatedContent = "";
      let latestCitations: Citation[] = [];
      let collectedSteps: CRAGStep[] = [];

      try {
        const response = await fetch(`${API_BASE}/chat/completions`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Accept: "text/event-stream",
          },
          body: JSON.stringify(body),
          signal: controller.signal,
        });

        if (!response.ok || !response.body) {
          let errDetail = `HTTP ${response.status}: ${response.statusText}`;
          try {
            const errJson = (await response.json()) as {
              detail?: string;
              message?: string;
            };
            errDetail = errJson.detail || errJson.message || errDetail;
          } catch {
            // ignore
          }
          throw new Error(errDetail);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let buffer = "";
        let streamEndedByError = false;

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const blocks = buffer.split(/\r?\n\r?\n/);
          buffer = blocks.pop() ?? "";

          for (const block of blocks) {
            const trimmedBlock = block.trim();
            if (!trimmedBlock) continue;
            const parsed = parseSSEBlock(trimmedBlock);
            if (!parsed) continue;

            if (parsed.event === "step") {
              const stepData = parsed.data as {
                node?: string;
                label?: string;
                status?: string;
                routing_mode?: string;
                target_domains?: string[];
                polysemy_collisions?: PolysemyCollision[];
                domain_queries?: Record<string, string>;
              };
              const nextStep: CRAGStep = {
                node: stepData.node || "step",
                label: stepData.label || stepData.node || "Processing",
                status: "completed",
                routing_mode: stepData.routing_mode,
                target_domains: stepData.target_domains,
                polysemy_collisions: stepData.polysemy_collisions,
                domain_queries: stepData.domain_queries,
              };
              collectedSteps = [
                ...collectedSteps.filter((s) => s.node !== nextStep.node),
                nextStep,
              ];
              setActiveSteps([...collectedSteps]);
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantTurnId
                    ? { ...m, steps: [...collectedSteps] }
                    : m
                )
              );
            } else if (parsed.event === "citations") {
              const citData = parsed.data as { citations?: Citation[] };
              if (Array.isArray(citData.citations)) {
                latestCitations = citData.citations;
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === assistantTurnId
                      ? { ...m, citations: latestCitations }
                      : m
                  )
                );
              }
            } else if (parsed.event === "token") {
              const tokData = parsed.data as {
                content?: string;
                token?: string;
              };
              const piece = tokData.content ?? tokData.token ?? "";
              accumulatedContent += piece;
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantTurnId
                    ? { ...m, content: accumulatedContent }
                    : m
                )
              );
            } else if (parsed.event === "done") {
              const doneData = parsed.data as {
                thread_id?: string;
                message?: string;
                citations?: Citation[];
                provider?: string;
                model?: string;
              };
              if (doneData.message && !accumulatedContent) {
                accumulatedContent = doneData.message;
              }
              if (Array.isArray(doneData.citations)) {
                latestCitations = doneData.citations;
              }
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantTurnId
                    ? {
                        ...m,
                        content: doneData.message || accumulatedContent,
                        citations: latestCitations,
                        steps: [...collectedSteps],
                        provider: doneData.provider || m.provider || null,
                        model: doneData.model || m.model || null,
                        isStreaming: false,
                      }
                    : m
                )
              );
              if (doneData.thread_id && doneData.thread_id !== activeThreadId) {
                skipNextThreadFetchRef.current = doneData.thread_id;
                onThreadIdResolved?.(doneData.thread_id);
              }
            } else if (parsed.event === "error") {
              const errData = parsed.data as {
                thread_id?: string;
                message?: string;
                error?: string;
              };
              const errMsg =
                errData.error ||
                errData.message ||
                "An unexpected error occurred during RAG execution.";
              setStreamError(errMsg);
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantTurnId
                    ? {
                        ...m,
                        content: errMsg,
                        isStreaming: false,
                        isError: true,
                      }
                    : m
                )
              );
              if (errData.thread_id && errData.thread_id !== activeThreadId) {
                skipNextThreadFetchRef.current = errData.thread_id;
                onThreadIdResolved?.(errData.thread_id);
              }
              streamEndedByError = true;
              break;
            }
          }

          if (streamEndedByError) {
            await reader.cancel();
            break;
          }
        }
      } catch (err: unknown) {
        if (err instanceof DOMException && err.name === "AbortError") {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantTurnId
                ? {
                    ...m,
                    content:
                      accumulatedContent || "(Generation cancelled by user.)",
                    isStreaming: false,
                  }
                : m
            )
          );
        } else {
          const errMsg =
            err instanceof Error ? err.message : "Failed to stream response.";
          setStreamError(errMsg);
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantTurnId
                ? {
                    ...m,
                    content: errMsg,
                    isStreaming: false,
                    isError: true,
                  }
                : m
            )
          );
        }
      } finally {
        abortControllerRef.current = null;
        setIsStreaming(false);
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantTurnId ? { ...m, isStreaming: false } : m
          )
        );
        void queryClient.invalidateQueries({ queryKey: ["chat-threads"] });
        void queryClient.invalidateQueries({ queryKey: ["system-stats"] });
        void queryClient.invalidateQueries({ queryKey: ["workspaceSettings"] });
      }
    },
    [activeThreadId, isStreaming, onThreadIdResolved, queryClient, stopStreaming]
  );

  return {
    messages,
    isStreaming,
    isLoadingHistory,
    activeSteps,
    streamError,
    sendMessage,
    stopStreaming,
    clearConversation,
    removeThread,
  };
}
