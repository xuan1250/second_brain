import { parseApiError } from "./api";
import type { Citation, CRAGStep, PolysemyCollision } from "./types";

export interface ParsedSSEFrame {
  event: string;
  data: unknown;
}

export function parseSSEBlock(rawBlock: string): ParsedSSEFrame | null {
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

export interface SSEStreamHandlers {
  onStep?: (step: CRAGStep) => void;
  onCitations?: (citations: Citation[]) => void;
  onToken?: (piece: string) => void;
  onDone?: (payload: {
    thread_id?: string;
    message?: string;
    citations?: Citation[];
    provider?: string;
    model?: string;
  }) => void;
  onErrorEvent?: (payload: {
    thread_id?: string;
    message?: string;
    error?: string;
  }) => void;
}

export async function readSSEStream(
  url: string,
  body: unknown,
  signal: AbortSignal,
  handlers: SSEStreamHandlers
): Promise<void> {
  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    body: JSON.stringify(body),
    signal,
  });

  if (!response.ok || !response.body) {
    throw new Error(await parseApiError(response));
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
          routing_mode?: string;
          target_domains?: string[];
          polysemy_collisions?: PolysemyCollision[];
          domain_queries?: Record<string, string>;
        };
        handlers.onStep?.({
          node: stepData.node || "step",
          label: stepData.label || stepData.node || "Processing",
          status: "completed",
          routing_mode: stepData.routing_mode,
          target_domains: stepData.target_domains,
          polysemy_collisions: stepData.polysemy_collisions,
          domain_queries: stepData.domain_queries,
        });
      } else if (parsed.event === "citations") {
        const citData = parsed.data as { citations?: Citation[] };
        if (Array.isArray(citData.citations)) {
          handlers.onCitations?.(citData.citations);
        }
      } else if (parsed.event === "token") {
        const tokData = parsed.data as {
          content?: string;
          token?: string;
        };
        const piece = tokData.content ?? tokData.token ?? "";
        handlers.onToken?.(piece);
      } else if (parsed.event === "done") {
        const doneData = parsed.data as {
          thread_id?: string;
          message?: string;
          citations?: Citation[];
          provider?: string;
          model?: string;
        };
        handlers.onDone?.(doneData);
      } else if (parsed.event === "error") {
        const errData = parsed.data as {
          thread_id?: string;
          message?: string;
          error?: string;
        };
        handlers.onErrorEvent?.(errData);
        streamEndedByError = true;
        break;
      }
    }

    if (streamEndedByError) {
      await reader.cancel();
      break;
    }
  }
}
