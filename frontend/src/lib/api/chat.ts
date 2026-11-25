import type {
  ThreadDeleteResponse,
  ThreadDetailResponse,
  ThreadListResponse,
} from "../types";
import { requestJson } from "./client";

export async function fetchChatThreads(params?: {
  limit?: number;
  offset?: number;
}): Promise<ThreadListResponse> {
  const query = new URLSearchParams();
  if (params?.limit !== undefined) query.set("limit", String(params.limit));
  if (params?.offset !== undefined) query.set("offset", String(params.offset));
  const qs = query.toString() ? `?${query.toString()}` : "";
  return requestJson<ThreadListResponse>(`/chat/threads${qs}`);
}

export async function fetchChatThread(
  threadId: string
): Promise<ThreadDetailResponse> {
  return requestJson<ThreadDetailResponse>(
    `/chat/threads/${encodeURIComponent(threadId)}`
  );
}

export async function deleteChatThread(
  threadId: string
): Promise<ThreadDeleteResponse> {
  return requestJson<ThreadDeleteResponse>(
    `/chat/threads/${encodeURIComponent(threadId)}`,
    { method: "DELETE" }
  );
}
