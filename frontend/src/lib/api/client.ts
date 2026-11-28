export const API_BASE = "/api/v1";

export async function parseApiError(response: Response): Promise<string> {
  let errorDetail = `HTTP ${response.status}: ${response.statusText}`;
  try {
    const payload = (await response.json()) as {
      detail?: string | unknown;
      message?: string;
    };
    if (payload?.detail) {
      errorDetail =
        typeof payload.detail === "string"
          ? payload.detail
          : JSON.stringify(payload.detail);
    } else if (payload?.message) {
      errorDetail = payload.message;
    }
  } catch {
    // Fallback to statusText
  }
  return errorDetail;
}

export async function requestJson<T>(
  path: string,
  init?: RequestInit
): Promise<T> {
  const url = path.startsWith("/") ? `${API_BASE}${path}` : `${API_BASE}/${path}`;
  const response = await fetch(url, init);
  if (!response.ok) {
    throw new Error(await parseApiError(response));
  }
  return (await response.json()) as T;
}

export function jsonInit(method: string, body?: unknown): RequestInit {
  if (body === undefined) {
    return { method };
  }
  return {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
}
