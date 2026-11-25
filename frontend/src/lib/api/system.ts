import type {
  SystemStats,
  VerifyKeyPayload,
  VerifyKeyResponse,
  WorkspaceSettings,
  WorkspaceSettingsUpdatePayload,
} from "../types";
import { jsonInit, requestJson } from "./client";

export async function fetchSystemStats(): Promise<SystemStats> {
  return requestJson<SystemStats>("/system/stats");
}

export async function pauseIngestion(): Promise<{
  status: string;
  ingestion_paused: boolean;
}> {
  return requestJson<{ status: string; ingestion_paused: boolean }>(
    "/system/ingestion/pause",
    { method: "POST" }
  );
}

export async function resumeIngestion(): Promise<{
  status: string;
  ingestion_paused: boolean;
}> {
  return requestJson<{ status: string; ingestion_paused: boolean }>(
    "/system/ingestion/resume",
    { method: "POST" }
  );
}

export async function fetchWorkspaceSettings(): Promise<WorkspaceSettings> {
  return requestJson<WorkspaceSettings>("/settings");
}

export async function updateWorkspaceSettings(
  payload: WorkspaceSettingsUpdatePayload
): Promise<WorkspaceSettings> {
  return requestJson<WorkspaceSettings>("/settings", jsonInit("PATCH", payload));
}

export async function verifyLlmKey(
  payload: VerifyKeyPayload
): Promise<VerifyKeyResponse> {
  return requestJson<VerifyKeyResponse>(
    "/settings/verify-key",
    jsonInit("POST", payload)
  );
}
