import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  fetchWorkspaceSettings,
  updateWorkspaceSettings,
  verifyLlmKey,
} from "../lib/api";
import { queryKeys } from "../lib/queryKeys";
import type {
  VerifyKeyPayload,
  VerifyKeyResponse,
  WorkspaceSettings,
  WorkspaceSettingsUpdatePayload,
} from "../lib/types";

export function useWorkspaceSettingsQuery() {
  return useQuery<WorkspaceSettings>({
    queryKey: queryKeys.workspaceSettings(),
    queryFn: () => fetchWorkspaceSettings(),
    staleTime: 60_000,
  });
}

export function useUpdateWorkspaceSettingsMutation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (payload: WorkspaceSettingsUpdatePayload) =>
      updateWorkspaceSettings(payload),
    onSuccess: (updatedSettings) => {
      queryClient.setQueryData(queryKeys.workspaceSettings(), updatedSettings);
      void queryClient.invalidateQueries({
        queryKey: queryKeys.workspaceSettings(),
      });
      void queryClient.invalidateQueries({ queryKey: queryKeys.systemStats() });
    },
  });
}

export function useVerifyLlmKeyMutation() {
  const queryClient = useQueryClient();
  return useMutation<VerifyKeyResponse, Error, VerifyKeyPayload>({
    mutationFn: (payload: VerifyKeyPayload) => verifyLlmKey(payload),
    onSettled: () => {
      void queryClient.invalidateQueries({
        queryKey: queryKeys.workspaceSettings(),
      });
      void queryClient.invalidateQueries({ queryKey: queryKeys.systemStats() });
    },
  });
}
