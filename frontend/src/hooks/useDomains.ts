import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  createDomain,
  createSubdomain,
  deleteDomain,
  deleteLexiconTerm,
  deleteSubdomain,
  fetchDomains,
  fetchLexicon,
  fetchWorkspaceSettings,
  updateDomain,
  updateSubdomain,
  updateWorkspaceSettings,
  upsertLexiconTerm,
  verifyLlmKey,
} from "../lib/api";
import { STARTER_DOMAINS_FALLBACK } from "../lib/domainTheme";
import type {
  DomainCreatePayload,
  DomainDeleteParams,
  DomainItem,
  DomainUpdatePayload,
  LexiconCreatePayload,
  SubdomainCreatePayload,
  SubdomainUpdatePayload,
  VerifyKeyPayload,
  VerifyKeyResponse,
  WorkspaceSettings,
  WorkspaceSettingsUpdatePayload,
} from "../lib/types";

/**
 * Hook to retrieve active domains with starter fallback to ensure UI never flashes empty.
 */
export function useDomainsQuery(includeInactive = false) {
  return useQuery<DomainItem[]>({
    queryKey: ["domains", { includeInactive }],
    queryFn: () => fetchDomains(includeInactive),
    placeholderData: STARTER_DOMAINS_FALLBACK,
    staleTime: 30_000,
  });
}

export function useCreateDomainMutation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (payload: DomainCreatePayload) => createDomain(payload),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["domains"] });
      void queryClient.invalidateQueries({ queryKey: ["documents"] });
      void queryClient.invalidateQueries({ queryKey: ["system-stats"] });
    },
  });
}

export function useUpdateDomainMutation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({
      slug,
      payload,
    }: {
      slug: string;
      payload: DomainUpdatePayload;
    }) => updateDomain(slug, payload),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["domains"] });
      void queryClient.invalidateQueries({ queryKey: ["documents"] });
    },
  });
}

export function useDeleteDomainMutation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({
      slug,
      params,
    }: {
      slug: string;
      params?: DomainDeleteParams;
    }) => deleteDomain(slug, params),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["domains"] });
      void queryClient.invalidateQueries({ queryKey: ["documents"] });
      void queryClient.invalidateQueries({ queryKey: ["system-stats"] });
    },
  });
}

export function useCreateSubdomainMutation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({
      domainSlug,
      payload,
    }: {
      domainSlug: string;
      payload: SubdomainCreatePayload;
    }) => createSubdomain(domainSlug, payload),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["domains"] });
    },
  });
}

export function useUpdateSubdomainMutation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({
      domainSlug,
      subSlug,
      payload,
    }: {
      domainSlug: string;
      subSlug: string;
      payload: SubdomainUpdatePayload;
    }) => updateSubdomain(domainSlug, subSlug, payload),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["domains"] });
    },
  });
}

export function useDeleteSubdomainMutation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({
      domainSlug,
      subSlug,
      targetSubdomain,
    }: {
      domainSlug: string;
      subSlug: string;
      targetSubdomain?: string | null;
    }) => deleteSubdomain(domainSlug, subSlug, targetSubdomain),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["domains"] });
      void queryClient.invalidateQueries({ queryKey: ["documents"] });
    },
  });
}

export function useLexiconQuery(params?: { domain_id?: string; search?: string }) {
  return useQuery({
    queryKey: ["lexicon", params],
    queryFn: () => fetchLexicon(params),
    staleTime: 15_000,
  });
}

export function useUpsertLexiconMutation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (payload: LexiconCreatePayload) => upsertLexiconTerm(payload),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["lexicon"] });
    },
  });
}

export function useDeleteLexiconMutation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => deleteLexiconTerm(id),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["lexicon"] });
    },
  });
}

export function useWorkspaceSettingsQuery() {
  return useQuery<WorkspaceSettings>({
    queryKey: ["workspaceSettings"],
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
      queryClient.setQueryData(["workspaceSettings"], updatedSettings);
      void queryClient.invalidateQueries({ queryKey: ["workspaceSettings"] });
      void queryClient.invalidateQueries({ queryKey: ["system-stats"] });
    },
  });
}

export function useVerifyLlmKeyMutation() {
  const queryClient = useQueryClient();
  return useMutation<VerifyKeyResponse, Error, VerifyKeyPayload>({
    mutationFn: (payload: VerifyKeyPayload) => verifyLlmKey(payload),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["workspaceSettings"] });
      void queryClient.invalidateQueries({ queryKey: ["system-stats"] });
    },
    onSettled: () => {
      void queryClient.invalidateQueries({ queryKey: ["workspaceSettings"] });
      void queryClient.invalidateQueries({ queryKey: ["system-stats"] });
    },
  });
}
