import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  createDomain,
  createSubdomain,
  deleteDomain,
  deleteSubdomain,
  fetchDomains,
  updateDomain,
} from "../lib/api";
import { STARTER_DOMAINS_FALLBACK } from "../lib/domainTheme";
import { queryKeys } from "../lib/queryKeys";
import type {
  DomainCreatePayload,
  DomainDeleteParams,
  DomainItem,
  DomainUpdatePayload,
  SubdomainCreatePayload,
} from "../lib/types";

/**
 * Retrieve active domains with starter fallback to ensure UI never flashes empty.
 */
export function useDomainsQuery(includeInactive = false) {
  return useQuery<DomainItem[]>({
    queryKey: queryKeys.domains(includeInactive),
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
      void queryClient.invalidateQueries({ queryKey: queryKeys.domains() });
      void queryClient.invalidateQueries({ queryKey: queryKeys.documents() });
      void queryClient.invalidateQueries({ queryKey: queryKeys.systemStats() });
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
      void queryClient.invalidateQueries({ queryKey: queryKeys.domains() });
      void queryClient.invalidateQueries({ queryKey: queryKeys.documents() });
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
      void queryClient.invalidateQueries({ queryKey: queryKeys.domains() });
      void queryClient.invalidateQueries({ queryKey: queryKeys.documents() });
      void queryClient.invalidateQueries({ queryKey: queryKeys.systemStats() });
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
      void queryClient.invalidateQueries({ queryKey: queryKeys.domains() });
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
      void queryClient.invalidateQueries({ queryKey: queryKeys.domains() });
      void queryClient.invalidateQueries({ queryKey: queryKeys.documents() });
    },
  });
}
