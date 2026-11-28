import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  deleteLexiconTerm,
  fetchLexicon,
  upsertLexiconTerm,
} from "../lib/api";
import { queryKeys } from "../lib/queryKeys";
import type { LexiconCreatePayload } from "../lib/types";

export function useLexiconQuery(params?: { domain_id?: string; search?: string }) {
  return useQuery({
    queryKey: queryKeys.lexicon(params),
    queryFn: () => fetchLexicon(params),
    staleTime: 15_000,
  });
}

export function useUpsertLexiconMutation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (payload: LexiconCreatePayload) => upsertLexiconTerm(payload),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: queryKeys.lexicon() });
    },
  });
}

export function useDeleteLexiconMutation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => deleteLexiconTerm(id),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: queryKeys.lexicon() });
    },
  });
}
