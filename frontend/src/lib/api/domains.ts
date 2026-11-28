import type {
  DomainCreatePayload,
  DomainDeleteParams,
  DomainDeleteResponse,
  DomainItem,
  DomainUpdatePayload,
  LexiconCreatePayload,
  LexiconItem,
  SubdomainCreatePayload,
  SubdomainItem,
} from "../types";
import { jsonInit, requestJson } from "./client";

export async function fetchDomains(
  includeInactive = false
): Promise<DomainItem[]> {
  const qs = includeInactive ? "?include_inactive=true" : "";
  return requestJson<DomainItem[]>(`/domains${qs}`);
}

export async function createDomain(
  payload: DomainCreatePayload
): Promise<DomainItem> {
  return requestJson<DomainItem>("/domains", jsonInit("POST", payload));
}

export async function updateDomain(
  slug: string,
  payload: DomainUpdatePayload
): Promise<DomainItem> {
  return requestJson<DomainItem>(
    `/domains/${encodeURIComponent(slug)}`,
    jsonInit("PATCH", payload)
  );
}

export async function deleteDomain(
  slug: string,
  params: DomainDeleteParams = {}
): Promise<DomainDeleteResponse> {
  const query = new URLSearchParams();
  if (params.target_domain) query.set("target_domain", params.target_domain);
  if (params.target_subdomain) {
    query.set("target_subdomain", params.target_subdomain);
  }
  if (params.re_index) query.set("re_index", String(params.re_index));
  const qs = query.toString() ? `?${query.toString()}` : "";
  return requestJson<DomainDeleteResponse>(
    `/domains/${encodeURIComponent(slug)}${qs}`,
    { method: "DELETE" }
  );
}

export async function createSubdomain(
  domainSlug: string,
  payload: SubdomainCreatePayload
): Promise<SubdomainItem> {
  return requestJson<SubdomainItem>(
    `/domains/${encodeURIComponent(domainSlug)}/subdomains`,
    jsonInit("POST", payload)
  );
}

export async function deleteSubdomain(
  domainSlug: string,
  subSlug: string,
  targetSubdomain?: string | null
): Promise<{ status: string; deleted_slug: string; target_subdomain?: string }> {
  const query = new URLSearchParams();
  if (targetSubdomain) query.set("target_subdomain", targetSubdomain);
  const qs = query.toString() ? `?${query.toString()}` : "";
  return requestJson<{
    status: string;
    deleted_slug: string;
    target_subdomain?: string;
  }>(
    `/domains/${encodeURIComponent(domainSlug)}/subdomains/${encodeURIComponent(subSlug)}${qs}`,
    { method: "DELETE" }
  );
}

export async function fetchLexicon(params?: {
  domain_id?: string;
  search?: string;
}): Promise<LexiconItem[]> {
  const query = new URLSearchParams();
  if (params?.domain_id) query.set("domain_id", params.domain_id);
  if (params?.search && params.search.trim()) {
    query.set("search", params.search.trim());
  }
  const qs = query.toString() ? `?${query.toString()}` : "";
  return requestJson<LexiconItem[]>(`/lexicon${qs}`);
}

export async function upsertLexiconTerm(
  payload: LexiconCreatePayload
): Promise<LexiconItem> {
  return requestJson<LexiconItem>("/lexicon", jsonInit("POST", payload));
}

export async function deleteLexiconTerm(
  id: string
): Promise<{ deleted: boolean; id: string }> {
  return requestJson<{ deleted: boolean; id: string }>(
    `/lexicon/${encodeURIComponent(id)}`,
    { method: "DELETE" }
  );
}
