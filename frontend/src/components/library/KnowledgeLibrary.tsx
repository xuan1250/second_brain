import React, { useEffect, useMemo, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import * as Dialog from "@radix-ui/react-dialog";
import {
  AlertTriangle,
  Check,
  CheckCircle2,
  CheckSquare,
  ChevronDown,
  Database,
  Edit2,
  FileSpreadsheet,
  FileText,
  Filter,
  Loader2,
  Pause,
  Play,
  Presentation,
  Rocket,
  RotateCcw,
  Search,
  ShieldCheck,
  Sparkles,
  Square,
  Trash2,
  X,
  Zap,
} from "lucide-react";
import {
  isDocumentProcessing,
  useApproveBatchMutation,
  useApproveClassificationMutation,
  useDeleteDocumentMutation,
  useDocuments,
  usePauseIngestionMutation,
  useResumeIngestionMutation,
  useRetryDocumentMutation,
  useSystemStats,
} from "../../hooks/useDocuments";
import { useDomainsQuery } from "../../hooks/useDomains";
import { resolveDomainMeta } from "../../lib/domainTheme";
import type {
  ApprovedClassificationRequest,
  DocumentItem,
  DomainItem,
  RecommendedDomainProfile,
} from "../../lib/types";
import { useWorkspace } from "../../lib/workspaceStore";
import { UploadDropzone } from "./UploadDropzone";

const FILE_TYPE_FILTERS = [
  { id: "all", label: "All" },
  { id: "pdf", label: "PDF" },
  { id: "docx", label: "DOCX" },
  { id: "xlsx", label: "XLSX" },
  { id: "pptx", label: "PPTX" },
] as const;

export type DomainFacetId = string;

export function getDomainKey(doc: DocumentItem): string {
  if (doc.primary_domain_id) {
    return doc.primary_domain_id.toLowerCase();
  }

  if (doc.classification_result) {
    try {
      const parsed = JSON.parse(doc.classification_result) as RecommendedDomainProfile;
      if (parsed.primary_domain) {
        return parsed.primary_domain.toLowerCase();
      }
    } catch {
      // ignore
    }
  }
  return "general";
}

export function parseClassificationResult(
  doc: DocumentItem
): RecommendedDomainProfile | null {
  if (!doc.classification_result) return null;
  try {
    return JSON.parse(doc.classification_result) as RecommendedDomainProfile;
  } catch {
    return null;
  }
}

function formatFileSize(bytes: number): string {
  if (!bytes || bytes <= 0) return "0 B";
  const kb = bytes / 1024;
  if (kb < 1024) return `${kb.toFixed(1)} KB`;
  return `${(kb / 1024).toFixed(2)} MB`;
}

function FileTypeIcon({ fileType }: { fileType: string }) {
  const ft = (fileType || "").toLowerCase();
  if (ft === "xlsx") {
    return <FileSpreadsheet className="h-4 w-4 text-emerald-400 shrink-0" />;
  }
  if (ft === "pptx") {
    return <Presentation className="h-4 w-4 text-amber-400 shrink-0" />;
  }
  if (ft === "docx") {
    return <FileText className="h-4 w-4 text-sky-400 shrink-0" />;
  }
  return <FileText className="h-4 w-4 text-rose-400 shrink-0" />;
}

export function AuthorityTierBadge({ tier }: { tier?: number | null }) {
  const t = tier ?? 3;
  let bg = "bg-sky-500/15 border-sky-500/30 text-sky-300";
  let title = "Tier 3: Standard Internal/Technical";

  if (t === 1) {
    bg = "bg-rose-500/15 border-rose-500/30 text-rose-300";
    title = "Tier 1: Binding Statutes / Primary Standards (IEEE, IEC, US Code)";
  } else if (t === 2) {
    bg = "bg-purple-500/15 border-purple-500/30 text-purple-300";
    title = "Tier 2: Regulatory Directives / Administrative Guidance (CFR, FASB)";
  } else if (t === 3) {
    bg = "bg-cyan-500/15 border-cyan-500/30 text-cyan-300";
    title = "Tier 3: Industry Practice / Standard Technical Reports";
  } else if (t === 4) {
    bg = "bg-emerald-500/15 border-emerald-500/30 text-emerald-300";
    title = "Tier 4: Working Papers / Draft Guidelines";
  } else if (t === 5) {
    bg = "bg-slate-500/15 border-slate-500/30 text-slate-400";
    title = "Tier 5: Informal Notes / Unverified Documents";
  }

  return (
    <span
      title={title}
      className={`inline-flex items-center rounded-full border px-1.5 py-0.2 text-[10px] font-mono font-semibold cursor-help ${bg}`}
    >
      T{t}
    </span>
  );
}

export function DomainBadge({
  domainKey,
  subdomain,
  domains,
}: {
  domainKey: string;
  subdomain?: string | null;
  domains?: DomainItem[] | null;
}) {
  const meta = resolveDomainMeta(domainKey, domains);
  const subLabel = subdomain ? ` › ${subdomain.split(".").pop()}` : "";

  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[10px] font-medium ${meta.theme.badge}`}
      title={`Domain: ${meta.name}${subdomain ? ` / Subdomain: ${subdomain}` : ""}`}
    >
      <span>{meta.icon}</span>
      <span>
        {meta.shortLabel}
        {subLabel}
      </span>
    </span>
  );
}

function StatusBadge({ doc }: { doc: DocumentItem }) {
  const status = (doc.status || "queued").toLowerCase();
  const stage = (doc.stage || status).toLowerCase();
  const processing = isDocumentProcessing(doc);

  if (status === "awaiting_approval") {
    return (
      <span className="inline-flex items-center gap-1 rounded-full bg-amber-500/15 border border-amber-500/40 px-2 py-0.5 text-[10px] font-semibold text-amber-300 animate-pulse">
        <ShieldCheck className="h-2.5 w-2.5 text-amber-400" />
        Awaiting Approval
      </span>
    );
  }

  if (status === "classifying") {
    return (
      <span className="inline-flex items-center gap-1 rounded-full bg-purple-500/15 border border-purple-500/30 px-2 py-0.5 text-[10px] font-medium text-purple-300 animate-pulse">
        <Loader2 className="h-2.5 w-2.5 animate-spin text-purple-400" />
        Classifying (Laya)
      </span>
    );
  }

  if (status === "ready") {
    return (
      <span className="inline-flex items-center gap-1 rounded-full bg-emerald-500/15 border border-emerald-500/30 px-2 py-0.5 text-[10px] font-medium text-emerald-300">
        <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
        Ready · {doc.chunk_count} ch
      </span>
    );
  }

  if (status === "failed") {
    const errText =
      doc.error_message || doc.error || "Document ingestion failed";
    return (
      <span
        title={errText}
        className="inline-flex items-center gap-1 rounded-full bg-rose-500/15 border border-rose-500/30 px-2 py-0.5 text-[10px] font-medium text-rose-300 cursor-help"
      >
        <AlertTriangle className="h-2.5 w-2.5 text-rose-400" />
        Failed
      </span>
    );
  }

  if (processing) {
    const pct =
      typeof doc.progress === "number" && doc.progress > 0
        ? ` ${Math.round(doc.progress <= 1 ? doc.progress * 100 : doc.progress)}%`
        : "";
    const stageLabel =
      stage.charAt(0).toUpperCase() + stage.slice(1).toLowerCase();
    return (
      <span className="inline-flex items-center gap-1 rounded-full bg-indigo-500/15 border border-indigo-500/30 px-2 py-0.5 text-[10px] font-medium text-indigo-300 animate-pulse">
        <Loader2 className="h-2.5 w-2.5 animate-spin text-indigo-400" />
        {stageLabel}
        {pct}
      </span>
    );
  }

  return (
    <span className="inline-flex items-center rounded-full bg-slate-800 px-2 py-0.5 text-[10px] text-slate-300">
      {doc.status}
    </span>
  );
}

function EngineBadge({ doc }: { doc: DocumentItem }) {
  if (doc.engine_fallback_reason) {
    return (
      <span
        title={doc.engine_fallback_reason}
        className="inline-flex items-center gap-1 rounded-full bg-amber-500/15 border border-amber-500/30 px-2 py-0.5 text-[10px] font-medium text-amber-300 cursor-help"
      >
        <AlertTriangle className="h-2.5 w-2.5 text-amber-400" />
        Fallback
      </span>
    );
  }

  if (doc.parsing_engine === "docling") {
    return (
      <span
        title="Docling GPU Vision Engine"
        className="inline-flex items-center gap-1 rounded-full bg-purple-500/15 border border-purple-500/30 px-2 py-0.5 text-[10px] font-medium text-purple-300"
      >
        <Rocket className="h-2.5 w-2.5 text-purple-400" />
        Docling GPU
      </span>
    );
  }

  if (doc.parsing_engine === "pymupdf4llm") {
    return (
      <span
        title="PyMuPDF Fast CPU Engine"
        className="inline-flex items-center gap-1 rounded-full bg-sky-500/15 border border-sky-500/30 px-2 py-0.5 text-[10px] font-medium text-sky-300"
      >
        <Zap className="h-2.5 w-2.5 text-sky-400" />
        Fast CPU
      </span>
    );
  }

  return null;
}

/**
 * Inline Approval Card for documents paused at `status === "awaiting_approval"`.
 */
function AwaitingApprovalCard({
  doc,
  onApprove,
  isApproving,
}: {
  doc: DocumentItem;
  onApprove: (docId: string, payload?: ApprovedClassificationRequest) => void;
  isApproving: boolean;
}) {
  const { data: domains = [] } = useDomainsQuery();
  const profile = useMemo(() => parseClassificationResult(doc), [doc]);
  const [isEditing, setIsEditing] = useState(false);

  const defaultDomain = profile?.primary_domain || getDomainKey(doc);
  const defaultSubdomain = profile?.primary_subdomain || doc.primary_subdomain_id || "";
  const defaultTier = profile?.authority_tier ?? doc.authority_tier ?? 3;
  const defaultJurisdiction = profile?.jurisdiction ?? doc.jurisdiction ?? "";

  const [domain, setDomain] = useState<string>(defaultDomain);
  const [subdomain, setSubdomain] = useState<string>(defaultSubdomain);
  const [tier, setTier] = useState<number>(defaultTier);
  const [jurisdiction, setJurisdiction] = useState<string>(defaultJurisdiction);

  const selectedDomainObj = useMemo(
    () => domains.find((d) => d.slug === domain),
    [domains, domain]
  );

  const confidencePct = profile?.confidence
    ? Math.round(profile.confidence * 100)
    : 80;

  const handleApproveRecommended = (e: React.MouseEvent) => {
    e.stopPropagation();
    onApprove(doc.id);
  };

  const handleSaveAndApprove = (e: React.MouseEvent) => {
    e.stopPropagation();
    onApprove(doc.id, {
      domain,
      subdomain: subdomain || undefined,
      authority_tier: tier,
      jurisdiction: jurisdiction || undefined,
    });
    setIsEditing(false);
  };

  return (
    <div
      onClick={(e) => e.stopPropagation()}
      className="mt-2 rounded-xl border border-amber-500/40 bg-amber-950/20 p-2.5 text-xs text-amber-100 space-y-2"
    >
      <div className="flex items-center justify-between gap-2 border-b border-amber-500/20 pb-1.5">
        <div className="flex items-center gap-1.5 font-semibold text-amber-300">
          <Sparkles className="h-3.5 w-3.5 text-amber-400" />
          <span>Laya Classification Recommendation</span>
        </div>
        <span className="rounded-full bg-amber-500/20 px-2 py-0.5 text-[10px] font-mono font-medium text-amber-200">
          {confidencePct}% conf
        </span>
      </div>

      {!isEditing ? (
        <div className="space-y-1.5">
          <div className="flex flex-wrap items-center gap-1.5">
            <DomainBadge
              domainKey={profile?.primary_domain || defaultDomain}
              subdomain={profile?.primary_subdomain || defaultSubdomain}
              domains={domains}
            />
            <AuthorityTierBadge tier={defaultTier} />
            {profile?.jurisdiction && (
              <span className="rounded bg-slate-800/80 px-1.5 py-0.5 text-[10px] text-slate-300 font-mono">
                {profile.jurisdiction}
              </span>
            )}
            {profile?.effective_year && (
              <span className="rounded bg-slate-800/80 px-1.5 py-0.5 text-[10px] text-slate-300 font-mono">
                {profile.effective_year}
              </span>
            )}
            {profile?.is_hybrid && (
              <span className="rounded bg-purple-950/60 border border-purple-500/30 px-1.5 py-0.5 text-[10px] text-purple-300">
                Hybrid: {profile.section_overrides?.length || 0} overrides
              </span>
            )}
          </div>

          {profile?.reasoning && (
            <p className="text-[10.5px] text-slate-300 italic line-clamp-2">
              "{profile.reasoning}"
            </p>
          )}

          <div className="flex items-center justify-end gap-1.5 pt-1">
            <button
              type="button"
              onClick={() => setIsEditing(true)}
              className="inline-flex items-center gap-1 rounded-lg border border-slate-700 bg-slate-800 px-2 py-1 text-[10.5px] font-medium text-slate-300 hover:bg-slate-700 hover:text-white transition"
            >
              <Edit2 className="h-3 w-3" />
              <span>Edit</span>
            </button>
            <button
              type="button"
              disabled={isApproving}
              onClick={handleApproveRecommended}
              className="inline-flex items-center gap-1 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white px-2.5 py-1 text-[10.5px] font-semibold transition shadow-xs disabled:opacity-50"
            >
              {isApproving ? (
                <Loader2 className="h-3 w-3 animate-spin" />
              ) : (
                <Check className="h-3 w-3" />
              )}
              <span>Approve Recommended</span>
            </button>
          </div>
        </div>
      ) : (
        <div className="space-y-2 pt-1">
          <div className="grid grid-cols-2 gap-2 text-[11px]">
            <div>
              <label className="block text-[10px] font-medium text-slate-400 mb-0.5">
                Primary Domain
              </label>
              <select
                value={domain}
                onChange={(e) => {
                  const nextD = e.target.value;
                  setDomain(nextD);
                  const found = domains.find((d) => d.slug === nextD);
                  if (found?.subdomains?.length) {
                    setSubdomain(found.subdomains[0].slug);
                  }
                }}
                className="w-full rounded border border-slate-700 bg-slate-900 px-2 py-1 text-slate-200 focus:outline-none"
              >
                {domains.map((d) => (
                  <option key={d.slug} value={d.slug}>
                    {d.icon} {d.name}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-[10px] font-medium text-slate-400 mb-0.5">
                Subdomain
              </label>
              {selectedDomainObj && selectedDomainObj.subdomains.length > 0 ? (
                <select
                  value={subdomain}
                  onChange={(e) => setSubdomain(e.target.value)}
                  className="w-full rounded border border-slate-700 bg-slate-900 px-2 py-1 text-slate-200 focus:outline-none"
                >
                  {selectedDomainObj.subdomains.map((s) => (
                    <option key={s.slug} value={s.slug}>
                      {s.name} ({s.slug})
                    </option>
                  ))}
                </select>
              ) : (
                <input
                  type="text"
                  value={subdomain}
                  onChange={(e) => setSubdomain(e.target.value)}
                  placeholder="e.g. electrical, statute"
                  className="w-full rounded border border-slate-700 bg-slate-900 px-2 py-1 text-slate-200 placeholder:text-slate-600 focus:outline-none"
                />
              )}
            </div>
            <div>
              <label className="block text-[10px] font-medium text-slate-400 mb-0.5">
                Authority Tier
              </label>
              <select
                value={tier}
                onChange={(e) => setTier(Number(e.target.value))}
                className="w-full rounded border border-slate-700 bg-slate-900 px-2 py-1 text-slate-200 focus:outline-none"
              >
                <option value={1}>T1 - Binding Statutes / Primary Standards</option>
                <option value={2}>T2 - Administrative Rules / Standards</option>
                <option value={3}>T3 - Standard Internal / Technical</option>
                <option value={4}>T4 - Drafts / Working Papers</option>
                <option value={5}>T5 - Informal / Unverified</option>
              </select>
            </div>
            <div>
              <label className="block text-[10px] font-medium text-slate-400 mb-0.5">
                Jurisdiction
              </label>
              <input
                type="text"
                value={jurisdiction}
                onChange={(e) => setJurisdiction(e.target.value)}
                placeholder="e.g. US, VN, Federal"
                className="w-full rounded border border-slate-700 bg-slate-900 px-2 py-1 text-slate-200 placeholder:text-slate-600 focus:outline-none"
              />
            </div>
          </div>

          <div className="flex items-center justify-end gap-1.5 pt-1">
            <button
              type="button"
              onClick={() => setIsEditing(false)}
              className="rounded-lg border border-slate-700 bg-slate-800 px-2 py-1 text-[10.5px] font-medium text-slate-400 hover:text-white"
            >
              Cancel
            </button>
            <button
              type="button"
              disabled={isApproving}
              onClick={handleSaveAndApprove}
              className="inline-flex items-center gap-1 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white px-2.5 py-1 text-[10.5px] font-semibold transition disabled:opacity-50"
            >
              {isApproving ? (
                <Loader2 className="h-3 w-3 animate-spin" />
              ) : (
                <Check className="h-3 w-3" />
              )}
              <span>Save & Approve</span>
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export const KnowledgeLibrary: React.FC = () => {
  const [searchText, setSearchText] = useState("");
  const [fileTypeFilter, setFileTypeFilter] = useState<string>("all");
  const [domainFacet, setDomainFacet] = useState<DomainFacetId>("all");
  const [isMoreOpen, setIsMoreOpen] = useState(false);
  const [docPendingDelete, setDocPendingDelete] =
    useState<DocumentItem | null>(null);

  const { data: domains = [] } = useDomainsQuery();

  // Reset domainFacet to "all" if active domain was deleted
  useEffect(() => {
    if (
      domainFacet !== "all" &&
      domainFacet !== "review" &&
      domains.length > 0 &&
      !domains.some((d) => d.slug === domainFacet)
    ) {
      setDomainFacet("all");
    }
  }, [domains, domainFacet]);

  // Facet bar inline vs overflow computation
  const STARTER_SLUGS = ["engineering", "economics", "law"];
  const specializedDomains = useMemo(
    () => domains.filter((d) => d.slug !== "general"),
    [domains]
  );

  const { inlineFacets, overflowFacets } = useMemo(() => {
    if (specializedDomains.length <= 4) {
      return {
        inlineFacets: domains,
        overflowFacets: [],
      };
    }

    const isCustomActive =
      domainFacet !== "all" &&
      domainFacet !== "review" &&
      !STARTER_SLUGS.includes(domainFacet) &&
      domainFacet !== "general";

    const starterInline = domains.filter(
      (d) => STARTER_SLUGS.includes(d.slug) || d.slug === "general"
    );
    const customInline = isCustomActive
      ? domains.filter((d) => d.slug === domainFacet)
      : [];

    const inlineSet = new Set([
      ...starterInline.map((d) => d.slug),
      ...customInline.map((d) => d.slug),
    ]);
    const overflow = domains.filter((d) => !inlineSet.has(d.slug));

    return {
      inlineFacets: [...starterInline, ...customInline],
      overflowFacets: overflow,
    };
  }, [domains, specializedDomains.length, domainFacet]);

  const {
    selectedDocumentId,
    selectDocument,
    selectedFilterDocIds,
    toggleFilterDocId,
    clearFilterDocIds,
    setSelectedFilterDocIds,
    activeCitation,
    clearActiveCitation,
  } = useWorkspace();

  const queryClient = useQueryClient();
  const { data, isLoading, isError, error } = useDocuments({
    file_type: fileTypeFilter === "all" ? undefined : fileTypeFilter,
    search: searchText || undefined,
  });

  const rawDocuments = data?.items ?? data?.documents ?? [];
  const anyProcessing = rawDocuments.some((doc) => isDocumentProcessing(doc));

  const { data: stats } = useSystemStats({
    refetchInterval: anyProcessing ? 2000 : 10000,
  });

  const prevProcessingRef = useRef(false);
  useEffect(() => {
    if (prevProcessingRef.current && !anyProcessing) {
      void queryClient.invalidateQueries({ queryKey: ["system-stats"] });
    }
    prevProcessingRef.current = anyProcessing;
  }, [anyProcessing, queryClient]);

  const deleteMutation = useDeleteDocumentMutation();
  const retryMutation = useRetryDocumentMutation();
  const pauseMutation = usePauseIngestionMutation();
  const resumeMutation = useResumeIngestionMutation();
  const approveClassificationMutation = useApproveClassificationMutation();
  const approveBatchMutation = useApproveBatchMutation();

  // Awaiting approval documents
  const awaitingApprovalDocs = useMemo(
    () =>
      rawDocuments.filter(
        (doc) => (doc.status || "").toLowerCase() === "awaiting_approval"
      ),
    [rawDocuments]
  );

  // Filter documents by Domain Facet
  const filteredDocuments = useMemo(() => {
    if (domainFacet === "all") return rawDocuments;
    if (domainFacet === "review") {
      return rawDocuments.filter((doc) => {
        const s = (doc.status || "").toLowerCase();
        return s === "awaiting_approval" || s === "classifying";
      });
    }
    return rawDocuments.filter((doc) => {
      const dk = getDomainKey(doc);
      return dk === domainFacet;
    });
  }, [rawDocuments, domainFacet]);

  const handleConfirmDelete = async () => {
    if (!docPendingDelete) return;
    const targetId = docPendingDelete.id;
    try {
      await deleteMutation.mutateAsync(targetId);
      setSelectedFilterDocIds((prev) => prev.filter((id) => id !== targetId));
      if (selectedDocumentId === targetId) {
        selectDocument(null);
      }
      if (activeCitation?.doc_id === targetId) {
        clearActiveCitation();
      }
    } finally {
      setDocPendingDelete(null);
    }
  };

  const handleApproveDoc = (
    docId: string,
    payload?: ApprovedClassificationRequest
  ) => {
    approveClassificationMutation.mutate({ docId, payload });
  };

  const handleApproveAllRecommended = () => {
    const ids = awaitingApprovalDocs.map((d) => d.id);
    if (ids.length > 0) {
      approveBatchMutation.mutate(ids);
    }
  };

  return (
    <div className="flex h-full flex-col bg-slate-950 border-r border-slate-800/80 select-none">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-slate-800/80 px-3.5 py-3">
        <div className="flex items-center gap-2">
          <div className="flex h-6 w-6 items-center justify-center rounded-md bg-indigo-500/20 text-indigo-400 border border-indigo-500/30">
            <Database className="h-3.5 w-3.5" />
          </div>
          <div>
            <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-200">
              Knowledge Library
            </h2>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {/* Pause / Resume Button */}
          <button
            type="button"
            disabled={pauseMutation.isPending || resumeMutation.isPending}
            onClick={() => {
              if (stats?.ingestion_paused) {
                void resumeMutation.mutate();
              } else {
                void pauseMutation.mutate();
              }
            }}
            title={
              stats?.ingestion_paused
                ? "Ingestion paused. Click to resume."
                : "Pause background ingestion"
            }
            className={`inline-flex items-center gap-1 rounded-md px-2 py-0.5 text-[11px] font-medium transition border ${
              stats?.ingestion_paused
                ? "border-amber-500/40 bg-amber-500/15 text-amber-300 hover:bg-amber-500/25"
                : "border-slate-800 bg-slate-900 text-slate-400 hover:text-slate-200 hover:bg-slate-800"
            }`}
          >
            {stats?.ingestion_paused ? (
              <>
                <Play className="h-3 w-3 text-amber-400 fill-amber-400" />
                <span>Resume</span>
              </>
            ) : (
              <>
                <Pause className="h-3 w-3 text-slate-400" />
                <span>Pause</span>
              </>
            )}
          </button>
          <span className="rounded-md bg-slate-900 border border-slate-800 px-2 py-0.5 text-[11px] font-medium text-slate-400">
            {data?.total ?? rawDocuments.length} docs
          </span>
        </div>
      </div>

      {/* Upload Dropzone */}
      <div className="p-3 border-b border-slate-800/60">
        <UploadDropzone />
      </div>

      {/* Search & Multi-Domain Facet & File-Type Filter Chips */}
      <div className="space-y-2 p-3 border-b border-slate-800/60">
        {/* Search Input */}
        <div className="relative">
          <Search className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-slate-500" />
          <input
            type="text"
            value={searchText}
            onChange={(e) => setSearchText(e.target.value)}
            placeholder="Search documents..."
            className="w-full rounded-lg border border-slate-800 bg-slate-900/90 pl-8 pr-7 py-1.5 text-xs text-slate-100 placeholder:text-slate-500 focus:border-indigo-500 focus:outline-none"
          />
          {searchText && (
            <button
              type="button"
              onClick={() => setSearchText("")}
              className="absolute right-2 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-300"
              aria-label="Clear search"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          )}
        </div>

        {/* Domain Facet Filter Strip */}
        <div className="flex items-center gap-1 overflow-x-auto pb-0.5">
          {/* All Facet */}
          <button
            type="button"
            onClick={() => setDomainFacet("all")}
            className={`inline-flex items-center gap-1 rounded-md px-2 py-0.5 text-[11px] font-medium transition shrink-0 ${
              domainFacet === "all"
                ? "bg-indigo-600 text-white shadow-xs font-semibold"
                : "bg-slate-900 text-slate-400 hover:bg-slate-800 hover:text-slate-200"
            }`}
          >
            <span>🌐</span>
            <span>All</span>
          </button>

          {/* Dynamic Inline Domain Facets */}
          {inlineFacets.map((dom) => {
            const active = domainFacet === dom.slug;
            const meta = resolveDomainMeta(dom.slug, domains);
            return (
              <button
                key={dom.id}
                type="button"
                onClick={() => setDomainFacet(dom.slug)}
                className={`inline-flex items-center gap-1 rounded-md px-2 py-0.5 text-[11px] font-medium transition shrink-0 ${
                  active
                    ? "bg-indigo-600 text-white shadow-xs font-semibold"
                    : "bg-slate-900 text-slate-400 hover:bg-slate-800 hover:text-slate-200"
                }`}
              >
                <span>{meta.icon}</span>
                <span>{meta.shortLabel}</span>
              </button>
            );
          })}

          {/* Overflow Popover (+N More) */}
          {overflowFacets.length > 0 && (
            <div className="relative shrink-0">
              <button
                type="button"
                onClick={() => setIsMoreOpen((v) => !v)}
                className="inline-flex items-center gap-1 rounded-md bg-slate-900 px-2 py-0.5 text-[11px] font-medium text-slate-400 hover:bg-slate-800 hover:text-slate-200 transition"
              >
                <span>+{overflowFacets.length} More</span>
                <ChevronDown className="h-3 w-3" />
              </button>
              {isMoreOpen && (
                <div
                  onClick={(e) => e.stopPropagation()}
                  className="absolute left-0 top-full z-30 mt-1 w-44 rounded-xl border border-slate-700 bg-slate-950 p-1.5 shadow-xl text-xs space-y-0.5"
                >
                  <div className="px-2 py-1 text-[10px] uppercase font-bold text-slate-500 border-b border-slate-800">
                    More Domains
                  </div>
                  {overflowFacets.map((ov) => {
                    const active = domainFacet === ov.slug;
                    const meta = resolveDomainMeta(ov.slug, domains);
                    return (
                      <button
                        key={ov.id}
                        type="button"
                        onClick={() => {
                          setDomainFacet(ov.slug);
                          setIsMoreOpen(false);
                        }}
                        className={`flex w-full items-center justify-between rounded-lg px-2 py-1.5 text-left text-xs transition ${
                          active
                            ? "bg-indigo-600 text-white font-semibold"
                            : "text-slate-300 hover:bg-slate-800 hover:text-white"
                        }`}
                      >
                        <span className="flex items-center gap-1.5 truncate">
                          <span>{meta.icon}</span>
                          <span className="truncate">{meta.name}</span>
                        </span>
                        {active && <Check className="h-3 w-3" />}
                      </button>
                    );
                  })}
                </div>
              )}
            </div>
          )}

          {/* Review Facet */}
          <button
            type="button"
            onClick={() => setDomainFacet("review")}
            className={`inline-flex items-center gap-1 rounded-md px-2 py-0.5 text-[11px] font-medium transition shrink-0 ${
              domainFacet === "review"
                ? "bg-amber-600 text-white shadow-xs font-semibold"
                : "bg-slate-900 text-slate-400 hover:bg-slate-800 hover:text-slate-200"
            }`}
          >
            <span>⏳</span>
            <span>Review</span>
            {awaitingApprovalDocs.length > 0 && (
              <span className="rounded-full bg-amber-500/30 text-amber-200 px-1.5 text-[9.5px] font-bold">
                {awaitingApprovalDocs.length}
              </span>
            )}
          </button>
        </div>

        {/* File Type Filter Chips */}
        <div className="flex flex-wrap items-center gap-1">
          {FILE_TYPE_FILTERS.map((chip) => {
            const active = fileTypeFilter === chip.id;
            return (
              <button
                key={chip.id}
                type="button"
                onClick={() => setFileTypeFilter(chip.id)}
                className={`rounded-md px-2 py-0.5 text-[10.5px] font-medium transition ${
                  active
                    ? "bg-slate-700 text-white shadow-xs"
                    : "bg-slate-900 text-slate-500 hover:bg-slate-800 hover:text-slate-300"
                }`}
              >
                {chip.label}
              </button>
            );
          })}
        </div>

        {selectedFilterDocIds.length > 0 && (
          <div className="flex items-center justify-between rounded-lg bg-indigo-950/50 border border-indigo-500/30 px-2.5 py-1 text-[11px] text-indigo-200">
            <span className="flex items-center gap-1.5">
              <Filter className="h-3 w-3 text-indigo-400" />
              Scoped to {selectedFilterDocIds.length} doc
              {selectedFilterDocIds.length > 1 ? "s" : ""}
            </span>
            <button
              type="button"
              onClick={clearFilterDocIds}
              className="text-indigo-300 hover:text-white underline text-[10px]"
            >
              Reset
            </button>
          </div>
        )}
      </div>

      {/* Sticky Batch Approval Banner */}
      {awaitingApprovalDocs.length > 0 && (
        <div className="sticky top-0 z-10 flex items-center justify-between gap-2 bg-gradient-to-r from-amber-950/90 to-amber-900/80 border-b border-amber-500/40 px-3 py-2 text-xs text-amber-100 backdrop-blur-xs">
          <div className="flex items-center gap-1.5 min-w-0">
            <ShieldCheck className="h-4 w-4 text-amber-400 shrink-0" />
            <span className="truncate font-medium">
              <strong>{awaitingApprovalDocs.length}</strong> document
              {awaitingApprovalDocs.length > 1 ? "s" : ""} awaiting domain approval
            </span>
          </div>
          <button
            type="button"
            disabled={approveBatchMutation.isPending}
            onClick={handleApproveAllRecommended}
            className="inline-flex items-center gap-1 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white px-2.5 py-1 text-[11px] font-semibold transition shrink-0 shadow-xs disabled:opacity-50"
          >
            {approveBatchMutation.isPending ? (
              <Loader2 className="h-3 w-3 animate-spin" />
            ) : (
              <CheckCircle2 className="h-3 w-3" />
            )}
            <span>Approve All Recommended</span>
          </button>
        </div>
      )}

      {/* Document List */}
      <div className="flex-1 overflow-y-auto p-2 space-y-1.5">
        {isLoading ? (
          <div className="flex flex-col items-center justify-center py-12 text-slate-400 gap-2">
            <Loader2 className="h-5 w-5 animate-spin text-indigo-400" />
            <span className="text-xs">Loading knowledge base...</span>
          </div>
        ) : isError ? (
          <div className="rounded-lg border border-rose-500/30 bg-rose-950/30 p-3 text-xs text-rose-200">
            Failed to load documents:{" "}
            {error instanceof Error ? error.message : "Unknown error"}
          </div>
        ) : filteredDocuments.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-12 px-4 text-center">
            <FileText className="h-8 w-8 text-slate-700 mb-2" />
            <p className="text-xs font-medium text-slate-300">
              No documents found
            </p>
            <p className="mt-1 text-[11px] text-slate-500">
              {domainFacet !== "all"
                ? `No documents match facet "${domainFacet}".`
                : "Upload PDF, Word, Excel, or PowerPoint files above to build your Second Brain."}
            </p>
          </div>
        ) : (
          filteredDocuments.map((doc) => {
            const isSelected = selectedDocumentId === doc.id;
            const isScoped = selectedFilterDocIds.includes(doc.id);
            const isCitedDoc = activeCitation?.doc_id === doc.id;
            const isAwaiting = (doc.status || "").toLowerCase() === "awaiting_approval";
            const domainKey = getDomainKey(doc);

            return (
              <div
                key={doc.id}
                onClick={() => selectDocument(doc)}
                className={`group relative flex flex-col rounded-xl border p-2.5 transition cursor-pointer ${
                  isAwaiting
                    ? "border-amber-500/50 bg-amber-950/15 hover:border-amber-400 hover:bg-amber-950/25"
                    : isSelected
                    ? "border-indigo-500/60 bg-indigo-950/30"
                    : isCitedDoc
                    ? "border-amber-500/50 bg-amber-950/20"
                    : "border-slate-800/60 bg-slate-900/40 hover:border-slate-700 hover:bg-slate-900/80"
                }`}
              >
                <div className="flex items-start gap-2.5">
                  {/* Checkbox for scoped RAG filter */}
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      toggleFilterDocId(doc.id);
                    }}
                    title={
                      isScoped
                        ? "Remove from RAG query scope"
                        : "Include in scoped RAG query"
                    }
                    className="mt-0.5 text-slate-400 hover:text-indigo-400 transition"
                  >
                    {isScoped ? (
                      <CheckSquare className="h-4 w-4 text-indigo-400" />
                    ) : (
                      <Square className="h-4 w-4" />
                    )}
                  </button>

                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-1.5">
                      <FileTypeIcon fileType={doc.file_type} />
                      <span
                        className="truncate text-xs font-medium text-slate-100"
                        title={doc.filename}
                      >
                        {doc.filename}
                      </span>
                    </div>

                    <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
                      <StatusBadge doc={doc} />
                      {!isAwaiting && (
                        <>
                          <DomainBadge
                            domainKey={domainKey}
                            subdomain={doc.primary_subdomain_id}
                            domains={domains}
                          />
                          <AuthorityTierBadge tier={doc.authority_tier} />
                        </>
                      )}
                      <EngineBadge doc={doc} />
                      <span className="text-[10px] uppercase text-slate-500 font-mono">
                        {doc.file_type}
                      </span>
                      <span className="text-[10px] text-slate-500">
                        · {formatFileSize(doc.size_bytes)}
                      </span>
                      {doc.page_count ? (
                        <span className="text-[10px] text-slate-500">
                          · {doc.page_count}p
                        </span>
                      ) : null}
                    </div>

                    {/* Inline Awaiting Approval Review Card */}
                    {isAwaiting && (
                      <AwaitingApprovalCard
                        doc={doc}
                        onApprove={handleApproveDoc}
                        isApproving={approveClassificationMutation.isPending}
                      />
                    )}

                    {doc.status === "failed" && (
                      <div
                        className="mt-2 space-y-1.5"
                        onClick={(e) => e.stopPropagation()}
                      >
                        {(doc.error_message || doc.error) && (
                          <div className="rounded-md bg-rose-950/40 border border-rose-500/20 px-2 py-1 text-[10px] text-rose-300">
                            <span className="font-semibold text-rose-200">
                              Failed at stage [{doc.stage || "unknown"}]:{" "}
                            </span>
                            <span className="break-words">
                              {doc.error_message || doc.error}
                            </span>
                          </div>
                        )}
                        <div className="flex flex-wrap items-center gap-1.5 pt-0.5">
                          <button
                            type="button"
                            disabled={retryMutation.isPending}
                            onClick={() => void retryMutation.mutate({ docId: doc.id })}
                            title="Retry with Stage 1 checkpoint if cached"
                            className="inline-flex items-center gap-1 rounded bg-indigo-600 hover:bg-indigo-500 text-white px-2 py-0.5 text-[10px] font-medium transition disabled:opacity-50"
                          >
                            <RotateCcw className="h-2.5 w-2.5" />
                            Retry
                          </button>

                          <button
                            type="button"
                            disabled={retryMutation.isPending}
                            onClick={() =>
                              void retryMutation.mutate({
                                docId: doc.id,
                                options: { preferFastPath: true, forceReparse: true },
                              })
                            }
                            title="Bypass Docling OCR, parse digital text via fast CPU (pymupdf4llm)"
                            className="rounded border border-slate-700 bg-slate-800 hover:bg-slate-700 text-slate-300 px-1.5 py-0.5 text-[10px] font-medium transition disabled:opacity-50"
                          >
                            Fast-Text Retry
                          </button>

                          <button
                            type="button"
                            disabled={retryMutation.isPending}
                            onClick={() =>
                              void retryMutation.mutate({
                                docId: doc.id,
                                options: { preferFastPath: false, forceReparse: true },
                              })
                            }
                            title="Force Docling + TableFormer full vision re-parsing"
                            className="rounded border border-slate-700 bg-slate-800 hover:bg-slate-700 text-slate-300 px-1.5 py-0.5 text-[10px] font-medium transition disabled:opacity-50"
                          >
                            Full Vision Retry
                          </button>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Delete button */}
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      setDocPendingDelete(doc);
                    }}
                    title={`Delete ${doc.filename}`}
                    className="opacity-70 group-hover:opacity-100 rounded-md p-1 text-slate-500 hover:bg-rose-500/15 hover:text-rose-400 transition"
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </button>
                </div>
              </div>
            );
          })
        )}
      </div>

      {/* Footer System Stats */}
      {stats && (
        <div className="border-t border-slate-800/80 bg-slate-900/50 px-3 py-2 text-[11px] text-slate-400 flex items-center justify-between">
          <span>
            Ready:{" "}
            <strong className="text-slate-200">
              {fileTypeFilter === "all" && !searchText && domainFacet === "all"
                ? Math.max(
                    rawDocuments.filter(
                      (d) => (d.status || "").toLowerCase() === "ready"
                    ).length,
                    stats.ready_documents
                  )
                : stats.ready_documents}
            </strong>
            /
            {fileTypeFilter === "all" && !searchText && domainFacet === "all"
              ? Math.max(
                  data?.total ?? rawDocuments.length,
                  stats.total_documents
                )
              : stats.total_documents}
          </span>
          <span>
            Chunks:{" "}
            <strong className="text-slate-200">
              {fileTypeFilter === "all" && !searchText && domainFacet === "all"
                ? Math.max(
                    rawDocuments.reduce(
                      (acc, d) => acc + (d.chunk_count || 0),
                      0
                    ),
                    stats.total_chunks
                  )
                : stats.total_chunks}
            </strong>
          </span>
          <span>
            Vectors:{" "}
            <strong className="text-slate-200">
              {stats.qdrant_vectors_count ?? stats.vector_count}
            </strong>
          </span>
        </div>
      )}

      {/* Delete Confirmation Dialog */}
      <Dialog.Root
        open={Boolean(docPendingDelete)}
        onOpenChange={(open) => {
          if (!open) setDocPendingDelete(null);
        }}
      >
        <Dialog.Portal>
          <Dialog.Overlay className="fixed inset-0 z-40 bg-black/70 backdrop-blur-xs" />
          <Dialog.Content className="fixed left-1/2 top-1/2 z-50 w-full max-w-md -translate-x-1/2 -translate-y-1/2 rounded-2xl border border-slate-800 bg-slate-900 p-5 shadow-2xl">
            <Dialog.Title className="text-sm font-semibold text-slate-100">
              Delete Document Atomically?
            </Dialog.Title>
            <Dialog.Description className="mt-2 text-xs leading-relaxed text-slate-400">
              This will permanently remove{" "}
              <span className="font-medium text-slate-200">
                {docPendingDelete?.filename}
              </span>{" "}
              from Qdrant hybrid vectors, SQLite metadata tables, and cached raw
              / Markdown files on disk.
            </Dialog.Description>

            <div className="mt-5 flex items-center justify-end gap-2">
              <Dialog.Close asChild>
                <button
                  type="button"
                  className="rounded-lg border border-slate-700 bg-slate-800 px-3 py-1.5 text-xs font-medium text-slate-200 hover:bg-slate-700"
                >
                  Cancel
                </button>
              </Dialog.Close>
              <button
                type="button"
                disabled={deleteMutation.isPending}
                onClick={() => void handleConfirmDelete()}
                className="inline-flex items-center gap-1.5 rounded-lg bg-rose-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-rose-500 disabled:opacity-50"
              >
                {deleteMutation.isPending ? (
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                ) : (
                  <Trash2 className="h-3.5 w-3.5" />
                )}
                Delete Document
              </button>
            </div>
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>
    </div>
  );
};
