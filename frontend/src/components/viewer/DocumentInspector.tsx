import React, { useEffect, useMemo, useState } from "react";
import * as Tabs from "@radix-ui/react-tabs";
import * as Dialog from "@radix-ui/react-dialog";
import {
  BookOpen,
  Check,
  CheckCircle2,
  Edit,
  FileCode2,
  FileText,
  Loader2,
  Rocket,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
  X,
  ZoomIn,
  ZoomOut,
} from "lucide-react";
import {
  isDocumentProcessing,
  useApproveClassificationMutation,
  useDocument,
  useRetryDocumentMutation,
  useUpdateDocumentMetadataMutation,
} from "../../hooks/useDocuments";
import { useDomainsQuery } from "../../hooks/useDomains";
import type {
  ApprovedClassificationRequest,
  SectionOverride,
  UpdateDocumentMetadataRequest,
} from "../../lib/types";
import { type InspectorTab, useWorkspace } from "../../lib/workspaceStore";
import {
  AuthorityTierBadge,
  DomainBadge,
  getDomainKey,
  parseClassificationResult,
} from "../library/KnowledgeLibrary";
import { MarkdownViewerPane } from "./MarkdownViewerPane";
import { PdfViewerPane } from "./PdfViewerPane";

export const DocumentInspector: React.FC = () => {
  const { data: domains = [] } = useDomainsQuery();

  const {
    selectedDocumentId,
    selectedDocument,
    inspectorTab,
    setInspectorTab,
    activeCitation,
    clearActiveCitation,
  } = useWorkspace();

  const [scale, setScale] = useState<number>(1.0);
  const [showMetadataDialog, setShowMetadataDialog] = useState(false);
  const [isAdjustingApproval, setIsAdjustingApproval] = useState(false);

  const { data: fetchedDoc } = useDocument(selectedDocumentId);
  const retryMutation = useRetryDocumentMutation();
  const approveMutation = useApproveClassificationMutation();
  const updateMetaMutation = useUpdateDocumentMetadataMutation();

  const activeDoc = fetchedDoc ?? selectedDocument;
  const isPdf =
    activeDoc?.file_type?.toLowerCase() === "pdf" ||
    activeDoc?.filename?.toLowerCase().endsWith(".pdf") ||
    activeCitation?.source_filename?.toLowerCase().endsWith(".pdf");

  const isProcessing = isDocumentProcessing(activeDoc);
  const isUpgrading =
    retryMutation.isPending ||
    (isProcessing && activeDoc?.parsing_engine === "pymupdf4llm");
  const canUpgrade =
    activeDoc?.file_type?.toLowerCase() === "pdf" &&
    activeDoc?.parsing_engine === "pymupdf4llm" &&
    (activeDoc?.status === "ready" || isUpgrading);

  const isAwaitingApproval =
    (activeDoc?.status || "").toLowerCase() === "awaiting_approval";

  const profile = useMemo(
    () => (activeDoc ? parseClassificationResult(activeDoc) : null),
    [activeDoc]
  );

  // Editable fields for adjusting approval or metadata
  const initialDomain = profile?.primary_domain || (activeDoc ? getDomainKey(activeDoc) : "general");
  const initialSubdomain = profile?.primary_subdomain || activeDoc?.primary_subdomain_id || "";
  const initialTier = profile?.authority_tier ?? activeDoc?.authority_tier ?? 3;
  const initialJurisdiction = profile?.jurisdiction ?? activeDoc?.jurisdiction ?? "";
  const initialYear = profile?.effective_year ?? activeDoc?.effective_year ?? undefined;

  const [editDomain, setEditDomain] = useState<string>(initialDomain);
  const [editSubdomain, setEditSubdomain] = useState<string>(initialSubdomain);
  const [editTier, setEditTier] = useState<number>(initialTier);
  const [editJurisdiction, setEditJurisdiction] = useState<string>(initialJurisdiction);
  const [editYear, setEditYear] = useState<number | undefined>(initialYear);
  const [localOverrides, setLocalOverrides] = useState<SectionOverride[]>(
    profile?.section_overrides || []
  );

  // Reset local edit state when active document changes
  useEffect(() => {
    const initDomain =
      profile?.primary_domain || (activeDoc ? getDomainKey(activeDoc) : "general");
    const initSubdomain =
      profile?.primary_subdomain || activeDoc?.primary_subdomain_id || "";
    const initTier = profile?.authority_tier ?? activeDoc?.authority_tier ?? 3;
    const initJurisdiction =
      profile?.jurisdiction ?? activeDoc?.jurisdiction ?? "";
    const initYear =
      profile?.effective_year ?? activeDoc?.effective_year ?? undefined;
    const initOverrides = profile?.section_overrides || [];

    setEditDomain(initDomain);
    setEditSubdomain(initSubdomain);
    setEditTier(initTier);
    setEditJurisdiction(initJurisdiction);
    setEditYear(initYear);
    setLocalOverrides(initOverrides);
  }, [activeDoc?.id, activeDoc?.classification_result, profile]);

  const selectedEditDomainObj = useMemo(
    () => domains.find((d) => d.slug === editDomain),
    [domains, editDomain]
  );

  const zoomOut = () =>
    setScale((s) => Math.max(0.6, Math.round((s - 0.15) * 100) / 100));
  const zoomIn = () =>
    setScale((s) => Math.min(2.0, Math.round((s + 0.15) * 100) / 100));
  const resetZoom = () => setScale(1.0);

  const handleApprove = (payload?: ApprovedClassificationRequest) => {
    if (!activeDoc) return;
    approveMutation.mutate({
      docId: activeDoc.id,
      payload: payload ?? {
        domain: editDomain,
        subdomain: editSubdomain || undefined,
        authority_tier: editTier,
        jurisdiction: editJurisdiction || undefined,
        effective_year: editYear,
        section_overrides: localOverrides.length > 0 ? localOverrides : undefined,
      },
    });
    setIsAdjustingApproval(false);
  };

  const handleSaveMetadata = () => {
    if (!activeDoc) return;
    const req: UpdateDocumentMetadataRequest = {
      domain: editDomain,
      subdomain: editSubdomain || undefined,
      authority_tier: editTier,
      jurisdiction: editJurisdiction || undefined,
      effective_year: editYear,
      section_overrides: localOverrides.length > 0 ? localOverrides : undefined,
    };
    updateMetaMutation.mutate({
      docId: activeDoc.id,
      payload: req,
    });
    setShowMetadataDialog(false);
  };

  const handleSectionOverride = (override: SectionOverride) => {
    setLocalOverrides((prev) => {
      const filtered = prev.filter(
        (o) => o.heading_path.toLowerCase() !== override.heading_path.toLowerCase()
      );
      return [...filtered, override];
    });
  };

  if (!selectedDocumentId) {
    return (
      <div className="flex h-full flex-col items-center justify-center bg-slate-950 p-8 text-center select-none">
        <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-slate-900 border border-slate-800 text-indigo-400 mb-3 shadow-inner">
          <BookOpen className="h-7 w-7" />
        </div>
        <h3 className="text-sm font-semibold text-slate-200">
          Document Inspector
        </h3>
        <p className="mt-1.5 max-w-sm text-xs leading-relaxed text-slate-400">
          Select a document in the <strong>Knowledge Library</strong> on the
          left, or click any inline citation pill <code className="text-indigo-300">[1]</code> in the{" "}
          <strong>Intelligent Assistant</strong> to inspect the exact source
          page or structured Markdown section.
        </p>
      </div>
    );
  }

  const displayFilename =
    activeDoc?.filename ||
    activeCitation?.source_filename ||
    `Document ${selectedDocumentId.slice(0, 8)}`;

  return (
    <Tabs.Root
      value={inspectorTab}
      onValueChange={(val) => setInspectorTab(val as InspectorTab)}
      className="flex h-full flex-col bg-slate-950 overflow-hidden"
    >
      {/* Full-Width Classification Review Bar when document is awaiting_approval */}
      {isAwaitingApproval && activeDoc && (
        <div className="border-b border-amber-500/40 bg-gradient-to-r from-amber-950/80 via-amber-900/60 to-slate-950 px-4 py-3 text-xs text-amber-100">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <div className="flex h-6 w-6 items-center justify-center rounded-md bg-amber-500/20 text-amber-300 border border-amber-500/30">
                <ShieldAlert className="h-4 w-4" />
              </div>
              <div>
                <span className="font-semibold text-amber-200">
                  Classification & Authority Review Required
                </span>
                <span className="ml-2 text-[10.5px] text-amber-300/80">
                  Confidence: {profile?.confidence ? Math.round(profile.confidence * 100) : 80}%
                </span>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => setIsAdjustingApproval((v) => !v)}
                className="inline-flex items-center gap-1 rounded-lg border border-amber-500/40 bg-amber-950/60 px-2.5 py-1 text-[11px] font-medium text-amber-200 hover:bg-amber-900/60 transition"
              >
                <Edit className="h-3 w-3" />
                <span>{isAdjustingApproval ? "Hide Editor" : "Adjust Metadata"}</span>
              </button>

              <button
                type="button"
                disabled={approveMutation.isPending}
                onClick={() => handleApprove()}
                className="inline-flex items-center gap-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white px-3 py-1 text-[11px] font-semibold transition shadow-xs disabled:opacity-50"
              >
                {approveMutation.isPending ? (
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                ) : (
                  <CheckCircle2 className="h-3.5 w-3.5" />
                )}
                <span>Approve Classification</span>
              </button>
            </div>
          </div>

          {/* Quick info row */}
          <div className="mt-2 flex flex-wrap items-center gap-2 text-[11px]">
            <span className="text-slate-400">Recommended:</span>
            <DomainBadge
              domainKey={editDomain}
              subdomain={editSubdomain}
            />
            <AuthorityTierBadge tier={editTier} />
            {editJurisdiction && (
              <span className="rounded bg-slate-900/80 border border-slate-800 px-1.5 py-0.5 text-[10px] text-slate-300 font-mono">
                Jurisdiction: {editJurisdiction}
              </span>
            )}
            {editYear && (
              <span className="rounded bg-slate-900/80 border border-slate-800 px-1.5 py-0.5 text-[10px] text-slate-300 font-mono">
                Year: {editYear}
              </span>
            )}
            {profile?.is_hybrid && (
              <span className="rounded bg-purple-950/60 border border-purple-500/30 px-2 py-0.5 text-[10px] text-purple-300">
                Hybrid: {profile.section_overrides?.length || 0} section overrides
              </span>
            )}
          </div>

          {/* Adjust editor expander */}
          {isAdjustingApproval && (
            <div className="mt-3 rounded-xl border border-amber-500/30 bg-slate-950/80 p-3 space-y-2.5">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-[11px]">
                <div>
                  <label className="block text-[10px] uppercase font-semibold text-slate-400 mb-1">
                    Domain
                  </label>
                  <select
                    value={editDomain}
                    onChange={(e) => {
                      const nextD = e.target.value;
                      setEditDomain(nextD);
                      const found = domains.find((d) => d.slug === nextD);
                      if (found?.subdomains?.length) {
                        setEditSubdomain(found.subdomains[0].slug);
                      }
                    }}
                    className="w-full rounded border border-slate-700 bg-slate-900 px-2 py-1 text-slate-200"
                  >
                    {domains.map((d) => (
                      <option key={d.slug} value={d.slug}>
                        {d.icon} {d.name}
                      </option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="block text-[10px] uppercase font-semibold text-slate-400 mb-1">
                    Subdomain
                  </label>
                  {selectedEditDomainObj && selectedEditDomainObj.subdomains.length > 0 ? (
                    <select
                      value={editSubdomain}
                      onChange={(e) => setEditSubdomain(e.target.value)}
                      className="w-full rounded border border-slate-700 bg-slate-900 px-2 py-1 text-slate-200"
                    >
                      {selectedEditDomainObj.subdomains.map((s) => (
                        <option key={s.slug} value={s.slug}>
                          {s.name} ({s.slug})
                        </option>
                      ))}
                    </select>
                  ) : (
                    <input
                      type="text"
                      value={editSubdomain}
                      onChange={(e) => setEditSubdomain(e.target.value)}
                      placeholder="e.g. electrical, statute"
                      className="w-full rounded border border-slate-700 bg-slate-900 px-2 py-1 text-slate-200"
                    />
                  )}
                </div>
                <div>
                  <label className="block text-[10px] uppercase font-semibold text-slate-400 mb-1">
                    Authority Tier
                  </label>
                  <select
                    value={editTier}
                    onChange={(e) => setEditTier(Number(e.target.value))}
                    className="w-full rounded border border-slate-700 bg-slate-900 px-2 py-1 text-slate-200"
                  >
                    <option value={1}>T1 - Binding Statutes / Primary Standards</option>
                    <option value={2}>T2 - Administrative Directives / Standards</option>
                    <option value={3}>T3 - Standard Internal / Technical</option>
                    <option value={4}>T4 - Drafts / Working Papers</option>
                    <option value={5}>T5 - Informal / Unverified</option>
                  </select>
                </div>
                <div>
                  <label className="block text-[10px] uppercase font-semibold text-slate-400 mb-1">
                    Jurisdiction
                  </label>
                  <input
                    type="text"
                    value={editJurisdiction}
                    onChange={(e) => setEditJurisdiction(e.target.value)}
                    placeholder="e.g. US, VN, Federal"
                    className="w-full rounded border border-slate-700 bg-slate-900 px-2 py-1 text-slate-200"
                  />
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Inspector Header */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800/80 bg-slate-950 px-4 py-2.5">
        <div className="flex items-center gap-2 min-w-0">
          <FileText className="h-4 w-4 text-indigo-400 shrink-0" />
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <h2
                className="truncate text-xs font-semibold text-slate-100"
                title={displayFilename}
              >
                {displayFilename}
              </h2>
              {activeDoc && (
                <DomainBadge
                  domainKey={getDomainKey(activeDoc)}
                  subdomain={activeDoc.primary_subdomain_id}
                />
              )}
              {activeDoc?.authority_tier && (
                <AuthorityTierBadge tier={activeDoc.authority_tier} />
              )}
            </div>
            {activeDoc && (
              <p className="text-[10px] text-slate-400">
                {activeDoc.file_type.toUpperCase()} · {activeDoc.chunk_count}{" "}
                chunks
                {activeDoc.page_count ? ` · ${activeDoc.page_count} pages` : ""}
                {activeDoc.jurisdiction ? ` · ${activeDoc.jurisdiction}` : ""}
              </p>
            )}
          </div>
        </div>

        {/* Controls: Edit Domain/Metadata + Upgrade + Zoom + Tab Switcher */}
        <div className="flex items-center gap-2">
          {activeDoc?.status === "ready" && (
            <button
              type="button"
              onClick={() => {
                setEditDomain(getDomainKey(activeDoc));
                setEditSubdomain(activeDoc.primary_subdomain_id || "");
                setEditTier(activeDoc.authority_tier || 3);
                setEditJurisdiction(activeDoc.jurisdiction || "");
                setEditYear(activeDoc.effective_year || undefined);
                setShowMetadataDialog(true);
              }}
              title="Edit domain taxonomy, authority tier, and jurisdiction"
              className="inline-flex items-center gap-1 rounded-lg border border-slate-800 bg-slate-900 px-2.5 py-1 text-xs font-medium text-slate-300 hover:bg-slate-800 hover:text-white transition"
            >
              <Edit className="h-3 w-3 text-indigo-400" />
              <span>Edit Domain / Metadata</span>
            </button>
          )}

          {canUpgrade && (
            <button
              type="button"
              disabled={isUpgrading}
              onClick={() => {
                if (activeDoc) {
                  retryMutation.mutate({
                    docId: activeDoc.id,
                    options: { forceReparse: true, preferFastPath: false },
                  });
                }
              }}
              title="Re-parse PDF with Docling GPU Vision (OCR & table structure analysis)"
              className="inline-flex items-center gap-1.5 rounded-lg border border-purple-500/30 bg-purple-950/40 hover:bg-purple-900/50 text-purple-200 px-2.5 py-1 text-xs font-medium transition disabled:opacity-60 disabled:cursor-not-allowed"
            >
              {isUpgrading ? (
                <>
                  <Loader2 className="h-3.5 w-3.5 animate-spin text-purple-400" />
                  <span>Upgrading...</span>
                </>
              ) : (
                <>
                  <Rocket className="h-3.5 w-3.5 text-purple-400" />
                  <span>Upgrade to GPU Vision</span>
                </>
              )}
            </button>
          )}

          {/* Zoom Controls */}
          <div className="flex items-center rounded-lg border border-slate-800 bg-slate-900/90 p-0.5 text-xs">
            <button
              type="button"
              onClick={zoomOut}
              className="rounded p-1 text-slate-400 hover:bg-slate-800 hover:text-slate-200"
              title="Zoom out"
              aria-label="Zoom out"
            >
              <ZoomOut className="h-3.5 w-3.5" />
            </button>
            <button
              type="button"
              onClick={resetZoom}
              className="px-1.5 font-mono text-[11px] text-slate-300 hover:text-white"
              title="Reset zoom"
            >
              {Math.round(scale * 100)}%
            </button>
            <button
              type="button"
              onClick={zoomIn}
              className="rounded p-1 text-slate-400 hover:bg-slate-800 hover:text-slate-200"
              title="Zoom in"
              aria-label="Zoom in"
            >
              <ZoomIn className="h-3.5 w-3.5" />
            </button>
          </div>

          {/* Radix Tabs Switcher */}
          <Tabs.List
            aria-label="Document view mode"
            className="inline-flex items-center rounded-lg border border-slate-800 bg-slate-900 p-0.5"
          >
            <Tabs.Trigger
              value="pdf"
              disabled={!isPdf}
              title={
                isPdf
                  ? "View original PDF page"
                  : "PDF view is only available for PDF files"
              }
              className="inline-flex items-center gap-1.5 rounded-md px-2.5 py-1 text-xs font-medium text-slate-400 transition data-[state=active]:bg-indigo-600 data-[state=active]:text-white disabled:opacity-40 disabled:cursor-not-allowed"
            >
              <FileText className="h-3.5 w-3.5" />
              <span>PDF View</span>
            </Tabs.Trigger>
            <Tabs.Trigger
              value="markdown"
              className="inline-flex items-center gap-1.5 rounded-md px-2.5 py-1 text-xs font-medium text-slate-400 transition data-[state=active]:bg-indigo-600 data-[state=active]:text-white"
            >
              <FileCode2 className="h-3.5 w-3.5" />
              <span>Markdown View</span>
            </Tabs.Trigger>
          </Tabs.List>
        </div>
      </div>

      {/* Active Citation Bridge Banner */}
      {activeCitation && activeCitation.doc_id === selectedDocumentId && (
        <div className="flex items-center justify-between gap-2 border-b border-indigo-500/30 bg-indigo-950/40 px-4 py-1.5 text-xs text-indigo-200">
          <div className="flex items-center gap-2 min-w-0">
            <Sparkles className="h-3.5 w-3.5 text-indigo-400 shrink-0" />
            <span className="truncate">
              {activeCitation.citation_index
                ? `Citation [${activeCitation.citation_index}]`
                : "Active Citation"}
              :{" "}
              <strong className="text-white">
                {activeCitation.source_filename}
              </strong>
              {activeCitation.domain && ` · ${activeCitation.domain}`}
              {activeCitation.authority_tier && ` · T${activeCitation.authority_tier}`}
              {activeCitation.page_numbers?.length > 0 &&
                ` · p.${activeCitation.page_numbers.join(", ")}`}
              {activeCitation.heading_path &&
                ` · §${activeCitation.heading_path}`}
            </span>
          </div>
          <button
            type="button"
            onClick={clearActiveCitation}
            className="rounded p-0.5 text-indigo-300 hover:bg-indigo-500/20 hover:text-white"
            title="Dismiss citation highlight"
          >
            <X className="h-3.5 w-3.5" />
          </button>
        </div>
      )}

      {/* Tab Content Panes */}
      <Tabs.Content value="pdf" className="flex-1 overflow-hidden outline-none">
        {isPdf ? (
          <PdfViewerPane
            docId={selectedDocumentId}
            filename={displayFilename}
            scale={scale}
            activeCitation={activeCitation}
          />
        ) : (
          <MarkdownViewerPane
            docId={selectedDocumentId}
            activeDoc={activeDoc}
            activeCitation={activeCitation}
            scale={scale}
            onUpdateSectionOverride={handleSectionOverride}
            sectionOverrides={localOverrides}
          />
        )}
      </Tabs.Content>

      <Tabs.Content
        value="markdown"
        className="flex-1 overflow-hidden outline-none"
      >
        <MarkdownViewerPane
          docId={selectedDocumentId}
          activeDoc={activeDoc}
          activeCitation={activeCitation}
          scale={scale}
          onUpdateSectionOverride={handleSectionOverride}
          sectionOverrides={localOverrides}
        />
      </Tabs.Content>

      {/* Edit Domain / Metadata Dialog */}
      <Dialog.Root
        open={showMetadataDialog}
        onOpenChange={(open) => setShowMetadataDialog(open)}
      >
        <Dialog.Portal>
          <Dialog.Overlay className="fixed inset-0 z-40 bg-black/70 backdrop-blur-xs" />
          <Dialog.Content className="fixed left-1/2 top-1/2 z-50 w-full max-w-md -translate-x-1/2 -translate-y-1/2 rounded-2xl border border-slate-800 bg-slate-900 p-5 shadow-2xl">
            <Dialog.Title className="text-sm font-semibold text-slate-100 flex items-center gap-2">
              <ShieldCheck className="h-4 w-4 text-indigo-400" />
              <span>Edit Document Domain & Metadata</span>
            </Dialog.Title>
            <Dialog.Description className="mt-1 text-xs text-slate-400">
              Update domain classification, authority tier, and jurisdiction for{" "}
              <strong className="text-slate-200">{activeDoc?.filename}</strong>.
            </Dialog.Description>

            <div className="mt-4 space-y-3 text-xs">
              <div>
                <label className="block text-[11px] font-medium text-slate-300 mb-1">
                  Primary Domain
                </label>
                <select
                  value={editDomain}
                  onChange={(e) => {
                    const nextD = e.target.value;
                    setEditDomain(nextD);
                    const found = domains.find((d) => d.slug === nextD);
                    if (found?.subdomains?.length) {
                      setEditSubdomain(found.subdomains[0].slug);
                    }
                  }}
                  className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-1.5 text-slate-200 focus:border-indigo-500 focus:outline-none"
                >
                  {domains.map((d) => (
                    <option key={d.slug} value={d.slug}>
                      {d.icon} {d.name} ({d.epistemic_type})
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-[11px] font-medium text-slate-300 mb-1">
                  Subdomain
                </label>
                {selectedEditDomainObj && selectedEditDomainObj.subdomains.length > 0 ? (
                  <select
                    value={editSubdomain}
                    onChange={(e) => setEditSubdomain(e.target.value)}
                    className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-1.5 text-slate-200 focus:border-indigo-500 focus:outline-none"
                  >
                    {selectedEditDomainObj.subdomains.map((s) => (
                      <option key={s.slug} value={s.slug}>
                        {s.name} ({s.slug})
                      </option>
                    ))}
                  </select>
                ) : (
                  <input
                    type="text"
                    value={editSubdomain}
                    onChange={(e) => setEditSubdomain(e.target.value)}
                    placeholder="e.g. electrical, finance, regulation"
                    className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-1.5 text-slate-200 focus:border-indigo-500 focus:outline-none"
                  />
                )}
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-[11px] font-medium text-slate-300 mb-1">
                    Authority Tier
                  </label>
                  <select
                    value={editTier}
                    onChange={(e) => setEditTier(Number(e.target.value))}
                    className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-1.5 text-slate-200 focus:border-indigo-500 focus:outline-none"
                  >
                    <option value={1}>T1 - Binding Statutes / Primary Standards</option>
                    <option value={2}>T2 - Administrative Rules / Standard Bodies</option>
                    <option value={3}>T3 - Standard Internal / Technical</option>
                    <option value={4}>T4 - Drafts / Working Papers</option>
                    <option value={5}>T5 - Informal / Unverified</option>
                  </select>
                </div>

                <div>
                  <label className="block text-[11px] font-medium text-slate-300 mb-1">
                    Jurisdiction / Standard Body
                  </label>
                  <input
                    type="text"
                    value={editJurisdiction}
                    onChange={(e) => setEditJurisdiction(e.target.value)}
                    placeholder="e.g. US, VN, IEEE, EPA"
                    className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-1.5 text-slate-200 focus:border-indigo-500 focus:outline-none"
                  />
                </div>
              </div>

              <div>
                <label className="block text-[11px] font-medium text-slate-300 mb-1">
                  Effective Year
                </label>
                <input
                  type="number"
                  value={editYear ?? ""}
                  onChange={(e) =>
                    setEditYear(e.target.value ? Number(e.target.value) : undefined)
                  }
                  placeholder="e.g. 2024"
                  className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-1.5 text-slate-200 focus:border-indigo-500 focus:outline-none"
                />
              </div>
            </div>

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
                disabled={updateMetaMutation.isPending}
                onClick={handleSaveMetadata}
                className="inline-flex items-center gap-1.5 rounded-lg bg-indigo-600 px-3 py-1.5 text-xs font-semibold text-white hover:bg-indigo-500 disabled:opacity-50"
              >
                {updateMetaMutation.isPending ? (
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                ) : (
                  <Check className="h-3.5 w-3.5" />
                )}
                Save Metadata
              </button>
            </div>
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>
    </Tabs.Root>
  );
};
