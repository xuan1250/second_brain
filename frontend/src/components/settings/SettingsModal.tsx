import React, { useEffect, useMemo, useRef, useState } from "react";
import * as Dialog from "@radix-ui/react-dialog";
import * as Tabs from "@radix-ui/react-tabs";
import {
  AlertCircle,
  AlertTriangle,
  BookOpen,
  Check,
  ChevronDown,
  ChevronRight,
  Eye,
  EyeOff,
  Key,
  Layers,
  Loader2,
  Lock,
  Plus,
  Search,
  Settings,
  ShieldCheck,
  Sliders,
  Trash2,
  X,
} from "lucide-react";
import {
  useCreateDomainMutation,
  useCreateSubdomainMutation,
  useDeleteDomainMutation,
  useDeleteLexiconMutation,
  useDeleteSubdomainMutation,
  useDomainsQuery,
  useLexiconQuery,
  useUpdateDomainMutation,
  useUpdateWorkspaceSettingsMutation,
  useUpsertLexiconMutation,
  useVerifyLlmKeyMutation,
  useWorkspaceSettingsQuery,
} from "../../hooks/useDomains";
import {
  ALLOWED_DOMAIN_COLORS,
  CHUNKING_STRATEGIES,
  DOMAIN_COLOR_PALETTE,
  EPISTEMIC_TYPES,
  resolveDomainMeta,
} from "../../lib/domainTheme";
import type {
  ChunkingStrategy,
  DomainColorToken,
  DomainCreatePayload,
  DomainItem,
  DomainUpdatePayload,
  EpistemicType,
  LexiconCreatePayload,
  LexiconItem,
  SubdomainCreatePayload,
  SubdomainItem,
} from "../../lib/types";
import { useWorkspace } from "../../lib/workspaceStore";

export interface SettingsModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  initialTab?: "domains" | "lexicon" | "ingestion";
}

export const SettingsModal: React.FC<SettingsModalProps> = ({
  open,
  onOpenChange,
  initialTab = "domains",
}) => {
  // Unconditional top-level hooks (AGENTS.md React Hook Ordering Invariant)
  const { data: domains = [], isLoading: isLoadingDomains } = useDomainsQuery(true);
  const { data: wsSettings } = useWorkspaceSettingsQuery();
  const { pruneDeletedDomain, setSelectedProvider, setSelectedModel } =
    useWorkspace();

  const createDomainMutation = useCreateDomainMutation();
  const updateDomainMutation = useUpdateDomainMutation();
  const deleteDomainMutation = useDeleteDomainMutation();
  const createSubdomainMutation = useCreateSubdomainMutation();
  const deleteSubdomainMutation = useDeleteSubdomainMutation();
  const upsertLexiconMutation = useUpsertLexiconMutation();
  const deleteLexiconMutation = useDeleteLexiconMutation();
  const updateSettingsMutation = useUpdateWorkspaceSettingsMutation();
  const verifyKeyMutation = useVerifyLlmKeyMutation();

  // Active Tab
  const [activeTab, setActiveTab] = useState<string>(initialTab);

  // Tab 1: Domains & Subdomains State
  const [editingDomain, setEditingDomain] = useState<DomainItem | null>(null);
  const [isCreatingDomain, setIsCreatingDomain] = useState<boolean>(false);
  const [expandedDomainId, setExpandedDomainId] = useState<string | null>(null);

  // Domain form inputs
  const [domainName, setDomainName] = useState("");
  const [domainSlug, setDomainSlug] = useState("");
  const [domainDescription, setDomainDescription] = useState("");
  const [domainIcon, setDomainIcon] = useState("📁");
  const [domainColor, setDomainColor] = useState<DomainColorToken>("cyan");
  const [domainEpistemic, setDomainEpistemic] = useState<EpistemicType>("empirical");
  const [domainPriority, setDomainPriority] = useState<number>(3);
  const [domainChunking, setDomainChunking] = useState<ChunkingStrategy>("standard_markdown");
  const [domainKeywords, setDomainKeywords] = useState("");
  const [domainReindex, setDomainReindex] = useState(false);
  const [domainFormError, setDomainFormError] = useState<string | null>(null);

  // Delete Domain Dialog State
  const [deletingDomain, setDeletingDomain] = useState<DomainItem | null>(null);
  const [targetDomainSlug, setTargetDomainSlug] = useState("general");
  const [targetSubdomainSlug, setTargetSubdomainSlug] = useState<string | null>(null);
  const [reindexOnDelete, setReindexOnDelete] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  // Subdomain Management State
  const [subdomainDomain, setSubdomainDomain] = useState<DomainItem | null>(null);
  const [subdomainSlugInput, setSubdomainSlugInput] = useState("");
  const [subdomainNameInput, setSubdomainNameInput] = useState("");
  const [subdomainStandardBodies, setSubdomainStandardBodies] = useState("");
  const [subdomainFormError, setSubdomainFormError] = useState<string | null>(null);

  // Delete Subdomain Dialog State
  const [deletingSubdomain, setDeletingSubdomain] = useState<{
    domain: DomainItem;
    subdomain: SubdomainItem;
  } | null>(null);
  const [targetSubdomainOnSubDelete, setTargetSubdomainOnSubDelete] = useState<string>("");

  // Tab 2: Polysemy Lexicon State
  const [lexiconFilterDomain, setLexiconFilterDomain] = useState<string>("all");
  const [lexiconSearch, setLexiconSearch] = useState<string>("");
  const { data: lexiconEntries = [], isLoading: isLoadingLexicon } = useLexiconQuery({
    domain_id: lexiconFilterDomain !== "all" ? lexiconFilterDomain : undefined,
    search: lexiconSearch.trim() || undefined,
  });

  const [editingLexicon, setEditingLexicon] = useState<LexiconItem | null>(null);
  const [lexiconTerm, setLexiconTerm] = useState("");
  const [lexiconDomainId, setLexiconDomainId] = useState("");
  const [lexiconSense, setLexiconSense] = useState("");
  const [lexiconWarning, setLexiconWarning] = useState("");
  const [lexiconFormError, setLexiconFormError] = useState<string | null>(null);

  // Tab 3: Ingestion & AI Defaults State
  const [uploadMode, setUploadMode] = useState<"auto" | "gpu" | "fast">("fast");
  const [approvalMode, setApprovalMode] = useState<
    "review_all" | "auto_high_confidence" | "auto_all"
  >("review_all");
  const [confidenceThreshold, setConfidenceThreshold] = useState<number>(0.85);
  const [llmProvider, setLlmProvider] = useState<string>("");
  const [llmModel, setLlmModel] = useState<string>("");
  const [compatBaseUrl, setCompatBaseUrl] = useState<string>("");
  const [apiKeyDrafts, setApiKeyDrafts] = useState<Record<string, string>>({});
  const [revealedKeys, setRevealedKeys] = useState<Record<string, boolean>>({});
  const [verifyingProvider, setVerifyingProvider] = useState<string | null>(null);
  const [verifyFeedback, setVerifyFeedback] = useState<
    Record<string, { valid: boolean; message: string }>
  >({});
  const [settingsSuccessMsg, setSettingsSuccessMsg] = useState<string | null>(null);

  const prevOpenRef = useRef(false);
  const hasHydratedRef = useRef(false);

  // Switch tab when modal opens or initialTab changes
  useEffect(() => {
    if (open && initialTab) {
      setActiveTab(initialTab);
    }
  }, [open, initialTab]);

  // Form Hydration Guard: Only reset settings when modal opens or first loads while open
  useEffect(() => {
    const justOpened = open && !prevOpenRef.current;
    prevOpenRef.current = open;

    if (!open) {
      hasHydratedRef.current = false;
      return;
    }

    if (wsSettings && (justOpened || !hasHydratedRef.current)) {
      setUploadMode(
        (wsSettings.default_upload_mode as "auto" | "gpu" | "fast") || "fast"
      );
      setApprovalMode(
        (wsSettings.default_approval_mode as
          | "review_all"
          | "auto_high_confidence"
          | "auto_all") || "review_all"
      );
      setConfidenceThreshold(wsSettings.high_confidence_threshold ?? 0.85);
      setLlmProvider(wsSettings.default_llm_provider || "");
      setLlmModel(wsSettings.default_llm_model || "");
      setCompatBaseUrl(wsSettings.llm_compat_base_url || "");
      hasHydratedRef.current = true;
    }
  }, [open, wsSettings]);

  // Active default provider and validation status for Tab 3 alert banner
  const activeDefaultProv =
    llmProvider || wsSettings?.default_llm_provider || "openai";
  const matchingOption = (wsSettings?.available_providers ?? []).find(
    (p) => p.provider === activeDefaultProv
  );
  const isMissingActiveKey = Boolean(
    matchingOption && !matchingOption.configured
  );
  const isInvalidActiveKey = Boolean(
    matchingOption && matchingOption.validation_status === "invalid"
  );

  // Active top-level domain count
  const activeDomainsCount = useMemo(
    () => domains.filter((d) => d.is_active).length,
    [domains]
  );

  // Available target domains for deletion reassignment
  const availableTargetDomains = useMemo(() => {
    if (!deletingDomain) return domains;
    return domains.filter((d) => d.slug !== deletingDomain.slug);
  }, [domains, deletingDomain]);

  // Target domain object and subdomains for deletion reassignment
  const targetDomainObj = useMemo(() => {
    return (
      domains.find((d) => d.slug === targetDomainSlug) ||
      domains.find((d) => d.slug === "general") ||
      domains[0]
    );
  }, [domains, targetDomainSlug]);

  // Auto-slug generator on create
  const handleDomainNameChange = (name: string) => {
    setDomainName(name);
    if (isCreatingDomain) {
      const generatedSlug = name
        .toLowerCase()
        .replace(/[^a-z0-9]/g, "-")
        .replace(/^-+|-+$/g, "")
        .slice(0, 32);
      setDomainSlug(generatedSlug);
    }
  };

  const openCreateDomain = () => {
    setEditingDomain(null);
    setIsCreatingDomain(true);
    setDomainName("");
    setDomainSlug("");
    setDomainDescription("");
    setDomainIcon("📁");
    setDomainColor("cyan");
    setDomainEpistemic("empirical");
    setDomainPriority(3);
    setDomainChunking("standard_markdown");
    setDomainKeywords("");
    setDomainReindex(false);
    setDomainFormError(null);
  };

  const openEditDomain = (d: DomainItem) => {
    setIsCreatingDomain(false);
    setEditingDomain(d);
    setDomainName(d.name);
    setDomainSlug(d.slug);
    setDomainDescription(d.description || "");
    setDomainIcon(d.icon || "📁");
    setDomainColor((d.color_token in DOMAIN_COLOR_PALETTE ? d.color_token : "slate") as DomainColorToken);
    setDomainEpistemic((d.epistemic_type as EpistemicType) || "general");
    setDomainPriority(d.authority_priority ?? 3);
    setDomainChunking((d.chunking_strategy as ChunkingStrategy) || "standard_markdown");
    setDomainKeywords(d.classifier_keywords || "");
    setDomainReindex(false);
    setDomainFormError(null);
  };

  const handleSaveDomain = async (e: React.FormEvent) => {
    e.preventDefault();
    setDomainFormError(null);

    if (!domainName.trim()) {
      setDomainFormError("Domain name is required.");
      return;
    }
    if (domainDescription.trim().length < 10) {
      setDomainFormError("Description must be at least 10 characters.");
      return;
    }

    try {
      if (isCreatingDomain) {
        if (!domainSlug.trim()) {
          setDomainFormError("Slug is required.");
          return;
        }
        const payload: DomainCreatePayload = {
          name: domainName.trim(),
          slug: domainSlug.trim().toLowerCase(),
          description: domainDescription.trim(),
          icon: domainIcon.trim() || "📁",
          color_token: domainColor,
          epistemic_type: domainEpistemic,
          authority_priority: domainPriority,
          chunking_strategy: domainChunking,
          classifier_keywords: domainKeywords.trim() || null,
        };
        await createDomainMutation.mutateAsync(payload);
      } else if (editingDomain) {
        const payload: DomainUpdatePayload = {
          name: editingDomain.is_system ? undefined : domainName.trim(),
          description: domainDescription.trim(),
          icon: domainIcon.trim() || "📁",
          color_token: domainColor,
          epistemic_type: editingDomain.is_system ? undefined : domainEpistemic,
          authority_priority: editingDomain.is_system ? undefined : domainPriority,
          chunking_strategy: editingDomain.is_system ? undefined : domainChunking,
          classifier_keywords: domainKeywords.trim() || null,
          re_index: domainReindex,
        };
        await updateDomainMutation.mutateAsync({
          slug: editingDomain.slug,
          payload,
        });
      }
      setIsCreatingDomain(false);
      setEditingDomain(null);
    } catch (err: unknown) {
      setDomainFormError(err instanceof Error ? err.message : "Failed to save domain.");
    }
  };

  const openDeleteDomain = (d: DomainItem) => {
    setDeletingDomain(d);
    setTargetDomainSlug(d.slug === "general" ? "engineering" : "general");
    setTargetSubdomainSlug(null);
    setReindexOnDelete(false);
    setDeleteError(null);
  };

  const handleConfirmDeleteDomain = async () => {
    if (!deletingDomain) return;
    setDeleteError(null);

    try {
      await deleteDomainMutation.mutateAsync({
        slug: deletingDomain.slug,
        params: {
          target_domain: targetDomainSlug,
          target_subdomain: targetSubdomainSlug || undefined,
          re_index: reindexOnDelete,
        },
      });
      pruneDeletedDomain(deletingDomain.slug);
      setDeletingDomain(null);
    } catch (err: unknown) {
      setDeleteError(err instanceof Error ? err.message : "Failed to delete domain.");
    }
  };

  const handleCreateSubdomain = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!subdomainDomain) return;
    setSubdomainFormError(null);

    if (!subdomainNameInput.trim()) {
      setSubdomainFormError("Subdomain name is required.");
      return;
    }
    if (!subdomainSlugInput.trim()) {
      setSubdomainFormError("Subdomain slug is required.");
      return;
    }

    try {
      const payload: SubdomainCreatePayload = {
        slug: subdomainSlugInput.trim().toLowerCase(),
        name: subdomainNameInput.trim(),
        standard_bodies: subdomainStandardBodies.trim() || null,
      };
      await createSubdomainMutation.mutateAsync({
        domainSlug: subdomainDomain.slug,
        payload,
      });
      setSubdomainSlugInput("");
      setSubdomainNameInput("");
      setSubdomainStandardBodies("");
    } catch (err: unknown) {
      setSubdomainFormError(err instanceof Error ? err.message : "Failed to create subdomain.");
    }
  };

  const handleConfirmDeleteSubdomain = async () => {
    if (!deletingSubdomain) return;
    try {
      await deleteSubdomainMutation.mutateAsync({
        domainSlug: deletingSubdomain.domain.slug,
        subSlug: deletingSubdomain.subdomain.slug,
        targetSubdomain: targetSubdomainOnSubDelete || undefined,
      });
      setDeletingSubdomain(null);
    } catch (err: unknown) {
      alert(err instanceof Error ? err.message : "Failed to delete subdomain.");
    }
  };

  // Lexicon Handlers
  const openCreateLexicon = () => {
    setEditingLexicon(null);
    setLexiconTerm("");
    setLexiconDomainId(domains[0]?.id || "");
    setLexiconSense("");
    setLexiconWarning("");
    setLexiconFormError(null);
  };

  const openEditLexicon = (item: LexiconItem) => {
    setEditingLexicon(item);
    setLexiconTerm(item.term);
    setLexiconDomainId(item.domain_id);
    setLexiconSense(item.specific_sense);
    setLexiconWarning(item.contrastive_warning);
    setLexiconFormError(null);
  };

  const handleSaveLexicon = async (e: React.FormEvent) => {
    e.preventDefault();
    setLexiconFormError(null);

    if (!lexiconTerm.trim() || !lexiconDomainId || !lexiconSense.trim() || !lexiconWarning.trim()) {
      setLexiconFormError("All lexicon fields are required.");
      return;
    }

    try {
      const payload: LexiconCreatePayload = {
        term: lexiconTerm.trim(),
        domain_id: lexiconDomainId,
        specific_sense: lexiconSense.trim(),
        contrastive_warning: lexiconWarning.trim(),
      };
      await upsertLexiconMutation.mutateAsync(payload);
      setEditingLexicon(null);
      setLexiconTerm("");
      setLexiconSense("");
      setLexiconWarning("");
    } catch (err: unknown) {
      setLexiconFormError(err instanceof Error ? err.message : "Failed to save lexicon term.");
    }
  };

  const handleDeleteLexicon = async (id: string) => {
    if (!confirm("Delete this lexicon term definition?")) return;
    try {
      await deleteLexiconMutation.mutateAsync(id);
    } catch (err: unknown) {
      alert(err instanceof Error ? err.message : "Failed to delete lexicon term.");
    }
  };

  // Settings Handlers
  const handleVerifyKey = async (provId: string) => {
    setVerifyingProvider(provId);
    setVerifyFeedback((prev) => {
      const next = { ...prev };
      delete next[provId];
      return next;
    });
    try {
      const draftKey = apiKeyDrafts[provId]?.trim() || null;
      const isDefaultProv =
        provId === (llmProvider || wsSettings?.default_llm_provider || "openai");
      const modelParam = isDefaultProv
        ? (llmModel.trim() || undefined)
        : undefined;
      const baseUrlParam =
        provId === "openai_compatible" ? (compatBaseUrl.trim() || null) : null;

      const res = await verifyKeyMutation.mutateAsync({
        provider: provId,
        api_key: draftKey,
        model: modelParam,
        base_url: baseUrlParam,
        save_if_valid: true,
      });

      if (res.valid) {
        setVerifyFeedback((prev) => ({
          ...prev,
          [provId]: {
            valid: true,
            message: "✓ Key verified & saved successfully",
          },
        }));
        setApiKeyDrafts((prev) => {
          const next = { ...prev };
          delete next[provId];
          return next;
        });
      } else {
        setVerifyFeedback((prev) => ({
          ...prev,
          [provId]: {
            valid: false,
            message: res.validation_error || "Verification failed",
          },
        }));
      }
    } catch (err: unknown) {
      setVerifyFeedback((prev) => ({
        ...prev,
        [provId]: {
          valid: false,
          message:
            err instanceof Error ? err.message : "Verification error",
        },
      }));
    } finally {
      setVerifyingProvider(null);
    }
  };

  const handleClearUiKey = async (provId: string) => {
    try {
      await updateSettingsMutation.mutateAsync({
        api_keys: { [provId]: "" },
      });
      setApiKeyDrafts((prev) => {
        const next = { ...prev };
        delete next[provId];
        return next;
      });
      setVerifyFeedback((prev) => ({
        ...prev,
        [provId]: {
          valid: true,
          message: "UI key cleared; reverted to .env fallback.",
        },
      }));
    } catch (err: unknown) {
      setVerifyFeedback((prev) => ({
        ...prev,
        [provId]: {
          valid: false,
          message:
            err instanceof Error ? err.message : "Failed to clear UI key",
        },
      }));
    }
  };

  const handleSaveSettings = async (e: React.FormEvent) => {
    e.preventDefault();
    setSettingsSuccessMsg(null);
    try {
      const apiKeysPayload: Record<string, string | null> = {};
      for (const [p, k] of Object.entries(apiKeyDrafts)) {
        if (k && k.trim()) {
          apiKeysPayload[p] = k.trim();
        }
      }

      const updated = await updateSettingsMutation.mutateAsync({
        default_upload_mode: uploadMode,
        default_approval_mode: approvalMode,
        high_confidence_threshold: confidenceThreshold,
        default_llm_provider: llmProvider || null,
        default_llm_model: llmModel || null,
        llm_compat_base_url: compatBaseUrl.trim() || null,
        ...(Object.keys(apiKeysPayload).length > 0
          ? { api_keys: apiKeysPayload }
          : {}),
      });
      setSelectedProvider(updated.default_llm_provider || "");
      setSelectedModel(updated.default_llm_model || "");
      setApiKeyDrafts({});
      setSettingsSuccessMsg("Workspace defaults saved successfully.");
      setTimeout(() => setSettingsSuccessMsg(null), 3000);
    } catch (err: unknown) {
      alert(
        err instanceof Error
          ? err.message
          : "Failed to update workspace settings."
      );
    }
  };

  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-50 bg-black/75 backdrop-blur-xs transition-opacity animate-in fade-in" />
        <Dialog.Content className="fixed left-1/2 top-1/2 z-50 w-full max-w-4xl -translate-x-1/2 -translate-y-1/2 rounded-2xl border border-slate-800 bg-slate-950 p-6 shadow-2xl focus:outline-none max-h-[90vh] flex flex-col">
          {/* Header */}
          <div className="flex items-center justify-between border-b border-slate-800/80 pb-4">
            <div className="flex items-center gap-2.5">
              <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-indigo-600/20 border border-indigo-500/30 text-indigo-400">
                <Settings className="h-5 w-5" />
              </div>
              <div>
                <Dialog.Title className="text-base font-bold text-white tracking-wide">
                  Workspace Settings & Dynamic Taxonomy
                </Dialog.Title>
                <Dialog.Description className="text-xs text-slate-400">
                  Manage domains, subdomains, polysemy definitions, and ingestion AI defaults.
                </Dialog.Description>
              </div>
            </div>
            <Dialog.Close asChild>
              <button
                type="button"
                className="rounded-lg p-1.5 text-slate-400 hover:bg-slate-800 hover:text-white transition"
              >
                <X className="h-4 w-4" />
              </button>
            </Dialog.Close>
          </div>

          {/* Radix Tabs Navigation */}
          <Tabs.Root
            value={activeTab}
            onValueChange={setActiveTab}
            className="flex-1 flex flex-col overflow-hidden pt-4"
          >
            <Tabs.List className="flex items-center gap-2 border-b border-slate-800/80 pb-2 text-xs font-medium">
              <Tabs.Trigger
                value="domains"
                className={`flex items-center gap-2 rounded-lg px-3 py-1.5 transition ${
                  activeTab === "domains"
                    ? "bg-indigo-600 text-white font-semibold shadow-xs"
                    : "text-slate-400 hover:bg-slate-900 hover:text-slate-200"
                }`}
              >
                <Layers className="h-3.5 w-3.5" />
                <span>Domains & Subdomains ({domains.length})</span>
              </Tabs.Trigger>

              <Tabs.Trigger
                value="lexicon"
                className={`flex items-center gap-2 rounded-lg px-3 py-1.5 transition ${
                  activeTab === "lexicon"
                    ? "bg-indigo-600 text-white font-semibold shadow-xs"
                    : "text-slate-400 hover:bg-slate-900 hover:text-slate-200"
                }`}
              >
                <BookOpen className="h-3.5 w-3.5" />
                <span>Polysemy Lexicon ({lexiconEntries.length})</span>
              </Tabs.Trigger>

              <Tabs.Trigger
                value="ingestion"
                className={`relative flex items-center gap-2 rounded-lg px-3 py-1.5 transition ${
                  activeTab === "ingestion"
                    ? "bg-indigo-600 text-white font-semibold shadow-xs"
                    : "text-slate-400 hover:bg-slate-900 hover:text-slate-200"
                }`}
              >
                <Sliders className="h-3.5 w-3.5" />
                <span>Ingestion & AI Defaults</span>
                {(isMissingActiveKey || isInvalidActiveKey) && (
                  <span
                    className={`h-2 w-2 rounded-full ${
                      isInvalidActiveKey ? "bg-rose-500 animate-pulse" : "bg-amber-400"
                    }`}
                  />
                )}
              </Tabs.Trigger>
            </Tabs.List>

            {/* TAB 1: DOMAINS & SUBDOMAINS */}
            <Tabs.Content
              value="domains"
              className="flex-1 overflow-y-auto pt-4 space-y-4 pr-1"
            >
              {/* Form Modal / Panel (Create or Edit) */}
              {(isCreatingDomain || editingDomain) && (
                <div className="rounded-xl border border-indigo-500/40 bg-slate-900/90 p-4 space-y-3">
                  <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                    <h4 className="text-xs font-bold text-white flex items-center gap-1.5">
                      {isCreatingDomain ? (
                        <>
                          <Plus className="h-3.5 w-3.5 text-indigo-400" />
                          <span>Create New Custom Domain</span>
                        </>
                      ) : (
                        <>
                          <Settings className="h-3.5 w-3.5 text-indigo-400" />
                          <span>Edit Domain: {editingDomain?.name}</span>
                          {editingDomain?.is_system && (
                            <span className="rounded bg-amber-500/20 text-amber-300 text-[10px] px-1.5 py-0.5 border border-amber-500/30">
                              System Domain (Protected Invariants)
                            </span>
                          )}
                        </>
                      )}
                    </h4>
                    <button
                      type="button"
                      onClick={() => {
                        setIsCreatingDomain(false);
                        setEditingDomain(null);
                      }}
                      className="text-slate-400 hover:text-white"
                    >
                      <X className="h-4 w-4" />
                    </button>
                  </div>

                  {domainFormError && (
                    <div className="rounded-lg bg-rose-950/40 border border-rose-500/40 p-2.5 text-xs text-rose-300 flex items-center gap-2">
                      <AlertCircle className="h-4 w-4 shrink-0" />
                      <span>{domainFormError}</span>
                    </div>
                  )}

                  <form onSubmit={handleSaveDomain} className="space-y-3 text-xs">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                      <div>
                        <label className="block text-[11px] font-semibold text-slate-300 mb-1">
                          Domain Name *
                        </label>
                        <input
                          type="text"
                          disabled={editingDomain?.is_system}
                          value={domainName}
                          onChange={(e) => handleDomainNameChange(e.target.value)}
                          placeholder="e.g. Healthcare, Geology"
                          className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-1.5 text-slate-200 disabled:opacity-50"
                        />
                      </div>

                      <div>
                        <label className="block text-[11px] font-semibold text-slate-300 mb-1">
                          Slug (Identifier) *
                        </label>
                        <input
                          type="text"
                          disabled={!isCreatingDomain}
                          value={domainSlug}
                          onChange={(e) => setDomainSlug(e.target.value.toLowerCase())}
                          placeholder="e.g. healthcare"
                          className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-1.5 text-slate-200 font-mono text-[11px] disabled:opacity-50"
                        />
                      </div>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                      <div>
                        <label className="block text-[11px] font-semibold text-slate-300 mb-1">
                          Icon (Emoji)
                        </label>
                        <input
                          type="text"
                          value={domainIcon}
                          onChange={(e) => setDomainIcon(e.target.value)}
                          maxLength={5}
                          className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-1.5 text-slate-200 text-center"
                        />
                      </div>

                      <div>
                        <label className="block text-[11px] font-semibold text-slate-300 mb-1">
                          Color Palette Swatch
                        </label>
                        <div className="flex flex-wrap items-center gap-1.5 pt-1">
                          {ALLOWED_DOMAIN_COLORS.map((c) => (
                            <button
                              key={c}
                              type="button"
                              onClick={() => setDomainColor(c)}
                              title={c}
                              className={`h-5 w-5 rounded-full ${DOMAIN_COLOR_PALETTE[c].swatchBg} transition ${
                                domainColor === c
                                  ? "ring-2 ring-white ring-offset-2 ring-offset-slate-950 scale-110"
                                  : "opacity-75 hover:opacity-100"
                              }`}
                            />
                          ))}
                        </div>
                      </div>

                      <div>
                        <label className="block text-[11px] font-semibold text-slate-300 mb-1">
                          Epistemic Modality
                        </label>
                        <select
                          disabled={editingDomain?.is_system}
                          value={domainEpistemic}
                          onChange={(e) => setDomainEpistemic(e.target.value as EpistemicType)}
                          className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-1.5 text-slate-200 disabled:opacity-50"
                        >
                          {EPISTEMIC_TYPES.map((ep) => (
                            <option key={ep.value} value={ep.value}>
                              {ep.label}
                            </option>
                          ))}
                        </select>
                      </div>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                      <div>
                        <label className="block text-[11px] font-semibold text-slate-300 mb-1">
                          Chunking Strategy Preset
                        </label>
                        <select
                          disabled={editingDomain?.is_system}
                          value={domainChunking}
                          onChange={(e) =>
                            setDomainChunking(e.target.value as ChunkingStrategy)
                          }
                          className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-1.5 text-slate-200 disabled:opacity-50"
                        >
                          {CHUNKING_STRATEGIES.map((cs) => (
                            <option key={cs.value} value={cs.value}>
                              {cs.label}
                            </option>
                          ))}
                        </select>
                      </div>

                      <div>
                        <label className="block text-[11px] font-semibold text-slate-300 mb-1">
                          Authority Priority (1 = Top)
                        </label>
                        <select
                          disabled={editingDomain?.is_system}
                          value={domainPriority}
                          onChange={(e) => setDomainPriority(Number(e.target.value))}
                          className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-1.5 text-slate-200 disabled:opacity-50"
                        >
                          <option value={1}>1 (Highest Priority)</option>
                          <option value={2}>2 (High Priority)</option>
                          <option value={3}>3 (Standard Priority)</option>
                          <option value={4}>4 (Secondary Priority)</option>
                          <option value={5}>5 (Supplemental)</option>
                        </select>
                      </div>
                    </div>

                    <div>
                      <label className="block text-[11px] font-semibold text-slate-300 mb-1">
                        Domain Description * (Used by Laya Zero-Shot Classifier)
                      </label>
                      <textarea
                        rows={2}
                        value={domainDescription}
                        onChange={(e) => setDomainDescription(e.target.value)}
                        placeholder="Detailed domain description and scope..."
                        className="w-full rounded-lg border border-slate-700 bg-slate-950 p-2 text-slate-200 placeholder:text-slate-600"
                      />
                    </div>

                    <div>
                      <label className="block text-[11px] font-semibold text-slate-300 mb-1">
                        Classifier Keyword Hints (Comma-separated, max 25)
                      </label>
                      <input
                        type="text"
                        value={domainKeywords}
                        onChange={(e) => setDomainKeywords(e.target.value)}
                        placeholder="e.g. clinic, diagnosis, pathology, pharmacology"
                        className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-1.5 text-slate-200 placeholder:text-slate-600"
                      />
                    </div>

                    {/* Optional Re-index Checkbox when updating chunking strategy on custom domain */}
                    {!isCreatingDomain &&
                      editingDomain &&
                      !editingDomain.is_system &&
                      editingDomain.document_count > 0 &&
                      domainChunking !== editingDomain.chunking_strategy && (
                        <div className="rounded-lg border border-amber-500/30 bg-amber-950/20 p-2.5">
                          <label className="flex items-center gap-2 cursor-pointer text-amber-200 font-medium">
                            <input
                              type="checkbox"
                              checked={domainReindex}
                              onChange={(e) => setDomainReindex(e.target.checked)}
                              className="rounded border-slate-700 bg-slate-900 text-indigo-600"
                            />
                            <span>
                              Re-index existing documents ({editingDomain.document_count}) in
                              background to apply new chunking strategy
                            </span>
                          </label>
                        </div>
                      )}

                    <div className="flex items-center justify-end gap-2 pt-1">
                      <button
                        type="button"
                        onClick={() => {
                          setIsCreatingDomain(false);
                          setEditingDomain(null);
                        }}
                        className="rounded-lg border border-slate-700 bg-slate-800 px-3 py-1.5 text-xs text-slate-300 hover:bg-slate-700"
                      >
                        Cancel
                      </button>
                      <button
                        type="submit"
                        disabled={
                          createDomainMutation.isPending || updateDomainMutation.isPending
                        }
                        className="inline-flex items-center gap-1.5 rounded-lg bg-indigo-600 px-4 py-1.5 text-xs font-semibold text-white hover:bg-indigo-500 disabled:opacity-50"
                      >
                        {createDomainMutation.isPending || updateDomainMutation.isPending ? (
                          <Loader2 className="h-3.5 w-3.5 animate-spin" />
                        ) : (
                          <Check className="h-3.5 w-3.5" />
                        )}
                        <span>Save Domain</span>
                      </button>
                    </div>
                  </form>
                </div>
              )}

              {/* Subdomain Management Panel (when opened) */}
              {subdomainDomain && (
                <div className="rounded-xl border border-slate-700 bg-slate-900/90 p-4 space-y-3">
                  <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                    <h4 className="text-xs font-bold text-white flex items-center gap-1.5">
                      <span>Subdomains for {subdomainDomain.icon} {subdomainDomain.name}</span>
                    </h4>
                    <button
                      type="button"
                      onClick={() => setSubdomainDomain(null)}
                      className="text-slate-400 hover:text-white"
                    >
                      <X className="h-4 w-4" />
                    </button>
                  </div>

                  {subdomainFormError && (
                    <div className="rounded-lg bg-rose-950/40 border border-rose-500/40 p-2 text-xs text-rose-300">
                      {subdomainFormError}
                    </div>
                  )}

                  {/* Add Subdomain Form */}
                  <form onSubmit={handleCreateSubdomain} className="grid grid-cols-1 md:grid-cols-4 gap-2 text-xs">
                    <div>
                      <input
                        type="text"
                        placeholder="Subdomain Name *"
                        value={subdomainNameInput}
                        onChange={(e) => setSubdomainNameInput(e.target.value)}
                        className="w-full rounded border border-slate-700 bg-slate-950 px-2 py-1 text-slate-200"
                      />
                    </div>
                    <div>
                      <input
                        type="text"
                        placeholder="Slug (e.g. cardiology) *"
                        value={subdomainSlugInput}
                        onChange={(e) => setSubdomainSlugInput(e.target.value)}
                        className="w-full rounded border border-slate-700 bg-slate-950 px-2 py-1 text-slate-200 font-mono text-[11px]"
                      />
                    </div>
                    <div>
                      <input
                        type="text"
                        placeholder="Standard Bodies (optional)"
                        value={subdomainStandardBodies}
                        onChange={(e) => setSubdomainStandardBodies(e.target.value)}
                        className="w-full rounded border border-slate-700 bg-slate-950 px-2 py-1 text-slate-200"
                      />
                    </div>
                    <div>
                      <button
                        type="submit"
                        disabled={createSubdomainMutation.isPending}
                        className="inline-flex items-center justify-center gap-1 w-full rounded bg-indigo-600 px-3 py-1 font-semibold text-white hover:bg-indigo-500 disabled:opacity-50"
                      >
                        <Plus className="h-3.5 w-3.5" />
                        <span>Add Subdomain</span>
                      </button>
                    </div>
                  </form>

                  {/* List of existing subdomains */}
                  <div className="divide-y divide-slate-800 border border-slate-800 rounded-lg overflow-hidden">
                    {subdomainDomain.subdomains.map((sub) => {
                      const isSingle = subdomainDomain.subdomains.length <= 1;
                      const isRootGeneral = sub.slug === "general.general";
                      const cannotDelete = isSingle || isRootGeneral;

                      return (
                        <div
                          key={sub.id}
                          className="flex items-center justify-between p-2 text-xs hover:bg-slate-800/40"
                        >
                          <div>
                            <span className="font-semibold text-slate-200">{sub.name}</span>
                            <span className="ml-2 font-mono text-[10px] text-slate-400">
                              {sub.slug}
                            </span>
                            {sub.standard_bodies && (
                              <span className="ml-2 text-[10.5px] text-slate-500">
                                ({sub.standard_bodies})
                              </span>
                            )}
                          </div>
                          <button
                            type="button"
                            disabled={cannotDelete}
                            onClick={() => {
                              const remaining = subdomainDomain.subdomains.filter(
                                (s) => s.slug !== sub.slug
                              );
                              setDeletingSubdomain({
                                domain: subdomainDomain,
                                subdomain: sub,
                              });
                              setTargetSubdomainOnSubDelete(remaining[0]?.slug || "");
                            }}
                            title={
                              isRootGeneral
                                ? "Root general.general cannot be deleted"
                                : isSingle
                                ? "Cannot delete the only remaining subdomain for this domain"
                                : "Delete subdomain and reassign records"
                            }
                            className="text-slate-400 hover:text-rose-400 disabled:opacity-30 disabled:hover:text-slate-400 p-1 rounded"
                          >
                            <Trash2 className="h-3.5 w-3.5" />
                          </button>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Action Toolbar */}
              <div className="flex items-center justify-between">
                <span className="text-xs text-slate-400">
                  Active Domains:{" "}
                  <strong className="text-white">{activeDomainsCount} / 10</strong>
                </span>
                <button
                  type="button"
                  disabled={activeDomainsCount >= 10 || isCreatingDomain}
                  onClick={openCreateDomain}
                  title={
                    activeDomainsCount >= 10
                      ? "Maximum limit of 10 active top-level domains reached"
                      : "Create a new custom domain"
                  }
                  className="inline-flex items-center gap-1.5 rounded-lg bg-indigo-600 px-3 py-1.5 text-xs font-semibold text-white hover:bg-indigo-500 disabled:opacity-40 transition"
                >
                  <Plus className="h-3.5 w-3.5" />
                  <span>New Domain</span>
                </button>
              </div>

              {/* Domains Grid / List */}
              {isLoadingDomains ? (
                <div className="flex items-center justify-center p-8 text-slate-400">
                  <Loader2 className="h-5 w-5 animate-spin mr-2" />
                  <span>Loading domains...</span>
                </div>
              ) : (
                <div className="space-y-2">
                  {domains.map((dom) => {
                    const meta = resolveDomainMeta(dom.slug, domains);
                    const isExpanded = expandedDomainId === dom.id;

                    return (
                      <div
                        key={dom.id}
                        className="rounded-xl border border-slate-800/90 bg-slate-900/60 p-3 hover:border-slate-700 transition"
                      >
                        <div className="flex items-center justify-between gap-3">
                          <div className="flex items-center gap-2.5">
                            <span className="text-xl select-none">{dom.icon}</span>
                            <div>
                              <div className="flex items-center gap-2">
                                <span className="font-bold text-white text-sm">
                                  {dom.name}
                                </span>
                                <span className="font-mono text-[10.5px] text-slate-400">
                                  {dom.slug}
                                </span>
                                {dom.is_system && (
                                  <span className="inline-flex items-center gap-1 rounded-full border border-amber-500/30 bg-amber-500/10 px-2 py-0.2 text-[9.5px] font-semibold text-amber-300">
                                    <Lock className="h-2.5 w-2.5" />
                                    <span>SYSTEM</span>
                                  </span>
                                )}
                                <span
                                  className={`rounded-full border px-2 py-0.2 text-[9.5px] font-medium ${meta.theme.badge}`}
                                >
                                  {meta.epistemicType}
                                </span>
                              </div>
                              <p className="text-xs text-slate-400 line-clamp-1 mt-0.5">
                                {dom.description || "No description provided."}
                              </p>
                            </div>
                          </div>

                          <div className="flex items-center gap-2 shrink-0">
                            <span className="text-xs text-slate-400 font-mono">
                              {dom.document_count} doc{dom.document_count === 1 ? "" : "s"}
                            </span>

                            <button
                              type="button"
                              onClick={() =>
                                setExpandedDomainId(isExpanded ? null : dom.id)
                              }
                              className="rounded-lg border border-slate-800 bg-slate-900 px-2.5 py-1 text-xs text-slate-300 hover:bg-slate-800 flex items-center gap-1"
                            >
                              <span>{dom.subdomains.length} Subdomains</span>
                              {isExpanded ? (
                                <ChevronDown className="h-3 w-3" />
                              ) : (
                                <ChevronRight className="h-3 w-3" />
                              )}
                            </button>

                            <button
                              type="button"
                              onClick={() => openEditDomain(dom)}
                              className="rounded-lg border border-slate-800 bg-slate-900 px-2.5 py-1 text-xs text-slate-300 hover:bg-slate-800"
                            >
                              Edit
                            </button>

                            <button
                              type="button"
                              disabled={dom.is_system}
                              onClick={() => openDeleteDomain(dom)}
                              title={
                                dom.is_system
                                  ? "System starter domain cannot be deleted"
                                  : "Delete domain & reassign documents"
                              }
                              className="rounded-lg border border-slate-800 bg-slate-900 p-1 text-slate-400 hover:text-rose-400 hover:bg-slate-800 disabled:opacity-30 disabled:hover:text-slate-400 transition"
                            >
                              <Trash2 className="h-3.5 w-3.5" />
                            </button>
                          </div>
                        </div>

                        {/* Expandable Subdomains View */}
                        {isExpanded && (
                          <div className="mt-3 border-t border-slate-800/80 pt-2.5 space-y-2">
                            <div className="flex items-center justify-between text-[11px] text-slate-400">
                              <span className="font-semibold uppercase text-slate-500">
                                Registered Subdomains
                              </span>
                              <button
                                type="button"
                                onClick={() => {
                                  setSubdomainDomain(dom);
                                  setSubdomainSlugInput("");
                                  setSubdomainNameInput("");
                                  setSubdomainStandardBodies("");
                                }}
                                className="text-indigo-400 hover:text-indigo-300 flex items-center gap-1"
                              >
                                <Plus className="h-3 w-3" />
                                <span>Manage Subdomains</span>
                              </button>
                            </div>
                            <div className="flex flex-wrap gap-1.5">
                              {dom.subdomains.map((sub) => (
                                <span
                                  key={sub.id}
                                  className="rounded-lg border border-slate-800 bg-slate-950/80 px-2.5 py-1 text-xs text-slate-300"
                                >
                                  <strong>{sub.name}</strong>{" "}
                                  <span className="font-mono text-[10.5px] text-slate-500">
                                    ({sub.slug})
                                  </span>
                                </span>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}
            </Tabs.Content>

            {/* TAB 2: POLYSEMY LEXICON */}
            <Tabs.Content
              value="lexicon"
              className="flex-1 overflow-y-auto pt-4 space-y-4 pr-1"
            >
              {/* Add / Edit Lexicon Form */}
              <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4 space-y-3">
                <h4 className="text-xs font-bold text-white flex items-center gap-1.5">
                  <Plus className="h-3.5 w-3.5 text-indigo-400" />
                  <span>
                    {editingLexicon ? `Edit Term: "${editingLexicon.term}"` : "Add Term to Polysemy Lexicon"}
                  </span>
                </h4>

                {lexiconFormError && (
                  <div className="rounded-lg bg-rose-950/40 border border-rose-500/40 p-2 text-xs text-rose-300">
                    {lexiconFormError}
                  </div>
                )}

                <form onSubmit={handleSaveLexicon} className="space-y-2 text-xs">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    <div>
                      <label className="block text-[11px] font-semibold text-slate-300 mb-1">
                        Term (Keyword) *
                      </label>
                      <input
                        type="text"
                        value={lexiconTerm}
                        onChange={(e) => setLexiconTerm(e.target.value)}
                        placeholder="e.g. depreciation, statute of limitations"
                        className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-1.5 text-slate-200"
                      />
                    </div>
                    <div>
                      <label className="block text-[11px] font-semibold text-slate-300 mb-1">
                        Owning Domain *
                      </label>
                      <select
                        value={lexiconDomainId}
                        onChange={(e) => setLexiconDomainId(e.target.value)}
                        className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-1.5 text-slate-200"
                      >
                        <option value="">Select Domain...</option>
                        {domains.map((d) => (
                          <option key={d.id} value={d.id}>
                            {d.icon} {d.name} ({d.slug})
                          </option>
                        ))}
                      </select>
                    </div>
                  </div>

                  <div>
                    <label className="block text-[11px] font-semibold text-slate-300 mb-1">
                      Domain-Specific Sense (Definition) *
                    </label>
                    <textarea
                      rows={2}
                      value={lexiconSense}
                      onChange={(e) => setLexiconSense(e.target.value)}
                      placeholder="e.g. Allocation of asset cost over useful life in GAAP accounting"
                      className="w-full rounded-lg border border-slate-700 bg-slate-950 p-2 text-slate-200"
                    />
                  </div>

                  <div>
                    <label className="block text-[11px] font-semibold text-slate-300 mb-1">
                      Contrastive Warning (Cross-Domain Collision Note) *
                    </label>
                    <textarea
                      rows={2}
                      value={lexiconWarning}
                      onChange={(e) => setLexiconWarning(e.target.value)}
                      placeholder="e.g. Do not confuse with physical mechanical wear-and-tear in engineering"
                      className="w-full rounded-lg border border-slate-700 bg-slate-950 p-2 text-slate-200"
                    />
                  </div>

                  <div className="flex items-center justify-end gap-2 pt-1">
                    {editingLexicon && (
                      <button
                        type="button"
                        onClick={openCreateLexicon}
                        className="rounded-lg border border-slate-700 bg-slate-800 px-3 py-1 text-xs text-slate-300"
                      >
                        Cancel
                      </button>
                    )}
                    <button
                      type="submit"
                      disabled={upsertLexiconMutation.isPending}
                      className="inline-flex items-center gap-1.5 rounded-lg bg-indigo-600 px-4 py-1.5 text-xs font-semibold text-white hover:bg-indigo-500 disabled:opacity-50"
                    >
                      {upsertLexiconMutation.isPending ? (
                        <Loader2 className="h-3.5 w-3.5 animate-spin" />
                      ) : (
                        <Check className="h-3.5 w-3.5" />
                      )}
                      <span>Save Term Definition</span>
                    </button>
                  </div>
                </form>
              </div>

              {/* Filter & Search Bar */}
              <div className="flex items-center gap-3">
                <div className="relative flex-1">
                  <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-slate-500" />
                  <input
                    type="text"
                    value={lexiconSearch}
                    onChange={(e) => setLexiconSearch(e.target.value)}
                    placeholder="Search terms, definitions or warnings..."
                    className="w-full rounded-lg border border-slate-800 bg-slate-900 pl-8 pr-3 py-1.5 text-xs text-slate-200 placeholder:text-slate-500"
                  />
                </div>
                <select
                  value={lexiconFilterDomain}
                  onChange={(e) => setLexiconFilterDomain(e.target.value)}
                  className="rounded-lg border border-slate-800 bg-slate-900 px-3 py-1.5 text-xs text-slate-200"
                >
                  <option value="all">All Domains</option>
                  {domains.map((d) => (
                    <option key={d.id} value={d.id}>
                      {d.icon} {d.name}
                    </option>
                  ))}
                </select>
              </div>

              {/* Lexicon Table */}
              {isLoadingLexicon ? (
                <div className="flex items-center justify-center p-8 text-slate-400">
                  <Loader2 className="h-5 w-5 animate-spin mr-2" />
                  <span>Loading lexicon...</span>
                </div>
              ) : lexiconEntries.length === 0 ? (
                <div className="rounded-xl border border-slate-800 bg-slate-900/30 p-8 text-center text-xs text-slate-500">
                  No polysemy definitions found. Add cross-domain terms above.
                </div>
              ) : (
                <div className="divide-y divide-slate-800 rounded-xl border border-slate-800 bg-slate-900/60 overflow-hidden">
                  {lexiconEntries.map((item) => {
                    const domObj = domains.find((d) => d.id === item.domain_id);
                    const meta = resolveDomainMeta(domObj?.slug, domains);

                    return (
                      <div
                        key={item.id}
                        className="p-3 text-xs flex items-start justify-between gap-4 hover:bg-slate-800/40 transition"
                      >
                        <div className="space-y-1">
                          <div className="flex items-center gap-2">
                            <span className="font-bold text-white text-sm">
                              {item.term}
                            </span>
                            <span
                              className={`rounded-full border px-2 py-0.2 text-[10px] font-medium ${meta.theme.badge}`}
                            >
                              {meta.icon} {meta.name}
                            </span>
                          </div>
                          <p className="text-slate-300">
                            <strong>Sense:</strong> {item.specific_sense}
                          </p>
                          <p className="text-amber-300/90 text-[11.5px]">
                            <strong>Warning:</strong> {item.contrastive_warning}
                          </p>
                        </div>

                        <div className="flex items-center gap-1 shrink-0 pt-1">
                          <button
                            type="button"
                            onClick={() => openEditLexicon(item)}
                            className="rounded p-1 text-slate-400 hover:bg-slate-800 hover:text-slate-200"
                          >
                            <Settings className="h-3.5 w-3.5" />
                          </button>
                          <button
                            type="button"
                            onClick={() => handleDeleteLexicon(item.id)}
                            className="rounded p-1 text-slate-400 hover:bg-slate-800 hover:text-rose-400"
                          >
                            <Trash2 className="h-3.5 w-3.5" />
                          </button>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </Tabs.Content>

            {/* TAB 3: INGESTION & AI DEFAULTS */}
            <Tabs.Content
              value="ingestion"
              className="flex-1 overflow-y-auto pt-4 space-y-4 pr-1"
            >
              {settingsSuccessMsg && (
                <div className="rounded-lg bg-emerald-950/40 border border-emerald-500/40 p-2.5 text-xs text-emerald-300 flex items-center gap-2">
                  <Check className="h-4 w-4 shrink-0 text-emerald-400" />
                  <span>{settingsSuccessMsg}</span>
                </div>
              )}

              {/* Surface 1: Proactive Default Provider Alert Banner */}
              {isMissingActiveKey && (
                <div className="rounded-xl border border-amber-500/40 bg-amber-950/30 p-3.5 text-xs text-amber-200 flex items-start gap-3">
                  <AlertTriangle className="h-5 w-5 shrink-0 text-amber-400 mt-0.5" />
                  <div className="space-y-1">
                    <div className="font-semibold text-amber-100">
                      Missing API Key for Default Provider ({matchingOption?.label || activeDefaultProv})
                    </div>
                    <p className="text-[11.5px] text-amber-300/90 leading-relaxed">
                      No API key was detected in Workspace Settings or <code className="bg-amber-950/80 px-1 py-0.5 rounded text-amber-200">backend/.env</code>.
                      Chat inference will fail until a valid key is provided and verified below.
                    </p>
                  </div>
                </div>
              )}

              {isInvalidActiveKey && (
                <div className="rounded-xl border border-rose-500/40 bg-rose-950/30 p-3.5 text-xs text-rose-200 flex items-start gap-3">
                  <AlertCircle className="h-5 w-5 shrink-0 text-rose-400 mt-0.5" />
                  <div className="space-y-1">
                    <div className="font-semibold text-rose-100">
                      Invalid API Key for Default Provider ({matchingOption?.label || activeDefaultProv})
                    </div>
                    <p className="text-[11.5px] text-rose-300/90 leading-relaxed">
                      {matchingOption?.validation_error ||
                        "Upstream rejected the configured key. Please check your credentials or enter a replacement key below."}
                    </p>
                  </div>
                </div>
              )}

              <form onSubmit={handleSaveSettings} className="space-y-4 text-xs">
                {/* Default Upload Engine */}
                <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4 space-y-2">
                  <label className="block text-xs font-bold text-white">
                    Default Parsing & Ingestion Engine
                  </label>
                  <p className="text-[11.5px] text-slate-400">
                    Determines document parsing pipeline when uploaded through drag & drop or API.
                  </p>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-2 pt-1">
                    {[
                      {
                        id: "auto" as const,
                        title: "Auto (Adaptive)",
                        desc: "Docling with PyMuPDF4LLM auto-fallback",
                      },
                      {
                        id: "gpu" as const,
                        title: "GPU / Deep Vision",
                        desc: "Force Docling for complex formulas and tables",
                      },
                      {
                        id: "fast" as const,
                        title: "Fast Path",
                        desc: "Force PyMuPDF4LLM for instant parsing",
                      },
                    ].map((opt) => (
                      <button
                        key={opt.id}
                        type="button"
                        onClick={() => setUploadMode(opt.id)}
                        className={`rounded-lg border p-3 text-left transition ${
                          uploadMode === opt.id
                            ? "border-indigo-500 bg-indigo-950/40 text-white"
                            : "border-slate-800 bg-slate-950/60 text-slate-400 hover:border-slate-700 hover:text-slate-200"
                        }`}
                      >
                        <div className="font-semibold text-xs text-white">{opt.title}</div>
                        <div className="text-[10.5px] text-slate-400 mt-0.5">{opt.desc}</div>
                      </button>
                    ))}
                  </div>
                </div>

                {/* Default Approval Gate */}
                <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4 space-y-2">
                  <label className="block text-xs font-bold text-white">
                    Default Human-in-the-Loop Approval Gate
                  </label>
                  <p className="text-[11.5px] text-slate-400">
                    Control when new documents require manual approval before vector indexing.
                  </p>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-2 pt-1">
                    {[
                      {
                        id: "review_all" as const,
                        title: "Review All (Default)",
                        desc: "Pause all documents at awaiting_approval",
                      },
                      {
                        id: "auto_high_confidence" as const,
                        title: "Auto-Approve High Conf",
                        desc: "Auto-approve if Laya confidence ≥ Threshold",
                      },
                      {
                        id: "auto_all" as const,
                        title: "Auto-Approve All",
                        desc: "Direct stream to indexing without review",
                      },
                    ].map((opt) => (
                      <button
                        key={opt.id}
                        type="button"
                        onClick={() => setApprovalMode(opt.id)}
                        className={`rounded-lg border p-3 text-left transition ${
                          approvalMode === opt.id
                            ? "border-emerald-500 bg-emerald-950/40 text-white"
                            : "border-slate-800 bg-slate-950/60 text-slate-400 hover:border-slate-700 hover:text-slate-200"
                        }`}
                      >
                        <div className="font-semibold text-xs text-white">{opt.title}</div>
                        <div className="text-[10.5px] text-slate-400 mt-0.5">{opt.desc}</div>
                      </button>
                    ))}
                  </div>
                </div>

                {/* High Confidence Threshold Slider */}
                <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4 space-y-2">
                  <div className="flex items-center justify-between">
                    <div>
                      <label className="block text-xs font-bold text-white">
                        Auto-Approve Confidence Threshold
                      </label>
                      <p className="text-[11.5px] text-slate-400">
                        Minimum zero-shot classifier confidence score required for auto-approval.
                      </p>
                    </div>
                    <span className="font-mono text-sm font-bold text-emerald-400">
                      {Math.round(confidenceThreshold * 100)}%
                    </span>
                  </div>
                  <input
                    type="range"
                    min={0.5}
                    max={0.99}
                    step={0.01}
                    value={confidenceThreshold}
                    onChange={(e) => setConfidenceThreshold(Number(e.target.value))}
                    className="w-full accent-emerald-500 cursor-pointer"
                  />
                  <div className="flex justify-between text-[10px] text-slate-500 font-mono">
                    <span>50% (Permissive)</span>
                    <span>85% (Balanced)</span>
                    <span>99% (Strict)</span>
                  </div>
                </div>

                {/* LLM Provider & Model Selector */}
                <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4 space-y-3">
                  <label className="block text-xs font-bold text-white">
                    Default Workspace LLM Provider & Model
                  </label>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    <div>
                      <label className="block text-[11px] font-semibold text-slate-300 mb-1">
                        LLM Provider
                      </label>
                      <select
                        value={llmProvider}
                        onChange={(e) => {
                          const nextProv = e.target.value;
                          setLlmProvider(nextProv);
                          if (!nextProv) {
                            setLlmModel("");
                            return;
                          }
                          const knownDefaults = new Set(
                            (wsSettings?.available_providers ?? [])
                              .map((p) => p.default_model)
                              .filter(Boolean)
                          );
                          const matched = (
                            wsSettings?.available_providers ?? []
                          ).find((p) => p.provider === nextProv);
                          if (
                            matched?.default_model &&
                            (!llmModel.trim() || knownDefaults.has(llmModel.trim()))
                          ) {
                            setLlmModel(matched.default_model);
                          }
                        }}
                        className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-1.5 text-slate-200"
                      >
                        <option value="">Default (From Backend Config)</option>
                        <option value="openai">OpenAI</option>
                        <option value="anthropic">Anthropic</option>
                        <option value="deepseek">DeepSeek</option>
                        <option value="openai_compatible">OpenAI-Compatible (Local / vLLM)</option>
                      </select>
                    </div>

                    <div>
                      <label className="block text-[11px] font-semibold text-slate-300 mb-1">
                        Model Name
                      </label>
                      <input
                        type="text"
                        value={llmModel}
                        onChange={(e) => setLlmModel(e.target.value)}
                        placeholder="e.g. gpt-4o, claude-3-5-sonnet-20241022"
                        className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-1.5 text-slate-200"
                      />
                    </div>
                  </div>
                </div>

                {/* API Keys & Live Verification */}
                <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4 space-y-3">
                  <div>
                    <label className="block text-xs font-bold text-white flex items-center gap-1.5">
                      <Key className="h-3.5 w-3.5 text-indigo-400" />
                      <span>API Keys & Provider Live Verification</span>
                    </label>
                    <p className="text-[11.5px] text-slate-400 mt-0.5">
                      Configure and live-test credentials for each model provider. Keys saved here override <code className="bg-slate-800 px-1 py-0.5 rounded text-slate-300">backend/.env</code>.
                    </p>
                  </div>

                  <div className="grid grid-cols-1 gap-3 pt-1">
                    {[
                      { id: "openai", label: "OpenAI" },
                      { id: "anthropic", label: "Anthropic" },
                      { id: "deepseek", label: "DeepSeek" },
                      {
                        id: "openai_compatible",
                        label: "OpenAI-Compatible (Local / vLLM)",
                      },
                    ].map((item) => {
                      const opt = (wsSettings?.available_providers ?? []).find(
                        (p) => p.provider === item.id
                      );
                      const keySource = opt?.key_source || "none";
                      const maskedKey = opt?.masked_key;
                      const valStatus = opt?.validation_status || "untested";
                      const valError = opt?.validation_error;
                      const isVerifying = verifyingProvider === item.id;
                      const feedback = verifyFeedback[item.id];
                      const isRevealed = Boolean(revealedKeys[item.id]);

                      return (
                        <div
                          key={item.id}
                          className="rounded-lg border border-slate-800/80 bg-slate-950/70 p-3.5 space-y-2.5"
                        >
                          {/* Card Header: Title + Badges */}
                          <div className="flex flex-wrap items-center justify-between gap-2">
                            <div className="flex items-center gap-2">
                              <span className="font-semibold text-xs text-white">
                                {opt?.label || item.label}
                              </span>
                              {item.id ===
                                (llmProvider ||
                                  wsSettings?.default_llm_provider ||
                                  "openai") && (
                                <span className="rounded-full bg-indigo-950 border border-indigo-500/30 px-2 py-0.5 text-[10px] font-medium text-indigo-300">
                                  Default
                                </span>
                              )}
                            </div>

                            <div className="flex items-center gap-1.5">
                              {/* Source Badge */}
                              {keySource === "ui" ? (
                                <span className="inline-flex items-center gap-1 rounded-md bg-emerald-950/80 border border-emerald-500/40 px-2 py-0.5 text-[10px] font-mono text-emerald-300">
                                  UI Key ({maskedKey || "saved"})
                                </span>
                              ) : keySource === "env" ? (
                                <span className="inline-flex items-center gap-1 rounded-md bg-sky-950/80 border border-sky-500/40 px-2 py-0.5 text-[10px] font-mono text-sky-300">
                                  .env ({maskedKey || "configured"})
                                </span>
                              ) : (
                                <span className="inline-flex items-center gap-1 rounded-md bg-amber-950/80 border border-amber-500/40 px-2 py-0.5 text-[10px] font-semibold text-amber-300">
                                  Missing Key
                                </span>
                              )}

                              {/* Validation Status Pill */}
                              {valStatus === "valid" ? (
                                <span className="inline-flex items-center gap-1 rounded-md bg-emerald-900/40 border border-emerald-600/40 px-2 py-0.5 text-[10px] font-medium text-emerald-300">
                                  ✓ Verified
                                </span>
                              ) : valStatus === "invalid" ? (
                                <span className="inline-flex items-center gap-1 rounded-md bg-rose-900/40 border border-rose-600/40 px-2 py-0.5 text-[10px] font-medium text-rose-300">
                                  ✗ Invalid Key
                                </span>
                              ) : valStatus === "missing" ? (
                                <span className="inline-flex items-center gap-1 rounded-md bg-amber-900/40 border border-amber-600/40 px-2 py-0.5 text-[10px] font-medium text-amber-300">
                                  Missing
                                </span>
                              ) : (
                                <span className="inline-flex items-center gap-1 rounded-md bg-slate-800 border border-slate-700 px-2 py-0.5 text-[10px] font-medium text-slate-400">
                                  Untested
                                </span>
                              )}
                            </div>
                          </div>

                          {/* Inputs: Key and Base URL */}
                          <div className="space-y-2">
                            <div className="relative flex items-center">
                              <input
                                type={isRevealed ? "text" : "password"}
                                autoComplete="off"
                                spellCheck={false}
                                value={apiKeyDrafts[item.id] ?? ""}
                                onChange={(e) =>
                                  setApiKeyDrafts((prev) => ({
                                    ...prev,
                                    [item.id]: e.target.value,
                                  }))
                                }
                                placeholder={
                                  maskedKey
                                    ? `Leave blank to keep ${maskedKey}`
                                    : `Enter ${item.label} API key...`
                                }
                                className="w-full rounded-lg border border-slate-700 bg-slate-900 px-3 py-1.5 pr-9 font-mono text-xs text-slate-200 placeholder:text-slate-500 placeholder:font-sans focus:border-indigo-500 focus:outline-none"
                              />
                              <button
                                type="button"
                                onClick={() =>
                                  setRevealedKeys((prev) => ({
                                    ...prev,
                                    [item.id]: !prev[item.id],
                                  }))
                                }
                                title={
                                  isRevealed
                                    ? "Hide API key"
                                    : "Show API key"
                                }
                                className="absolute right-2 text-slate-400 hover:text-slate-200"
                              >
                                {isRevealed ? (
                                  <EyeOff className="h-4 w-4" />
                                ) : (
                                  <Eye className="h-4 w-4" />
                                )}
                              </button>
                            </div>

                            {item.id === "openai_compatible" && (
                              <div>
                                <label className="block text-[11px] font-semibold text-slate-400 mb-1">
                                  Base URL (vLLM / Ollama / Local / Custom)
                                </label>
                                <input
                                  type="text"
                                  autoComplete="off"
                                  spellCheck={false}
                                  value={compatBaseUrl}
                                  onChange={(e) =>
                                    setCompatBaseUrl(e.target.value)
                                  }
                                  placeholder="http://localhost:11434/v1"
                                  className="w-full rounded-lg border border-slate-700 bg-slate-900 px-3 py-1.5 font-mono text-xs text-slate-200 placeholder:text-slate-500 placeholder:font-sans focus:border-indigo-500 focus:outline-none"
                                />
                              </div>
                            )}
                          </div>

                          {/* Action Buttons & Feedback */}
                          <div className="flex flex-wrap items-center justify-between gap-2 pt-1">
                            <div className="flex items-center gap-2">
                              <button
                                type="button"
                                onClick={() => handleVerifyKey(item.id)}
                                disabled={
                                  isVerifying || verifyKeyMutation.isPending
                                }
                                className="inline-flex items-center gap-1.5 rounded-lg border border-indigo-500/50 bg-indigo-600/20 px-3 py-1 text-xs font-semibold text-indigo-300 hover:bg-indigo-600/30 disabled:opacity-50 transition"
                              >
                                {isVerifying ? (
                                  <>
                                    <Loader2 className="h-3.5 w-3.5 animate-spin" />
                                    <span>Verifying...</span>
                                  </>
                                ) : (
                                  <>
                                    <ShieldCheck className="h-3.5 w-3.5 text-indigo-400" />
                                    <span>Verify & Save</span>
                                  </>
                                )}
                              </button>

                              {keySource === "ui" && (
                                <button
                                  type="button"
                                  onClick={() => handleClearUiKey(item.id)}
                                  disabled={updateSettingsMutation.isPending}
                                  title="Remove UI override and revert to backend/.env"
                                  className="inline-flex items-center gap-1 rounded-lg border border-slate-700 bg-slate-900 px-2.5 py-1 text-xs text-slate-400 hover:border-rose-500/50 hover:text-rose-300 transition"
                                >
                                  <span>Clear UI Key</span>
                                </button>
                              )}
                            </div>

                            {feedback && (
                              <div
                                className={`rounded-md px-2.5 py-1 text-[11px] flex items-center gap-1.5 ${
                                  feedback.valid
                                    ? "bg-emerald-950/60 text-emerald-300 border border-emerald-500/40"
                                    : "bg-rose-950/60 text-rose-300 border border-rose-500/40"
                                }`}
                              >
                                {feedback.valid ? (
                                  <Check className="h-3.5 w-3.5 shrink-0 text-emerald-400" />
                                ) : (
                                  <AlertCircle className="h-3.5 w-3.5 shrink-0 text-rose-400" />
                                )}
                                <span>{feedback.message}</span>
                              </div>
                            )}

                            {!feedback &&
                              valStatus === "invalid" &&
                              valError && (
                                <div className="rounded-md px-2.5 py-1 text-[11px] bg-rose-950/60 text-rose-300 border border-rose-500/40 flex items-center gap-1.5">
                                  <AlertCircle className="h-3.5 w-3.5 shrink-0 text-rose-400" />
                                  <span>{valError}</span>
                                </div>
                              )}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>

                <div className="flex items-center justify-end gap-2 pt-2">
                  <button
                    type="submit"
                    disabled={updateSettingsMutation.isPending}
                    className="inline-flex items-center gap-1.5 rounded-lg bg-indigo-600 px-4 py-2 text-xs font-semibold text-white hover:bg-indigo-500 disabled:opacity-50 shadow-xs"
                  >
                    {updateSettingsMutation.isPending ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <Check className="h-4 w-4" />
                    )}
                    <span>Save Workspace Settings</span>
                  </button>
                </div>
              </form>
            </Tabs.Content>
          </Tabs.Root>

          {/* Delete & Reassign Custom Domain Modal */}
          {deletingDomain && (
            <Dialog.Root open={Boolean(deletingDomain)} onOpenChange={() => setDeletingDomain(null)}>
              <Dialog.Portal>
                <Dialog.Overlay className="fixed inset-0 z-60 bg-black/80 backdrop-blur-xs" />
                <Dialog.Content className="fixed left-1/2 top-1/2 z-60 w-full max-w-lg -translate-x-1/2 -translate-y-1/2 rounded-2xl border border-rose-500/40 bg-slate-950 p-6 shadow-2xl focus:outline-none">
                  <div className="flex items-center gap-2.5 text-rose-400 mb-2">
                    <AlertTriangle className="h-5 w-5" />
                    <Dialog.Title className="text-base font-bold text-white">
                      Delete Custom Domain: {deletingDomain.name}
                    </Dialog.Title>
                  </div>
                  <Dialog.Description className="text-xs text-slate-300 mb-4 leading-relaxed">
                    Deleting domain <strong className="text-white">{deletingDomain.name}</strong> will
                    permanently remove the domain. All existing{" "}
                    <strong className="text-rose-300">{deletingDomain.document_count} documents</strong> and
                    their vectors will be atomically reassigned to the target domain below.
                  </Dialog.Description>

                  {deleteError && (
                    <div className="rounded-lg bg-rose-950/40 border border-rose-500/40 p-2.5 text-xs text-rose-300 mb-3">
                      {deleteError}
                    </div>
                  )}

                  <div className="space-y-3 text-xs mb-5">
                    <div>
                      <label className="block text-[11px] font-semibold text-slate-300 mb-1">
                        Reassign Documents To Target Domain *
                      </label>
                      <select
                        value={targetDomainSlug}
                        onChange={(e) => {
                          setTargetDomainSlug(e.target.value);
                          setTargetSubdomainSlug(null);
                        }}
                        className="w-full rounded-lg border border-slate-700 bg-slate-900 px-3 py-1.5 text-slate-200"
                      >
                        {availableTargetDomains.map((d) => (
                          <option key={d.slug} value={d.slug}>
                            {d.icon} {d.name} ({d.slug})
                          </option>
                        ))}
                      </select>
                    </div>

                    <div>
                      <label className="block text-[11px] font-semibold text-slate-300 mb-1">
                        Target Subdomain (Optional)
                      </label>
                      <select
                        value={targetSubdomainSlug || ""}
                        onChange={(e) => setTargetSubdomainSlug(e.target.value || null)}
                        className="w-full rounded-lg border border-slate-700 bg-slate-900 px-3 py-1.5 text-slate-200"
                      >
                        <option value="">Default Subdomain</option>
                        {targetDomainObj?.subdomains.map((s) => (
                          <option key={s.slug} value={s.slug}>
                            {s.name} ({s.slug})
                          </option>
                        ))}
                      </select>
                    </div>

                    <div className="rounded-lg border border-slate-800 bg-slate-900/60 p-3">
                      <label className="flex items-center gap-2 cursor-pointer text-slate-300">
                        <input
                          type="checkbox"
                          checked={reindexOnDelete}
                          onChange={(e) => setReindexOnDelete(e.target.checked)}
                          className="rounded border-slate-700 bg-slate-900 text-indigo-600"
                        />
                        <span>
                          Re-index affected documents in background to regenerate chunk contextual headers
                        </span>
                      </label>
                    </div>
                  </div>

                  <div className="flex items-center justify-end gap-2">
                    <button
                      type="button"
                      onClick={() => setDeletingDomain(null)}
                      className="rounded-lg border border-slate-700 bg-slate-800 px-3 py-1.5 text-xs text-slate-300 hover:bg-slate-700"
                    >
                      Cancel
                    </button>
                    <button
                      type="button"
                      disabled={deleteDomainMutation.isPending}
                      onClick={handleConfirmDeleteDomain}
                      className="inline-flex items-center gap-1.5 rounded-lg bg-rose-600 px-4 py-1.5 text-xs font-semibold text-white hover:bg-rose-500 disabled:opacity-50"
                    >
                      {deleteDomainMutation.isPending ? (
                        <Loader2 className="h-3.5 w-3.5 animate-spin" />
                      ) : (
                        <Trash2 className="h-3.5 w-3.5" />
                      )}
                      <span>Delete & Reassign</span>
                    </button>
                  </div>
                </Dialog.Content>
              </Dialog.Portal>
            </Dialog.Root>
          )}

          {/* Delete Subdomain Confirmation Dialog */}
          {deletingSubdomain && (
            <Dialog.Root open={Boolean(deletingSubdomain)} onOpenChange={() => setDeletingSubdomain(null)}>
              <Dialog.Portal>
                <Dialog.Overlay className="fixed inset-0 z-60 bg-black/80 backdrop-blur-xs" />
                <Dialog.Content className="fixed left-1/2 top-1/2 z-60 w-full max-w-md -translate-x-1/2 -translate-y-1/2 rounded-2xl border border-rose-500/40 bg-slate-950 p-6 shadow-2xl focus:outline-none">
                  <Dialog.Title className="text-sm font-bold text-white mb-2">
                    Delete Subdomain: {deletingSubdomain.subdomain.name}
                  </Dialog.Title>
                  <Dialog.Description className="text-xs text-slate-300 mb-4">
                    Reassign affected records in domain{" "}
                    <strong>{deletingSubdomain.domain.name}</strong> to another subdomain:
                  </Dialog.Description>

                  <div className="space-y-3 text-xs mb-5">
                    <div>
                      <label className="block text-[11px] font-semibold text-slate-300 mb-1">
                        Target Subdomain *
                      </label>
                      <select
                        value={targetSubdomainOnSubDelete}
                        onChange={(e) => setTargetSubdomainOnSubDelete(e.target.value)}
                        className="w-full rounded-lg border border-slate-700 bg-slate-900 px-3 py-1.5 text-slate-200"
                      >
                        {deletingSubdomain.domain.subdomains
                          .filter((s) => s.slug !== deletingSubdomain.subdomain.slug)
                          .map((s) => (
                            <option key={s.slug} value={s.slug}>
                              {s.name} ({s.slug})
                            </option>
                          ))}
                      </select>
                    </div>
                  </div>

                  <div className="flex items-center justify-end gap-2">
                    <button
                      type="button"
                      onClick={() => setDeletingSubdomain(null)}
                      className="rounded-lg border border-slate-700 bg-slate-800 px-3 py-1.5 text-xs text-slate-300"
                    >
                      Cancel
                    </button>
                    <button
                      type="button"
                      disabled={deleteSubdomainMutation.isPending}
                      onClick={handleConfirmDeleteSubdomain}
                      className="rounded-lg bg-rose-600 px-3 py-1.5 text-xs font-semibold text-white hover:bg-rose-500 disabled:opacity-50"
                    >
                      Delete Subdomain
                    </button>
                  </div>
                </Dialog.Content>
              </Dialog.Portal>
            </Dialog.Root>
          )}
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
};
