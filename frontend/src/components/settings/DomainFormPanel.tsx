import React, { useState } from "react";
import { Check, Loader2, Plus, Settings, X } from "lucide-react";
import {
  useCreateDomainMutation,
  useUpdateDomainMutation,
} from "../../hooks/useDomains";
import { DOMAIN_COLOR_PALETTE } from "../../lib/domainTheme";
import type {
  ChunkingStrategy,
  DomainColorToken,
  DomainItem,
  EpistemicType,
} from "../../lib/types";
import { AlertBanner } from "../common/AlertBanner";
import { DomainAppearanceFields } from "./DomainAppearanceFields";

export interface DomainFormPanelProps {
  editingDomain: DomainItem | null;
  onClose: () => void;
}

function toSlug(name: string): string {
  return name
    .toLowerCase()
    .replace(/[^a-z0-9]/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 32);
}

export const DomainFormPanel: React.FC<DomainFormPanelProps> = ({
  editingDomain,
  onClose,
}) => {
  const isCreating = editingDomain === null;
  const createDomainMutation = useCreateDomainMutation();
  const updateDomainMutation = useUpdateDomainMutation();

  const [domainName, setDomainName] = useState(editingDomain?.name ?? "");
  const [domainSlug, setDomainSlug] = useState(editingDomain?.slug ?? "");
  const [domainDescription, setDomainDescription] = useState(
    editingDomain?.description ?? ""
  );
  const [domainIcon, setDomainIcon] = useState(editingDomain?.icon || "📁");
  const [domainColor, setDomainColor] = useState<DomainColorToken>(
    editingDomain && editingDomain.color_token in DOMAIN_COLOR_PALETTE
      ? (editingDomain.color_token as DomainColorToken)
      : isCreating
      ? "cyan"
      : "slate"
  );
  const [domainEpistemic, setDomainEpistemic] = useState<EpistemicType>(
    (editingDomain?.epistemic_type as EpistemicType) ||
      (isCreating ? "empirical" : "general")
  );
  const [domainPriority, setDomainPriority] = useState<number>(
    editingDomain?.authority_priority ?? 3
  );
  const [domainChunking, setDomainChunking] = useState<ChunkingStrategy>(
    (editingDomain?.chunking_strategy as ChunkingStrategy) ||
      "standard_markdown"
  );
  const [domainKeywords, setDomainKeywords] = useState(
    editingDomain?.classifier_keywords ?? ""
  );
  const [domainReindex, setDomainReindex] = useState(false);
  const [domainFormError, setDomainFormError] = useState<string | null>(null);

  const handleDomainNameChange = (name: string) => {
    setDomainName(name);
    if (isCreating) setDomainSlug(toSlug(name));
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
      if (isCreating) {
        if (!domainSlug.trim()) {
          setDomainFormError("Slug is required.");
          return;
        }
        await createDomainMutation.mutateAsync({
          name: domainName.trim(),
          slug: domainSlug.trim().toLowerCase(),
          description: domainDescription.trim(),
          icon: domainIcon.trim() || "📁",
          color_token: domainColor,
          epistemic_type: domainEpistemic,
          authority_priority: domainPriority,
          chunking_strategy: domainChunking,
          classifier_keywords: domainKeywords.trim() || null,
        });
      } else if (editingDomain) {
        await updateDomainMutation.mutateAsync({
          slug: editingDomain.slug,
          payload: {
            name: editingDomain.is_system ? undefined : domainName.trim(),
            description: domainDescription.trim(),
            icon: domainIcon.trim() || "📁",
            color_token: domainColor,
            epistemic_type: editingDomain.is_system
              ? undefined
              : domainEpistemic,
            authority_priority: editingDomain.is_system
              ? undefined
              : domainPriority,
            chunking_strategy: editingDomain.is_system
              ? undefined
              : domainChunking,
            classifier_keywords: domainKeywords.trim() || null,
            re_index: domainReindex,
          },
        });
      }
      onClose();
    } catch (err: unknown) {
      setDomainFormError(
        err instanceof Error ? err.message : "Failed to save domain."
      );
    }
  };

  const isPending =
    createDomainMutation.isPending || updateDomainMutation.isPending;

  return (
    <div className="rounded-xl border border-indigo-500/40 bg-slate-900/90 p-4 space-y-3">
      <div className="flex items-center justify-between border-b border-slate-800 pb-2">
        <h4 className="text-xs font-bold text-white flex items-center gap-1.5">
          {isCreating ? (
            <><Plus className="h-3.5 w-3.5 text-indigo-400" /><span>Create New Custom Domain</span></>
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
        <button type="button" onClick={onClose} className="text-slate-400 hover:text-white">
          <X className="h-4 w-4" />
        </button>
      </div>

      {domainFormError && <AlertBanner variant="error" message={domainFormError} />}

      <form onSubmit={handleSaveDomain} className="space-y-3 text-xs">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <div>
            <label className="block text-[11px] font-semibold text-slate-300 mb-1">Domain Name *</label>
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
            <label className="block text-[11px] font-semibold text-slate-300 mb-1">Slug (Identifier) *</label>
            <input
              type="text"
              disabled={!isCreating}
              value={domainSlug}
              onChange={(e) => setDomainSlug(e.target.value.toLowerCase())}
              placeholder="e.g. healthcare"
              className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-1.5 text-slate-200 font-mono text-[11px] disabled:opacity-50"
            />
          </div>
        </div>

        <DomainAppearanceFields
          isSystemDomain={Boolean(editingDomain?.is_system)}
          domainIcon={domainIcon}
          onDomainIconChange={setDomainIcon}
          domainColor={domainColor}
          onDomainColorChange={setDomainColor}
          domainEpistemic={domainEpistemic}
          onDomainEpistemicChange={setDomainEpistemic}
          domainChunking={domainChunking}
          onDomainChunkingChange={setDomainChunking}
          domainPriority={domainPriority}
          onDomainPriorityChange={setDomainPriority}
        />

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

        {!isCreating &&
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
                  Re-index existing documents ({editingDomain.document_count})
                  in background to apply new chunking strategy
                </span>
              </label>
            </div>
          )}

        <div className="flex items-center justify-end gap-2 pt-1">
          <button
            type="button"
            onClick={onClose}
            className="rounded-lg border border-slate-700 bg-slate-800 px-3 py-1.5 text-xs text-slate-300 hover:bg-slate-700"
          >
            Cancel
          </button>
          <button
            type="submit"
            disabled={isPending}
            className="inline-flex items-center gap-1.5 rounded-lg bg-indigo-600 px-4 py-1.5 text-xs font-semibold text-white hover:bg-indigo-500 disabled:opacity-50"
          >
            {isPending ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            ) : (
              <Check className="h-3.5 w-3.5" />
            )}
            <span>Save Domain</span>
          </button>
        </div>
      </form>
    </div>
  );
};
