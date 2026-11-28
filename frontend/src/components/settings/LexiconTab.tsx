import React, { useState } from "react";
import { Settings, Trash2 } from "lucide-react";
import {
  useDeleteLexiconMutation,
  useLexiconQuery,
  useUpsertLexiconMutation,
} from "../../hooks/useLexicon";
import { resolveDomainMeta } from "../../lib/domainTheme";
import type {
  DomainItem,
  LexiconCreatePayload,
  LexiconItem,
} from "../../lib/types";
import { ConfirmDialog } from "../common/ConfirmDialog";
import { EmptyState } from "../common/EmptyState";
import { LoadingState } from "../common/LoadingState";
import { SearchInput } from "../common/SearchInput";
import { LexiconFormCard } from "./LexiconFormCard";

export interface LexiconTabProps {
  domains: DomainItem[];
}

export const LexiconTab: React.FC<LexiconTabProps> = ({ domains }) => {
  const [lexiconFilterDomain, setLexiconFilterDomain] = useState<string>("all");
  const [lexiconSearch, setLexiconSearch] = useState<string>("");

  const { data: lexiconEntries = [], isLoading: isLoadingLexicon } =
    useLexiconQuery({
      domain_id: lexiconFilterDomain !== "all" ? lexiconFilterDomain : undefined,
      search: lexiconSearch.trim() || undefined,
    });

  const upsertLexiconMutation = useUpsertLexiconMutation();
  const deleteLexiconMutation = useDeleteLexiconMutation();

  const [editingLexicon, setEditingLexicon] = useState<LexiconItem | null>(
    null
  );
  const [lexiconTerm, setLexiconTerm] = useState("");
  const [lexiconDomainId, setLexiconDomainId] = useState("");
  const [lexiconSense, setLexiconSense] = useState("");
  const [lexiconWarning, setLexiconWarning] = useState("");
  const [lexiconFormError, setLexiconFormError] = useState<string | null>(null);

  const [deletingItem, setDeletingItem] = useState<LexiconItem | null>(null);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  const resetForm = () => {
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

    if (
      !lexiconTerm.trim() ||
      !lexiconDomainId ||
      !lexiconSense.trim() ||
      !lexiconWarning.trim()
    ) {
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
      setLexiconFormError(
        err instanceof Error ? err.message : "Failed to save lexicon term."
      );
    }
  };

  const handleConfirmDelete = async () => {
    if (!deletingItem) return;
    setDeleteError(null);
    try {
      await deleteLexiconMutation.mutateAsync(deletingItem.id);
      setDeletingItem(null);
    } catch (err: unknown) {
      setDeleteError(
        err instanceof Error ? err.message : "Failed to delete lexicon term."
      );
    }
  };

  return (
    <>
      <LexiconFormCard
        domains={domains}
        editingLexicon={editingLexicon}
        term={lexiconTerm}
        onTermChange={setLexiconTerm}
        domainId={lexiconDomainId}
        onDomainIdChange={setLexiconDomainId}
        sense={lexiconSense}
        onSenseChange={setLexiconSense}
        warning={lexiconWarning}
        onWarningChange={setLexiconWarning}
        error={lexiconFormError}
        isPending={upsertLexiconMutation.isPending}
        onCancelEdit={resetForm}
        onSubmit={handleSaveLexicon}
      />

      <div className="flex items-center gap-3">
        <SearchInput
          className="flex-1"
          value={lexiconSearch}
          onChange={setLexiconSearch}
          placeholder="Search terms, definitions or warnings..."
        />
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

      {isLoadingLexicon ? (
        <LoadingState label="Loading lexicon..." className="p-8" />
      ) : lexiconEntries.length === 0 ? (
        <EmptyState
          className="rounded-xl border border-slate-800 bg-slate-900/30 p-8"
          title="No polysemy definitions found"
          description="Add cross-domain terms above."
        />
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
                    onClick={() => {
                      setDeleteError(null);
                      setDeletingItem(item);
                    }}
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

      <ConfirmDialog
        open={Boolean(deletingItem)}
        onOpenChange={(open) => {
          if (!open) setDeletingItem(null);
        }}
        zIndexClass="z-60"
        title={`Delete Lexicon Term: "${deletingItem?.term ?? ""}"?`}
        description="This removes the domain-specific disambiguation definition from the polysemy lexicon."
        confirmLabel="Delete Term"
        confirmIcon={<Trash2 className="h-3.5 w-3.5" />}
        isPending={deleteLexiconMutation.isPending}
        error={deleteError}
        onConfirm={handleConfirmDelete}
      />
    </>
  );
};
