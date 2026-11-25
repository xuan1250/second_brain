import React from "react";
import { Check, Loader2, Plus } from "lucide-react";
import type { DomainItem, LexiconItem } from "../../lib/types";
import { AlertBanner } from "../common/AlertBanner";

export interface LexiconFormCardProps {
  domains: DomainItem[];
  editingLexicon: LexiconItem | null;
  term: string;
  onTermChange: (v: string) => void;
  domainId: string;
  onDomainIdChange: (v: string) => void;
  sense: string;
  onSenseChange: (v: string) => void;
  warning: string;
  onWarningChange: (v: string) => void;
  error: string | null;
  isPending: boolean;
  onCancelEdit: () => void;
  onSubmit: (e: React.FormEvent) => void;
}

export const LexiconFormCard: React.FC<LexiconFormCardProps> = ({
  domains,
  editingLexicon,
  term,
  onTermChange,
  domainId,
  onDomainIdChange,
  sense,
  onSenseChange,
  warning,
  onWarningChange,
  error,
  isPending,
  onCancelEdit,
  onSubmit,
}) => {
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4 space-y-3">
      <h4 className="text-xs font-bold text-white flex items-center gap-1.5">
        <Plus className="h-3.5 w-3.5 text-indigo-400" />
        <span>
          {editingLexicon
            ? `Edit Term: "${editingLexicon.term}"`
            : "Add Term to Polysemy Lexicon"}
        </span>
      </h4>

      {error && <AlertBanner variant="error" message={error} />}

      <form onSubmit={onSubmit} className="space-y-2 text-xs">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <div>
            <label className="block text-[11px] font-semibold text-slate-300 mb-1">
              Term (Keyword) *
            </label>
            <input
              type="text"
              value={term}
              onChange={(e) => onTermChange(e.target.value)}
              placeholder="e.g. depreciation, statute of limitations"
              className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-1.5 text-slate-200"
            />
          </div>
          <div>
            <label className="block text-[11px] font-semibold text-slate-300 mb-1">
              Owning Domain *
            </label>
            <select
              value={domainId}
              onChange={(e) => onDomainIdChange(e.target.value)}
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
            value={sense}
            onChange={(e) => onSenseChange(e.target.value)}
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
            value={warning}
            onChange={(e) => onWarningChange(e.target.value)}
            placeholder="e.g. Do not confuse with physical mechanical wear-and-tear in engineering"
            className="w-full rounded-lg border border-slate-700 bg-slate-950 p-2 text-slate-200"
          />
        </div>

        <div className="flex items-center justify-end gap-2 pt-1">
          {editingLexicon && (
            <button
              type="button"
              onClick={onCancelEdit}
              className="rounded-lg border border-slate-700 bg-slate-800 px-3 py-1 text-xs text-slate-300"
            >
              Cancel
            </button>
          )}
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
            <span>Save Term Definition</span>
          </button>
        </div>
      </form>
    </div>
  );
};
