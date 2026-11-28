import React from "react";
import { Sparkles, X } from "lucide-react";
import type { ActiveCitation } from "../../lib/types";

export const CitationBridgeBanner: React.FC<{
  citation: ActiveCitation;
  onDismiss: () => void;
}> = ({ citation, onDismiss }) => {
  return (
    <div className="flex items-center justify-between gap-2 border-b border-indigo-500/30 bg-indigo-950/40 px-4 py-1.5 text-xs text-indigo-200">
      <div className="flex items-center gap-2 min-w-0">
        <Sparkles className="h-3.5 w-3.5 text-indigo-400 shrink-0" />
        <span className="truncate">
          {citation.citation_index
            ? `Citation [${citation.citation_index}]`
            : "Active Citation"}
          : <strong className="text-white">{citation.source_filename}</strong>
          {citation.domain && ` · ${citation.domain}`}
          {citation.authority_tier && ` · T${citation.authority_tier}`}
          {citation.page_numbers?.length > 0 &&
            ` · p.${citation.page_numbers.join(", ")}`}
          {citation.heading_path && ` · §${citation.heading_path}`}
        </span>
      </div>
      <button
        type="button"
        onClick={onDismiss}
        className="rounded p-0.5 text-indigo-300 hover:bg-indigo-500/20 hover:text-white"
        title="Dismiss citation highlight"
      >
        <X className="h-3.5 w-3.5" />
      </button>
    </div>
  );
};
