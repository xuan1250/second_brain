import React from "react";
import { Sparkles } from "lucide-react";

export interface CitationExcerptCardProps {
  title: React.ReactNode;
  rightMeta?: React.ReactNode;
  excerpt?: string | null;
  showSparkles?: boolean;
  className?: string;
}

export const CitationExcerptCard: React.FC<CitationExcerptCardProps> = ({
  title,
  rightMeta,
  excerpt,
  showSparkles = false,
  className = "",
}) => {
  return (
    <div
      className={`rounded-xl border border-indigo-500/40 bg-indigo-950/35 p-3 text-xs text-indigo-100 ${className}`}
    >
      <div className="flex items-center justify-between gap-2 text-[11px] font-semibold text-indigo-300">
        <span className="flex items-center gap-1.5 min-w-0 truncate">
          {showSparkles && (
            <Sparkles className="h-3.5 w-3.5 text-indigo-400 shrink-0" />
          )}
          <span className="truncate">{title}</span>
        </span>
        {rightMeta && <span className="shrink-0">{rightMeta}</span>}
      </div>
      {excerpt && (
        <p className="mt-1.5 text-slate-200 italic leading-relaxed">
          "{excerpt}"
        </p>
      )}
    </div>
  );
};
