import React from "react";
import { Database, Pause, Play } from "lucide-react";

export interface LibraryHeaderProps {
  totalDocs: number;
  ingestionPaused?: boolean;
  isTogglePending: boolean;
  onToggleIngestion: () => void;
}

export const LibraryHeader: React.FC<LibraryHeaderProps> = ({
  totalDocs,
  ingestionPaused,
  isTogglePending,
  onToggleIngestion,
}) => {
  return (
    <div className="flex items-center justify-between border-b border-slate-800/80 px-3.5 py-3">
      <div className="flex items-center gap-2">
        <div className="flex h-6 w-6 items-center justify-center rounded-md bg-indigo-500/20 text-indigo-400 border border-indigo-500/30">
          <Database className="h-3.5 w-3.5" />
        </div>
        <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-200">
          Knowledge Library
        </h2>
      </div>
      <div className="flex items-center gap-2">
        <button
          type="button"
          disabled={isTogglePending}
          onClick={onToggleIngestion}
          title={
            ingestionPaused
              ? "Ingestion paused. Click to resume."
              : "Pause background ingestion"
          }
          className={`inline-flex items-center gap-1 rounded-md px-2 py-0.5 text-[11px] font-medium transition border ${
            ingestionPaused
              ? "border-amber-500/40 bg-amber-500/15 text-amber-300 hover:bg-amber-500/25"
              : "border-slate-800 bg-slate-900 text-slate-400 hover:text-slate-200 hover:bg-slate-800"
          }`}
        >
          {ingestionPaused ? (
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
          {totalDocs} docs
        </span>
      </div>
    </div>
  );
};
