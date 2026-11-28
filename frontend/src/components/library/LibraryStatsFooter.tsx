import React from "react";
import type { DocumentItem, SystemStats } from "../../lib/types";

export interface LibraryStatsFooterProps {
  stats: SystemStats;
  rawDocuments: DocumentItem[];
  totalFromList?: number;
  isUnfiltered: boolean;
}

export const LibraryStatsFooter: React.FC<LibraryStatsFooterProps> = ({
  stats,
  rawDocuments,
  totalFromList,
  isUnfiltered,
}) => {
  const readyCount = isUnfiltered
    ? Math.max(
        rawDocuments.filter((d) => (d.status || "").toLowerCase() === "ready")
          .length,
        stats.ready_documents
      )
    : stats.ready_documents;

  const totalCount = isUnfiltered
    ? Math.max(totalFromList ?? rawDocuments.length, stats.total_documents)
    : stats.total_documents;

  const chunkCount = isUnfiltered
    ? Math.max(
        rawDocuments.reduce((acc, d) => acc + (d.chunk_count || 0), 0),
        stats.total_chunks
      )
    : stats.total_chunks;

  return (
    <div className="border-t border-slate-800/80 bg-slate-900/50 px-3 py-2 text-[11px] text-slate-400 flex items-center justify-between">
      <span>
        Ready: <strong className="text-slate-200">{readyCount}</strong>/
        {totalCount}
      </span>
      <span>
        Chunks: <strong className="text-slate-200">{chunkCount}</strong>
      </span>
      <span>
        Vectors:{" "}
        <strong className="text-slate-200">
          {stats.qdrant_vectors_count ?? stats.vector_count}
        </strong>
      </span>
    </div>
  );
};
