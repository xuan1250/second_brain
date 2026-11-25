import React from "react";
import * as Tabs from "@radix-ui/react-tabs";
import {
  Edit,
  FileCode2,
  FileText,
  Loader2,
  Rocket,
  ZoomIn,
  ZoomOut,
} from "lucide-react";
import { getDomainKey } from "../../lib/documentMeta";
import type { DocumentItem, DomainItem } from "../../lib/types";
import { AuthorityTierBadge, DomainBadge } from "../common/DocumentBadges";

export interface InspectorToolbarProps {
  activeDoc: DocumentItem | null | undefined;
  displayFilename: string;
  domains: DomainItem[];
  isPdf: boolean;
  canUpgrade: boolean;
  isUpgrading: boolean;
  scale: number;
  onEditMetadata: () => void;
  onUpgradeToGpu: () => void;
  onZoomOut: () => void;
  onResetZoom: () => void;
  onZoomIn: () => void;
}

export const InspectorToolbar: React.FC<InspectorToolbarProps> = ({
  activeDoc,
  displayFilename,
  domains,
  isPdf,
  canUpgrade,
  isUpgrading,
  scale,
  onEditMetadata,
  onUpgradeToGpu,
  onZoomOut,
  onResetZoom,
  onZoomIn,
}) => {
  return (
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
                domains={domains}
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

      <div className="flex items-center gap-2">
        {activeDoc?.status === "ready" && (
          <button
            type="button"
            onClick={onEditMetadata}
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
            onClick={onUpgradeToGpu}
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

        <div className="flex items-center rounded-lg border border-slate-800 bg-slate-900/90 p-0.5 text-xs">
          <button
            type="button"
            onClick={onZoomOut}
            className="rounded p-1 text-slate-400 hover:bg-slate-800 hover:text-slate-200"
            title="Zoom out"
            aria-label="Zoom out"
          >
            <ZoomOut className="h-3.5 w-3.5" />
          </button>
          <button
            type="button"
            onClick={onResetZoom}
            className="px-1.5 font-mono text-[11px] text-slate-300 hover:text-white"
            title="Reset zoom"
          >
            {Math.round(scale * 100)}%
          </button>
          <button
            type="button"
            onClick={onZoomIn}
            className="rounded p-1 text-slate-400 hover:bg-slate-800 hover:text-slate-200"
            title="Zoom in"
            aria-label="Zoom in"
          >
            <ZoomIn className="h-3.5 w-3.5" />
          </button>
        </div>

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
  );
};
