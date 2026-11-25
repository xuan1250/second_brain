import {
  AlertTriangle,
  FileSpreadsheet,
  FileText,
  Layers,
  Loader2,
  Presentation,
  Rocket,
  ShieldCheck,
  Zap,
} from "lucide-react";
import { isDocumentProcessing } from "../../hooks/useDocuments";
import { useDomainsQuery } from "../../hooks/useDomains";
import { getAuthorityTierMeta } from "../../lib/documentMeta";
import { resolveDomainMeta } from "../../lib/domainTheme";
import { formatPercent, formatSubdomainLeaf } from "../../lib/format";
import type { DocumentItem, DomainItem } from "../../lib/types";

export function FileTypeIcon({ fileType }: { fileType: string }) {
  const ft = (fileType || "").toLowerCase();
  if (ft === "xlsx") {
    return <FileSpreadsheet className="h-4 w-4 text-emerald-400 shrink-0" />;
  }
  if (ft === "pptx") {
    return <Presentation className="h-4 w-4 text-amber-400 shrink-0" />;
  }
  if (ft === "docx") {
    return <FileText className="h-4 w-4 text-sky-400 shrink-0" />;
  }
  return <FileText className="h-4 w-4 text-rose-400 shrink-0" />;
}

export function AuthorityTierBadge({ tier }: { tier?: number | null }) {
  const meta = getAuthorityTierMeta(tier);

  return (
    <span
      title={meta.title}
      className={`inline-flex items-center rounded-full border px-1.5 py-0.2 text-[10px] font-mono font-semibold cursor-help ${meta.badgeClass}`}
    >
      {meta.shortLabel}
    </span>
  );
}

export function DomainBadge({
  domainKey,
  subdomain,
  domains,
}: {
  domainKey: string;
  subdomain?: string | null;
  domains?: DomainItem[] | null;
}) {
  // Unconditional top-level hook call before any conditional fallback (AGENTS.md)
  const { data: queriedDomains } = useDomainsQuery();
  const effectiveDomains = domains ?? queriedDomains;

  const meta = resolveDomainMeta(domainKey, effectiveDomains);
  const leaf = formatSubdomainLeaf(subdomain);
  const subLabel = leaf ? ` › ${leaf}` : "";

  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[10px] font-medium ${meta.theme.badge}`}
      title={`Domain: ${meta.name}${subdomain ? ` / Subdomain: ${subdomain}` : ""}`}
    >
      <span>{meta.icon}</span>
      <span>
        {meta.shortLabel}
        {subLabel}
      </span>
    </span>
  );
}

export function StatusBadge({ doc }: { doc: DocumentItem }) {
  const status = (doc.status || "queued").toLowerCase();
  const stage = (doc.stage || status).toLowerCase();
  const processing = isDocumentProcessing(doc);

  if (status === "awaiting_approval") {
    return (
      <span className="inline-flex items-center gap-1 rounded-full bg-amber-500/15 border border-amber-500/40 px-2 py-0.5 text-[10px] font-semibold text-amber-300 animate-pulse">
        <ShieldCheck className="h-2.5 w-2.5 text-amber-400" />
        Awaiting Approval
      </span>
    );
  }

  if (status === "classifying") {
    return (
      <span className="inline-flex items-center gap-1 rounded-full bg-purple-500/15 border border-purple-500/30 px-2 py-0.5 text-[10px] font-medium text-purple-300 animate-pulse">
        <Loader2 className="h-2.5 w-2.5 animate-spin text-purple-400" />
        Classifying (Laya)
      </span>
    );
  }

  if (status === "ready") {
    return (
      <span className="inline-flex items-center gap-1 rounded-full bg-emerald-500/15 border border-emerald-500/30 px-2 py-0.5 text-[10px] font-medium text-emerald-300">
        <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
        Ready · {doc.chunk_count} ch
      </span>
    );
  }

  if (status === "failed") {
    const errText =
      doc.error_message || doc.error || "Document ingestion failed";
    return (
      <span
        title={errText}
        className="inline-flex items-center gap-1 rounded-full bg-rose-500/15 border border-rose-500/30 px-2 py-0.5 text-[10px] font-medium text-rose-300 cursor-help"
      >
        <AlertTriangle className="h-2.5 w-2.5 text-rose-400" />
        Failed
      </span>
    );
  }

  if (processing) {
    const pct =
      typeof doc.progress === "number" && doc.progress > 0
        ? ` ${formatPercent(doc.progress)}`
        : "";
    const stageLabel =
      stage.charAt(0).toUpperCase() + stage.slice(1).toLowerCase();
    return (
      <span className="inline-flex items-center gap-1 rounded-full bg-indigo-500/15 border border-indigo-500/30 px-2 py-0.5 text-[10px] font-medium text-indigo-300 animate-pulse">
        <Loader2 className="h-2.5 w-2.5 animate-spin text-indigo-400" />
        {stageLabel}
        {pct}
      </span>
    );
  }

  return (
    <span className="inline-flex items-center rounded-full bg-slate-800 px-2 py-0.5 text-[10px] text-slate-300">
      {doc.status}
    </span>
  );
}

export function EngineBadge({ doc }: { doc: DocumentItem }) {
  if (doc.engine_fallback_reason) {
    return (
      <span
        title={doc.engine_fallback_reason}
        className="inline-flex items-center gap-1 rounded-full bg-amber-500/15 border border-amber-500/30 px-2 py-0.5 text-[10px] font-medium text-amber-300 cursor-help"
      >
        <AlertTriangle className="h-2.5 w-2.5 text-amber-400" />
        Fallback
      </span>
    );
  }

  if (doc.parsing_engine === "docling") {
    return (
      <span
        title="Docling GPU Vision Engine"
        className="inline-flex items-center gap-1 rounded-full bg-purple-500/15 border border-purple-500/30 px-2 py-0.5 text-[10px] font-medium text-purple-300"
      >
        <Rocket className="h-2.5 w-2.5 text-purple-400" />
        Docling GPU
      </span>
    );
  }

  if (doc.parsing_engine === "pymupdf4llm") {
    return (
      <span
        title="PyMuPDF Fast CPU Engine"
        className="inline-flex items-center gap-1 rounded-full bg-sky-500/15 border border-sky-500/30 px-2 py-0.5 text-[10px] font-medium text-sky-300"
      >
        <Zap className="h-2.5 w-2.5 text-sky-400" />
        Fast CPU
      </span>
    );
  }

  if (doc.parsing_engine === "engineering_cad") {
    return (
      <span
        title="Engineering CAD Geometry & OCR Engine"
        className="inline-flex items-center gap-1 rounded-full bg-cyan-500/15 border border-cyan-500/30 px-2 py-0.5 text-[10px] font-medium text-cyan-300"
      >
        <Layers className="h-2.5 w-2.5 text-cyan-400" />
        Engineering CAD
      </span>
    );
  }

  return null;
}
