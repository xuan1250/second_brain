import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Document, Page, pdfjs } from "react-pdf";
import {
  ChevronLeft,
  ChevronRight,
  ExternalLink,
  Sparkles,
} from "lucide-react";
import { getDocumentRawUrl } from "../../lib/api";
import type { ActiveCitation } from "../../lib/types";
import { CitationExcerptCard } from "../common/CitationExcerptCard";
import { LoadingState } from "../common/LoadingState";

pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/build/pdf.worker.min.mjs",
  import.meta.url
).toString();

export interface PdfViewerPaneProps {
  docId: string;
  filename: string;
  scale: number;
  activeCitation: ActiveCitation | null;
}

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function extractHighlightTokens(excerpt: string | undefined | null): string[] {
  if (!excerpt) return [];
  const words = excerpt
    .replace(/[^\p{L}\p{N}\s.-]/gu, " ")
    .split(/\s+/)
    .map((w) => w.trim())
    .filter((w) => w.length >= 4);
  return Array.from(new Set(words)).slice(0, 12);
}

export const PdfViewerPane: React.FC<PdfViewerPaneProps> = ({
  docId,
  filename,
  scale,
  activeCitation,
}) => {
  const [numPages, setNumPages] = useState<number>(1);
  const [pageNumber, setPageNumber] = useState<number>(1);
  const [useNativeFallback, setUseNativeFallback] = useState<boolean>(false);
  const [pulseActive, setPulseActive] = useState<boolean>(false);
  const containerRef = useRef<HTMLDivElement | null>(null);

  const rawUrl = useMemo(() => getDocumentRawUrl(docId), [docId]);

  const isCitedDoc = activeCitation?.doc_id === docId;
  const citedPage =
    isCitedDoc &&
    activeCitation?.page_numbers &&
    activeCitation.page_numbers.length > 0
      ? activeCitation.page_numbers[0]
      : null;

  useEffect(() => {
    if (citedPage && citedPage >= 1) {
      setPageNumber(citedPage);
    } else {
      setPageNumber(1);
    }
    setUseNativeFallback(false);
  }, [docId, citedPage]);

  useEffect(() => {
    if (isCitedDoc && citedPage && citedPage >= 1) {
      setPageNumber(Math.min(citedPage, numPages || citedPage));
      setPulseActive(true);
      containerRef.current?.scrollTo({ top: 0, behavior: "smooth" });
      const timer = window.setTimeout(() => setPulseActive(false), 4500);
      return () => window.clearTimeout(timer);
    }
  }, [isCitedDoc, citedPage, numPages, activeCitation?.triggered_at]);

  const highlightTokens = useMemo(() => {
    if (!isCitedDoc || !activeCitation?.excerpt) return [];
    if (citedPage && citedPage !== pageNumber) return [];
    return extractHighlightTokens(activeCitation.excerpt);
  }, [isCitedDoc, activeCitation?.excerpt, citedPage, pageNumber]);

  const customTextRenderer = useCallback(
    ({ str }: { str: string }) => {
      const safe = escapeHtml(str);
      if (highlightTokens.length === 0) return safe;
      const lower = str.toLowerCase();
      const matched = highlightTokens.some((tok) =>
        lower.includes(tok.toLowerCase())
      );
      if (!matched) return safe;
      return `<mark class="pdf-citation-mark">${safe}</mark>`;
    },
    [highlightTokens]
  );

  return (
    <div className="flex h-full flex-col bg-slate-950">
      <div className="flex items-center justify-between border-b border-slate-800/80 bg-slate-900/60 px-3 py-1.5 text-xs">
        <div className="flex items-center gap-1.5">
          <button
            type="button"
            disabled={pageNumber <= 1}
            onClick={() => setPageNumber((p) => Math.max(1, p - 1))}
            className="rounded-md border border-slate-700/80 bg-slate-800/80 p-1 text-slate-300 hover:bg-slate-700 disabled:opacity-40"
            aria-label="Previous page"
          >
            <ChevronLeft className="h-3.5 w-3.5" />
          </button>
          <span className="px-1.5 font-mono text-[11px] text-slate-300">
            Page <strong>{pageNumber}</strong> of <strong>{numPages}</strong>
          </span>
          <button
            type="button"
            disabled={pageNumber >= numPages}
            onClick={() => setPageNumber((p) => Math.min(numPages, p + 1))}
            className="rounded-md border border-slate-700/80 bg-slate-800/80 p-1 text-slate-300 hover:bg-slate-700 disabled:opacity-40"
            aria-label="Next page"
          >
            <ChevronRight className="h-3.5 w-3.5" />
          </button>

          {citedPage && (
            <button
              type="button"
              onClick={() => setPageNumber(citedPage)}
              className="ml-2 inline-flex items-center gap-1 rounded-full bg-indigo-500/20 border border-indigo-500/40 px-2 py-0.5 text-[11px] font-medium text-indigo-300 hover:bg-indigo-500/30"
            >
              <Sparkles className="h-3 w-3 text-indigo-400" />
              Cited p.{citedPage}
            </button>
          )}
        </div>

        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => setUseNativeFallback((v) => !v)}
            className="rounded-md border border-slate-800 bg-slate-900 px-2 py-0.5 text-[11px] text-slate-400 hover:text-slate-200"
          >
            {useNativeFallback ? "Use Interactive PDF" : "Native Frame"}
          </button>
          <a
            href={`${rawUrl}#page=${pageNumber}`}
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-1 text-[11px] text-slate-400 hover:text-indigo-300"
            title={`Open ${filename} in new tab`}
          >
            <ExternalLink className="h-3 w-3" />
            Raw
          </a>
        </div>
      </div>

      <div
        ref={containerRef}
        className="flex-1 overflow-auto p-4 flex flex-col items-center bg-slate-950/90"
      >
        {useNativeFallback ? (
          <iframe
            key={`${rawUrl}-${pageNumber}`}
            src={`${rawUrl}#page=${pageNumber}`}
            title={filename}
            className="h-full w-full rounded-lg border border-slate-800 bg-white"
          />
        ) : (
          <div
            className={`transition-all duration-300 rounded-lg ${
              pulseActive ? "citation-pulse-highlight p-1" : ""
            }`}
          >
            <Document
              file={rawUrl}
              onLoadSuccess={({ numPages: loadedPages }) => {
                setNumPages(loadedPages || 1);
                if (citedPage && citedPage <= loadedPages) {
                  setPageNumber(citedPage);
                }
              }}
              onLoadError={() => {
                setUseNativeFallback(true);
              }}
              loading={
                <LoadingState label="Rendering PDF document..." className="py-20" />
              }
            >
              <Page
                pageNumber={pageNumber}
                scale={scale}
                renderTextLayer={true}
                renderAnnotationLayer={false}
                customTextRenderer={customTextRenderer}
                className="shadow-2xl rounded-md overflow-hidden"
              />
            </Document>
          </div>
        )}

        {isCitedDoc && activeCitation?.excerpt && (
          <CitationExcerptCard
            className="mt-4 w-full max-w-2xl shadow-lg"
            title={
              <>
                Active Citation Excerpt
                {activeCitation.heading_path
                  ? ` · §${activeCitation.heading_path}`
                  : ""}
              </>
            }
            rightMeta={citedPage ? `Page ${citedPage}` : undefined}
            excerpt={activeCitation.excerpt}
          />
        )}
      </div>
    </div>
  );
};
