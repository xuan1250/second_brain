import React, { useMemo } from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import { Cpu, Loader2 } from "lucide-react";
import { useDomainsQuery } from "../../hooks/useDomains";
import { resolveDomainMeta } from "../../lib/domainTheme";
import type { ChatMessage, Citation } from "../../lib/types";
import { AlertBanner } from "../common/AlertBanner";
import { CitationPill } from "./CitationPill";
import { CRAGStepper } from "./CRAGStepper";
import { SourcesDrawer } from "./SourcesDrawer";

function injectCitationLinks(markdown: string): string {
  if (!markdown) return "";
  const parts = markdown.split(/(```[\s\S]*?```|`[^`\n]*`)/g);
  return parts
    .map((part, idx) => {
      if (idx % 2 === 1) return part;
      return part.replace(/\[(\d+)\](?!\()/g, "[$1](#citation-$1)");
    })
    .join("");
}

export const AssistantMessageContent: React.FC<{
  message: ChatMessage;
}> = ({ message }) => {
  const { data: domains = [] } = useDomainsQuery();

  const citationMap = useMemo(() => {
    const map = new Map<number, Citation>();
    for (const c of message.citations || []) {
      map.set(c.citation_index, c);
    }
    return map;
  }, [message.citations]);

  const markdownWithLinks = useMemo(
    () => injectCitationLinks(message.content),
    [message.content]
  );

  const markdownComponents = useMemo<Components>(
    () => ({
      a: ({ href, children }) => {
        if (href && href.startsWith("#citation-")) {
          const idxNum = Number.parseInt(href.replace("#citation-", ""), 10);
          if (!Number.isNaN(idxNum)) {
            return (
              <CitationPill
                index={idxNum}
                citation={citationMap.get(idxNum)}
              />
            );
          }
        }
        return (
          <a
            href={href}
            target="_blank"
            rel="noreferrer"
            className="text-indigo-400 underline hover:text-indigo-300"
          >
            {children}
          </a>
        );
      },
      h2: ({ children }) => {
        const text = String(children);
        const lower = text.toLowerCase();
        let barCls =
          "border-l-4 border-indigo-500 bg-indigo-950/20 text-indigo-200";

        if (
          lower.includes("conflict") ||
          lower.includes("trade-off") ||
          lower.includes("xung đột") ||
          lower.includes("matrix")
        ) {
          barCls =
            "border-l-4 border-violet-400 bg-violet-950/40 text-violet-200";
        } else {
          for (const d of domains) {
            const dName = d.name.toLowerCase();
            const dSlug = d.slug.toLowerCase();
            const dEpistemic = (d.epistemic_type || "").toLowerCase();
            if (
              lower.includes(dName) ||
              lower.includes(dSlug) ||
              (dEpistemic && lower.includes(dEpistemic))
            ) {
              barCls = resolveDomainMeta(d.slug, domains).theme.heading.h2;
              break;
            }
          }
        }

        return (
          <h2
            className={`mt-4 mb-2 rounded-r-lg px-2.5 py-1 text-sm font-bold ${barCls}`}
          >
            {children}
          </h2>
        );
      },
      h3: ({ children }) => {
        const text = String(children);
        const lower = text.toLowerCase();
        let barCls = "text-indigo-300";

        if (
          lower.includes("conflict") ||
          lower.includes("priority") ||
          lower.includes("matrix")
        ) {
          barCls = "text-violet-300 border-b border-violet-500/30 pb-0.5";
        } else {
          for (const d of domains) {
            const dName = d.name.toLowerCase();
            const dSlug = d.slug.toLowerCase();
            const dEpistemic = (d.epistemic_type || "").toLowerCase();
            if (
              lower.includes(dName) ||
              lower.includes(dSlug) ||
              (dEpistemic && lower.includes(dEpistemic))
            ) {
              barCls = resolveDomainMeta(d.slug, domains).theme.heading.h3;
              break;
            }
          }
        }

        return (
          <h3 className={`mt-3 mb-1 text-xs font-semibold ${barCls}`}>
            {children}
          </h3>
        );
      },
      table: ({ children }) => (
        <div className="my-2.5 overflow-x-auto rounded-xl border border-slate-700/80 bg-slate-950/70 p-1 shadow-inner">
          <table className="w-full border-collapse text-xs">{children}</table>
        </div>
      ),
    }),
    [citationMap, domains]
  );

  if (message.isError) {
    return (
      <AlertBanner
        variant="error"
        title="Assistant Error"
        message={message.content}
      />
    );
  }

  return (
    <div>
      {message.steps && message.steps.length > 0 && (
        <CRAGStepper steps={message.steps} isStreaming={message.isStreaming} />
      )}

      {message.content ? (
        <div className="prose prose-invert markdown-prose-shared max-w-none text-xs leading-relaxed text-slate-100 space-y-2 [&_p]:my-1.5 [&_ul]:pl-4 [&_ol]:pl-4 [&_pre]:bg-slate-950 [&_pre]:p-2.5">
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={markdownComponents}
          >
            {markdownWithLinks}
          </ReactMarkdown>
        </div>
      ) : message.isStreaming ? (
        <div className="flex items-center gap-2 py-1 text-xs text-slate-400">
          <Loader2 className="h-3.5 w-3.5 animate-spin text-indigo-400" />
          <span>Running Multi-Domain CRAG pipeline...</span>
        </div>
      ) : null}

      <SourcesDrawer citations={message.citations} />

      {message.model && (
        <div className="mt-2 flex items-center gap-1.5 text-[10px] text-slate-500 font-mono">
          <Cpu className="h-3 w-3 text-indigo-400/80" />
          <span>
            {message.provider ? `${message.provider} / ` : ""}
            <strong className="text-slate-400 font-semibold">
              {message.model}
            </strong>
          </span>
        </div>
      )}
    </div>
  );
};
