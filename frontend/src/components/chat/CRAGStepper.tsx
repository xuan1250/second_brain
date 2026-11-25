import React, { useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Loader2,
} from "lucide-react";
import type { CRAGStep } from "../../lib/types";

const CRAG_PIPELINE_STEPS: { node: string; label: string }[] = [
  { node: "rewrite_query", label: "Rewriting & Decomposing" },
  { node: "retrieve", label: "Federated search" },
  { node: "rerank", label: "Reranking (FlashRank)" },
  { node: "grade_context", label: "Grading context" },
  { node: "synthesize", label: "Synthesizing" },
];

export interface CRAGStepperProps {
  steps: CRAGStep[];
  isStreaming?: boolean;
}

export const CRAGStepper: React.FC<CRAGStepperProps> = ({
  steps,
  isStreaming,
}) => {
  const [showSubqueries, setShowSubqueries] = useState(false);

  if ((!steps || steps.length === 0) && !isStreaming) return null;

  const completedNodes = new Set(steps.map((s) => s.node));
  const hasDirectReply = completedNodes.has("direct_reply");
  const hasFallback = completedNodes.has("fallback_notice");

  const displaySteps = hasDirectReply
    ? [
        { node: "rewrite_query", label: "Rewriting & Intent" },
        { node: "direct_reply", label: "Replying directly" },
      ]
    : hasFallback
    ? [
        ...CRAG_PIPELINE_STEPS.slice(0, 4),
        { node: "fallback_notice", label: "Fallback notice" },
      ]
    : CRAG_PIPELINE_STEPS;

  const stepWithPolysemy = steps.find(
    (s) => s.polysemy_collisions && s.polysemy_collisions.length > 0
  );
  const collisions = stepWithPolysemy?.polysemy_collisions || [];

  const stepWithQueries = steps.find(
    (s) => s.domain_queries && Object.keys(s.domain_queries).length > 0
  );
  const domainQueries = stepWithQueries?.domain_queries || {};

  const activeRoutingMode =
    steps.find((s) => s.routing_mode)?.routing_mode ||
    (Object.keys(domainQueries).length > 1 ? "federated" : "single");

  return (
    <div className="mb-2.5 rounded-xl border border-slate-800/90 bg-slate-900/70 p-2.5 text-[10px] space-y-2">
      <div className="flex flex-wrap items-center gap-1.5">
        {displaySteps.map((step, idx) => {
          const done = completedNodes.has(step.node);
          const prevDone =
            idx === 0 || completedNodes.has(displaySteps[idx - 1].node);
          const isCurrent = Boolean(isStreaming && !done && prevDone);

          return (
            <React.Fragment key={step.node}>
              {idx > 0 && <span className="text-slate-600">→</span>}
              <span
                className={`inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 font-medium transition ${
                  done
                    ? "bg-emerald-500/10 text-emerald-300"
                    : isCurrent
                    ? "bg-indigo-500/20 text-indigo-300 animate-pulse"
                    : "text-slate-500"
                }`}
              >
                {done ? (
                  <CheckCircle2 className="h-2.5 w-2.5 text-emerald-400" />
                ) : isCurrent ? (
                  <Loader2 className="h-2.5 w-2.5 animate-spin text-indigo-400" />
                ) : null}
                {step.label}
              </span>
            </React.Fragment>
          );
        })}

        {activeRoutingMode && (
          <span
            className={`ml-auto inline-flex items-center gap-1 rounded-full px-2 py-0.2 font-mono text-[9.5px] font-semibold border ${
              activeRoutingMode === "federated"
                ? "border-purple-500/40 bg-purple-500/15 text-purple-300"
                : "border-sky-500/40 bg-sky-500/15 text-sky-300"
            }`}
          >
            {activeRoutingMode === "federated" ? "🔀 Federated" : "🎯 Single"}
          </span>
        )}
      </div>

      {collisions.length > 0 && (
        <div className="rounded-lg border border-amber-500/30 bg-amber-950/30 p-2 text-amber-200 space-y-1">
          <div className="flex items-center gap-1.5 font-semibold text-amber-300">
            <AlertTriangle className="h-3.5 w-3.5 text-amber-400 shrink-0" />
            <span>Cross-Domain Polysemic Collisions Detected:</span>
          </div>
          {collisions.map((col, cIdx) => (
            <div key={cIdx} className="text-[10px] text-amber-100/90 pl-5">
              <strong>"{col.term}"</strong> across [
              {Array.isArray(col.domains) ? col.domains.join(", ") : ""}]:{" "}
              <span>{col.contrastive_warning}</span>
            </div>
          ))}
        </div>
      )}

      {Object.keys(domainQueries).length > 0 && (
        <div className="border-t border-slate-800/80 pt-1.5">
          <button
            type="button"
            onClick={() => setShowSubqueries((v) => !v)}
            className="flex items-center gap-1 text-[10px] text-indigo-300 hover:text-indigo-200 transition"
          >
            {showSubqueries ? (
              <ChevronDown className="h-3 w-3" />
            ) : (
              <ChevronRight className="h-3 w-3" />
            )}
            <span className="font-medium">
              Domain Sub-Queries ({Object.keys(domainQueries).length})
            </span>
          </button>
          {showSubqueries && (
            <div className="mt-1.5 grid gap-1 pl-4">
              {Object.entries(domainQueries).map(([dom, qText]) => (
                <div key={dom} className="text-[10px]">
                  <span className="font-semibold uppercase text-indigo-400">
                    [{dom}]:
                  </span>{" "}
                  <span className="text-slate-300 font-mono italic">
                    "{qText}"
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};
