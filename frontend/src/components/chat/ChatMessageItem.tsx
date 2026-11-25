import React from "react";
import { Bot, User } from "lucide-react";
import { resolveDomainMeta } from "../../lib/domainTheme";
import type { ChatMessage, DomainItem } from "../../lib/types";
import { ErrorBoundary } from "../common/ErrorBoundary";
import { AssistantMessageContent } from "./AssistantMessageContent";

export interface ChatMessageItemProps {
  message: ChatMessage;
  domains: DomainItem[];
}

export const ChatMessageItem: React.FC<ChatMessageItemProps> = ({
  message,
  domains,
}) => {
  const isUser = message.role === "user";

  let scopeBadge: React.ReactNode = null;
  if (isUser && message.domain_filter) {
    const df = message.domain_filter;
    const hasDomains = Array.isArray(df.domains) && df.domains.length > 0;
    const isT2 = df.max_authority_tier === 2;

    if (hasDomains) {
      const domNames = df.domains
        .map((d) => resolveDomainMeta(d, domains).shortLabel)
        .join(", ");
      scopeBadge = (
        <span className="inline-flex items-center gap-1 rounded-full border border-indigo-500/40 bg-indigo-950/60 px-2 py-0.2 text-[9.5px] font-mono text-indigo-300">
          <span>{df.mode === "federated" ? "🔀" : "🎯"}</span>
          <span>{domNames}</span>
          {isT2 && <span className="text-amber-300">(T1–2)</span>}
        </span>
      );
    } else if (isT2) {
      scopeBadge = (
        <span className="inline-flex items-center gap-1 rounded-full border border-amber-500/40 bg-amber-950/60 px-2 py-0.2 text-[9.5px] font-mono text-amber-300">
          <span>🛡️ T1–2 Only</span>
        </span>
      );
    }
  }

  return (
    <div className={`flex gap-2.5 ${isUser ? "justify-end" : "justify-start"}`}>
      {!isUser && (
        <div className="flex h-6 w-6 shrink-0 items-center justify-center rounded-lg bg-indigo-600/20 border border-indigo-500/30 text-indigo-400 mt-0.5">
          <Bot className="h-3.5 w-3.5" />
        </div>
      )}

      <div
        className={`max-w-[88%] rounded-2xl px-3.5 py-2.5 ${
          isUser
            ? "bg-indigo-600 text-white text-xs leading-relaxed"
            : "bg-slate-900/90 border border-slate-800/90 text-slate-100 w-full"
        }`}
      >
        {isUser ? (
          <div>
            {scopeBadge && <div className="mb-1.5">{scopeBadge}</div>}
            <p className="whitespace-pre-wrap break-words">{message.content}</p>
          </div>
        ) : (
          <ErrorBoundary fallbackTitle="Response Render Error">
            <AssistantMessageContent message={message} />
          </ErrorBoundary>
        )}
      </div>

      {isUser && (
        <div className="flex h-6 w-6 shrink-0 items-center justify-center rounded-lg bg-slate-800 text-slate-300 mt-0.5">
          <User className="h-3.5 w-3.5" />
        </div>
      )}
    </div>
  );
};
