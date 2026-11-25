import React, { useEffect, useState } from "react";
import * as Dialog from "@radix-ui/react-dialog";
import * as Tabs from "@radix-ui/react-tabs";
import { BookOpen, Layers, Settings, Sliders, X } from "lucide-react";
import { useActiveProviderStatus } from "../../hooks/useActiveProviderStatus";
import { useDomainsQuery } from "../../hooks/useDomains";
import { useLexiconQuery } from "../../hooks/useLexicon";
import { useWorkspaceSettingsQuery } from "../../hooks/useWorkspaceSettings";
import { StatusDot } from "../common/StatusDot";
import { DomainsTab } from "./DomainsTab";
import { IngestionDefaultsTab } from "./IngestionDefaultsTab";
import { LexiconTab } from "./LexiconTab";

export interface SettingsModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  initialTab?: "domains" | "lexicon" | "ingestion";
}

export const SettingsModal: React.FC<SettingsModalProps> = ({
  open,
  onOpenChange,
  initialTab = "domains",
}) => {
  const { data: domains = [], isLoading: isLoadingDomains } =
    useDomainsQuery(true);
  const { data: wsSettings } = useWorkspaceSettingsQuery();
  const { data: lexiconEntries = [] } = useLexiconQuery();

  const [activeTab, setActiveTab] = useState<string>(initialTab);

  useEffect(() => {
    if (open && initialTab) {
      setActiveTab(initialTab);
    }
  }, [open, initialTab]);

  const { hasKeyIssue, isInvalidKey } = useActiveProviderStatus({ wsSettings });

  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-50 bg-black/75 backdrop-blur-xs transition-opacity animate-in fade-in" />
        <Dialog.Content className="fixed left-1/2 top-1/2 z-50 w-full max-w-4xl -translate-x-1/2 -translate-y-1/2 rounded-2xl border border-slate-800 bg-slate-950 p-6 shadow-2xl focus:outline-none max-h-[90vh] flex flex-col">
          <div className="flex items-center justify-between border-b border-slate-800/80 pb-4">
            <div className="flex items-center gap-2.5">
              <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-indigo-600/20 border border-indigo-500/30 text-indigo-400">
                <Settings className="h-5 w-5" />
              </div>
              <div>
                <Dialog.Title className="text-base font-bold text-white tracking-wide">
                  Workspace Settings & Dynamic Taxonomy
                </Dialog.Title>
                <Dialog.Description className="text-xs text-slate-400">
                  Manage domains, subdomains, polysemy definitions, and
                  ingestion AI defaults.
                </Dialog.Description>
              </div>
            </div>
            <Dialog.Close asChild>
              <button
                type="button"
                className="rounded-lg p-1.5 text-slate-400 hover:bg-slate-800 hover:text-white transition"
              >
                <X className="h-4 w-4" />
              </button>
            </Dialog.Close>
          </div>

          <Tabs.Root
            value={activeTab}
            onValueChange={setActiveTab}
            className="flex-1 flex flex-col overflow-hidden pt-4"
          >
            <Tabs.List className="flex items-center gap-2 border-b border-slate-800/80 pb-2 text-xs font-medium">
              <Tabs.Trigger
                value="domains"
                className={`flex items-center gap-2 rounded-lg px-3 py-1.5 transition ${
                  activeTab === "domains"
                    ? "bg-indigo-600 text-white font-semibold shadow-xs"
                    : "text-slate-400 hover:bg-slate-900 hover:text-slate-200"
                }`}
              >
                <Layers className="h-3.5 w-3.5" />
                <span>Domains & Subdomains ({domains.length})</span>
              </Tabs.Trigger>

              <Tabs.Trigger
                value="lexicon"
                className={`flex items-center gap-2 rounded-lg px-3 py-1.5 transition ${
                  activeTab === "lexicon"
                    ? "bg-indigo-600 text-white font-semibold shadow-xs"
                    : "text-slate-400 hover:bg-slate-900 hover:text-slate-200"
                }`}
              >
                <BookOpen className="h-3.5 w-3.5" />
                <span>Polysemy Lexicon ({lexiconEntries.length})</span>
              </Tabs.Trigger>

              <Tabs.Trigger
                value="ingestion"
                className={`relative flex items-center gap-2 rounded-lg px-3 py-1.5 transition ${
                  activeTab === "ingestion"
                    ? "bg-indigo-600 text-white font-semibold shadow-xs"
                    : "text-slate-400 hover:bg-slate-900 hover:text-slate-200"
                }`}
              >
                <Sliders className="h-3.5 w-3.5" />
                <span>Ingestion & AI Defaults</span>
                {hasKeyIssue && (
                  <StatusDot
                    tone={isInvalidKey ? "error" : "warning"}
                    pulse={isInvalidKey}
                  />
                )}
              </Tabs.Trigger>
            </Tabs.List>

            <Tabs.Content
              value="domains"
              className="flex-1 overflow-y-auto pt-4 space-y-4 pr-1"
            >
              <DomainsTab domains={domains} isLoading={isLoadingDomains} />
            </Tabs.Content>

            <Tabs.Content
              value="lexicon"
              className="flex-1 overflow-y-auto pt-4 space-y-4 pr-1"
            >
              <LexiconTab domains={domains} />
            </Tabs.Content>

            <Tabs.Content
              value="ingestion"
              className="flex-1 overflow-y-auto pt-4 space-y-4 pr-1"
            >
              <IngestionDefaultsTab open={open} wsSettings={wsSettings} />
            </Tabs.Content>
          </Tabs.Root>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
};
