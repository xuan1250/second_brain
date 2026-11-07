import React, { useRef, useState } from "react";
import {
  type ImperativePanelHandle,
  Panel,
  PanelGroup,
  PanelResizeHandle,
} from "react-resizable-panels";
import {
  BrainCircuit,
  PanelLeftClose,
  PanelLeftOpen,
  PanelRightClose,
  PanelRightOpen,
  Settings,
} from "lucide-react";
import { useSystemStats } from "../../hooks/useDocuments";
import { ChatAssistant } from "../chat/ChatAssistant";
import { KnowledgeLibrary } from "../library/KnowledgeLibrary";
import { DocumentInspector } from "../viewer/DocumentInspector";
import { SettingsModal } from "../settings/SettingsModal";
import { ErrorBoundary } from "../common/ErrorBoundary";

export const WorkspaceLayout: React.FC = () => {
  const leftPanelRef = useRef<ImperativePanelHandle | null>(null);
  const rightPanelRef = useRef<ImperativePanelHandle | null>(null);

  const [leftCollapsed, setLeftCollapsed] = useState(false);
  const [rightCollapsed, setRightCollapsed] = useState(false);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);

  const { data: stats } = useSystemStats();

  const toggleLeftPanel = () => {
    const panel = leftPanelRef.current;
    if (!panel) return;
    if (panel.isCollapsed()) {
      panel.expand();
    } else {
      panel.collapse();
    }
  };

  const toggleRightPanel = () => {
    const panel = rightPanelRef.current;
    if (!panel) return;
    if (panel.isCollapsed()) {
      panel.expand();
    } else {
      panel.collapse();
    }
  };

  const classifierName =
    stats?.classifier?.model_name?.split("/").pop() || "Laya-Multilingual";
  const isClassifierResident = Boolean(
    stats?.classifier?.resident || stats?.memory?.classifier_resident
  );

  return (
    <div className="flex h-screen w-screen flex-col overflow-hidden bg-slate-950 text-slate-100">
      {/* Top Application Bar with Telemetry */}
      <header className="flex h-11 shrink-0 items-center justify-between border-b border-slate-800/90 bg-slate-950 px-3.5 select-none">
        <div className="flex items-center gap-2.5">
          <button
            type="button"
            onClick={toggleLeftPanel}
            title={
              leftCollapsed
                ? "Expand Knowledge Library"
                : "Collapse Knowledge Library"
            }
            className="rounded-lg border border-slate-800 bg-slate-900 p-1.5 text-slate-400 hover:bg-slate-800 hover:text-slate-200 transition"
          >
            {leftCollapsed ? (
              <PanelLeftOpen className="h-3.5 w-3.5" />
            ) : (
              <PanelLeftClose className="h-3.5 w-3.5" />
            )}
          </button>

          <div className="flex items-center gap-2">
            <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-gradient-to-br from-indigo-500 to-violet-600 text-white shadow-sm">
              <BrainCircuit className="h-4 w-4" />
            </div>
            <div>
              <h1 className="text-xs font-bold tracking-wide text-white">
                SECOND BRAIN WORKSPACE
              </h1>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2.5 text-[11px] text-slate-400">
          {/* Laya Domain Classifier Telemetry Badge */}
          {stats?.classifier && (
            <span
              title={`Classifier: ${stats.classifier.model_name || "convaiinnovations/laya-multilingual"}${
                isClassifierResident ? " (Resident in Memory)" : " (On-Demand)"
              }${
                stats.classifier.high_confidence_threshold
                  ? ` · Auto-approve threshold: ${Math.round(stats.classifier.high_confidence_threshold * 100)}%`
                  : ""
              }`}
              className="hidden sm:inline-flex items-center gap-1.5 rounded-full border border-slate-800 bg-slate-900/90 px-2.5 py-0.5 cursor-help"
            >
              <span
                className={`h-1.5 w-1.5 rounded-full ${
                  isClassifierResident
                    ? "bg-emerald-400 shadow-[0_0_6px_rgba(52,211,153,0.8)] animate-pulse"
                    : "bg-amber-400"
                }`}
              />
              <span className="font-medium text-slate-300">
                🧠 Classifier: {classifierName}
              </span>
            </span>
          )}

          {stats?.embedding?.model_name && (
            <span className="hidden md:inline-flex items-center gap-1.5 rounded-full border border-slate-800 bg-slate-900/90 px-2.5 py-0.5">
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
              <span>Embedder: {stats.embedding.model_name}</span>
            </span>
          )}

          {stats?.reranker?.model_name && (
            <span className="hidden lg:inline-flex items-center gap-1.5 rounded-full border border-slate-800 bg-slate-900/90 px-2.5 py-0.5">
              <span>Reranker: {stats.reranker.model_name}</span>
            </span>
          )}

          <button
            type="button"
            onClick={() => setIsSettingsOpen(true)}
            title="Workspace Settings & Dynamic Taxonomy"
            className="flex items-center gap-1.5 rounded-lg border border-slate-800 bg-slate-900 px-2.5 py-1 text-slate-300 hover:bg-slate-800 hover:text-white transition cursor-pointer"
          >
            <Settings className="h-3.5 w-3.5 text-indigo-400" />
            <span className="font-medium">Settings</span>
          </button>

          <button
            type="button"
            onClick={toggleRightPanel}
            title={
              rightCollapsed
                ? "Expand Intelligent Assistant"
                : "Collapse Intelligent Assistant"
            }
            className="rounded-lg border border-slate-800 bg-slate-900 p-1.5 text-slate-400 hover:bg-slate-800 hover:text-slate-200 transition"
          >
            {rightCollapsed ? (
              <PanelRightOpen className="h-3.5 w-3.5" />
            ) : (
              <PanelRightClose className="h-3.5 w-3.5" />
            )}
          </button>
        </div>

        <SettingsModal
          open={isSettingsOpen}
          onOpenChange={setIsSettingsOpen}
        />
      </header>

      {/* Three-Pane Resizable Split Workspace */}
      <div className="flex-1 overflow-hidden">
        <PanelGroup direction="horizontal" autoSaveId="second-brain-workspace">
          {/* Pane 1: Knowledge Library */}
          <Panel
            ref={leftPanelRef}
            id="pane-library"
            order={1}
            defaultSize={23}
            minSize={16}
            maxSize={40}
            collapsible={true}
            collapsedSize={0}
            onCollapse={() => setLeftCollapsed(true)}
            onExpand={() => setLeftCollapsed(false)}
          >
            <ErrorBoundary fallbackTitle="Knowledge Library Error">
              <KnowledgeLibrary />
            </ErrorBoundary>
          </Panel>

          <PanelResizeHandle className="w-1 bg-slate-900 hover:bg-indigo-500/60 active:bg-indigo-500 transition-colors cursor-col-resize" />

          {/* Pane 2: Document Inspector */}
          <Panel
            id="pane-inspector"
            order={2}
            defaultSize={45}
            minSize={25}
          >
            <ErrorBoundary fallbackTitle="Document Inspector Error">
              <DocumentInspector />
            </ErrorBoundary>
          </Panel>

          <PanelResizeHandle className="w-1 bg-slate-900 hover:bg-indigo-500/60 active:bg-indigo-500 transition-colors cursor-col-resize" />

          {/* Pane 3: Intelligent Assistant */}
          <Panel
            ref={rightPanelRef}
            id="pane-assistant"
            order={3}
            defaultSize={32}
            minSize={22}
            maxSize={50}
            collapsible={true}
            collapsedSize={0}
            onCollapse={() => setRightCollapsed(true)}
            onExpand={() => setRightCollapsed(false)}
          >
            <ErrorBoundary fallbackTitle="Intelligent Assistant Error">
              <ChatAssistant />
            </ErrorBoundary>
          </Panel>
        </PanelGroup>
      </div>
    </div>
  );
};
