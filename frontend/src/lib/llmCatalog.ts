export type IngestionMode = "auto" | "gpu" | "fast";
export type ApprovalMode = "review_all" | "auto_high_confidence" | "auto_all";

export interface LlmProviderCatalogItem {
  id: string;
  label: string;
  shortLabel: string;
  defaultModelFallback: string;
}

export const LLM_PROVIDERS: LlmProviderCatalogItem[] = [
  {
    id: "openai",
    label: "OpenAI",
    shortLabel: "OpenAI",
    defaultModelFallback: "gpt-4o",
  },
  {
    id: "anthropic",
    label: "Anthropic",
    shortLabel: "Anthropic",
    defaultModelFallback: "claude-sonnet-5-5",
  },
  {
    id: "deepseek",
    label: "DeepSeek",
    shortLabel: "DeepSeek",
    defaultModelFallback: "deepseek-v4-pro",
  },
  {
    id: "openai_compatible",
    label: "OpenAI-Compatible (Local / vLLM)",
    shortLabel: "OpenAI-Compatible",
    defaultModelFallback: "glm-4-plus",
  },
];

export const MODEL_SUGGESTIONS: Record<string, string[]> = {
  openai: ["gpt-5", "gpt-5-mini", "gpt-4o", "gpt-4o-mini", "o3-mini", "o1"],
  deepseek: [
    "deepseek-v4-pro",
    "deepseek-flash",
    "deepseek-chat",
    "deepseek-reasoner",
  ],
  anthropic: [
    "claude-sonnet-5-5",
    "claude-opus-5-5",
    "claude-opus-4-8",
    "claude-sonnet-4-6",
    "claude-sonnet-4-5-20250929",
    "claude-haiku-4-5-20251001",
  ],
  openai_compatible: ["glm-4-plus", "qwen-plus", "llama-3.3-70b"],
};

export const INGESTION_MODE_OPTIONS: {
  id: IngestionMode;
  compactLabel: string;
  compactTooltip: string;
  compactActiveClass: string;
  settingsTitle: string;
  settingsDesc: string;
}[] = [
  {
    id: "auto",
    compactLabel: "⚡ Auto Engine",
    compactTooltip: "⚡ Auto (Recommended): Balanced heuristic routing",
    compactActiveClass: "bg-indigo-600 text-white shadow-xs font-semibold",
    settingsTitle: "Auto (Adaptive)",
    settingsDesc: "Docling with PyMuPDF4LLM auto-fallback",
  },
  {
    id: "gpu",
    compactLabel: "🚀 GPU Vision",
    compactTooltip: "🚀 GPU Vision: Docling OCR + layout analysis",
    compactActiveClass: "bg-purple-600 text-white shadow-xs font-semibold",
    settingsTitle: "GPU / Deep Vision",
    settingsDesc: "Force Docling for complex formulas and tables",
  },
  {
    id: "fast",
    compactLabel: "📄 Fast CPU",
    compactTooltip: "📄 Fast CPU: PyMuPDF high-speed extraction",
    compactActiveClass: "bg-sky-600 text-white shadow-xs font-semibold",
    settingsTitle: "Fast Path",
    settingsDesc: "Force PyMuPDF4LLM for instant parsing",
  },
];

export const APPROVAL_MODE_OPTIONS: {
  id: ApprovalMode;
  settingsTitle: string;
  settingsDesc: string;
}[] = [
  {
    id: "review_all",
    settingsTitle: "Review All (Default)",
    settingsDesc: "Pause all documents at awaiting_approval",
  },
  {
    id: "auto_high_confidence",
    settingsTitle: "Auto-Approve High Conf",
    settingsDesc: "Auto-approve if Laya confidence ≥ Threshold",
  },
  {
    id: "auto_all",
    settingsTitle: "Auto-Approve All",
    settingsDesc: "Direct stream to indexing without review",
  },
];
