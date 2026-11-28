import type { DomainColorToken } from "./types";

export interface DomainColorTheme {
  token: DomainColorToken;
  label: string;
  swatchBg: string;
  badge: string;
  activePill: string;
  citation: {
    border: string;
    bg: string;
    text: string;
    activeBorder: string;
    activeBg: string;
    activeText: string;
  };
  heading: {
    h2: string;
    h3: string;
  };
}

/**
 * Static Tailwind v4 palette lookup table.
 * NO dynamic string interpolation (bg-${color}-500) to ensure static scanner catches all classes.
 */
export const DOMAIN_COLOR_PALETTE: Record<DomainColorToken, DomainColorTheme> = {
  cyan: {
    token: "cyan",
    label: "Cyan",
    swatchBg: "bg-cyan-500",
    badge: "bg-cyan-500/15 border-cyan-500/30 text-cyan-300",
    activePill: "bg-cyan-500/25 border-cyan-400 text-cyan-200",
    citation: {
      border: "border-cyan-500/40",
      bg: "bg-cyan-500/15",
      text: "text-cyan-300",
      activeBorder: "border-cyan-400",
      activeBg: "bg-cyan-500/30",
      activeText: "text-cyan-100",
    },
    heading: {
      h2: "border-l-4 border-cyan-400 bg-cyan-950/30 text-cyan-200",
      h3: "text-cyan-300 border-b border-cyan-500/30 pb-0.5",
    },
  },
  emerald: {
    token: "emerald",
    label: "Emerald",
    swatchBg: "bg-emerald-500",
    badge: "bg-emerald-500/15 border-emerald-500/30 text-emerald-300",
    activePill: "bg-emerald-500/25 border-emerald-400 text-emerald-200",
    citation: {
      border: "border-emerald-500/40",
      bg: "bg-emerald-500/15",
      text: "text-emerald-300",
      activeBorder: "border-emerald-400",
      activeBg: "bg-emerald-500/30",
      activeText: "text-emerald-100",
    },
    heading: {
      h2: "border-l-4 border-emerald-400 bg-emerald-950/30 text-emerald-200",
      h3: "text-emerald-300 border-b border-emerald-500/30 pb-0.5",
    },
  },
  amber: {
    token: "amber",
    label: "Amber",
    swatchBg: "bg-amber-500",
    badge: "bg-amber-500/15 border-amber-500/30 text-amber-300",
    activePill: "bg-amber-500/25 border-amber-400 text-amber-200",
    citation: {
      border: "border-amber-500/40",
      bg: "bg-amber-500/15",
      text: "text-amber-300",
      activeBorder: "border-amber-400",
      activeBg: "bg-amber-500/30",
      activeText: "text-amber-100",
    },
    heading: {
      h2: "border-l-4 border-amber-400 bg-amber-950/30 text-amber-200",
      h3: "text-amber-300 border-b border-amber-500/30 pb-0.5",
    },
  },
  rose: {
    token: "rose",
    label: "Rose",
    swatchBg: "bg-rose-500",
    badge: "bg-rose-500/15 border-rose-500/30 text-rose-300",
    activePill: "bg-rose-500/25 border-rose-400 text-rose-200",
    citation: {
      border: "border-rose-500/40",
      bg: "bg-rose-500/15",
      text: "text-rose-300",
      activeBorder: "border-rose-400",
      activeBg: "bg-rose-500/30",
      activeText: "text-rose-100",
    },
    heading: {
      h2: "border-l-4 border-rose-400 bg-rose-950/30 text-rose-200",
      h3: "text-rose-300 border-b border-rose-500/30 pb-0.5",
    },
  },
  violet: {
    token: "violet",
    label: "Violet",
    swatchBg: "bg-violet-500",
    badge: "bg-violet-500/15 border-violet-500/30 text-violet-300",
    activePill: "bg-violet-500/25 border-violet-400 text-violet-200",
    citation: {
      border: "border-violet-500/40",
      bg: "bg-violet-500/15",
      text: "text-violet-300",
      activeBorder: "border-violet-400",
      activeBg: "bg-violet-500/30",
      activeText: "text-violet-100",
    },
    heading: {
      h2: "border-l-4 border-violet-400 bg-violet-950/30 text-violet-200",
      h3: "text-violet-300 border-b border-violet-500/30 pb-0.5",
    },
  },
  blue: {
    token: "blue",
    label: "Blue",
    swatchBg: "bg-blue-500",
    badge: "bg-blue-500/15 border-blue-500/30 text-blue-300",
    activePill: "bg-blue-500/25 border-blue-400 text-blue-200",
    citation: {
      border: "border-blue-500/40",
      bg: "bg-blue-500/15",
      text: "text-blue-300",
      activeBorder: "border-blue-400",
      activeBg: "bg-blue-500/30",
      activeText: "text-blue-100",
    },
    heading: {
      h2: "border-l-4 border-blue-400 bg-blue-950/30 text-blue-200",
      h3: "text-blue-300 border-b border-blue-500/30 pb-0.5",
    },
  },
  teal: {
    token: "teal",
    label: "Teal",
    swatchBg: "bg-teal-500",
    badge: "bg-teal-500/15 border-teal-500/30 text-teal-300",
    activePill: "bg-teal-500/25 border-teal-400 text-teal-200",
    citation: {
      border: "border-teal-500/40",
      bg: "bg-teal-500/15",
      text: "text-teal-300",
      activeBorder: "border-teal-400",
      activeBg: "bg-teal-500/30",
      activeText: "text-teal-100",
    },
    heading: {
      h2: "border-l-4 border-teal-400 bg-teal-950/30 text-teal-200",
      h3: "text-teal-300 border-b border-teal-500/30 pb-0.5",
    },
  },
  orange: {
    token: "orange",
    label: "Orange",
    swatchBg: "bg-orange-500",
    badge: "bg-orange-500/15 border-orange-500/30 text-orange-300",
    activePill: "bg-orange-500/25 border-orange-400 text-orange-200",
    citation: {
      border: "border-orange-500/40",
      bg: "bg-orange-500/15",
      text: "text-orange-300",
      activeBorder: "border-orange-400",
      activeBg: "bg-orange-500/30",
      activeText: "text-orange-100",
    },
    heading: {
      h2: "border-l-4 border-orange-400 bg-orange-950/30 text-orange-200",
      h3: "text-orange-300 border-b border-orange-500/30 pb-0.5",
    },
  },
  fuchsia: {
    token: "fuchsia",
    label: "Fuchsia",
    swatchBg: "bg-fuchsia-500",
    badge: "bg-fuchsia-500/15 border-fuchsia-500/30 text-fuchsia-300",
    activePill: "bg-fuchsia-500/25 border-fuchsia-400 text-fuchsia-200",
    citation: {
      border: "border-fuchsia-500/40",
      bg: "bg-fuchsia-500/15",
      text: "text-fuchsia-300",
      activeBorder: "border-fuchsia-400",
      activeBg: "bg-fuchsia-500/30",
      activeText: "text-fuchsia-100",
    },
    heading: {
      h2: "border-l-4 border-fuchsia-400 bg-fuchsia-950/30 text-fuchsia-200",
      h3: "text-fuchsia-300 border-b border-fuchsia-500/30 pb-0.5",
    },
  },
  slate: {
    token: "slate",
    label: "Slate",
    swatchBg: "bg-slate-500",
    badge: "bg-slate-500/15 border-slate-500/30 text-slate-300",
    activePill: "bg-slate-500/25 border-slate-400 text-slate-200",
    citation: {
      border: "border-slate-500/40",
      bg: "bg-slate-500/15",
      text: "text-slate-300",
      activeBorder: "border-slate-400",
      activeBg: "bg-slate-500/30",
      activeText: "text-slate-100",
    },
    heading: {
      h2: "border-l-4 border-slate-400 bg-slate-950/30 text-slate-200",
      h3: "text-slate-300 border-b border-slate-500/30 pb-0.5",
    },
  },
  indigo: {
    token: "indigo",
    label: "Indigo",
    swatchBg: "bg-indigo-500",
    badge: "bg-indigo-500/15 border-indigo-500/30 text-indigo-300",
    activePill: "bg-indigo-500/25 border-indigo-400 text-indigo-200",
    citation: {
      border: "border-indigo-500/40",
      bg: "bg-indigo-500/15",
      text: "text-indigo-300",
      activeBorder: "border-indigo-400",
      activeBg: "bg-indigo-500/30",
      activeText: "text-indigo-100",
    },
    heading: {
      h2: "border-l-4 border-indigo-400 bg-indigo-950/30 text-indigo-200",
      h3: "text-indigo-300 border-b border-indigo-500/30 pb-0.5",
    },
  },
};

export const ALLOWED_DOMAIN_COLORS: DomainColorToken[] = [
  "cyan",
  "emerald",
  "amber",
  "rose",
  "violet",
  "blue",
  "teal",
  "orange",
  "fuchsia",
  "slate",
];
