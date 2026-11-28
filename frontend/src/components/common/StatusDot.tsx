import React from "react";

export type StatusDotTone = "ok" | "warning" | "error" | "idle";

export interface StatusDotProps {
  tone: StatusDotTone;
  pulse?: boolean;
  glow?: boolean;
  size?: "sm" | "md";
  className?: string;
}

const TONE_CLASSES: Record<StatusDotTone, string> = {
  ok: "bg-emerald-400",
  warning: "bg-amber-400",
  error: "bg-rose-500",
  idle: "bg-slate-500",
};

export const StatusDot: React.FC<StatusDotProps> = ({
  tone,
  pulse = false,
  glow = false,
  size = "md",
  className = "",
}) => {
  const sizeCls = size === "sm" ? "h-1.5 w-1.5" : "h-2 w-2";
  const glowCls =
    glow && tone === "ok" ? "shadow-[0_0_6px_rgba(52,211,153,0.8)]" : "";
  const pulseCls = pulse ? "animate-pulse" : "";

  return (
    <span
      className={`inline-block rounded-full shrink-0 ${sizeCls} ${TONE_CLASSES[tone]} ${glowCls} ${pulseCls} ${className}`}
    />
  );
};
