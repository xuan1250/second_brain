import React from "react";
import { AlertCircle, AlertTriangle, Check, Info, X } from "lucide-react";

export type AlertBannerVariant = "error" | "warning" | "success" | "info";

export interface AlertBannerProps {
  variant?: AlertBannerVariant;
  title?: React.ReactNode;
  message: React.ReactNode;
  action?: React.ReactNode;
  onDismiss?: () => void;
  className?: string;
}

const VARIANT_STYLES: Record<
  AlertBannerVariant,
  { container: string; title: string; icon: React.ReactNode }
> = {
  error: {
    container: "border-rose-500/40 bg-rose-950/40 text-rose-200",
    title: "text-rose-100",
    icon: <AlertCircle className="h-4 w-4 shrink-0 text-rose-400 mt-0.5" />,
  },
  warning: {
    container: "border-amber-500/40 bg-amber-950/35 text-amber-200",
    title: "text-amber-100",
    icon: <AlertTriangle className="h-4 w-4 shrink-0 text-amber-400 mt-0.5" />,
  },
  success: {
    container: "border-emerald-500/40 bg-emerald-950/40 text-emerald-300",
    title: "text-emerald-200",
    icon: <Check className="h-4 w-4 shrink-0 text-emerald-400 mt-0.5" />,
  },
  info: {
    container: "border-indigo-500/40 bg-indigo-950/35 text-indigo-200",
    title: "text-indigo-100",
    icon: <Info className="h-4 w-4 shrink-0 text-indigo-400 mt-0.5" />,
  },
};

export const AlertBanner: React.FC<AlertBannerProps> = ({
  variant = "error",
  title,
  message,
  action,
  onDismiss,
  className = "",
}) => {
  const styles = VARIANT_STYLES[variant];

  return (
    <div
      className={`flex items-start justify-between gap-2.5 rounded-xl border p-3 text-xs ${styles.container} ${className}`}
    >
      <div className="flex items-start gap-2.5 min-w-0">
        {styles.icon}
        <div className="space-y-0.5 min-w-0">
          {title && (
            <div className={`font-semibold leading-tight ${styles.title}`}>
              {title}
            </div>
          )}
          <div className="break-words leading-relaxed">{message}</div>
        </div>
      </div>

      <div className="flex items-center gap-1.5 shrink-0">
        {action}
        {onDismiss && (
          <button
            type="button"
            onClick={onDismiss}
            aria-label="Dismiss message"
            className="rounded p-0.5 opacity-75 hover:opacity-100 transition"
          >
            <X className="h-3.5 w-3.5" />
          </button>
        )}
      </div>
    </div>
  );
};
