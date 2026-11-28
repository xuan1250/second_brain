import React from "react";
import { AlertTriangle, Loader2 } from "lucide-react";
import { AlertBanner } from "./AlertBanner";
import { ModalShell } from "./ModalShell";

export interface ConfirmDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: React.ReactNode;
  description: React.ReactNode;
  confirmLabel?: string;
  cancelLabel?: string;
  confirmIcon?: React.ReactNode;
  variant?: "danger" | "primary";
  isPending?: boolean;
  error?: string | null;
  zIndexClass?: "z-50" | "z-60" | string;
  maxWidthClass?: string;
  onConfirm: () => void | Promise<void>;
  children?: React.ReactNode;
}

export const ConfirmDialog: React.FC<ConfirmDialogProps> = ({
  open,
  onOpenChange,
  title,
  description,
  confirmLabel = "Confirm",
  cancelLabel = "Cancel",
  confirmIcon,
  variant = "danger",
  isPending = false,
  error,
  zIndexClass = "z-50",
  maxWidthClass = "max-w-md",
  onConfirm,
  children,
}) => {
  const isDanger = variant === "danger";

  return (
    <ModalShell
      open={open}
      onOpenChange={onOpenChange}
      title={title}
      description={description}
      icon={
        isDanger ? (
          <AlertTriangle className="h-4 w-4 text-rose-400" />
        ) : undefined
      }
      zIndexClass={zIndexClass}
      maxWidthClass={maxWidthClass}
      overlayClassName={
        zIndexClass === "z-60"
          ? "bg-black/80 backdrop-blur-xs"
          : "bg-black/70 backdrop-blur-xs"
      }
      contentClassName={
        isDanger
          ? "border-rose-500/40 bg-slate-950 p-6"
          : "border-slate-800 bg-slate-900 p-5"
      }
      footer={
        <>
          <button
            type="button"
            onClick={() => onOpenChange(false)}
            className="rounded-lg border border-slate-700 bg-slate-800 px-3 py-1.5 text-xs font-medium text-slate-200 hover:bg-slate-700"
          >
            {cancelLabel}
          </button>
          <button
            type="button"
            disabled={isPending}
            onClick={() => void onConfirm()}
            className={`inline-flex items-center gap-1.5 rounded-lg px-3.5 py-1.5 text-xs font-semibold text-white disabled:opacity-50 ${
              isDanger
                ? "bg-rose-600 hover:bg-rose-500"
                : "bg-indigo-600 hover:bg-indigo-500"
            }`}
          >
            {isPending ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            ) : (
              confirmIcon
            )}
            <span>{confirmLabel}</span>
          </button>
        </>
      }
    >
      {error && (
        <AlertBanner variant="error" message={error} className="mt-3" />
      )}
      {children && <div className="mt-4">{children}</div>}
    </ModalShell>
  );
};
