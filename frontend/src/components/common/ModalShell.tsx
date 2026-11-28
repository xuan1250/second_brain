import React from "react";
import * as Dialog from "@radix-ui/react-dialog";
import { X } from "lucide-react";

export interface ModalShellProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: React.ReactNode;
  description?: React.ReactNode;
  icon?: React.ReactNode;
  children: React.ReactNode;
  footer?: React.ReactNode;
  maxWidthClass?: string;
  zIndexClass?: "z-50" | "z-60" | string;
  overlayClassName?: string;
  contentClassName?: string;
  showCloseButton?: boolean;
}

export const ModalShell: React.FC<ModalShellProps> = ({
  open,
  onOpenChange,
  title,
  description,
  icon,
  children,
  footer,
  maxWidthClass = "max-w-md",
  zIndexClass = "z-50",
  overlayClassName = "bg-black/70 backdrop-blur-xs",
  contentClassName = "border-slate-800 bg-slate-900 p-5",
  showCloseButton = false,
}) => {
  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay
          className={`fixed inset-0 ${zIndexClass} ${overlayClassName}`}
        />
        <Dialog.Content
          className={`fixed left-1/2 top-1/2 ${zIndexClass} w-full ${maxWidthClass} -translate-x-1/2 -translate-y-1/2 rounded-2xl border shadow-2xl focus:outline-none ${contentClassName}`}
        >
          <div className="flex items-start justify-between gap-3">
            <div className="flex items-start gap-2.5 min-w-0">
              {icon && <div className="shrink-0 mt-0.5">{icon}</div>}
              <div className="min-w-0">
                <Dialog.Title className="text-sm font-semibold text-slate-100">
                  {title}
                </Dialog.Title>
                {description && (
                  <Dialog.Description className="mt-1 text-xs leading-relaxed text-slate-400">
                    {description}
                  </Dialog.Description>
                )}
              </div>
            </div>
            {showCloseButton && (
              <Dialog.Close asChild>
                <button
                  type="button"
                  aria-label="Close dialog"
                  className="rounded-lg p-1.5 text-slate-400 hover:bg-slate-800 hover:text-white transition"
                >
                  <X className="h-4 w-4" />
                </button>
              </Dialog.Close>
            )}
          </div>

          {children}

          {footer && (
            <div className="mt-5 flex items-center justify-end gap-2">
              {footer}
            </div>
          )}
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
};
