import React, { useEffect, useRef } from "react";

export interface PopoverMenuProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  trigger: React.ReactNode;
  children: React.ReactNode;
  align?: "left" | "right";
  widthClass?: string;
  headerLabel?: string;
  className?: string;
}

export const PopoverMenu: React.FC<PopoverMenuProps> = ({
  open,
  onOpenChange,
  trigger,
  children,
  align = "left",
  widthClass = "w-44",
  headerLabel,
  className = "relative shrink-0",
}) => {
  const containerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!open) return;

    const handlePointerDown = (event: MouseEvent) => {
      if (
        containerRef.current &&
        !containerRef.current.contains(event.target as Node)
      ) {
        onOpenChange(false);
      }
    };

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        onOpenChange(false);
      }
    };

    document.addEventListener("mousedown", handlePointerDown);
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("mousedown", handlePointerDown);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [open, onOpenChange]);

  return (
    <div ref={containerRef} className={className}>
      {trigger}
      {open && (
        <div
          onClick={(e) => e.stopPropagation()}
          className={`absolute ${
            align === "right" ? "right-0" : "left-0"
          } top-full z-30 mt-1 ${widthClass} rounded-xl border border-slate-700 bg-slate-950 p-1.5 shadow-xl text-xs space-y-0.5`}
        >
          {headerLabel && (
            <div className="px-2 py-1 text-[10px] uppercase font-bold text-slate-500 border-b border-slate-800">
              {headerLabel}
            </div>
          )}
          {children}
        </div>
      )}
    </div>
  );
};
