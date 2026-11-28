import React from "react";
import { Loader2 } from "lucide-react";

export interface LoadingStateProps {
  label?: string;
  size?: "sm" | "md";
  className?: string;
}

export const LoadingState: React.FC<LoadingStateProps> = ({
  label = "Loading...",
  size = "md",
  className = "py-12",
}) => {
  const iconCls = size === "sm" ? "h-4 w-4" : "h-5 w-5";
  return (
    <div
      className={`flex flex-col items-center justify-center text-slate-400 gap-2 ${className}`}
    >
      <Loader2 className={`${iconCls} animate-spin text-indigo-400`} />
      <span className="text-xs">{label}</span>
    </div>
  );
};
