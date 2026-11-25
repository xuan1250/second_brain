import React from "react";

export interface EmptyStateProps {
  icon?: React.ReactNode;
  title: string;
  description?: React.ReactNode;
  children?: React.ReactNode;
  className?: string;
}

export const EmptyState: React.FC<EmptyStateProps> = ({
  icon,
  title,
  description,
  children,
  className = "py-12 px-4",
}) => {
  return (
    <div
      className={`flex flex-col items-center justify-center text-center ${className}`}
    >
      {icon && <div className="mb-2.5">{icon}</div>}
      <h3 className="text-xs font-semibold text-slate-200">{title}</h3>
      {description && (
        <p className="mt-1 max-w-sm text-[11px] leading-relaxed text-slate-400">
          {description}
        </p>
      )}
      {children}
    </div>
  );
};
