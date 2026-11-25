import React from "react";
import {
  ALLOWED_DOMAIN_COLORS,
  CHUNKING_STRATEGIES,
  DOMAIN_COLOR_PALETTE,
  EPISTEMIC_TYPES,
} from "../../lib/domainTheme";
import type {
  ChunkingStrategy,
  DomainColorToken,
  EpistemicType,
} from "../../lib/types";

export interface DomainAppearanceFieldsProps {
  isSystemDomain: boolean;
  domainIcon: string;
  onDomainIconChange: (value: string) => void;
  domainColor: DomainColorToken;
  onDomainColorChange: (value: DomainColorToken) => void;
  domainEpistemic: EpistemicType;
  onDomainEpistemicChange: (value: EpistemicType) => void;
  domainChunking: ChunkingStrategy;
  onDomainChunkingChange: (value: ChunkingStrategy) => void;
  domainPriority: number;
  onDomainPriorityChange: (value: number) => void;
}

export const DomainAppearanceFields: React.FC<DomainAppearanceFieldsProps> = ({
  isSystemDomain,
  domainIcon,
  onDomainIconChange,
  domainColor,
  onDomainColorChange,
  domainEpistemic,
  onDomainEpistemicChange,
  domainChunking,
  onDomainChunkingChange,
  domainPriority,
  onDomainPriorityChange,
}) => {
  return (
    <>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        <div>
          <label className="block text-[11px] font-semibold text-slate-300 mb-1">
            Icon (Emoji)
          </label>
          <input
            type="text"
            value={domainIcon}
            onChange={(e) => onDomainIconChange(e.target.value)}
            maxLength={5}
            className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-1.5 text-slate-200 text-center"
          />
        </div>

        <div>
          <label className="block text-[11px] font-semibold text-slate-300 mb-1">
            Color Palette Swatch
          </label>
          <div className="flex flex-wrap items-center gap-1.5 pt-1">
            {ALLOWED_DOMAIN_COLORS.map((c) => (
              <button
                key={c}
                type="button"
                onClick={() => onDomainColorChange(c)}
                title={c}
                className={`h-5 w-5 rounded-full ${DOMAIN_COLOR_PALETTE[c].swatchBg} transition ${
                  domainColor === c
                    ? "ring-2 ring-white ring-offset-2 ring-offset-slate-950 scale-110"
                    : "opacity-75 hover:opacity-100"
                }`}
              />
            ))}
          </div>
        </div>

        <div>
          <label className="block text-[11px] font-semibold text-slate-300 mb-1">
            Epistemic Modality
          </label>
          <select
            disabled={isSystemDomain}
            value={domainEpistemic}
            onChange={(e) =>
              onDomainEpistemicChange(e.target.value as EpistemicType)
            }
            className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-1.5 text-slate-200 disabled:opacity-50"
          >
            {EPISTEMIC_TYPES.map((ep) => (
              <option key={ep.value} value={ep.value}>
                {ep.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        <div>
          <label className="block text-[11px] font-semibold text-slate-300 mb-1">
            Chunking Strategy Preset
          </label>
          <select
            disabled={isSystemDomain}
            value={domainChunking}
            onChange={(e) =>
              onDomainChunkingChange(e.target.value as ChunkingStrategy)
            }
            className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-1.5 text-slate-200 disabled:opacity-50"
          >
            {CHUNKING_STRATEGIES.map((cs) => (
              <option key={cs.value} value={cs.value}>
                {cs.label}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="block text-[11px] font-semibold text-slate-300 mb-1">
            Authority Priority (1 = Top)
          </label>
          <select
            disabled={isSystemDomain}
            value={domainPriority}
            onChange={(e) => onDomainPriorityChange(Number(e.target.value))}
            className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-1.5 text-slate-200 disabled:opacity-50"
          >
            <option value={1}>1 (Highest Priority)</option>
            <option value={2}>2 (High Priority)</option>
            <option value={3}>3 (Standard Priority)</option>
            <option value={4}>4 (Secondary Priority)</option>
            <option value={5}>5 (Supplemental)</option>
          </select>
        </div>
      </div>
    </>
  );
};
