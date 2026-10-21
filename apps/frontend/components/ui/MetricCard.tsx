import React from "react";
import { Sparkline } from "./Sparkline";

interface MetricCardProps {
  label: string;
  value: number | string;
  unit?: string;
  nominalRange?: string;
  history?: number[];
  accentColor?: string;
  isWarning?: boolean;
  isCritical?: boolean;
  delta?: number;
  icon?: React.ReactNode;
}

export const MetricCard: React.FC<MetricCardProps> = ({
  label,
  value,
  unit = "",
  nominalRange,
  history,
  accentColor = "#38BDF8",
  isWarning = false,
  isCritical = false,
  delta,
  icon,
}) => {
  const borderStatus = isCritical
    ? "border-rose-500/50 bg-rose-950/10"
    : isWarning
    ? "border-amber-500/40 bg-amber-950/10"
    : "border-slate-800/80 bg-[#0E1320]";

  const valueStatusColor = isCritical
    ? "text-rose-400"
    : isWarning
    ? "text-amber-400"
    : "text-slate-100";

  return (
    <div
      className={`scada-panel p-3.5 flex flex-col justify-between transition-colors ${borderStatus}`}
    >
      <div className="flex items-start justify-between gap-2">
        <span className="text-[11px] font-mono uppercase tracking-wider text-slate-400 truncate">
          {label}
        </span>
        {icon && <div className="text-slate-500">{icon}</div>}
      </div>

      <div className="my-2 flex items-baseline justify-between">
        <div className="flex items-baseline gap-1.5">
          <span
            className={`text-2xl font-bold font-mono tracking-tight tabular-nums ${valueStatusColor}`}
          >
            {value}
          </span>
          {unit && <span className="text-xs font-mono text-slate-400">{unit}</span>}
        </div>

        {delta !== undefined && (
          <span
            className={`text-[11px] font-mono font-medium ${
              delta > 0
                ? "text-rose-400"
                : delta < 0
                ? "text-emerald-400"
                : "text-slate-400"
            }`}
          >
            {delta > 0 ? `+${delta.toFixed(2)}` : delta.toFixed(2)}
          </span>
        )}
      </div>

      {history && history.length > 1 && (
        <div className="my-1">
          <Sparkline
            data={history}
            color={isCritical ? "#F43F5E" : isWarning ? "#F59E0B" : accentColor}
            height={28}
          />
        </div>
      )}

      {nominalRange && (
        <div className="pt-2 border-t border-slate-800/60 flex items-center justify-between text-[10px] font-mono text-slate-400">
          <span>SPEC RANGE</span>
          <span className="text-slate-300 font-medium">{nominalRange}</span>
        </div>
      )}
    </div>
  );
};
