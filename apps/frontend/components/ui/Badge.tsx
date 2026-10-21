import React from "react";

export type StatusType = "NORMAL" | "WARNING" | "CRITICAL" | "OFFLINE" | "INFO";

interface BadgeProps {
  status?: StatusType;
  label?: string;
  className?: string;
  size?: "sm" | "md";
  showDot?: boolean;
}

export const StatusBadge: React.FC<BadgeProps> = ({
  status = "NORMAL",
  label,
  className = "",
  size = "md",
  showDot = true,
}) => {
  const displayLabel = label || status;

  const styleConfig: Record<
    StatusType,
    { bg: string; text: string; border: string; dot: string; dotPulse: boolean }
  > = {
    NORMAL: {
      bg: "bg-emerald-950/40",
      text: "text-emerald-400",
      border: "border-emerald-500/30",
      dot: "bg-emerald-400",
      dotPulse: false,
    },
    WARNING: {
      bg: "bg-amber-950/40",
      text: "text-amber-400",
      border: "border-amber-500/30",
      dot: "bg-amber-400",
      dotPulse: true,
    },
    CRITICAL: {
      bg: "bg-rose-950/50",
      text: "text-rose-300",
      border: "border-rose-500/40",
      dot: "bg-rose-400",
      dotPulse: true,
    },
    OFFLINE: {
      bg: "bg-slate-900",
      text: "text-slate-400",
      border: "border-slate-800",
      dot: "bg-slate-500",
      dotPulse: false,
    },
    INFO: {
      bg: "bg-sky-950/40",
      text: "text-sky-300",
      border: "border-sky-500/30",
      dot: "bg-sky-400",
      dotPulse: false,
    },
  };

  const current = styleConfig[status] || styleConfig.INFO;
  const sizeClasses =
    size === "sm"
      ? "px-2 py-0.5 text-[10px] tracking-wide"
      : "px-2.5 py-1 text-[11px] tracking-wider";

  return (
    <span
      className={`inline-flex items-center gap-1.5 font-mono font-medium uppercase rounded border ${current.bg} ${current.text} ${current.border} ${sizeClasses} ${className}`}
    >
      {showDot && (
        <span className="relative flex h-1.5 w-1.5">
          {current.dotPulse && (
            <span
              className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${current.dot}`}
            />
          )}
          <span className={`relative inline-flex rounded-full h-1.5 w-1.5 ${current.dot}`} />
        </span>
      )}
      <span>{displayLabel}</span>
    </span>
  );
};
