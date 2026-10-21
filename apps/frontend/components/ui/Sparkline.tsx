import React from "react";

interface SparklineProps {
  data: number[];
  color?: string;
  height?: number;
  fillOpacity?: number;
  showMinMax?: boolean;
  unit?: string;
}

export const Sparkline: React.FC<SparklineProps> = ({
  data,
  color = "#38BDF8",
  height = 36,
  fillOpacity = 0.15,
  showMinMax = false,
  unit = "",
}) => {
  if (!data || data.length < 2) {
    return <div className="h-9 flex items-center justify-center text-[10px] text-slate-600 font-mono">NO DATA</div>;
  }

  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min === 0 ? 1 : max - min;

  // ViewBox coordinates
  const svgWidth = 120;
  const svgHeight = height;
  const padding = 3;

  const points = data.map((val, idx) => {
    const x = padding + (idx / (data.length - 1)) * (svgWidth - padding * 2);
    const y = svgHeight - padding - ((val - min) / range) * (svgHeight - padding * 2);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });

  const pathD = `M ${points.join(" L ")}`;
  const areaD = `${pathD} L ${svgWidth - padding},${svgHeight} L ${padding},${svgHeight} Z`;

  const gradientId = `sparkline-grad-${Math.random().toString(36).substring(2, 8)}`;

  return (
    <div className="flex flex-col gap-0.5 w-full">
      <div className="relative w-full overflow-hidden" style={{ height: `${height}px` }}>
        <svg
          viewBox={`0 0 ${svgWidth} ${svgHeight}`}
          preserveAspectRatio="none"
          className="w-full h-full overflow-visible"
        >
          <defs>
            <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={color} stopOpacity={fillOpacity} />
              <stop offset="100%" stopColor={color} stopOpacity={0.0} />
            </linearGradient>
          </defs>
          <path d={areaD} fill={`url(#${gradientId})`} />
          <path
            d={pathD}
            fill="none"
            stroke={color}
            strokeWidth="1.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
          {/* Latest point indicator */}
          {points.length > 0 && (
            <circle
              cx={points[points.length - 1].split(",")[0]}
              cy={points[points.length - 1].split(",")[1]}
              r="2"
              fill="#FFFFFF"
              stroke={color}
              strokeWidth="1"
            />
          )}
        </svg>
      </div>

      {showMinMax && (
        <div className="flex justify-between items-center text-[9px] font-mono text-slate-400 px-0.5">
          <span>MIN: {min.toFixed(1)}{unit}</span>
          <span>MAX: {max.toFixed(1)}{unit}</span>
        </div>
      )}
    </div>
  );
};
