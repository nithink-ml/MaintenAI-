"use client";

import React, { useState } from "react";
import {
  Check,
  CheckCircle2,
  Clock,
  MessageSquare,
  ShieldAlert,
} from "lucide-react";
import { StatusBadge, StatusType } from "../ui/Badge";

export interface AlertItem {
  alert_id: string;
  machine_id: string;
  severity: "CRITICAL" | "WARNING";
  title: string;
  description: string;
  timestamp: string;
  acknowledged: boolean;
  acknowledgedBy?: string;
  acknowledgedAt?: string;
  threshold?: string;
  triggerValue?: string;
}

interface AlertsCenterProps {
  alerts: AlertItem[];
  onAcknowledgeAlert: (alertId: string) => void;
  onOpenCopilotForAlert?: (machineId: string, alertTitle: string) => void;
}

export const AlertsCenter: React.FC<AlertsCenterProps> = ({
  alerts,
  onAcknowledgeAlert,
  onOpenCopilotForAlert,
}) => {
  const [filter, setFilter] = useState<"ALL" | "PENDING" | "CRITICAL" | "ACKNOWLEDGED">("PENDING");

  const filteredAlerts = alerts.filter((a) => {
    if (filter === "PENDING") return !a.acknowledged;
    if (filter === "CRITICAL") return a.severity === "CRITICAL";
    if (filter === "ACKNOWLEDGED") return a.acknowledged;
    return true;
  });

  const pendingCount = alerts.filter((a) => !a.acknowledged).length;
  const criticalCount = alerts.filter((a) => a.severity === "CRITICAL" && !a.acknowledged).length;

  return (
    <div className="space-y-4">
      {/* Control Header */}
      <div className="scada-panel p-4 bg-[#0B0F1A] flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <ShieldAlert className="w-5 h-5 text-rose-400" />
            <h2 className="text-base font-bold font-mono uppercase text-white tracking-wide">
              Deterministic Alert Center
            </h2>
            <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-rose-950/40 border border-rose-500/30 text-rose-400 font-bold tabular-nums">
              {pendingCount} PENDING
            </span>
          </div>
          <p className="text-xs font-mono text-slate-400 mt-1">
            Deterministic rule engine with hysteresis (2-3 consecutive cycles) and 15-min deduplication window.
          </p>
        </div>

        {/* Filters */}
        <div className="flex items-center gap-1.5 bg-[#121826] p-1 rounded border border-white/[0.08] text-xs font-mono">
          {[
            { label: `PENDING (${pendingCount})`, value: "PENDING" },
            { label: `CRITICAL (${criticalCount})`, value: "CRITICAL" },
            { label: "ACKNOWLEDGED", value: "ACKNOWLEDGED" },
            { label: "ALL", value: "ALL" },
          ].map((item) => (
            <button
              key={item.value}
              onClick={() =>
                setFilter(item.value as "ALL" | "PENDING" | "CRITICAL" | "ACKNOWLEDGED")
              }
              className={`px-2.5 py-1 rounded text-[11px] font-medium transition-colors ${
                filter === item.value
                  ? "bg-[#1E283D] text-white font-semibold"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              {item.label}
            </button>
          ))}
        </div>
      </div>

      {/* Alert Cards Feed */}
      <div className="space-y-3">
        {filteredAlerts.length === 0 ? (
          <div className="scada-panel p-12 text-center bg-[#0B0F1A]">
            <CheckCircle2 className="w-8 h-8 text-emerald-400 mx-auto mb-2" />
            <div className="text-sm font-bold font-mono text-white">NO ACTIVE ALERTS</div>
            <p className="text-xs font-mono text-slate-400 mt-1">
              All monitored equipment telemetry is within nominal threshold specifications.
            </p>
          </div>
        ) : (
          filteredAlerts.map((alert) => {
            const isCrit = alert.severity === "CRITICAL";
            return (
              <div
                key={alert.alert_id}
                className={`scada-panel p-4 transition-all ${
                  alert.acknowledged
                    ? "bg-[#090D16] border-white/[0.04] opacity-70"
                    : isCrit
                    ? "bg-[#140D14] border-rose-500/40"
                    : "bg-[#14120D] border-amber-500/40"
                }`}
              >
                <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
                  {/* Left info */}
                  <div className="space-y-1.5 flex-1">
                    <div className="flex items-center gap-2 text-[11px] font-mono">
                      <StatusBadge
                        status={alert.severity as StatusType}
                        size="sm"
                        label={alert.severity}
                      />
                      <span className="font-bold text-white tracking-wider">
                        {alert.machine_id.toUpperCase()}
                      </span>
                      <span className="text-slate-600">|</span>
                      <span className="text-slate-400">{alert.alert_id}</span>
                      <span className="text-slate-600">|</span>
                      <span className="text-slate-400 flex items-center gap-1">
                        <Clock className="w-3 h-3 text-slate-500" />
                        {alert.timestamp}
                      </span>
                    </div>

                    <h3 className="text-sm font-bold text-slate-100">{alert.title}</h3>
                    <p className="text-xs text-slate-300 font-sans leading-relaxed">
                      {alert.description}
                    </p>

                    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[10px] font-mono text-slate-400 pt-1">
                      <span>RULE: <strong className="text-slate-300">HYSTERESIS_N3</strong></span>
                      <span>DEDUP COOLDOWN: <strong className="text-emerald-400">ACTIVE (15m)</strong></span>
                      {alert.acknowledged && alert.acknowledgedBy && (
                        <span className="text-emerald-400 font-semibold flex items-center gap-1">
                          <Check className="w-3 h-3" /> Signed off by {alert.acknowledgedBy} at {alert.acknowledgedAt}
                        </span>
                      )}
                    </div>
                  </div>

                  {/* Actions */}
                  <div className="flex items-center gap-2">
                    {onOpenCopilotForAlert && (
                      <button
                        onClick={() =>
                          onOpenCopilotForAlert(alert.machine_id, alert.title)
                        }
                        className="px-3 py-1.5 rounded text-xs font-mono font-medium bg-[#141C2E] hover:bg-[#1C263D] text-sky-300 border border-sky-500/20 flex items-center gap-1.5 transition-colors"
                      >
                        <MessageSquare className="w-3.5 h-3.5 text-sky-400" />
                        Route to Copilot
                      </button>
                    )}

                    {!alert.acknowledged ? (
                      <button
                        onClick={() => onAcknowledgeAlert(alert.alert_id)}
                        className="px-3 py-1.5 rounded text-xs font-mono font-bold bg-[#1C283F] hover:bg-emerald-950 text-slate-200 hover:text-emerald-300 border border-white/[0.1] hover:border-emerald-500/30 flex items-center gap-1.5 transition-colors"
                      >
                        <Check className="w-3.5 h-3.5" />
                        Acknowledge
                      </button>
                    ) : (
                      <span className="text-[11px] font-mono text-slate-500 px-2 py-1 bg-white/[0.02] rounded border border-white/[0.04]">
                        ACKNOWLEDGED
                      </span>
                    )}
                  </div>
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};
