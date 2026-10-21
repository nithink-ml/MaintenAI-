"use client";

import React, { useState, useEffect } from "react";
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  Cpu,
  Database,
  Layers,
  Pause,
  Play,
  Radio,
  ShieldAlert,
  Sparkles,
} from "lucide-react";

export type NavTab = "fleet" | "diagnostics" | "alerts" | "copilot";

interface HeaderProps {
  activeTab: NavTab;
  setActiveTab: (tab: NavTab) => void;
  isSimulating: boolean;
  setIsSimulating: (sim: boolean) => void;
  simSpeed: number;
  setSimSpeed: (speed: number) => void;
  criticalCount: number;
  warningCount: number;
  normalCount: number;
  unacknowledgedAlertsCount: number;
  isBackendConnected?: boolean;
}

export const Header: React.FC<HeaderProps> = ({
  activeTab,
  setActiveTab,
  isSimulating,
  setIsSimulating,
  simSpeed,
  setSimSpeed,
  criticalCount,
  warningCount,
  normalCount,
  unacknowledgedAlertsCount,
  isBackendConnected = false,
}) => {
  const [utcTime, setUtcTime] = useState<string>("");
  const [systemStatus, setSystemStatus] = useState<{
    timescaledb?: { status: string; sync_state?: string; total_records?: number };
    redis?: { status: string; stream_state?: string; cached_keys?: number };
    ml_runtime?: { status: string };
    simulation?: { mode?: string };
  } | null>(null);

  useEffect(() => {
    const updateTime = () => {
      const now = new Date();
      setUtcTime(now.toUTCString().replace("GMT", "UTC"));
    };
    updateTime();
    const timer = setInterval(updateTime, 1000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    let isMounted = true;
    const fetchStatus = async () => {
      try {
        const res = await fetch("http://localhost:8000/api/v1/system/status");
        if (res.ok) {
          const data = await res.json();
          if (isMounted) setSystemStatus(data.services || null);
        } else if (isMounted) {
          setSystemStatus(null);
        }
      } catch {
        if (isMounted) setSystemStatus(null);
      }
    };
    fetchStatus();
    const timer = setInterval(fetchStatus, 4000);
    return () => {
      isMounted = false;
      clearInterval(timer);
    };
  }, []);

  return (
    <header className="border-b border-white/[0.08] bg-[#0A0E17]/95 backdrop-blur-md sticky top-0 z-50 px-4 lg:px-6 py-2.5">
      {/* Upper Row: Brand & Hardware Infrastructure Status */}
      <div className="flex flex-wrap items-center justify-between gap-4 pb-2 border-b border-white/[0.05]">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded bg-[#151D2E] border border-white/[0.12] flex items-center justify-center text-sky-400">
            <Activity className="w-4 h-4" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-sm font-bold tracking-tight text-white uppercase font-mono">
                FactoryPulse
              </span>
              <span className="text-[10px] font-mono text-slate-400 px-1.5 py-0.5 rounded bg-white/[0.05] border border-white/[0.08]">
                SCADA-OS v2.4
              </span>
              <span className="text-[10px] font-mono uppercase tracking-wider px-2 py-0.5 rounded bg-amber-500/10 text-amber-300 border border-amber-500/30 flex items-center gap-1">
                <span className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse" />
                SYNTHETIC DEVELOPMENT SIMULATION
              </span>
            </div>
            <div className="text-[10px] font-mono text-slate-400">
              EQUIPMENT INTELLIGENCE &amp; PREDICTIVE MAINTENANCE
            </div>
          </div>
        </div>

        {/* Live Hardware Telemetry Pills */}
        <div className="hidden xl:flex items-center gap-3 text-[11px] font-mono">
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-[#0E1422] border border-white/[0.06] text-slate-300">
            <span
              className={`w-2 h-2 rounded-full ${
                isBackendConnected ? "bg-emerald-400 animate-pulse" : "bg-amber-400"
              }`}
            />
            <span className="text-slate-400">Gateway:</span>
            <span
              className={
                isBackendConnected ? "text-emerald-400 font-semibold" : "text-amber-400 font-semibold"
              }
            >
              {isBackendConnected ? "LIVE HTTP:8000" : "STANDALONE SIM"}
            </span>
          </div>

          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-[#0E1422] border border-white/[0.06] text-slate-300">
            <Database
              className={`w-3.5 h-3.5 ${
                systemStatus?.timescaledb?.status === "online" ? "text-emerald-400" : "text-rose-400"
              }`}
            />
            <span className="text-slate-400">TimescaleDB:</span>
            <span
              className={
                systemStatus?.timescaledb?.status === "online"
                  ? "text-emerald-400 font-semibold"
                  : isBackendConnected
                  ? "text-rose-400 font-semibold"
                  : "text-slate-400 font-semibold"
              }
            >
              {systemStatus?.timescaledb?.status === "online"
                ? systemStatus.timescaledb.sync_state || "HYPERTABLE SYNC"
                : isBackendConnected
                ? "OFFLINE"
                : "STANDALONE"}
            </span>
          </div>

          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-[#0E1422] border border-white/[0.06] text-slate-300">
            <Radio
              className={`w-3.5 h-3.5 ${
                systemStatus?.redis?.status === "online" ? "text-sky-400 animate-pulse" : "text-rose-400"
              }`}
            />
            <span className="text-slate-400">Redis Cache:</span>
            <span
              className={
                systemStatus?.redis?.status === "online"
                  ? "text-sky-400 font-semibold"
                  : isBackendConnected
                  ? "text-rose-400 font-semibold"
                  : "text-slate-400 font-semibold"
              }
            >
              {systemStatus?.redis?.status === "online"
                ? systemStatus.redis.stream_state || "100Hz STREAM"
                : isBackendConnected
                ? "OFFLINE"
                : "IN-MEMORY"}
            </span>
          </div>

          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-[#0E1422] border border-white/[0.06] text-slate-300">
            <Cpu
              className={`w-3.5 h-3.5 ${
                systemStatus?.ml_runtime?.status ? "text-indigo-400" : "text-rose-400"
              }`}
            />
            <span className="text-slate-400">ML Runtime:</span>
            <span
              className={
                systemStatus?.ml_runtime?.status
                  ? "text-indigo-300 font-semibold"
                  : isBackendConnected
                  ? "text-rose-400 font-semibold"
                  : "text-slate-400 font-semibold"
              }
            >
              {systemStatus?.ml_runtime?.status === "active" ||
              systemStatus?.ml_runtime?.status === "fallback_active"
                ? "HIST-GBM ACTIVE"
                : isBackendConnected
                ? "OFFLINE"
                : "LOCAL FALLBACK"}
            </span>
          </div>

          <div className="text-[11px] text-slate-400 tabular-nums font-mono px-2 py-1 bg-black/40 rounded border border-white/[0.04]">
            {utcTime || "2026-09-22 10:15:00 UTC"}
          </div>
        </div>
      </div>

      {/* Lower Row: Navigation & Fleet Operational Metrics */}
      <div className="flex flex-wrap items-center justify-between gap-3 pt-2">
        {/* Navigation Tabs */}
        <nav className="flex items-center gap-1 bg-[#0E1424] p-1 rounded-lg border border-white/[0.08]">
          <button
            onClick={() => setActiveTab("fleet")}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-mono transition-all ${
              activeTab === "fleet"
                ? "bg-[#1C253B] text-white border border-white/[0.12] shadow-sm"
                : "text-slate-400 hover:text-slate-200 hover:bg-white/[0.02]"
            }`}
          >
            <Layers className="w-3.5 h-3.5 text-sky-400" />
            <span>[1] Fleet Matrix</span>
          </button>

          <button
            onClick={() => setActiveTab("diagnostics")}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-mono transition-all ${
              activeTab === "diagnostics"
                ? "bg-[#1C253B] text-white border border-white/[0.12] shadow-sm"
                : "text-slate-400 hover:text-slate-200 hover:bg-white/[0.02]"
            }`}
          >
            <Activity className="w-3.5 h-3.5 text-indigo-400" />
            <span>[2] Diagnostics</span>
          </button>

          <button
            onClick={() => setActiveTab("alerts")}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-mono transition-all relative ${
              activeTab === "alerts"
                ? "bg-[#1C253B] text-white border border-white/[0.12] shadow-sm"
                : "text-slate-400 hover:text-slate-200 hover:bg-white/[0.02]"
            }`}
          >
            <AlertTriangle className="w-3.5 h-3.5 text-amber-400" />
            <span>[3] Alerts</span>
            {unacknowledgedAlertsCount > 0 && (
              <span className="ml-1 px-1.5 py-0.2 rounded-full text-[10px] font-bold bg-rose-500/20 text-rose-400 border border-rose-500/30 tabular-nums">
                {unacknowledgedAlertsCount}
              </span>
            )}
          </button>

          <button
            onClick={() => setActiveTab("copilot")}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-mono transition-all ${
              activeTab === "copilot"
                ? "bg-[#1C253B] text-white border border-white/[0.12] shadow-sm"
                : "text-slate-400 hover:text-slate-200 hover:bg-white/[0.02]"
            }`}
          >
            <Sparkles className="w-3.5 h-3.5 text-cyan-400" />
            <span>[4] Copilot AI</span>
          </button>
        </nav>

        {/* Fleet Status Counters & Simulation Control */}
        <div className="flex items-center gap-3">
          {/* Status Counts */}
          <div className="hidden sm:flex items-center gap-2 bg-[#0E1424] px-3 py-1.5 rounded-lg border border-white/[0.08] text-xs font-mono">
            <span className="flex items-center gap-1 text-rose-400 font-semibold">
              <ShieldAlert className="w-3.5 h-3.5" />
              <span>{criticalCount} CRIT</span>
            </span>
            <span className="text-slate-700">|</span>
            <span className="flex items-center gap-1 text-amber-400 font-semibold">
              <AlertTriangle className="w-3.5 h-3.5" />
              <span>{warningCount} WARN</span>
            </span>
            <span className="text-slate-700">|</span>
            <span className="flex items-center gap-1 text-emerald-400 font-semibold">
              <CheckCircle2 className="w-3.5 h-3.5" />
              <span>{normalCount} NORM</span>
            </span>
          </div>

          {/* Rate Selector */}
          <div className="flex items-center bg-[#0E1424] rounded-lg border border-white/[0.08] p-0.5 text-xs font-mono">
            {[1, 2, 5].map((speed) => (
              <button
                key={speed}
                onClick={() => setSimSpeed(speed)}
                className={`px-2 py-1 rounded text-[10px] font-semibold transition-colors ${
                  simSpeed === speed
                    ? "bg-[#1D253A] text-sky-400 border border-white/[0.1]"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                {speed}x
              </button>
            ))}
          </div>

          {/* Simulation Toggle */}
          <button
            onClick={() => setIsSimulating(!isSimulating)}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-mono font-medium border transition-all ${
              isSimulating
                ? "bg-emerald-950/40 border-emerald-500/30 text-emerald-300 hover:bg-emerald-900/40"
                : "bg-amber-950/40 border-amber-500/30 text-amber-300 hover:bg-amber-900/40"
            }`}
          >
            {isSimulating ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
            <span className="hidden md:inline">{isSimulating ? "SIMULATION ACTIVE" : "SIMULATION PAUSED"}</span>
          </button>
        </div>
      </div>
    </header>
  );
};
