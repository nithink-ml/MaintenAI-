"use client";

import React, { useState } from "react";
import {
  Activity,
  AlertTriangle,
  Clock,
  Cpu,
  Flame,
  Gauge,
  RotateCcw,
  Sliders,
  Thermometer,
  Wrench,
  Zap,
} from "lucide-react";
import { StatusBadge, StatusType } from "../ui/Badge";
import { MetricCard } from "../ui/MetricCard";
import { Sparkline } from "../ui/Sparkline";
import { MachineState } from "../fleet/FleetMatrix";

interface MachineDetailProps {
  machine: MachineState;
  onInjectFault: (machineId: string, faultType: string) => void;
  onResetMachine?: (machineId: string) => void;
  onOpenCopilotWithPrompt?: (prompt: string) => void;
}

export const MachineDetail: React.FC<MachineDetailProps> = ({
  machine,
  onInjectFault,
  onResetMachine,
  onOpenCopilotWithPrompt,
}) => {
  const [selectedChartMetric, setSelectedChartMetric] = useState<"temp" | "vib" | "pressure">("vib");
  const [modelsMeta, setModelsMeta] = useState<{
    anomaly?: { model_family?: string; model_version?: string; feature_count?: number };
    rul?: { model_family?: string; model_version?: string; feature_count?: number; is_fallback?: boolean };
    failure_classifier?: { model_family?: string; model_version?: string; feature_count?: number; is_fallback?: boolean };
  } | null>(null);

  React.useEffect(() => {
    let isMounted = true;
    fetch("http://localhost:8000/api/v1/models/metadata")
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (isMounted && data) setModelsMeta(data);
      })
      .catch(() => {});
    return () => {
      isMounted = false;
    };
  }, []);

  const chartData = machine.history[selectedChartMetric];

  return (
    <div className="space-y-4">
      {/* Top Header Card: Asset Specs & Status */}
      <div className="scada-panel p-4 bg-[#0B0F1A] border-white/[0.08]">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 rounded bg-[#131A2B] border border-white/[0.1] flex items-center justify-center text-sky-400">
              <Gauge className="w-6 h-6" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="text-base font-bold font-mono tracking-tight text-white">
                  {machine.machine_id.toUpperCase()}
                </span>
                <span className="text-slate-600">|</span>
                <span className="text-sm font-semibold text-slate-200">{machine.name}</span>
                <StatusBadge status={machine.status as StatusType} size="sm" />
              </div>
              <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] font-mono text-slate-400 mt-1">
                <span>CATEGORY: <strong className="text-slate-300 uppercase">{machine.equipment_type}</strong></span>
                <span>CUMULATIVE CYCLES: <strong className="text-slate-300">#{machine.operating_cycle}</strong></span>
                <span>SAMPLING: <strong className="text-emerald-400">100Hz REALTIME</strong></span>
              </div>
            </div>
          </div>

          {/* Fault Injection Bench */}
          <div className="flex flex-col sm:flex-row sm:items-center gap-2 bg-[#121826] p-2 rounded-lg border border-white/[0.08]">
            <span className="text-[10px] font-mono text-slate-400 uppercase tracking-wider px-1 flex items-center gap-1">
              <Sliders className="w-3 h-3 text-amber-400" />
              Hardware Test Bench:
            </span>
            <div className="flex flex-wrap items-center gap-1.5">
              <button
                onClick={() => onInjectFault(machine.machine_id, "thermal_runaway")}
                className="px-2 py-1 rounded text-[10px] font-mono font-semibold bg-rose-950/40 hover:bg-rose-900/50 text-rose-300 border border-rose-500/30 transition-colors flex items-center gap-1"
                title="Inject sudden thermal breakdown"
              >
                <Flame className="w-3 h-3 text-rose-400" />
                Thermal Spike
              </button>
              <button
                onClick={() => onInjectFault(machine.machine_id, "bearing_spall")}
                className="px-2 py-1 rounded text-[10px] font-mono font-semibold bg-amber-950/40 hover:bg-amber-900/50 text-amber-300 border border-amber-500/30 transition-colors flex items-center gap-1"
                title="Inject outer-race bearing spalling"
              >
                <Zap className="w-3 h-3 text-amber-400" />
                Bearing Spall
              </button>
              <button
                onClick={() => onInjectFault(machine.machine_id, "tool_wear")}
                className="px-2 py-1 rounded text-[10px] font-mono font-semibold bg-indigo-950/40 hover:bg-indigo-900/50 text-indigo-300 border border-indigo-500/30 transition-colors flex items-center gap-1"
                title="Inject progressive cutting tool degradation"
              >
                <Wrench className="w-3 h-3 text-indigo-400" />
                Tool Wear
              </button>
              {onResetMachine && (
                <button
                  onClick={() => onResetMachine(machine.machine_id)}
                  className="px-2 py-1 rounded text-[10px] font-mono text-slate-400 hover:text-white bg-slate-800/80 hover:bg-slate-700/80 border border-white/[0.08] transition-colors"
                  title="Reset to nominal baseline"
                >
                  <RotateCcw className="w-3 h-3" />
                </button>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Live Physical Telemetry Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <MetricCard
          label="Exhaust / Surface Temp"
          value={machine.temperature.toFixed(1)}
          unit="°C"
          nominalRange="580 - 620°C"
          history={machine.history.temp}
          accentColor="#F97316"
          isWarning={machine.temperature > 630}
          isCritical={machine.temperature > 640}
          icon={<Thermometer className="w-4 h-4 text-orange-400" />}
        />

        <MetricCard
          label="Acoustic Vibration RMS"
          value={machine.vibration.toFixed(2)}
          unit="g"
          nominalRange="< 2.5 g"
          history={machine.history.vib}
          accentColor="#38BDF8"
          isWarning={machine.vibration > 2.8}
          isCritical={machine.vibration > 3.4}
          icon={<Activity className="w-4 h-4 text-sky-400" />}
        />

        <MetricCard
          label="Internal Pressure"
          value={machine.pressure.toFixed(1)}
          unit="psia"
          nominalRange="450 - 550 psia"
          history={machine.history.pressure}
          accentColor="#818CF8"
          icon={<Gauge className="w-4 h-4 text-indigo-400" />}
        />

        <MetricCard
          label="Operational Rotor Speed"
          value={machine.rpm.toFixed(0)}
          unit="RPM"
          nominalRange="2200 - 2500"
          accentColor="#10B981"
          icon={<RotateCcw className="w-4 h-4 text-emerald-400" />}
        />
      </div>

      {/* Waveform Historical Trend Chart with Metric Switcher */}
      <div className="scada-panel p-4 bg-[#0B0F1A]">
        <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-white/[0.06]">
          <div className="flex items-center gap-2">
            <span className="text-xs font-mono font-bold uppercase tracking-wider text-slate-200">
              Sensor Waveform Trajectory (Sliding Window: 30 Cycles)
            </span>
            <span className="text-[10px] font-mono text-slate-400">
              Zero-leakage temporal buffer
            </span>
          </div>

          <div className="flex items-center bg-[#121826] p-0.5 rounded border border-white/[0.08] text-xs font-mono">
            <button
              onClick={() => setSelectedChartMetric("vib")}
              className={`px-2.5 py-1 rounded text-[10px] transition-colors ${
                selectedChartMetric === "vib"
                  ? "bg-[#1E283D] text-sky-300 font-bold"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              Vibration (g)
            </button>
            <button
              onClick={() => setSelectedChartMetric("temp")}
              className={`px-2.5 py-1 rounded text-[10px] transition-colors ${
                selectedChartMetric === "temp"
                  ? "bg-[#1E283D] text-orange-300 font-bold"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              Temperature (°C)
            </button>
            <button
              onClick={() => setSelectedChartMetric("pressure")}
              className={`px-2.5 py-1 rounded text-[10px] transition-colors ${
                selectedChartMetric === "pressure"
                  ? "bg-[#1E283D] text-indigo-300 font-bold"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              Pressure (psia)
            </button>
          </div>
        </div>

        <div className="mt-4">
          <Sparkline
            data={chartData}
            height={64}
            color={
              selectedChartMetric === "vib"
                ? "#38BDF8"
                : selectedChartMetric === "temp"
                ? "#FB923C"
                : "#818CF8"
            }
            showMinMax
            unit={
              selectedChartMetric === "vib"
                ? "g"
                : selectedChartMetric === "temp"
                ? "°C"
                : "psia"
            }
          />
        </div>
      </div>

      {/* Multi-Model Predictive Intelligence Section */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-3">
        {/* Model 1: Isolation Forest Anomaly Detector */}
        <div className="scada-panel p-4 bg-[#0B0F1A] flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between pb-2 border-b border-white/[0.06]">
              <span className="text-xs font-mono font-bold uppercase text-slate-200 flex items-center gap-1.5">
                <Cpu className="w-3.5 h-3.5 text-indigo-400" />
                Anomaly Detector
              </span>
              <span className="text-[10px] font-mono text-slate-400">
                {modelsMeta?.anomaly?.model_family || "ISOLATION FOREST"}
              </span>
            </div>

            <div className="my-3">
              <div className="flex items-baseline justify-between mb-1">
                <span className="text-[11px] font-mono text-slate-400">ANOMALY SCORE</span>
                <span
                  className={`text-2xl font-mono font-bold tabular-nums ${
                    machine.anomaly_score >= 0.85
                      ? "text-rose-400"
                      : machine.anomaly_score >= 0.7
                      ? "text-amber-400"
                      : "text-emerald-400"
                  }`}
                >
                  {(machine.anomaly_score * 100).toFixed(0)}%
                </span>
              </div>

              {/* Multi-tier progress bar */}
              <div className="w-full h-2 rounded bg-[#131929] overflow-hidden border border-white/[0.05]">
                <div
                  className={`h-full transition-all duration-300 ${
                    machine.anomaly_score >= 0.85
                      ? "bg-rose-500"
                      : machine.anomaly_score >= 0.7
                      ? "bg-amber-500"
                      : "bg-emerald-500"
                  }`}
                  style={{ width: `${machine.anomaly_score * 100}%` }}
                />
              </div>

              <div className="flex justify-between text-[10px] font-mono text-slate-400 mt-1.5">
                <span>NORMAL &lt; 70%</span>
                <span>WARN 70%</span>
                <span>CRIT 85%</span>
              </div>
            </div>
          </div>

          <div className="pt-2 border-t border-white/[0.06] text-[10px] font-mono text-slate-400">
            Features: {modelsMeta?.anomaly?.feature_count || 53} | Contamination: 0.05 | Hysteresis: 2 cycles
          </div>
        </div>

        {/* Model 2: HistGradientBoosting RUL Estimator (Fallback) */}
        <div className="scada-panel p-4 bg-[#0B0F1A] flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between pb-2 border-b border-white/[0.06]">
              <span className="text-xs font-mono font-bold uppercase text-slate-200 flex items-center gap-1.5">
                <Clock className="w-3.5 h-3.5 text-sky-400" />
                Remaining Useful Life
              </span>
              <span className="text-[10px] font-mono text-amber-300/90 bg-amber-500/10 px-1.5 py-0.5 rounded border border-amber-500/20">
                {modelsMeta?.rul?.model_family ? `${modelsMeta.rul.model_family} (Fallback)` : "HIST-GBM FALLBACK"}
              </span>
            </div>

            <div className="my-3 flex items-center justify-between">
              <div>
                <span className="text-[10px] font-mono text-slate-400 uppercase">
                  Cycles to Overhaul
                </span>
                <div className="text-3xl font-bold font-mono text-white tabular-nums tracking-tight mt-0.5">
                  {machine.predicted_rul.toFixed(1)}
                  <span className="text-xs font-mono text-slate-400 font-normal ml-1">CYC</span>
                </div>
                <div className="text-[10px] font-mono text-slate-400 mt-0.5">
                  ~{(machine.predicted_rul * 1.6).toFixed(0)} operating duty hours
                </div>
              </div>

              <span
                className={`px-2.5 py-1 rounded text-xs font-mono font-bold border ${
                  machine.predicted_rul <= 15
                    ? "bg-rose-950/40 text-rose-300 border-rose-500/40"
                    : machine.predicted_rul <= 30
                    ? "bg-amber-950/40 text-amber-300 border-amber-500/40"
                    : "bg-emerald-950/40 text-emerald-300 border-emerald-500/40"
                }`}
              >
                {machine.predicted_rul <= 15
                  ? "CRITICAL"
                  : machine.predicted_rul <= 30
                  ? "WARNING"
                  : "HEALTHY"}
              </span>
            </div>
          </div>

          <div className="pt-2 border-t border-white/[0.06] text-[10px] font-mono text-slate-400">
            Features: {modelsMeta?.rul?.feature_count || 133} | Tree Regressor | Version: {modelsMeta?.rul?.model_version || "v1.0.0-fallback"}
          </div>
        </div>

        {/* Model 3: HistGradientBoosting Multi-target Failure Classifier (Fallback) */}
        <div className="scada-panel p-4 bg-[#0B0F1A] flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between pb-2 border-b border-white/[0.06]">
              <span className="text-xs font-mono font-bold uppercase text-slate-200 flex items-center gap-1.5">
                <AlertTriangle className="w-3.5 h-3.5 text-amber-400" />
                Failure Mode Matrix
              </span>
              <span className="text-[10px] font-mono text-amber-300/90 bg-amber-500/10 px-1.5 py-0.5 rounded border border-amber-500/20">
                {modelsMeta?.failure_classifier?.model_family ? `${modelsMeta.failure_classifier.model_family} (Fallback)` : "HIST-GBM MULTI"}
              </span>
            </div>

            <div className="my-2 space-y-1.5">
              {Object.entries(machine.failure_modes).map(([mode, prob]) => (
                <div key={mode}>
                  <div className="flex justify-between text-[11px] font-mono text-slate-300 mb-0.5">
                    <span className="truncate max-w-[170px]">{mode}</span>
                    <span className="font-bold tabular-nums text-slate-200">
                      {(prob * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div className="w-full h-1.5 rounded bg-[#131929] overflow-hidden">
                    <div
                      className={`h-full transition-all duration-300 ${
                        prob >= 0.7
                          ? "bg-rose-500"
                          : prob >= 0.4
                          ? "bg-amber-500"
                          : "bg-sky-500"
                      }`}
                      style={{ width: `${prob * 100}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="pt-2 border-t border-white/[0.06] flex items-center justify-between text-[10px] font-mono text-slate-400">
            <span>Features: {modelsMeta?.failure_classifier?.feature_count || 13} | Multi-target</span>
            {onOpenCopilotWithPrompt && (
              <button
                onClick={() =>
                  onOpenCopilotWithPrompt(
                    `Analyze failure risk modes for ${machine.machine_id}`
                  )
                }
                className="text-sky-400 hover:text-sky-300 font-semibold"
              >
                Inspect SOP &rarr;
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
