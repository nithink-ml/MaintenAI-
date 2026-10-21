"use client";

import React, { useState, useMemo } from "react";
import {
  ChevronRight,
  Grid,
  List,
  Search,
} from "lucide-react";
import { StatusBadge, StatusType } from "../ui/Badge";
import { Sparkline } from "../ui/Sparkline";

export interface MachineState {
  machine_id: string;
  name: string;
  equipment_type: "turbofan" | "bearing" | "cnc_mill";
  status: "NORMAL" | "WARNING" | "CRITICAL";
  temperature: number;
  pressure: number;
  rpm: number;
  vibration: number;
  current: number;
  operating_cycle: number;
  anomaly_score: number;
  predicted_rul: number;
  failure_probability: number;
  failure_modes: Record<string, number>;
  history: { temp: number[]; vib: number[]; pressure: number[] };
}

interface FleetMatrixProps {
  machines: MachineState[];
  selectedMachineId: string;
  onSelectMachine: (id: string) => void;
  onInspectDiagnostics?: (id: string) => void;
}

export const FleetMatrix: React.FC<FleetMatrixProps> = ({
  machines,
  selectedMachineId,
  onSelectMachine,
  onInspectDiagnostics,
}) => {
  const [viewMode, setViewMode] = useState<"grid" | "table">("grid");
  const [searchQuery, setSearchQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState<string>("ALL");
  const [typeFilter, setTypeFilter] = useState<string>("ALL");

  const filteredMachines = useMemo(() => {
    return machines.filter((m) => {
      const matchesSearch =
        m.machine_id.toLowerCase().includes(searchQuery.toLowerCase()) ||
        m.name.toLowerCase().includes(searchQuery.toLowerCase());
      const matchesStatus = statusFilter === "ALL" || m.status === statusFilter;
      const matchesType = typeFilter === "ALL" || m.equipment_type === typeFilter;
      return matchesSearch && matchesStatus && matchesType;
    });
  }, [machines, searchQuery, statusFilter, typeFilter]);

  return (
    <div className="space-y-4">
      {/* Control Bar: Search, Filters & View Toggle */}
      <div className="flex flex-col md:flex-row items-stretch md:items-center justify-between gap-3 bg-[#0B0F1A] p-3 rounded-lg border border-white/[0.08]">
        {/* Search */}
        <div className="relative flex-1 max-w-md">
          <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search by Machine ID, Asset Name, or Type..."
            className="w-full bg-[#121826] border border-white/[0.08] focus:border-sky-500 rounded pl-9 pr-3 py-1.5 text-xs font-mono text-slate-100 placeholder:text-slate-400 focus:outline-none transition-colors"
          />
        </div>

        {/* Filter Chips & View Mode */}
        <div className="flex flex-wrap items-center gap-2">
          {/* Status Filter */}
          <div className="flex items-center gap-1 bg-[#121826] p-0.5 rounded border border-white/[0.08] text-xs font-mono">
            {["ALL", "CRITICAL", "WARNING", "NORMAL"].map((status) => (
              <button
                key={status}
                onClick={() => setStatusFilter(status)}
                className={`px-2 py-1 rounded text-[10px] font-medium transition-colors ${
                  statusFilter === status
                    ? "bg-[#1E283D] text-white font-semibold"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                {status}
              </button>
            ))}
          </div>

          {/* Equipment Type Filter */}
          <div className="hidden lg:flex items-center gap-1 bg-[#121826] p-0.5 rounded border border-white/[0.08] text-xs font-mono">
            {[
              { label: "ALL", value: "ALL" },
              { label: "TURBOFAN", value: "turbofan" },
              { label: "BEARING", value: "bearing" },
              { label: "CNC", value: "cnc_mill" },
            ].map((item) => (
              <button
                key={item.value}
                onClick={() => setTypeFilter(item.value)}
                className={`px-2 py-1 rounded text-[10px] font-medium transition-colors ${
                  typeFilter === item.value
                    ? "bg-[#1E283D] text-sky-300 font-semibold"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                {item.label}
              </button>
            ))}
          </div>

          {/* View Toggle */}
          <div className="flex items-center bg-[#121826] p-0.5 rounded border border-white/[0.08]">
            <button
              onClick={() => setViewMode("grid")}
              className={`p-1.5 rounded transition-colors ${
                viewMode === "grid" ? "bg-[#1E283D] text-white" : "text-slate-400 hover:text-slate-200"
              }`}
              title="Card Grid View"
            >
              <Grid className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={() => setViewMode("table")}
              className={`p-1.5 rounded transition-colors ${
                viewMode === "table" ? "bg-[#1E283D] text-white" : "text-slate-400 hover:text-slate-200"
              }`}
              title="SCADA Density Table View"
            >
              <List className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      </div>

      {/* Grid View */}
      {viewMode === "grid" ? (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3.5">
          {filteredMachines.map((m) => {
            const isSelected = m.machine_id === selectedMachineId;
            return (
              <div
                key={m.machine_id}
                onClick={() => onSelectMachine(m.machine_id)}
                className={`scada-panel p-4 cursor-pointer transition-all duration-150 flex flex-col justify-between ${
                  isSelected
                    ? "bg-[#141B2D] border-sky-500/60 ring-1 ring-sky-500/40"
                    : "hover:bg-[#121826] hover:border-white/[0.14]"
                }`}
              >
                <div>
                  {/* Top Bar */}
                  <div className="flex items-start justify-between gap-2 mb-2">
                    <div>
                      <div className="flex items-center gap-1.5">
                        <span className="text-xs font-mono font-bold text-white tracking-wider">
                          {m.machine_id.toUpperCase()}
                        </span>
                        <span className="text-[10px] font-mono text-slate-400 uppercase">
                          • {m.equipment_type}
                        </span>
                      </div>
                      <h3 className="text-sm font-semibold text-slate-200 line-clamp-1 mt-0.5">
                        {m.name}
                      </h3>
                    </div>
                    <StatusBadge status={m.status as StatusType} size="sm" />
                  </div>

                  {/* Core Metrics Comparison */}
                  <div className="grid grid-cols-2 gap-2 my-3">
                    <div className="bg-[#090D17] p-2.5 rounded border border-white/[0.05]">
                      <span className="text-[10px] font-mono text-slate-400 block uppercase">
                        Predicted RUL
                      </span>
                      <div className="flex items-baseline gap-1 mt-0.5">
                        <span className="text-lg font-bold font-mono text-white tabular-nums">
                          {m.predicted_rul.toFixed(1)}
                        </span>
                        <span className="text-[10px] font-mono text-slate-400">CYCLES</span>
                      </div>
                    </div>

                    <div className="bg-[#090D17] p-2.5 rounded border border-white/[0.05]">
                      <span className="text-[10px] font-mono text-slate-400 block uppercase">
                        Anomaly Index
                      </span>
                      <div className="flex items-baseline gap-1 mt-0.5">
                        <span
                          className={`text-lg font-bold font-mono tabular-nums ${
                            m.anomaly_score >= 0.8
                              ? "text-rose-400"
                              : m.anomaly_score >= 0.7
                              ? "text-amber-400"
                              : "text-emerald-400"
                          }`}
                        >
                          {(m.anomaly_score * 100).toFixed(0)}%
                        </span>
                        <span className="text-[10px] font-mono text-slate-400">ISOF</span>
                      </div>
                    </div>
                  </div>

                  {/* Waveform Sparkline */}
                  <div className="my-2">
                    <div className="flex justify-between items-center text-[10px] font-mono text-slate-400 mb-1">
                      <span>VIBRATION ACCEL</span>
                      <span className="text-slate-300 tabular-nums">{m.vibration.toFixed(2)} g</span>
                    </div>
                    <Sparkline
                      data={m.history.vib}
                      color={
                        m.status === "CRITICAL"
                          ? "#F43F5E"
                          : m.status === "WARNING"
                          ? "#F59E0B"
                          : "#10B981"
                      }
                      height={28}
                    />
                  </div>
                </div>

                {/* Footer Strip */}
                <div className="mt-3 pt-2.5 border-t border-white/[0.06] flex items-center justify-between text-[11px] font-mono text-slate-400">
                  <div className="flex items-center gap-3">
                    <span>TEMP: <strong className="text-slate-300">{m.temperature.toFixed(1)}°C</strong></span>
                    <span>CYC: <strong className="text-slate-300">#{m.operating_cycle}</strong></span>
                  </div>
                  {onInspectDiagnostics && (
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        onInspectDiagnostics(m.machine_id);
                      }}
                      className="text-sky-400 hover:text-sky-300 flex items-center text-[10px] uppercase font-semibold"
                    >
                      Diagnostics <ChevronRight className="w-3 h-3" />
                    </button>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      ) : (
        /* SCADA Density Table View */
        <div className="scada-panel overflow-x-auto">
          <table className="w-full text-left text-xs font-mono">
            <thead className="bg-[#090D17] border-b border-white/[0.08] text-slate-400 text-[10px] uppercase tracking-wider">
              <tr>
                <th className="px-4 py-3">Asset ID / Name</th>
                <th className="px-4 py-3">Category</th>
                <th className="px-4 py-3">Status</th>
                <th className="px-4 py-3 text-right">Temp (°C)</th>
                <th className="px-4 py-3 text-right">Vib (g RMS)</th>
                <th className="px-4 py-3 text-right">Pressure</th>
                <th className="px-4 py-3 text-right">Cycle</th>
                <th className="px-4 py-3 text-right">Anomaly %</th>
                <th className="px-4 py-3 text-right">Predicted RUL</th>
                <th className="px-4 py-3 text-center">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/[0.05]">
              {filteredMachines.map((m) => {
                const isSelected = m.machine_id === selectedMachineId;
                return (
                  <tr
                    key={m.machine_id}
                    onClick={() => onSelectMachine(m.machine_id)}
                    className={`cursor-pointer transition-colors ${
                      isSelected ? "bg-[#141B2D]" : "hover:bg-[#101624]"
                    }`}
                  >
                    <td className="px-4 py-3">
                      <div className="font-bold text-white">{m.machine_id.toUpperCase()}</div>
                      <div className="text-[11px] text-slate-400 truncate max-w-[180px] font-sans">
                        {m.name}
                      </div>
                    </td>
                    <td className="px-4 py-3 text-slate-300 uppercase text-[10px]">
                      {m.equipment_type}
                    </td>
                    <td className="px-4 py-3">
                      <StatusBadge status={m.status as StatusType} size="sm" />
                    </td>
                    <td className="px-4 py-3 text-right tabular-nums text-slate-200">
                      {m.temperature.toFixed(1)}
                    </td>
                    <td className="px-4 py-3 text-right tabular-nums text-slate-200">
                      {m.vibration.toFixed(2)}
                    </td>
                    <td className="px-4 py-3 text-right tabular-nums text-slate-200">
                      {m.pressure.toFixed(1)}
                    </td>
                    <td className="px-4 py-3 text-right tabular-nums text-slate-400">
                      #{m.operating_cycle}
                    </td>
                    <td className="px-4 py-3 text-right tabular-nums font-bold">
                      <span
                        className={
                          m.anomaly_score >= 0.8
                            ? "text-rose-400"
                            : m.anomaly_score >= 0.7
                            ? "text-amber-400"
                            : "text-emerald-400"
                        }
                      >
                        {(m.anomaly_score * 100).toFixed(0)}%
                      </span>
                    </td>
                    <td className="px-4 py-3 text-right tabular-nums font-bold text-slate-100">
                      {m.predicted_rul.toFixed(1)} cyc
                    </td>
                    <td className="px-4 py-3 text-center">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          if (onInspectDiagnostics) onInspectDiagnostics(m.machine_id);
                        }}
                        className="px-2.5 py-1 rounded bg-[#182033] hover:bg-sky-950 text-sky-300 text-[10px] font-mono border border-sky-500/20"
                      >
                        Inspect
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};
