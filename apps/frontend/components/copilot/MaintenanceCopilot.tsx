"use client";

import React, { useState, useRef, useEffect } from "react";
import {
  BookOpen,
  Check,
  Send,
  ShieldCheck,
  Sparkles,
  Wrench,
  X,
} from "lucide-react";
import { MachineState } from "../fleet/FleetMatrix";

export interface WorkOrder {
  work_order_id: string;
  machine_id: string;
  title: string;
  category: string;
  priority: "HIGH" | "MEDIUM" | "LOW";
  procedures: string[];
  required_parts: string[];
  citations: string[];
  estimated_downtime_hours: number;
  status: "PENDING_APPROVAL" | "APPROVED" | "REJECTED";
  approved_by?: string;
  approved_at?: string;
}

export interface ChatMessage {
  id: string;
  sender: "user" | "ai";
  text: string;
  timestamp: string;
  citations?: { title: string; source: string; score: number }[];
  work_order?: WorkOrder;
}

interface MaintenanceCopilotProps {
  selectedMachine: MachineState;
  messages: ChatMessage[];
  onSendMessage: (text: string) => void;
  onApproveWorkOrder: (msgId: string, woId: string, approved: boolean) => void;
  isTyping: boolean;
}

export const MaintenanceCopilot: React.FC<MaintenanceCopilotProps> = ({
  selectedMachine,
  messages,
  onSendMessage,
  onApproveWorkOrder,
  isTyping,
}) => {
  const [inputText, setInputText] = useState("");
  const chatBottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    chatBottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isTyping]);

  const handleSubmit = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!inputText.trim()) return;
    onSendMessage(inputText);
    setInputText("");
  };

  const handleSuggestedPrompt = (prompt: string) => {
    onSendMessage(prompt);
  };

  return (
    <div className="scada-panel p-4 bg-[#0A0E17] flex flex-col h-[840px] border-white/[0.08] relative">
      {/* Header */}
      <div className="flex items-center justify-between pb-3 border-b border-white/[0.06]">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded bg-[#141B2D] border border-white/[0.1] flex items-center justify-center text-sky-400">
            <Sparkles className="w-4 h-4" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h3 className="text-xs font-bold font-mono uppercase text-white tracking-wide">
                Maintenance Copilot Agent
              </h3>
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
            </div>
            <p className="text-[10px] font-mono text-slate-400">
              LangGraph Multi-Tool Agent + Qdrant RAG Grounding
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 text-[11px] font-mono">
          <span className="text-slate-400 hidden sm:inline">TARGET ASSET:</span>
          <span className="px-2 py-0.5 rounded bg-[#141C2E] border border-sky-500/30 text-sky-300 font-bold">
            {selectedMachine.machine_id.toUpperCase()}
          </span>
        </div>
      </div>

      {/* Suggested Quick Diagnostic Queries */}
      <div className="py-2.5 flex items-center gap-2 overflow-x-auto text-[10px] font-mono border-b border-white/[0.04]">
        <span className="text-slate-500 uppercase tracking-wider whitespace-nowrap">
          Quick Prompts:
        </span>
        <button
          onClick={() =>
            handleSuggestedPrompt(
              `Run diagnostic assessment on ${selectedMachine.machine_id}`
            )
          }
          className="px-2.5 py-1 rounded bg-[#121826] hover:bg-[#182033] text-slate-300 hover:text-white border border-white/[0.06] transition-colors whitespace-nowrap"
        >
          🔍 Assess {selectedMachine.machine_id}
        </button>
        <button
          onClick={() =>
            handleSuggestedPrompt(
              `What are the official SOP repair steps for ${selectedMachine.machine_id}?`
            )
          }
          className="px-2.5 py-1 rounded bg-[#121826] hover:bg-[#182033] text-slate-300 hover:text-white border border-white/[0.06] transition-colors whitespace-nowrap"
        >
          📖 Retrieve SOP Manual
        </button>
        <button
          onClick={() =>
            handleSuggestedPrompt(
              `Draft an emergency work order for ${selectedMachine.machine_id}`
            )
          }
          className="px-2.5 py-1 rounded bg-[#121826] hover:bg-[#182033] text-slate-300 hover:text-white border border-white/[0.06] transition-colors whitespace-nowrap"
        >
          📋 Draft Work Order
        </button>
      </div>

      {/* Message Feed */}
      <div className="flex-1 overflow-y-auto space-y-4 pr-1.5 py-3 text-xs font-mono">
        {messages.map((msg) => {
          const isUser = msg.sender === "user";
          return (
            <div
              key={msg.id}
              className={`flex flex-col ${isUser ? "items-end" : "items-start"}`}
            >
              <div
                className={`max-w-[94%] p-3.5 rounded-lg border ${
                  isUser
                    ? "bg-[#182238] border-sky-500/30 text-slate-100 rounded-br-none"
                    : "bg-[#0E1422] border-white/[0.07] text-slate-200 rounded-bl-none shadow-sm"
                }`}
              >
                {/* Message Header */}
                <div className="flex items-center justify-between text-[10px] text-slate-400 mb-2 pb-1.5 border-b border-white/[0.05]">
                  <span className="font-bold tracking-wider text-slate-300 uppercase">
                    {isUser ? "OPERATOR // HUMAN" : "MAINTENANCE COPILOT // AGENT"}
                  </span>
                  <span>{msg.timestamp}</span>
                </div>

                {/* Body */}
                <div className="whitespace-pre-line font-sans text-xs leading-relaxed text-slate-200">
                  {msg.text}
                </div>

                {/* Grounded Citations Accordion */}
                {msg.citations && msg.citations.length > 0 && (
                  <div className="mt-3 pt-2.5 border-t border-white/[0.06] space-y-2">
                    <div className="flex items-center gap-1.5 text-[11px] font-mono font-semibold text-sky-400">
                      <BookOpen className="w-3.5 h-3.5" />
                      Grounded Manual Evidence (Qdrant Vector Index):
                    </div>
                    {msg.citations.map((cite, idx) => (
                      <div
                        key={idx}
                        className="bg-[#090D17] p-2.5 rounded border border-white/[0.06] text-[10px] font-mono space-y-1"
                      >
                        <div className="flex justify-between items-center text-slate-200 font-bold">
                          <span>{cite.title}</span>
                          <span className="text-sky-400">
                            {(cite.score * 100).toFixed(0)}% MATCH
                          </span>
                        </div>
                        <div className="text-slate-400 flex items-center justify-between">
                          <span>SOURCE: {cite.source}</span>
                          <span className="text-emerald-400">GROUNDED &bull; UNMODIFIED</span>
                        </div>
                      </div>
                    ))}
                  </div>
                )}

                {/* Human-in-the-Loop Work Order Draft */}
                {msg.work_order && (
                  <div className="mt-3.5 p-3.5 rounded bg-[#090D17] border border-sky-500/40 space-y-3">
                    <div className="flex items-center justify-between border-b border-white/[0.08] pb-2">
                      <div className="flex items-center gap-2">
                        <Wrench className="w-4 h-4 text-sky-400" />
                        <span className="font-mono font-bold text-white text-xs">
                          {msg.work_order.work_order_id}
                        </span>
                      </div>
                      <span
                        className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded uppercase border ${
                          msg.work_order.status === "APPROVED"
                            ? "bg-emerald-950/40 text-emerald-300 border-emerald-500/40"
                            : msg.work_order.status === "REJECTED"
                            ? "bg-rose-950/40 text-rose-300 border-rose-500/40"
                            : "bg-amber-950/40 text-amber-300 border-amber-500/40 animate-pulse"
                        }`}
                      >
                        {msg.work_order.status}
                      </span>
                    </div>

                    <div>
                      <div className="text-xs font-bold text-white font-sans">
                        {msg.work_order.title}
                      </div>
                      <div className="text-[10px] font-mono text-slate-400 mt-0.5">
                        DOWNTIME IMPACT: {msg.work_order.estimated_downtime_hours} HOURS | PRIORITY: {msg.work_order.priority}
                      </div>
                    </div>

                    {/* Step-by-Step SOP Procedures */}
                    <div className="space-y-1 text-[11px] font-mono">
                      <span className="text-slate-400 uppercase text-[10px]">
                        Required SOP Procedures:
                      </span>
                      {msg.work_order.procedures.map((proc, pIdx) => (
                        <div
                          key={pIdx}
                          className="text-slate-300 pl-2 border-l border-white/[0.1] text-[10px] py-0.5"
                        >
                          {proc}
                        </div>
                      ))}
                    </div>

                    {/* Parts Required */}
                    <div className="text-[10px] font-mono text-slate-400 pt-1 border-t border-white/[0.04]">
                      <span className="uppercase">Parts: </span>
                      <span className="text-sky-300 font-semibold">
                        {msg.work_order.required_parts.join(", ")}
                      </span>
                    </div>

                    {/* Authorization Action Buttons */}
                    {msg.work_order.status === "PENDING_APPROVAL" ? (
                      <div className="pt-2 flex items-center gap-2">
                        <button
                          onClick={() =>
                            onApproveWorkOrder(msg.id, msg.work_order!.work_order_id, true)
                          }
                          className="flex-1 py-1.5 rounded bg-emerald-700 hover:bg-emerald-600 text-white font-mono font-bold text-xs flex items-center justify-center gap-1.5 transition-colors shadow"
                        >
                          <Check className="w-3.5 h-3.5" />
                          Authorize &amp; Dispatch Work Order
                        </button>
                        <button
                          onClick={() =>
                            onApproveWorkOrder(msg.id, msg.work_order!.work_order_id, false)
                          }
                          className="px-3 py-1.5 rounded bg-[#182033] hover:bg-rose-950 text-slate-300 hover:text-rose-400 text-xs font-mono transition-colors"
                          title="Reject draft"
                        >
                          <X className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    ) : (
                      <div className="pt-1 flex items-center justify-between text-[11px] font-mono">
                        <span className="text-emerald-400 font-bold flex items-center gap-1">
                          <ShieldCheck className="w-4 h-4 text-emerald-400" />
                          Digitally Authorized by {msg.work_order.approved_by || "Chief Engineer"}
                        </span>
                        <span className="text-slate-500">AUDIT LOG RECORDED</span>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          );
        })}

        {isTyping && (
          <div className="flex items-center gap-2 text-xs font-mono text-slate-400 bg-[#0E1422] p-3 rounded border border-white/[0.06] w-fit">
            <Sparkles className="w-3.5 h-3.5 text-sky-400 animate-spin" />
            <span>Agent querying telemetry models &amp; retrieving RAG manuals...</span>
          </div>
        )}
        <div ref={chatBottomRef} />
      </div>

      {/* Input Bar */}
      <form onSubmit={handleSubmit} className="pt-3 border-t border-white/[0.06] flex items-center gap-2">
        <input
          type="text"
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          placeholder={`Enter diagnostic command or query regarding ${selectedMachine.machine_id}...`}
          className="flex-1 bg-[#121826] border border-white/[0.08] focus:border-sky-500 rounded px-3 py-2 text-xs font-mono text-slate-100 placeholder:text-slate-500 focus:outline-none transition-colors"
        />
        <button
          type="submit"
          disabled={!inputText.trim()}
          className="px-3.5 py-2 rounded bg-sky-600 hover:bg-sky-500 disabled:opacity-40 text-white text-xs font-mono font-bold flex items-center gap-1.5 transition-colors"
        >
          <Send className="w-3.5 h-3.5" />
          <span>SEND</span>
        </button>
      </form>
    </div>
  );
};
