import { useEffect, useState } from "react";
import { MessageSquare } from "lucide-react";
import { api } from "../lib/api";
import { useDashboard } from "../lib/context";
import type { KPIs } from "../lib/types";

export default function NavBar() {
  const { chatOpen, setChatOpen } = useDashboard();
  const [kpis, setKpis] = useState<KPIs | null>(null);

  useEffect(() => {
    const poll = () => {
      api.getKPIs().then(setKpis).catch(() => setKpis(null));
    };
    poll();
    const id = setInterval(poll, 10_000);
    return () => clearInterval(id);
  }, []);

  const dbOk = kpis?.db_connected ?? false;
  const sciOk = kpis?.science_container_available ?? false;

  return (
    <nav className="h-12 flex items-center justify-between px-4 border-b border-zinc-800 bg-zinc-900/80 backdrop-blur-sm shrink-0">
      {/* Branding */}
      <div className="flex items-center gap-3">
        <span className="font-semibold text-sm tracking-tight text-cyan-400">
          Tokyo Eye
        </span>
        <span className="text-[10px] text-zinc-500 uppercase tracking-widest">
          v5 Dashboard
        </span>
      </div>

      {/* Status indicators + chat toggle */}
      <div className="flex items-center gap-4 text-xs text-zinc-400">
        <div className="flex items-center gap-1.5">
          <span
            className={`w-2 h-2 rounded-full ${dbOk ? "bg-emerald-400" : "bg-red-400"}`}
          />
          <span>DB</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span
            className={`w-2 h-2 rounded-full ${sciOk ? "bg-emerald-400" : "bg-red-400"}`}
          />
          <span>Science</span>
        </div>
        <span className="text-zinc-600">|</span>
        <span className="text-zinc-500">Researcher</span>
        <span className="text-zinc-600">|</span>
        <button
          onClick={() => setChatOpen(!chatOpen)}
          className={`flex items-center gap-1.5 px-2 py-1 rounded-md transition-colors ${
            chatOpen
              ? "bg-cyan-900/40 text-cyan-300 border border-cyan-800/40"
              : "hover:bg-zinc-800 text-zinc-400 hover:text-zinc-200"
          }`}
          aria-label={chatOpen ? "Close agent chat" : "Open agent chat"}
        >
          <MessageSquare size={14} />
          <span className="text-[11px]">Agent</span>
        </button>
      </div>
    </nav>
  );
}
