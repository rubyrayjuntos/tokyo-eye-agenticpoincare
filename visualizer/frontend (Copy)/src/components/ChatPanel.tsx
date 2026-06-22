/**
 * ChatPanel — sidebar chat interface for talking to the agent.
 * 
 * Displays conversation history, sends messages, shows tool results
 * and viewport directives inline.
 */
import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Send, Bot, User, Loader2, Zap, Eye } from 'lucide-react';
import { sendChat, ViewportDirective } from '../lib/agentClient';
import { cn } from '../lib/utils';

interface ChatMessage {
  id: string;
  role: 'user' | 'agent';
  content: string;
  timestamp: Date;
  directives?: ViewportDirective[];
  toolResults?: any[];
  loading?: boolean;
}

interface ChatPanelProps {
  sessionId: string;
  structureId: string;
  currentMetric: string;
  curvature: number;
  onDirective: (directive: ViewportDirective) => void;
  agentConnected: boolean;
}

export default function ChatPanel({
  sessionId,
  structureId,
  currentMetric,
  curvature,
  onDirective,
  agentConnected,
}: ChatPanelProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [sending, setSending] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = useCallback(async () => {
    const text = input.trim();
    if (!text || sending) return;

    const userMsg: ChatMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content: text,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setSending(true);

    // Add loading placeholder
    const loadingId = crypto.randomUUID();
    setMessages(prev => [...prev, {
      id: loadingId,
      role: 'agent',
      content: '',
      timestamp: new Date(),
      loading: true,
    }]);

    try {
      const response = await sendChat(text, sessionId, {
        structure_id: structureId,
        current_metric: currentMetric,
        curvature,
      });

      // Replace loading with actual response
      const agentMsg: ChatMessage = {
        id: loadingId,
        role: 'agent',
        content: response.response,
        timestamp: new Date(),
        directives: response.viewport_directives as ViewportDirective[] | undefined,
        toolResults: response.tool_results,
      };

      setMessages(prev => prev.map(m => m.id === loadingId ? agentMsg : m));

      // Apply viewport directives
      if (response.viewport_directives) {
        for (const d of response.viewport_directives) {
          onDirective(d as ViewportDirective);
        }
      }
    } catch (e) {
      setMessages(prev => prev.map(m =>
        m.id === loadingId
          ? { ...m, content: `Error: ${e instanceof Error ? e.message : 'Unknown error'}`, loading: false }
          : m
      ));
    } finally {
      setSending(false);
    }
  }, [input, sending, sessionId, structureId, currentMetric, curvature, onDirective]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="flex flex-col h-full bg-slate-900/95 backdrop-blur-sm border-l border-slate-700/50">
      {/* Header */}
      <div className="px-3 py-2 border-b border-slate-700/50 flex items-center gap-2">
        <Bot className="w-4 h-4 text-cyan-400" />
        <span className="text-xs font-medium text-slate-300">Agent</span>
        <div className={cn(
          "w-1.5 h-1.5 rounded-full ml-auto",
          agentConnected ? "bg-emerald-400" : "bg-red-400"
        )} />
        <span className="text-[10px] text-slate-500">
          {agentConnected ? 'connected' : 'offline'}
        </span>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-3 py-2 space-y-3">
        {messages.length === 0 && (
          <div className="text-center text-slate-500 text-xs mt-8 space-y-2">
            <Bot className="w-8 h-8 mx-auto text-slate-600" />
            <p>Ask about source leaks, uncertainty,<br />or run the DTIE pipeline.</p>
            <div className="space-y-1 text-[10px] text-slate-600">
              <p>"Show me source leaks in this structure"</p>
              <p>"What are the high uncertainty residues?"</p>
              <p>"Run the full pipeline on 4OBE"</p>
            </div>
          </div>
        )}

        {messages.map(msg => (
          <div key={msg.id} className={cn(
            "flex gap-2",
            msg.role === 'user' ? 'justify-end' : 'justify-start'
          )}>
            {msg.role === 'agent' && (
              <Bot className="w-4 h-4 text-cyan-400 mt-0.5 shrink-0" />
            )}
            <div className={cn(
              "max-w-[85%] rounded-lg px-2.5 py-1.5 text-xs",
              msg.role === 'user'
                ? 'bg-cyan-900/50 text-cyan-100'
                : 'bg-slate-800 text-slate-200'
            )}>
              {msg.loading ? (
                <Loader2 className="w-3 h-3 animate-spin text-slate-400" />
              ) : (
                <>
                  <p className="whitespace-pre-wrap">{msg.content}</p>

                  {/* Tool results indicator */}
                  {msg.toolResults && msg.toolResults.length > 0 && (
                    <div className="mt-1.5 flex items-center gap-1 text-[10px] text-amber-400/70">
                      <Zap className="w-3 h-3" />
                      <span>{msg.toolResults.length} tool result(s)</span>
                    </div>
                  )}

                  {/* Viewport directives indicator */}
                  {msg.directives && msg.directives.length > 0 && (
                    <div className="mt-1 flex items-center gap-1 text-[10px] text-emerald-400/70">
                      <Eye className="w-3 h-3" />
                      <span>{msg.directives.length} viewport update(s)</span>
                    </div>
                  )}
                </>
              )}
            </div>
            {msg.role === 'user' && (
              <User className="w-4 h-4 text-slate-400 mt-0.5 shrink-0" />
            )}
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="px-3 py-2 border-t border-slate-700/50">
        <div className="flex items-center gap-2 bg-slate-800 rounded-lg px-2 py-1.5">
          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={agentConnected ? "Ask the agent..." : "Agent offline"}
            disabled={!agentConnected || sending}
            className="flex-1 bg-transparent text-xs text-slate-200 placeholder-slate-500 outline-none"
          />
          <button
            onClick={handleSend}
            disabled={!input.trim() || sending || !agentConnected}
            className={cn(
              "p-1 rounded transition-colors",
              input.trim() && agentConnected
                ? "text-cyan-400 hover:bg-cyan-900/30"
                : "text-slate-600"
            )}
          >
            <Send className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>
    </div>
  );
}
