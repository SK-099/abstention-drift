"use client";

import { useEffect, useRef, useState } from "react";
import { ArrowUp, ChevronDown, MessageSquarePlus, Square, Trash2 } from "lucide-react";
import { api, Citation, Conversation, Final, StoredMessage, streamChat } from "@/lib/api";
import ChatMessage, { UiMessage } from "./ChatMessage";
import ModelPicker from "./ModelPicker";

let keySeq = 0;
const newKey = () => `m${++keySeq}`;

function fromStored(ms: StoredMessage[]): UiMessage[] {
  return ms.map((m) => {
    const meta = m.meta as Record<string, unknown>;
    if (m.role === "user") return { key: m.id, role: "user", content: m.content };
    const trace = (meta.trace as string[]) || [];
    return {
      key: m.id,
      role: "assistant",
      content: m.content,
      messageId: m.id,
      model: (meta.model as string) || ((meta.versions as Record<string, string>)?.llm_model ?? undefined),
      guardrails: (meta.guardrails as UiMessage["guardrails"]) || [],
      steps: trace.map((t) => ({ node: t.split(/[[:]/)[0], label: t.split(/[[:]/)[0].replace("_", " "), detail: t })),
      final: {
        decision: (m.decision || "answer") as Final["decision"],
        p_unanswerable: m.p_unanswerable,
        latency_ms: m.latency_ms || 0,
        attempts: (meta.attempts as number) || 0,
        query: (meta.query as string) || null,
        citations: (meta.citations as Citation[]) || [],
      },
    };
  });
}

export default function ChatPanel({
  notebookId,
  notebookName,
  documentIds,
  hasSources,
  builtin,
  onCite,
  onNotesChanged,
  ask,
  suggestions: guideSuggestions,
}: {
  notebookId: string;
  notebookName: string;
  documentIds: string[] | null;
  hasSources: boolean;
  builtin: boolean;
  onCite: (c: Citation) => void;
  onNotesChanged: () => void;
  ask?: { q: string; n: number } | null;
  suggestions?: string[] | null;
}) {
  const [model, setModel] = useState<string | null>(null);
  const [convs, setConvs] = useState<Conversation[]>([]);
  const [convId, setConvId] = useState<string | null>(null);
  const [messages, setMessages] = useState<UiMessage[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [showConvs, setShowConvs] = useState(false);
  // switching local models: one model fits on a small GPU, so load the new one while the user types
  const [loading, setLoading] = useState<{ name: string; done?: number | null; failed?: boolean } | null>(null);
  const warm = (id: string, name: string) => {
    setLoading({ name });
    api
      .warmModel(id)
      .then((r) => {
        setLoading(r.loaded ? { name, done: r.seconds } : null);
        if (r.loaded) setTimeout(() => setLoading((l) => (l && l.name === name && l.done !== undefined ? null : l)), 6000);
      })
      .catch(() => setLoading(null));
  };
  const abort = useRef<AbortController | null>(null);
  const bottom = useRef<HTMLDivElement>(null);

  const loadConvs = () => api.conversations(notebookId).then(setConvs).catch(() => setConvs([]));

  useEffect(() => {
    loadConvs();
    setConvId(null);
    setMessages([]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [notebookId]);

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  async function openConv(id: string) {
    setShowConvs(false);
    const c = await api.conversation(id);
    setConvId(id);
    setMessages(fromStored(c.messages));
  }

  async function removeConv(id: string) {
    await api.deleteConversation(id).catch(() => undefined);
    if (id === convId) {
      setConvId(null);
      setMessages([]);
    }
    loadConvs();
  }

  function patch(key: string, f: (m: UiMessage) => UiMessage) {
    setMessages((ms) => ms.map((m) => (m.key === key ? f(m) : m)));
  }

  // a question clicked in a source guide
  useEffect(() => {
    if (ask) send(ask.q);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ask?.n]);

  async function send(text?: string) {
    const q = (text ?? input).trim();
    if (!q || busy) return;
    if (documentIds && documentIds.length === 0 && !builtin) return;
    setInput("");
    setBusy(true);
    const ak = newKey();
    setMessages((ms) => [
      ...ms,
      { key: newKey(), role: "user", content: q },
      { key: ak, role: "assistant", content: "", steps: [], streaming: true, model: model || undefined },
    ]);
    const ctl = new AbortController();
    abort.current = ctl;
    try {
      for await (const ev of streamChat(
        {
          message: q,
          collection: notebookId,
          conversation_id: convId || undefined,
          model: model || undefined,
          document_ids: builtin ? undefined : documentIds || undefined,
        },
        ctl.signal,
      )) {
        if (ev.event === "meta") {
          setConvId(ev.data.conversation_id);
          patch(ak, (m) => ({ ...m, model: ev.data.model }));
        } else if (ev.event === "step") patch(ak, (m) => ({ ...m, steps: [...(m.steps || []), ev.data] }));
        else if (ev.event === "token") patch(ak, (m) => ({ ...m, content: m.content + ev.data.text }));
        else if (ev.event === "guardrail") patch(ak, (m) => ({ ...m, guardrails: [...(m.guardrails || []), ev.data] }));
        else if (ev.event === "final")
          patch(ak, (m) => ({
            ...m,
            content: ev.data.answer, // the checked final answer replaces streamed tokens
            final: ev.data,
            messageId: ev.data.message_id,
            model: ev.data.model || m.model,
            streaming: false,
          }));
        else if (ev.event === "error") patch(ak, (m) => ({ ...m, error: ev.data.message, streaming: false }));
      }
    } catch (e) {
      if ((e as Error).name !== "AbortError") patch(ak, (m) => ({ ...m, error: (e as Error).message }));
    } finally {
      patch(ak, (m) => ({ ...m, streaming: false }));
      setBusy(false);
      abort.current = null;
      loadConvs();
    }
  }

  const noneSelected = !builtin && hasSources && documentIds !== null && documentIds.length === 0;
  const suggestions = builtin
    ? ["How do I create a bucket in IBM Cloud Object Storage?", "What is the capital gains tax rate?"]
    : guideSuggestions || ["What are the key findings?", "Which methods were used?", "What are the main limitations?"];

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center gap-2 border-b border-line px-3 py-2">
        <ModelPicker
          value={model}
          onChange={(id, info, picked) => {
            setModel(id);
            if (picked && id.startsWith("local:")) warm(id, info?.name || id.split(":").slice(1).join(":"));
          }}
        />
        <div className="relative ml-auto">
          <button
            onClick={() => setShowConvs((s) => !s)}
            className="inline-flex items-center gap-1 rounded-lg px-2.5 py-1.5 text-sm text-muted hover:bg-panel-2"
          >
            Chats ({convs.length}) <ChevronDown size={15} />
          </button>
          {showConvs && (
            <div className="absolute right-0 top-full z-30 mt-1 w-72 rounded-xl border border-line bg-panel py-1 shadow-xl">
              {convs.length === 0 && <p className="px-3 py-2 text-sm text-muted">No chats yet</p>}
              {convs.map((c) => (
                <div key={c.id} className="group flex items-center hover:bg-panel-2">
                  <button onClick={() => openConv(c.id)} className="min-w-0 flex-1 truncate px-3 py-2 text-left text-sm" title={c.title}>
                    {c.id === convId ? "• " : ""}
                    {c.title}
                  </button>
                  <button onClick={() => removeConv(c.id)} className="mr-2 rounded p-1 text-muted opacity-0 hover:text-bad group-hover:opacity-100" title="Delete chat">
                    <Trash2 size={14} />
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
        <button
          onClick={() => {
            setConvId(null);
            setMessages([]);
          }}
          title="New chat"
          className="rounded-lg p-1.5 text-muted hover:bg-panel-2"
        >
          <MessageSquarePlus size={18} />
        </button>
      </div>

      <div className="scroll-thin flex-1 overflow-y-auto">
        <div className="mx-auto flex max-w-3xl flex-col gap-6 px-4 py-6">
          {messages.length === 0 && (
            <div className="mt-8 text-center">
              <h2 className="text-2xl font-semibold">{notebookName}</h2>
              <p className="mx-auto mt-2 max-w-md text-sm text-muted">
                {builtin
                  ? "Ask about this benchmark collection. Answers cite the passages they come from."
                  : hasSources
                    ? "Ask anything about your sources. Answers are grounded in them and cited; if your sources do not contain the answer, the assistant says so."
                    : "Add sources on the left to start."}
              </p>
              {(builtin || hasSources) && (
                <div className="mt-6 flex flex-wrap justify-center gap-2">
                  {suggestions.map((s) => (
                    <button key={s} onClick={() => send(s)} className="rounded-full border border-line bg-panel px-3 py-1.5 text-sm hover:border-accent">
                      {s}
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}
          {messages.map((m) => (
            <ChatMessage key={m.key} m={m} notebookId={notebookId} onCite={onCite} onSaved={onNotesChanged} />
          ))}
          <div ref={bottom} />
        </div>
      </div>

      <div className="border-t border-line px-4 py-3">
        <div className="mx-auto max-w-3xl">
          {noneSelected && <p className="mb-2 text-xs text-warn">Tick at least one source to ask a question.</p>}
          {loading && (
            <p className="mb-2 flex items-center gap-1.5 text-xs text-muted">
              {loading.done === undefined ? (
                <>
                  <span className="h-2 w-2 animate-pulse rounded-full bg-accent" /> Loading {loading.name} into
                  memory… the first answer after switching models can take a minute or two.
                </>
              ) : (
                <>
                  <span className="h-2 w-2 rounded-full bg-ok" /> {loading.name} is ready
                  {loading.done ? ` (loaded in ${Math.round(loading.done)} s)` : ""}.
                </>
              )}
            </p>
          )}
          <div className="flex items-end gap-2 rounded-2xl border border-line bg-panel-2 px-3 py-2 focus-within:border-accent">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  send();
                }
              }}
              rows={1}
              placeholder={builtin || hasSources ? "Ask about your sources…" : "Add a source first"}
              disabled={!builtin && !hasSources}
              className="max-h-40 min-h-[1.75rem] flex-1 resize-none bg-transparent py-1 text-[15px] outline-none"
            />
            {busy ? (
              <button onClick={() => abort.current?.abort()} title="Stop" className="rounded-full bg-ink p-2 text-panel">
                <Square size={14} />
              </button>
            ) : (
              <button
                onClick={() => send()}
                disabled={!input.trim() || noneSelected || (!builtin && !hasSources)}
                title="Send"
                className="rounded-full bg-accent p-2 text-white disabled:opacity-40"
              >
                <ArrowUp size={16} />
              </button>
            )}
          </div>
          <p className="mt-1.5 text-center text-[11px] text-muted">
            Answers come only from your sources and can still be wrong. Check the citations.
          </p>
        </div>
      </div>
    </div>
  );
}
