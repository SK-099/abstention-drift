"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { ArrowLeft, BookOpen, MessageSquare, Share2, Sparkles } from "lucide-react";
import { api, Citation, Notebook, SourceDoc } from "@/lib/api";
import ChatPanel from "./ChatPanel";
import ShareDialog from "./ShareDialog";
import SourcesPanel from "./SourcesPanel";
import SourceViewer from "./SourceViewer";
import StudioPanel from "./StudioPanel";

type Tab = "sources" | "chat" | "studio";

export default function NotebookView({ notebookId }: { notebookId: string }) {
  const [nb, setNb] = useState<Notebook | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [sources, setSources] = useState<SourceDoc[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [viewing, setViewing] = useState<{ doc: SourceDoc; passageId: string | null } | null>(null);
  const [notesKey, setNotesKey] = useState(0);
  const [tab, setTab] = useState<Tab>("chat");
  const [title, setTitle] = useState("");
  const [ask, setAsk] = useState<{ q: string; n: number } | null>(null);
  const askQuestion = (q: string) => {
    setAsk((a) => ({ q, n: (a?.n || 0) + 1 }));
    setTab("chat");
  };
  const known = useRef<Set<string>>(new Set());

  useEffect(() => {
    api
      .notebook(notebookId)
      .then((found) => {
        setNb(found);
        setTitle(found.name);
      })
      .catch((e) => setError(e.status === 404 ? "Notebook not found, or it is not shared with you" : e.message));
  }, [notebookId]);

  const loadSources = useCallback(async () => {
    const docs = await api.sources(notebookId).catch(() => [] as SourceDoc[]);
    setSources(docs);
    // newly indexed sources are ticked automatically (like NotebookLM)
    const fresh = docs.filter((d) => d.status === "indexed" && !known.current.has(d.id)).map((d) => d.id);
    fresh.forEach((id) => known.current.add(id));
    setSelected((sel) => {
      const next = new Set([...sel].filter((id) => docs.some((d) => d.id === id && d.status === "indexed")));
      fresh.forEach((id) => next.add(id));
      return next;
    });
  }, [notebookId]);

  useEffect(() => {
    known.current = new Set();
    loadSources();
  }, [loadSources]);

  // poll while sources are being processed
  const pending = sources.some(
    (s) => s.status === "queued" || s.status === "processing" || s.guide_status === "pending",
  );
  useEffect(() => {
    if (!pending) return;
    const t = setInterval(loadSources, 2500);
    return () => clearInterval(t);
  }, [pending, loadSources]);

  const builtin = !!nb?.is_public;
  const canEdit = !builtin && (nb?.role === "owner" || nb?.role === "editor");
  const [sharing, setSharing] = useState(false);
  const indexed = sources.filter((s) => s.status === "indexed");
  const documentIds = useMemo(() => (builtin ? null : [...selected]), [builtin, selected]);
  // chat suggestions from the source guides of the ticked sources (like NotebookLM)
  const suggestions = useMemo(() => {
    const ready = indexed.filter((d) => selected.has(d.id) && d.guide_status === "ready" && d.guide);
    if (!ready.length) return null;
    const qs = ready.flatMap((d) => d.guide!.questions.slice(0, ready.length > 2 ? 1 : 2)).slice(0, 3);
    return [ready.length > 1 ? "Summarise my sources" : "Summarise this source", ...qs];
  }, [indexed, selected]);

  const openCitation = (c: Citation) => {
    const doc = sources.find((d) => d.id === c.source_document);
    if (doc) {
      setViewing({ doc, passageId: c.document_id });
      setTab("sources");
    } else {
      window.alert(`${c.title || c.document_id}\n\n${c.snippet}`);
    }
  };

  async function rename() {
    const t = title.trim();
    if (!nb || !t || t === nb.name) return setTitle(nb?.name || "");
    const updated = await api.renameNotebook(nb.id, t).catch(() => null);
    if (updated) setNb(updated);
  }

  if (error) {
    return (
      <div className="p-8">
        <Link href="/" className="text-accent">
          ← All notebooks
        </Link>
        <p className="mt-4 text-bad">{error}</p>
      </div>
    );
  }

  const panel = "min-h-0 overflow-hidden rounded-2xl border border-line bg-panel";
  return (
    <div className="flex h-dvh flex-col">
      <header className="flex items-center gap-3 px-4 py-2.5">
        <Link href="/" className="rounded-lg p-1.5 hover:bg-panel" title="All notebooks">
          <ArrowLeft size={18} />
        </Link>
        <div className="flex h-8 w-8 items-center justify-center rounded-full bg-accent-soft text-accent">
          <BookOpen size={16} />
        </div>
        {builtin || !canEdit ? (
          <h1 className="min-w-0 flex-1 truncate text-lg font-semibold">{nb?.name}</h1>
        ) : (
          <input
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            onBlur={rename}
            onKeyDown={(e) => e.key === "Enter" && (e.target as HTMLInputElement).blur()}
            className="min-w-0 flex-1 truncate rounded-md bg-transparent px-1 text-lg font-semibold outline-none hover:bg-panel focus:bg-panel"
            title="Rename notebook"
          />
        )}
        {nb && !builtin && nb.role === "viewer" && (
          <span className="rounded-full border border-line px-2 py-0.5 text-xs text-muted">View only</span>
        )}
        {nb && !builtin && nb.role === "editor" && (
          <span className="rounded-full border border-line px-2 py-0.5 text-xs text-muted">Shared with you · editor</span>
        )}
        {nb && nb.role === "owner" && !builtin && (
          <button
            onClick={() => setSharing(true)}
            className="inline-flex items-center gap-1.5 rounded-lg border border-line bg-panel px-3 py-1.5 text-sm hover:bg-panel-2"
          >
            <Share2 size={15} /> Share
          </button>
        )}
        {sharing && nb && <ShareDialog notebookId={nb.id} name={nb.name} onClose={() => setSharing(false)} />}
      </header>

      <nav className="flex gap-1 px-4 pb-2 lg:hidden">
        {(
          [
            ["sources", "Sources", BookOpen],
            ["chat", "Chat", MessageSquare],
            ["studio", "Studio", Sparkles],
          ] as const
        ).map(([k, label, Icon]) => (
          <button
            key={k}
            onClick={() => setTab(k)}
            className={`flex flex-1 items-center justify-center gap-1.5 rounded-lg py-2 text-sm ${tab === k ? "bg-panel font-semibold shadow-sm" : "text-muted"}`}
          >
            <Icon size={15} /> {label}
          </button>
        ))}
      </nav>

      <div className="grid min-h-0 flex-1 grid-cols-1 gap-3 px-4 pb-4 lg:grid-cols-[18rem_minmax(0,1fr)_20rem] xl:grid-cols-[20rem_minmax(0,1fr)_22rem]">
        <section className={`${panel} ${tab === "sources" ? "" : "hidden"} lg:block`}>
          {viewing ? (
            <SourceViewer
              notebookId={notebookId}
              doc={sources.find((d) => d.id === viewing.doc.id) || viewing.doc}
              passageId={viewing.passageId}
              onBack={() => setViewing(null)}
              onAsk={askQuestion}
              canEdit={canEdit}
              onChanged={loadSources}
            />
          ) : (
            <SourcesPanel
              notebookId={notebookId}
              builtin={builtin}
              sources={sources}
              selected={selected}
              setSelected={setSelected}
              onChanged={loadSources}
              onOpen={(doc) => setViewing({ doc, passageId: null })}
              canEdit={canEdit}
            />
          )}
        </section>
        <section className={`${panel} ${tab === "chat" ? "" : "hidden"} lg:block`}>
          {nb && (
            <ChatPanel
              notebookId={notebookId}
              notebookName={nb.name}
              documentIds={documentIds}
              hasSources={indexed.length > 0}
              builtin={builtin}
              onCite={openCitation}
              onNotesChanged={() => setNotesKey((k) => k + 1)}
              ask={ask}
              suggestions={suggestions}
            />
          )}
        </section>
        <section className={`${panel} ${tab === "studio" ? "" : "hidden"} lg:block`}>
          <StudioPanel
            notebookId={notebookId}
            refreshKey={notesKey}
            onCite={openCitation}
            onAsk={askQuestion}
            documentIds={documentIds}
            builtin={builtin}
            canEdit={canEdit}
            hasSources={indexed.length > 0}
          />
        </section>
      </div>
    </div>
  );
}
