"use client";

import { useEffect, useRef, useState } from "react";
import { ArrowLeft, ChevronDown, ExternalLink, Loader2, MessageCircleQuestion, RefreshCw, Sparkles } from "lucide-react";
import { api, Passage, SourceDoc } from "@/lib/api";

/** "4:00" or "Title › 4:00" -> 240 */
function seconds(section: string | null | undefined): number | null {
  const last = (section || "").split("\u203a").pop()?.trim() || "";
  if (!/^\d+(:\d{2}){1,2}$/.test(last)) return null;
  return last.split(":").reduce((a, p) => a * 60 + Number(p), 0);
}

function originalHref(notebookId: string, doc: SourceDoc, cited: Passage | undefined): string {
  if (doc.source_type === "youtube" && doc.source_url) {
    const t = seconds(cited?.section);
    if (t === null) return doc.source_url;
    return doc.source_url + (doc.source_url.includes("?") ? "&" : "?") + `t=${t}s`;
  }
  if (doc.source_type === "url" && doc.source_url) return doc.source_url;
  const isPdf = doc.filename.toLowerCase().endsWith(".pdf");
  return api.fileUrl(notebookId, doc.id) + (isPdf && cited?.page ? `#page=${cited.page}` : "");
}

/** Source guide: summary, key topics, and questions the user can click to ask. */
function Guide({
  notebookId,
  doc,
  onAsk,
  canEdit,
  onChanged,
}: {
  notebookId: string;
  doc: SourceDoc;
  onAsk?: (q: string) => void;
  canEdit: boolean;
  onChanged?: () => void;
}) {
  const [open, setOpen] = useState(true);
  const [busy, setBusy] = useState(false);
  const status = busy ? "pending" : doc.guide_status;
  const g = doc.guide;
  if (status === "none" || status === undefined) return null;
  const regenerate = async () => {
    setBusy(true);
    await api.regenerateGuide(notebookId, doc.id).catch(() => undefined);
    setBusy(false);
    onChanged?.();
  };
  return (
    <div className="mb-4 rounded-xl border border-line bg-panel-2 p-3">
      <button onClick={() => setOpen((o) => !o)} className="flex w-full items-center gap-1.5 text-left text-sm font-semibold">
        <Sparkles size={14} className="text-accent" /> Source guide
        <ChevronDown size={14} className={`ml-auto text-muted transition ${open ? "" : "-rotate-90"}`} />
      </button>
      {open && status === "pending" && (
        <p className="mt-2 flex items-center gap-2 text-xs text-muted">
          <Loader2 size={12} className="animate-spin" /> Writing the summary…
        </p>
      )}
      {open && status === "failed" && (
        <p className="mt-2 text-xs text-muted">
          The guide could not be written.{" "}
          {canEdit && (
            <button onClick={regenerate} className="text-accent hover:underline">
              Try again
            </button>
          )}
        </p>
      )}
      {open && status === "ready" && g && (
        <div className="mt-2 space-y-2 text-sm">
          <p className="whitespace-pre-line leading-relaxed">{g.summary}</p>
          {g.topics.length > 0 && (
            <div className="flex flex-wrap gap-1.5">
              {g.topics.map((t) => (
                <span key={t} className="rounded-full border border-line bg-panel px-2 py-0.5 text-xs">
                  {t}
                </span>
              ))}
            </div>
          )}
          {g.questions.length > 0 && onAsk && (
            <div className="flex flex-col items-start gap-1 pt-1">
              {g.questions.map((q) => (
                <button
                  key={q}
                  onClick={() => onAsk(q)}
                  className="inline-flex items-start gap-1.5 rounded-lg px-1 py-0.5 text-left text-xs text-accent hover:bg-accent-soft"
                >
                  <MessageCircleQuestion size={13} className="mt-px shrink-0" /> {q}
                </button>
              ))}
            </div>
          )}
          {canEdit && (
            <button onClick={regenerate} className="inline-flex items-center gap-1 text-[11px] text-muted hover:text-ink">
              <RefreshCw size={11} /> Write again
            </button>
          )}
        </div>
      )}
    </div>
  );
}

/** Reads one source as its indexed passages; the cited passage is highlighted and scrolled into view. */
export default function SourceViewer({
  notebookId,
  doc,
  passageId,
  onBack,
  onAsk,
  canEdit = false,
  onChanged,
}: {
  notebookId: string;
  doc: SourceDoc;
  passageId: string | null;
  onBack: () => void;
  onAsk?: (q: string) => void;
  canEdit?: boolean;
  onChanged?: () => void;
}) {
  const [passages, setPassages] = useState<Passage[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const hit = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setPassages(null);
    api
      .passages(notebookId, doc.id)
      .then(setPassages)
      .catch((e) => setError(e.message));
  }, [notebookId, doc.id]);

  useEffect(() => {
    hit.current?.scrollIntoView({ block: "center", behavior: "smooth" });
  }, [passages, passageId]);

  const cited = passages?.find((p) => p.passage_id === passageId);
  const fileHref = originalHref(notebookId, doc, cited);
  let lastSection: string | null = null;
  let prevWords: string[] = [];

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center gap-2 border-b border-line px-3 py-3">
        <button onClick={onBack} className="rounded-md p-1 hover:bg-panel-2" title="Back to sources">
          <ArrowLeft size={18} />
        </button>
        <h2 className="min-w-0 flex-1 truncate font-semibold" title={doc.filename}>
          {doc.filename}
        </h2>
        <a
          href={fileHref}
          target="_blank"
          rel="noreferrer"
          className="inline-flex items-center gap-1 rounded-md px-2 py-1 text-xs text-accent hover:bg-accent-soft"
          title={doc.source_url ? "Open the original page" : "Open the original file"}
        >
          Original <ExternalLink size={12} />
        </a>
      </div>
      <div className="scroll-thin flex-1 overflow-y-auto px-4 py-3 text-sm leading-relaxed">
        {(!passageId || passageId.endsWith("-guide")) && <Guide key={doc.id} notebookId={notebookId} doc={doc} onAsk={onAsk} canEdit={canEdit} onChanged={onChanged} />}
        {error && <p className="text-bad">{error}</p>}
        {!passages && !error && (
          <p className="flex items-center gap-2 text-muted">
            <Loader2 size={14} className="animate-spin" /> Loading…
          </p>
        )}
        {passages?.map((p) => {
          const showSection = p.section && p.section !== lastSection;
          lastSection = p.section;
          const raw = p.section && p.text.startsWith(p.section) ? p.text.slice(p.section.length).trim() : p.text;
          const words = raw.split(/\s+/);
          const body = (showSection ? words : words.slice(overlap(prevWords, words))).join(" ");
          prevWords = words;
          const active = p.passage_id === passageId;
          return (
            <div key={p.passage_id}>
              {showSection && (
                <h3 className="mb-1 mt-4 text-xs font-semibold uppercase tracking-wide text-muted">{p.section}</h3>
              )}
              <div
                ref={active ? hit : undefined}
                className={`mb-2 rounded-lg px-2 py-1.5 ${active ? "bg-accent-soft ring-1 ring-accent" : ""}`}
              >
                {p.page != null && <span className="mr-2 text-[11px] text-muted">p. {p.page}</span>}
                {body}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

/** Number of leading words of `cur` that repeat the end of `prev` (chunks overlap by a few dozen words). */
function overlap(prev: string[], cur: string[]): number {
  for (let n = Math.min(60, prev.length, cur.length - 1); n >= 5; n--) {
    let same = true;
    for (let i = 0; i < n; i++) {
      if (prev[prev.length - n + i] !== cur[i]) {
        same = false;
        break;
      }
    }
    if (same) return n;
  }
  return 0;
}
