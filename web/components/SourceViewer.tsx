"use client";

import { useEffect, useRef, useState } from "react";
import { ArrowLeft, ExternalLink, Loader2 } from "lucide-react";
import { api, Passage, SourceDoc } from "@/lib/api";

/** Reads one source as its indexed passages; the cited passage is highlighted and scrolled into view. */
export default function SourceViewer({
  notebookId,
  doc,
  passageId,
  onBack,
}: {
  notebookId: string;
  doc: SourceDoc;
  passageId: string | null;
  onBack: () => void;
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
  const isPdf = doc.filename.toLowerCase().endsWith(".pdf");
  const fileHref = api.fileUrl(notebookId, doc.id) + (isPdf && cited?.page ? `#page=${cited.page}` : "");
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
          title="Open the original file"
        >
          Original <ExternalLink size={12} />
        </a>
      </div>
      <div className="scroll-thin flex-1 overflow-y-auto px-4 py-3 text-sm leading-relaxed">
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
