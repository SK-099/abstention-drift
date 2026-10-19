"use client";

import { useEffect, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  AlertTriangle,
  Check,
  ChevronLeft,
  ChevronRight,
  Copy,
  Download,
  Loader2,
  RotateCcw,
  Shuffle,
  Trash2,
  Volume2,
  X,
} from "lucide-react";
import { api, Artifact, Citation, MindNode } from "@/lib/api";
import { KIND_META } from "./studioKinds";

type Src = NonNullable<Artifact["data"]["sources"]>[number];

function citationFor(s: Src): Citation {
  // opens the source viewer on the source guide
  return { index: s.index, document_id: `${s.document_id}-guide`, title: s.title, snippet: "", score: null, source_document: s.document_id };
}

/** Markdown in which "[2]" becomes a clickable chip that opens source 2. */
function CitedMarkdown({ md, sources, onCite }: { md: string; sources: Src[]; onCite: (c: Citation) => void }) {
  const linked = md.replace(/\[(\d{1,2})\](?!\()/g, (m, n) => (sources.some((s) => s.index === Number(n)) ? `[${n}](#cite-${n})` : m));
  return (
    <div className="prose-answer text-[15px] leading-relaxed">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a: ({ href, children }) => {
            const m = /^#cite-(\d+)$/.exec(href || "");
            if (!m) return <a href={href} target="_blank" rel="noreferrer">{children}</a>;
            const s = sources.find((x) => x.index === Number(m[1]));
            return (
              <button
                onClick={() => s && onCite(citationFor(s))}
                title={s?.title}
                className="mx-0.5 inline-flex h-5 min-w-5 items-center justify-center rounded-full bg-accent-soft px-1 align-text-top text-[11px] font-semibold text-accent hover:bg-accent hover:text-white"
              >
                {m[1]}
              </button>
            );
          },
        }}
      >
        {linked}
      </ReactMarkdown>
    </div>
  );
}

function SourceChips({ nums, sources, onCite }: { nums: number[]; sources: Src[]; onCite: (c: Citation) => void }) {
  return (
    <>
      {nums.map((n) => {
        const s = sources.find((x) => x.index === n);
        return s ? (
          <button
            key={n}
            onClick={() => onCite(citationFor(s))}
            title={s.title}
            className="ml-1 inline-flex h-5 min-w-5 items-center justify-center rounded-full bg-accent-soft px-1 text-[11px] font-semibold text-accent"
          >
            {n}
          </button>
        ) : null;
      })}
    </>
  );
}

// ------------------------------------------------------------------------------------------- mind map
type Laid = { node: MindNode; id: string; depth: number; y: number; x: number; hidden: number };

function layout(root: MindNode, collapsed: Set<string>) {
  const out: Laid[] = [];
  const edges: [Laid, Laid][] = [];
  let leaf = 0;
  const walk = (n: MindNode, id: string, depth: number): Laid => {
    const kids = collapsed.has(id) ? [] : n.children;
    const me: Laid = { node: n, id, depth, y: 0, x: depth * 230, hidden: collapsed.has(id) ? n.children.length : 0 };
    out.push(me);
    if (!kids.length) me.y = leaf++ * 46;
    else {
      const placed = kids.map((c, i) => walk(c, `${id}.${i}`, depth + 1));
      placed.forEach((c) => edges.push([me, c]));
      me.y = (placed[0].y + placed[placed.length - 1].y) / 2;
    }
    return me;
  };
  walk(root, "0", 0);
  return { nodes: out, edges, height: Math.max(1, leaf) * 46, width: (Math.max(...out.map((n) => n.depth)) + 1) * 230 };
}

function MindMap({ tree, onAsk }: { tree: MindNode; onAsk?: (q: string) => void }) {
  const [collapsed, setCollapsed] = useState<Set<string>>(() => {
    // start with the second level collapsed when the map is big
    const big = tree.children.reduce((a, c) => a + c.children.length, 0) > 14;
    return new Set(big ? tree.children.map((_, i) => `0.${i}`) : []);
  });
  const { nodes, edges, height, width } = useMemo(() => layout(tree, collapsed), [tree, collapsed]);
  const toggle = (id: string) =>
    setCollapsed((c) => {
      const n = new Set(c);
      if (n.has(id)) n.delete(id);
      else n.add(id);
      return n;
    });
  const W = 200;
  return (
    <div className="scroll-thin overflow-auto rounded-xl border border-line bg-panel-2 p-4">
      <svg width={width + 20} height={height + 20} className="block">
        <g transform="translate(10,14)">
          {edges.map(([a, b]) => (
            <path
              key={`${a.id}-${b.id}`}
              d={`M${a.x + W},${a.y + 15} C${a.x + W + 18},${a.y + 15} ${b.x - 18},${b.y + 15} ${b.x},${b.y + 15}`}
              fill="none"
              className="stroke-line"
              strokeWidth={1.5}
            />
          ))}
          {nodes.map((n) => {
            const hasKids = n.node.children.length > 0;
            return (
              <g key={n.id} transform={`translate(${n.x},${n.y})`}>
                <foreignObject width={W} height={32}>
                  <button
                    onClick={() => onAsk?.(`Explain "${n.node.label}" in the context of ${tree.label}`)}
                    title={onAsk ? "Ask about this in the chat" : n.node.label}
                    className={`h-[30px] w-full truncate rounded-lg border px-2.5 text-left text-xs ${
                      n.depth === 0
                        ? "border-accent bg-accent text-white font-semibold"
                        : n.depth === 1
                          ? "border-accent/40 bg-accent-soft font-medium"
                          : "border-line bg-panel"
                    }`}
                  >
                    {n.node.label}
                  </button>
                </foreignObject>
                {hasKids && n.depth > 0 && (
                  <g transform={`translate(${W + 2},6)`} onClick={() => toggle(n.id)} className="cursor-pointer">
                    <circle cx={9} cy={9} r={9} className="fill-panel stroke-line" />
                    <text x={9} y={13} textAnchor="middle" className="fill-muted text-[10px] font-semibold">
                      {n.hidden ? `+${n.hidden}` : "‹"}
                    </text>
                  </g>
                )}
              </g>
            );
          })}
        </g>
      </svg>
      <p className="mt-2 text-[11px] text-muted">Click a topic to ask about it in the chat; use the round buttons to fold branches.</p>
    </div>
  );
}

// ------------------------------------------------------------------------------------------- flashcards
function Flashcards({ cards, sources, onCite }: { cards: NonNullable<Artifact["data"]["cards"]>; sources: Src[]; onCite: (c: Citation) => void }) {
  const [order, setOrder] = useState(() => cards.map((_, i) => i));
  const [i, setI] = useState(0);
  const [flipped, setFlipped] = useState(false);
  const c = cards[order[i]];
  const go = (d: number) => {
    setFlipped(false);
    setI((x) => (x + d + cards.length) % cards.length);
  };
  return (
    <div className="flex flex-col items-center gap-4">
      <div
        role="button"
        tabIndex={0}
        onClick={() => setFlipped((f) => !f)}
        onKeyDown={(e) => (e.key === " " || e.key === "Enter") && setFlipped((f) => !f)}
        className={`flex min-h-56 cursor-pointer w-full max-w-xl flex-col items-center justify-center rounded-2xl border-2 px-8 py-10 text-center shadow-sm transition ${
          flipped ? "border-accent bg-accent-soft" : "border-line bg-panel"
        }`}
      >
        <span className="mb-3 text-[11px] font-semibold uppercase tracking-wide text-muted">{flipped ? "Answer" : "Question"}</span>
        <span className="text-xl font-medium">{flipped ? c.back : c.front}</span>
        {flipped && c.sources.length > 0 && (
          <span className="mt-3" onClick={(e) => e.stopPropagation()}>
            <SourceChips nums={c.sources} sources={sources} onCite={onCite} />
          </span>
        )}
        {!flipped && <span className="mt-4 text-xs text-muted">Click to see the answer</span>}
      </div>
      <div className="flex items-center gap-3">
        <button onClick={() => go(-1)} className="rounded-full border border-line p-2 hover:bg-panel-2" title="Previous">
          <ChevronLeft size={18} />
        </button>
        <span className="w-16 text-center text-sm text-muted">
          {i + 1} / {cards.length}
        </span>
        <button onClick={() => go(1)} className="rounded-full border border-line p-2 hover:bg-panel-2" title="Next">
          <ChevronRight size={18} />
        </button>
        <button
          onClick={() => {
            setOrder((o) => [...o].sort(() => Math.random() - 0.5));
            setI(0);
            setFlipped(false);
          }}
          className="ml-2 inline-flex items-center gap-1 rounded-lg border border-line px-2 py-1.5 text-xs hover:bg-panel-2"
        >
          <Shuffle size={13} /> Shuffle
        </button>
      </div>
    </div>
  );
}

// ------------------------------------------------------------------------------------------- quiz
function Quiz({ questions, sources, onCite }: { questions: NonNullable<Artifact["data"]["questions"]>; sources: Src[]; onCite: (c: Citation) => void }) {
  const [picked, setPicked] = useState<(number | null)[]>(() => questions.map(() => null));
  const done = picked.filter((p) => p !== null).length;
  const score = picked.filter((p, i) => p === questions[i].answer).length;
  return (
    <div className="space-y-5">
      {questions.map((q, i) => (
        <div key={i} className="rounded-xl border border-line bg-panel p-4">
          <p className="mb-3 font-medium">
            {i + 1}. {q.question}
          </p>
          <div className="grid gap-2">
            {q.options.map((o, j) => {
              const chosen = picked[i] === j;
              const show = picked[i] !== null;
              const right = j === q.answer;
              return (
                <button
                  key={j}
                  disabled={show}
                  onClick={() => setPicked((p) => p.map((x, k) => (k === i ? j : x)))}
                  className={`flex items-center gap-2 rounded-lg border px-3 py-2 text-left text-sm ${
                    show && right
                      ? "border-ok bg-ok/10"
                      : show && chosen
                        ? "border-bad bg-bad/10"
                        : "border-line hover:border-accent"
                  }`}
                >
                  <span className="w-5 font-semibold text-muted">{"ABCD"[j]}</span> {o}
                  {show && right && <Check size={15} className="ml-auto text-ok" />}
                  {show && chosen && !right && <X size={15} className="ml-auto text-bad" />}
                </button>
              );
            })}
          </div>
          {picked[i] !== null && q.why && (
            <p className="mt-2 text-sm text-muted">
              {q.why}
              <SourceChips nums={q.sources} sources={sources} onCite={onCite} />
            </p>
          )}
        </div>
      ))}
      <div className="flex items-center gap-3 text-sm">
        <span className="font-medium">
          Score: {score} / {questions.length}
          {done < questions.length && <span className="text-muted"> ({questions.length - done} left)</span>}
        </span>
        <button
          onClick={() => setPicked(questions.map(() => null))}
          className="inline-flex items-center gap-1 rounded-lg border border-line px-2 py-1 text-xs hover:bg-panel-2"
        >
          <RotateCcw size={12} /> Retry
        </button>
      </div>
    </div>
  );
}

// ------------------------------------------------------------------------------------------- timeline
function Timeline({ events, sources, onCite }: { events: NonNullable<Artifact["data"]["events"]>; sources: Src[]; onCite: (c: Citation) => void }) {
  if (!events.length) return <p className="text-muted">The sources contain no dated events.</p>;
  return (
    <ol className="relative ml-3 border-l-2 border-line">
      {events.map((e, i) => (
        <li key={i} className="mb-5 ml-5">
          <span className="absolute -left-[7px] mt-1.5 h-3 w-3 rounded-full border-2 border-panel bg-accent" />
          <div className="text-sm font-semibold text-accent">{e.date}</div>
          <div className="text-[15px]">
            {e.event}
            <SourceChips nums={e.sources} sources={sources} onCite={onCite} />
          </div>
        </li>
      ))}
    </ol>
  );
}

// ------------------------------------------------------------------------------------------- audio
function AudioOverview({ a, notebookId, onChanged, canEdit }: { a: Artifact; notebookId: string; onChanged: () => void; canEdit: boolean }) {
  const [url, setUrl] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    let u: string | null = null;
    if (a.data.audio)
      api
        .audioBlobUrl(notebookId, a.id)
        .then((x) => {
          u = x;
          setUrl(x);
        })
        .catch(() => setUrl(null));
    return () => {
      if (u) URL.revokeObjectURL(u);
    };
  }, [a.id, a.data.audio, notebookId]);
  return (
    <div className="space-y-4">
      {a.data.audio ? (
        <div className="rounded-xl border border-line bg-panel-2 p-4">
          {url ? <audio controls src={url} className="w-full" /> : <Loader2 className="animate-spin text-muted" size={18} />}
          {a.data.duration_s && (
            <p className="mt-2 text-xs text-muted">
              {Math.floor(a.data.duration_s / 60)}:{String(Math.round(a.data.duration_s % 60)).padStart(2, "0")} · voices by
              Kokoro
            </p>
          )}
        </div>
      ) : (
        <div className="flex items-start gap-2 rounded-xl border border-warn/40 bg-warn/10 p-3 text-sm">
          <Volume2 size={16} className="mt-0.5 shrink-0 text-warn" />
          <div>
            <p>{a.data.audio_error || "No audio yet."}</p>
            {canEdit && (
              <button
                disabled={busy}
                onClick={async () => {
                  setBusy(true);
                  await api.speak(notebookId, a.id).catch(() => undefined);
                  setBusy(false);
                  onChanged();
                }}
                className="mt-2 inline-flex items-center gap-1.5 rounded-lg bg-accent px-3 py-1.5 text-xs font-medium text-white disabled:opacity-50"
              >
                {busy && <Loader2 size={12} className="animate-spin" />} Speak again
              </button>
            )}
          </div>
        </div>
      )}
      <div className="space-y-2">
        <h3 className="text-sm font-semibold">Script</h3>
        {(a.data.turns || []).map((t, i) => (
          <p key={i} className="text-[15px] leading-relaxed">
            <span className={`mr-1.5 font-semibold ${t.speaker === "ALEX" ? "text-accent" : "text-ok"}`}>
              {t.speaker === "ALEX" ? "Alex" : "Sam"}:
            </span>
            {t.text}
          </p>
        ))}
      </div>
    </div>
  );
}

// ------------------------------------------------------------------------------------------- progress
function utc(iso: string) {
  return new Date(/[zZ]|[+-]\d\d:?\d\d$/.test(iso) ? iso : iso + "Z").getTime();
}

function Elapsed({ since }: { since: string }) {
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, []);
  const s = Math.max(0, Math.round((now - utc(since)) / 1000));
  return (
    <span className="tabular-nums">
      {Math.floor(s / 60)}:{String(s % 60).padStart(2, "0")}
    </span>
  );
}

function Progress({ a }: { a: Artifact }) {
  const model = a.model ? a.model.split(":").slice(1).join(":") : "the default model";
  const n = (a.data.sources || []).length;
  const speaking = a.data.stage === "speaking";
  const writing = a.data.stage === "writing" || a.data.stage === "retrying";
  const again = a.data.stage === "retrying" ? " (second try: the first was not in the right format)" : "";
  const [done, total] = a.data.progress || [0, 0];
  const steps =
    a.kind === "audio"
      ? [
          ["Reading your sources", !a.data.stage],
          [`Writing the script with ${model}${again}`, writing],
          ["Speaking it with Kokoro", speaking],
        ]
      : [
          ["Reading your sources", !a.data.stage],
          [`Writing with ${model}${again}`, writing],
        ];
  const current = steps.findIndex(([, on]) => on);
  return (
    <div className="space-y-4">
      <ol className="space-y-2">
        {steps.map(([label], i) => (
          <li key={String(label)} className="flex items-center gap-2 text-sm">
            {i < current ? (
              <Check size={16} className="text-ok" />
            ) : i === current ? (
              <Loader2 size={16} className="animate-spin text-accent" />
            ) : (
              <span className="ml-1 h-2 w-2 rounded-full bg-line" />
            )}
            <span className={i === current ? "font-medium" : "text-muted"}>{label}</span>
            {i === current && i === steps.length - 1 && speaking && total > 0 && (
              <span className="text-muted">
                · turn {done} of {total}
              </span>
            )}
          </li>
        ))}
      </ol>
      {speaking && total > 0 && (
        <div className="h-1.5 w-full max-w-md overflow-hidden rounded-full bg-panel-2">
          <div className="h-full bg-accent transition-all" style={{ width: `${Math.round((100 * done) / total)}%` }} />
        </div>
      )}
      <p className="text-xs text-muted">
        Running for <Elapsed since={a.created_at} />
        {n ? ` · ${n} source${n === 1 ? "" : "s"}` : ""}. On a 4 GB GPU this takes a few minutes; larger models
        (gemma3:4b, phi4-mini) and audio take longest. You can close this window; the result appears in the Studio.
      </p>
      {speaking && (a.data.turns || []).length > 0 && (
        <div className="space-y-2 border-t border-line pt-3">
          <h3 className="text-sm font-semibold">Script</h3>
          {(a.data.turns || []).map((t, i) => (
            <p key={i} className={`text-[15px] leading-relaxed ${i < done ? "" : "text-muted"}`}>
              <span className={`mr-1.5 font-semibold ${t.speaker === "ALEX" ? "text-accent" : "text-ok"}`}>
                {t.speaker === "ALEX" ? "Alex" : "Sam"}:
              </span>
              {t.text}
            </p>
          ))}
        </div>
      )}
    </div>
  );
}

// ------------------------------------------------------------------------------------------- the dialog
export default function ArtifactView({
  notebookId,
  artifact: a,
  onClose,
  onCite,
  onAsk,
  onChanged,
  canEdit,
}: {
  notebookId: string;
  artifact: Artifact;
  onClose: () => void;
  onCite: (c: Citation) => void;
  onAsk?: (q: string) => void;
  onChanged: () => void;
  canEdit: boolean;
}) {
  const meta = KIND_META[a.kind];
  const sources = a.data.sources || [];
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    const esc = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", esc);
    return () => window.removeEventListener("keydown", esc);
  }, [onClose]);

  const download = () => {
    const blob = new Blob([`# ${a.title}\n\n${a.content}\n`], { type: "text/markdown" });
    const u = URL.createObjectURL(blob);
    const el = document.createElement("a");
    el.href = u;
    el.download = `${a.title.replace(/[^\w\- ]+/g, "").trim() || a.kind}.md`;
    el.click();
    URL.revokeObjectURL(u);
  };

  const cite = (c: Citation) => {
    onCite(c);
    onClose();
  };
  const ask = onAsk
    ? (q: string) => {
        onAsk(q);
        onClose();
      }
    : undefined;

  let body: React.ReactNode;
  if (a.status === "pending" || a.status === "running") body = <Progress a={a} />;
  else if (a.status === "failed")
    body = (
      <p className="flex items-start gap-2 text-bad">
        <AlertTriangle size={16} className="mt-0.5 shrink-0" /> {a.error || "Generation failed."}
      </p>
    );
  else if (a.kind === "mindmap" && a.data.tree) body = <MindMap tree={a.data.tree} onAsk={ask} />;
  else if (a.kind === "flashcards" && a.data.cards?.length) body = <Flashcards cards={a.data.cards} sources={sources} onCite={cite} />;
  else if (a.kind === "quiz" && a.data.questions?.length) body = <Quiz questions={a.data.questions} sources={sources} onCite={cite} />;
  else if (a.kind === "timeline") body = <Timeline events={a.data.events || []} sources={sources} onCite={cite} />;
  else if (a.kind === "audio") body = <AudioOverview a={a} notebookId={notebookId} onChanged={onChanged} canEdit={canEdit} />;
  else body = <CitedMarkdown md={a.content} sources={sources} onCite={cite} />;

  return (
    <div className="fixed inset-0 z-50 flex items-stretch justify-center bg-black/40 p-0 sm:p-6" onMouseDown={onClose}>
      <div
        role="dialog"
        aria-label={a.title}
        onMouseDown={(e) => e.stopPropagation()}
        className="flex w-full max-w-4xl flex-col overflow-hidden bg-panel shadow-2xl sm:rounded-2xl sm:border sm:border-line"
      >
        <div className="flex items-center gap-3 border-b border-line px-5 py-3">
          <div className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-lg ${meta.tint}`}>
            <meta.icon size={16} />
          </div>
          <div className="min-w-0 flex-1">
            <h2 className="truncate font-semibold">{a.title || meta.label}</h2>
            <p className="truncate text-xs text-muted">
              {meta.label} · {sources.length} source{sources.length === 1 ? "" : "s"}
              {a.model ? ` · ${a.model.split(":").slice(1).join(":")}` : ""} ·{" "}
              {new Date(/[zZ]|[+-]\d\d:?\d\d$/.test(a.created_at) ? a.created_at : a.created_at + "Z").toLocaleString()}
            </p>
          </div>
          {a.status === "ready" && a.kind !== "audio" && (
            <>
              <button
                onClick={() => {
                  navigator.clipboard.writeText(a.content).then(() => setCopied(true));
                  setTimeout(() => setCopied(false), 1500);
                }}
                className="rounded-md p-1.5 text-muted hover:bg-panel-2"
                title="Copy as Markdown"
              >
                {copied ? <Check size={16} /> : <Copy size={16} />}
              </button>
              <button onClick={download} className="rounded-md p-1.5 text-muted hover:bg-panel-2" title="Download as Markdown">
                <Download size={16} />
              </button>
            </>
          )}
          {canEdit && (
            <button
              onClick={async () => {
                if (!window.confirm(`Delete "${a.title || meta.label}"?`)) return;
                await api.deleteArtifact(notebookId, a.id).catch(() => undefined);
                onChanged();
                onClose();
              }}
              className="rounded-md p-1.5 text-muted hover:text-bad"
              title="Delete"
            >
              <Trash2 size={16} />
            </button>
          )}
          <button onClick={onClose} className="rounded-md p-1.5 text-muted hover:bg-panel-2" title="Close">
            <X size={18} />
          </button>
        </div>
        <div className="scroll-thin flex-1 overflow-y-auto px-6 py-5">{body}</div>
        {a.status === "ready" && sources.length > 0 && a.kind !== "audio" && (
          <div className="flex flex-wrap items-center gap-1.5 border-t border-line px-5 py-2.5 text-xs text-muted">
            Sources:
            {sources.map((s) => (
              <button
                key={s.index}
                onClick={() => cite(citationFor(s))}
                className="rounded-full border border-line px-2 py-0.5 hover:border-accent"
              >
                [{s.index}] {s.title}
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
