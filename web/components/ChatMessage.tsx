"use client";

import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  BookmarkPlus,
  Check,
  ChevronDown,
  ChevronRight,
  Copy,
  HelpCircle,
  Loader2,
  ShieldAlert,
  ShieldX,
  ThumbsDown,
  ThumbsUp,
} from "lucide-react";
import { api, Citation, Final, GuardEvent, Step } from "@/lib/api";

export type UiMessage = {
  key: string;
  role: "user" | "assistant";
  content: string;
  steps?: Step[];
  final?: Partial<Final> & { citations?: Citation[] };
  guardrails?: GuardEvent[];
  streaming?: boolean;
  error?: string;
  model?: string;
  messageId?: string;
};

const DECISION: Record<string, { label: string; cls: string; icon: React.ReactNode; hint: string }> = {
  abstain: {
    label: "Not in your sources",
    cls: "bg-warn-soft text-warn",
    icon: <ShieldAlert size={14} />,
    hint: "The assistant searched your sources (and searched again) but found no support, so it did not answer.",
  },
  clarify: {
    label: "Needs clarification",
    cls: "bg-accent-soft text-accent",
    icon: <HelpCircle size={14} />,
    hint: "The question was ambiguous, so the assistant asked what you meant.",
  },
  blocked: {
    label: "Blocked by guardrails",
    cls: "bg-bad-soft text-bad",
    icon: <ShieldX size={14} />,
    hint: "The question looked like a prompt injection or abusive content.",
  },
};

/** Evidence strength = 1 - P(unanswerable) from the answerability gate. */
function EvidenceMeter({ p }: { p: number }) {
  const strength = Math.max(0, Math.min(1, 1 - p));
  const color = strength >= 0.6 ? "bg-ok" : strength >= 0.4 ? "bg-warn" : "bg-bad";
  const label = strength >= 0.6 ? "Strong" : strength >= 0.4 ? "Moderate" : "Weak";
  return (
    <span className="inline-flex items-center gap-1.5 text-xs text-muted" title={`P(unanswerable) = ${p.toFixed(3)}`}>
      Evidence
      <span className="relative h-1.5 w-16 overflow-hidden rounded-full bg-line">
        <span className={`absolute inset-y-0 left-0 ${color}`} style={{ width: `${Math.round(strength * 100)}%` }} />
      </span>
      {label}
    </span>
  );
}

function modelLabel(id?: string) {
  if (!id) return "";
  const [, ...rest] = id.split(":");
  return rest.join(":");
}

export default function ChatMessage({
  m,
  notebookId,
  onCite,
  onSaved,
}: {
  m: UiMessage;
  notebookId: string;
  onCite: (c: Citation) => void;
  onSaved: () => void;
}) {
  const [showSteps, setShowSteps] = useState(false);
  const [copied, setCopied] = useState(false);
  const [rated, setRated] = useState<0 | 1 | -1>(0);
  const [askReason, setAskReason] = useState(false);
  const [saved, setSaved] = useState(false);

  if (m.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] whitespace-pre-wrap rounded-2xl rounded-br-md bg-accent-soft px-4 py-2.5 text-[15px]">
          {m.content}
        </div>
      </div>
    );
  }

  const f = m.final;
  const decision = f?.decision;
  const badge = decision && decision !== "answer" ? DECISION[decision] : null;
  const citations = f?.citations || [];
  const steps = m.steps || [];
  // "[1]" markers in the answer become clickable citation chips
  const withChips = m.content.replace(/\[(\d{1,2})\]/g, (all, n) =>
    citations.some((c) => c.index === Number(n)) ? `[${n}](#cite-${n})` : all,
  );

  async function rate(r: 1 | -1, reason?: string) {
    if (!m.messageId) return;
    setRated(r);
    setAskReason(false);
    await api.feedback(m.messageId, r, reason).catch(() => setRated(0));
  }

  async function save() {
    await api.createNote(notebookId, {
      title: (f?.query || m.content).slice(0, 80),
      content: m.content,
      kind: "answer",
      source_message_id: m.messageId || null,
      citations,
    });
    setSaved(true);
    onSaved();
  }

  return (
    <div className="group">
      {badge && (
        <div className={`mb-2 inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium ${badge.cls}`} title={badge.hint}>
          {badge.icon} {badge.label}
        </div>
      )}
      {m.streaming && !m.content && (
        <div className="flex items-center gap-2 text-sm text-muted">
          <Loader2 size={15} className="animate-spin" />
          {steps.length ? steps[steps.length - 1].label : "Thinking"}…
        </div>
      )}
      {m.content && (
        <div className="prose-answer text-[15px] leading-relaxed">
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={{
              a: ({ href, children }) => {
                const n = href?.startsWith("#cite-") ? Number(href.slice(6)) : null;
                const c = n ? citations.find((x) => x.index === n) : null;
                if (c)
                  return (
                    <button
                      onClick={() => onCite(c)}
                      className="mx-0.5 inline-flex h-5 min-w-5 items-center justify-center rounded-full bg-accent-soft px-1 align-text-top text-[11px] font-semibold text-accent hover:bg-accent hover:text-white"
                    >
                      {children}
                    </button>
                  );
                return (
                  <a href={href} target="_blank" rel="noreferrer" className="text-accent underline">
                    {children}
                  </a>
                );
              },
            }}
          >
            {withChips}
          </ReactMarkdown>
        </div>
      )}
      {m.error && <div className="mt-2 rounded-lg bg-bad-soft px-3 py-2 text-sm text-bad">{m.error}</div>}

      {citations.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-2">
          {citations.map((c) => (
            <button
              key={c.index}
              onClick={() => onCite(c)}
              className="flex max-w-full items-center gap-2 rounded-lg border border-line bg-panel-2 px-2.5 py-1.5 text-left text-xs hover:border-accent"
              title={c.snippet}
            >
              <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-accent-soft font-semibold text-accent">
                {c.index}
              </span>
              <span className="min-w-0">
                <span className="block max-w-[14rem] truncate font-medium">{c.title || c.document_id}</span>
                <span className="block max-w-[14rem] truncate text-muted">
                  {[c.page != null ? `p. ${c.page}` : null, c.section].filter(Boolean).join(" · ") || c.snippet.slice(0, 60)}
                </span>
              </span>
            </button>
          ))}
        </div>
      )}

      {(steps.length > 0 || f) && !m.streaming && (
        <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-2">
          {f?.p_unanswerable != null && <EvidenceMeter p={f.p_unanswerable} />}
          {steps.length > 0 && (
            <button onClick={() => setShowSteps((s) => !s)} className="inline-flex items-center gap-1 text-xs text-muted hover:text-ink">
              {showSteps ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
              How I got this ({steps.length} step{steps.length === 1 ? "" : "s"}{f?.attempts ? `, searched again ${f.attempts}×` : ""})
            </button>
          )}
          {m.model && <span className="text-xs text-muted">{modelLabel(m.model)}</span>}
          {f?.latency_ms ? <span className="text-xs text-muted">{(f.latency_ms / 1000).toFixed(1)} s</span> : null}
          <span className="ml-auto flex items-center gap-1 opacity-70 group-hover:opacity-100">
            <IconBtn title={saved ? "Saved to notes" : "Save to notes"} onClick={save} disabled={saved || !m.content}>
              {saved ? <Check size={15} /> : <BookmarkPlus size={15} />}
            </IconBtn>
            <IconBtn
              title="Copy"
              onClick={() => {
                navigator.clipboard?.writeText(m.content);
                setCopied(true);
                setTimeout(() => setCopied(false), 1200);
              }}
            >
              {copied ? <Check size={15} /> : <Copy size={15} />}
            </IconBtn>
            <IconBtn title="Good answer" onClick={() => rate(1)} active={rated === 1} disabled={!m.messageId}>
              <ThumbsUp size={15} />
            </IconBtn>
            <IconBtn title="Bad answer" onClick={() => setAskReason((a) => !a)} active={rated === -1} disabled={!m.messageId}>
              <ThumbsDown size={15} />
            </IconBtn>
          </span>
        </div>
      )}
      {askReason && (
        <div className="mt-2 flex flex-wrap gap-2 text-xs">
          {[
            ["wrong_refusal", "Should have answered"],
            ["wrong_answer", "Wrong answer"],
            ["bad_citation", "Bad citation"],
            ["unclear", "Unclear"],
            ["other", "Other"],
          ].map(([k, label]) => (
            <button key={k} onClick={() => rate(-1, k)} className="rounded-full border border-line px-2.5 py-1 hover:border-accent">
              {label}
            </button>
          ))}
        </div>
      )}
      {showSteps && (
        <ol className="mt-2 space-y-1 border-l-2 border-line pl-3 text-xs text-muted">
          {steps.map((s, i) => (
            <li key={i}>
              <span className="font-medium text-ink">{s.label}</span>
              {s.detail ? ` — ${s.detail.replace(/^[a-z_]+(\[\d+\])?:\s*/, "")}` : ""}
            </li>
          ))}
          {(m.guardrails || []).map((g, i) => (
            <li key={`g${i}`} className="text-warn">
              Guardrail ({g.stage}): {g.rule.replace("_", " ")} → {g.action}
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}

function IconBtn({
  children,
  title,
  onClick,
  active,
  disabled,
}: {
  children: React.ReactNode;
  title: string;
  onClick: () => void;
  active?: boolean;
  disabled?: boolean;
}) {
  return (
    <button
      title={title}
      onClick={onClick}
      disabled={disabled}
      className={`rounded-md p-1.5 hover:bg-panel-2 disabled:opacity-40 ${active ? "text-accent" : "text-muted"}`}
    >
      {children}
    </button>
  );
}
