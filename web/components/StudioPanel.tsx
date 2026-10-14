"use client";

import { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { AlertTriangle, ArrowLeft, Loader2, Pin, PinOff, Plus, SlidersHorizontal, StickyNote, Trash2 } from "lucide-react";
import { api, Artifact, ArtifactKind, Citation, Note } from "@/lib/api";
import ArtifactView from "./ArtifactView";
import { loadSavedModel } from "./ModelPicker";
import { KIND_META, KIND_ORDER } from "./studioKinds";

function ago(iso: string) {
  const utc = /[zZ]|[+-]\d\d:?\d\d$/.test(iso) ? iso : iso + "Z"; // SQLite returns naive UTC times
  const s = (Date.now() - new Date(utc).getTime()) / 1000;
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.round(s / 60)} min ago`;
  if (s < 86400) return `${Math.round(s / 3600)} h ago`;
  return new Date(iso).toLocaleDateString();
}

export default function StudioPanel({
  notebookId,
  refreshKey,
  onCite,
  onAsk,
  documentIds,
  builtin,
  canEdit,
  hasSources,
}: {
  notebookId: string;
  refreshKey: number;
  onCite: (c: Citation) => void;
  onAsk?: (q: string) => void;
  documentIds: string[] | null;
  builtin: boolean;
  canEdit: boolean;
  hasSources: boolean;
}) {
  const [artifacts, setArtifacts] = useState<Artifact[]>([]);
  const [open, setOpen] = useState<string | null>(null);
  const [custom, setCustom] = useState(false);
  const [instructions, setInstructions] = useState("");
  const [studioError, setStudioError] = useState<string | null>(null);
  const loadStudio = () =>
    api
      .studio(notebookId)
      .then(setArtifacts)
      .catch(() => setArtifacts([]));
  const working = artifacts.some((a) => a.status === "pending" || a.status === "running");
  useEffect(() => {
    if (!builtin) loadStudio();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [notebookId, builtin]);
  useEffect(() => {
    if (!working) return;
    const t = setInterval(loadStudio, 3000);
    return () => clearInterval(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [working]);

  async function generate(kind: ArtifactKind) {
    setStudioError(null);
    try {
      const saved = loadSavedModel();
      const a = await api.createArtifact(notebookId, {
        kind,
        document_ids: documentIds || undefined,
        model: saved || undefined,
        instructions: instructions.trim() || undefined,
      });
      setArtifacts((xs) => [a, ...xs.filter((x) => x.id !== a.id)]);
      loadStudio();
    } catch (e) {
      setStudioError((e as Error).message);
    }
  }
  const canGenerate = canEdit && hasSources && (documentIds === null || documentIds.length > 0);
  const current = artifacts.find((a) => a.id === open);
  const [notes, setNotes] = useState<Note[] | null>(null);
  const [editing, setEditing] = useState<Note | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = () =>
    api
      .notes(notebookId)
      .then(setNotes)
      .catch((e) => setError(e.message));

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [notebookId, refreshKey]);

  async function addNote() {
    const n = await api.createNote(notebookId, { title: "New note", content: "" });
    await load();
    setEditing(n);
  }

  if (editing) {
    return (
      <NoteEditor
        notebookId={notebookId}
        note={editing}
        onCite={onCite}
        onClose={() => {
          setEditing(null);
          load();
        }}
      />
    );
  }

  return (
    <div className="flex h-full flex-col">
      <div className="border-b border-line px-4 py-3">
        <h2 className="font-semibold">Studio</h2>
      </div>
      <div className="scroll-thin flex-1 overflow-y-auto px-3 py-3">
        <div className="grid grid-cols-2 gap-2">
          {KIND_ORDER.map((k) => {
            const m = KIND_META[k];
            return (
              <button
                key={k}
                disabled={!canGenerate}
                onClick={() => generate(k)}
                title={
                  canGenerate
                    ? m.hint
                    : builtin
                      ? "The Studio works on your own notebooks"
                      : !canEdit
                        ? "You can view this notebook but not generate"
                        : "Add and tick sources first"
                }
                className="flex items-center gap-2 rounded-xl border border-line bg-panel-2 px-2.5 py-2 text-left text-xs hover:border-accent disabled:cursor-not-allowed disabled:opacity-50"
              >
                <span className={`flex h-6 w-6 shrink-0 items-center justify-center rounded-md ${m.tint}`}>
                  <m.icon size={14} />
                </span>
                {m.label}
              </button>
            );
          })}
        </div>
        {canEdit && (
          <div className="mt-2 px-1">
            <button
              onClick={() => setCustom((c) => !c)}
              className="inline-flex items-center gap-1 text-[11px] text-muted hover:text-ink"
            >
              <SlidersHorizontal size={11} /> {custom ? "Hide instructions" : "Customise (optional)"}
            </button>
            {custom && (
              <input
                value={instructions}
                onChange={(e) => setInstructions(e.target.value)}
                maxLength={500}
                placeholder="e.g. focus on the methods; for beginners"
                className="mt-1 w-full rounded-lg border border-line bg-panel-2 px-2 py-1.5 text-xs outline-none focus:border-accent"
              />
            )}
          </div>
        )}
        {builtin && <p className="mt-2 px-1 text-[11px] text-muted">The Studio works on your own notebooks.</p>}
        {studioError && <p className="mt-2 px-1 text-xs text-bad">{studioError}</p>}

        {artifacts.length > 0 && (
          <div className="mt-4 space-y-1.5">
            {artifacts.map((a) => {
              const m = KIND_META[a.kind];
              const busy = a.status === "pending" || a.status === "running";
              return (
                <button
                  key={a.id}
                  onClick={() => setOpen(a.id)}
                  className="flex w-full items-center gap-2 rounded-xl border border-line bg-panel px-2.5 py-2 text-left hover:border-accent"
                >
                  <span className={`flex h-7 w-7 shrink-0 items-center justify-center rounded-md ${m.tint}`}>
                    <m.icon size={15} />
                  </span>
                  <span className="min-w-0 flex-1">
                    <span className="block truncate text-sm font-medium">{busy ? `Generating ${m.label.toLowerCase()}…` : a.title || m.label}</span>
                    <span className="block truncate text-[11px] text-muted">
                      {a.status === "failed" ? "Failed · " : ""}
                      {(a.data.sources || []).length || a.document_ids.length || "all"} sources · {ago(a.created_at)}
                    </span>
                  </span>
                  {busy && <Loader2 size={14} className="shrink-0 animate-spin text-muted" />}
                  {a.status === "failed" && <AlertTriangle size={14} className="shrink-0 text-bad" />}
                </button>
              );
            })}
          </div>
        )}
        {current && (
          <ArtifactView
            notebookId={notebookId}
            artifact={current}
            onClose={() => setOpen(null)}
            onCite={onCite}
            onAsk={onAsk}
            onChanged={loadStudio}
            canEdit={canEdit}
          />
        )}

        <div className="mb-2 mt-5 flex items-center justify-between px-1">
          <h3 className="text-sm font-semibold">Notes</h3>
          <button onClick={addNote} className="inline-flex items-center gap-1 rounded-lg border border-line px-2 py-1 text-xs hover:bg-panel-2">
            <Plus size={13} /> Add note
          </button>
        </div>
        {error && <p className="px-1 text-xs text-bad">{error}</p>}
        {notes === null && !error && (
          <p className="flex items-center gap-2 px-1 text-sm text-muted">
            <Loader2 size={14} className="animate-spin" /> Loading…
          </p>
        )}
        {notes?.length === 0 && (
          <p className="px-1 text-sm text-muted">Save an answer with the bookmark button, or write your own notes.</p>
        )}
        <div className="space-y-2">
          {notes?.map((n) => (
            <div
              key={n.id}
              onClick={() => setEditing(n)}
              className="group cursor-pointer rounded-xl border border-line bg-panel px-3 py-2.5 hover:border-accent"
            >
              <div className="flex items-start gap-2">
                <StickyNote size={15} className="mt-0.5 shrink-0 text-muted" />
                <div className="min-w-0 flex-1">
                  <div className="truncate text-sm font-medium">{n.title}</div>
                  <div className="line-clamp-2 text-xs text-muted">{n.content || "Empty note"}</div>
                </div>
                <button
                  onClick={async (e) => {
                    e.stopPropagation();
                    await api.updateNote(notebookId, n.id, { pinned: !n.pinned });
                    load();
                  }}
                  title={n.pinned ? "Unpin" : "Pin"}
                  className={`shrink-0 rounded p-1 ${n.pinned ? "text-accent" : "text-muted opacity-0 group-hover:opacity-100"}`}
                >
                  {n.pinned ? <Pin size={14} /> : <PinOff size={14} />}
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function NoteEditor({
  notebookId,
  note,
  onClose,
  onCite,
}: {
  notebookId: string;
  note: Note;
  onClose: () => void;
  onCite: (c: Citation) => void;
}) {
  const [title, setTitle] = useState(note.title);
  const [content, setContent] = useState(note.content);
  const [preview, setPreview] = useState(note.kind === "answer");
  const [saving, setSaving] = useState(false);

  async function save() {
    setSaving(true);
    await api.updateNote(notebookId, note.id, { title: title.trim() || "Untitled note", content }).catch(() => undefined);
    setSaving(false);
    onClose();
  }

  async function remove() {
    if (!window.confirm("Delete this note?")) return;
    await api.deleteNote(notebookId, note.id).catch(() => undefined);
    onClose();
  }

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center gap-2 border-b border-line px-3 py-3">
        <button onClick={save} className="rounded-md p-1 hover:bg-panel-2" title="Save and close">
          {saving ? <Loader2 size={18} className="animate-spin" /> : <ArrowLeft size={18} />}
        </button>
        <span className="text-sm text-muted">Studio › Note</span>
        <button onClick={remove} className="ml-auto rounded-md p-1 text-muted hover:text-bad" title="Delete note">
          <Trash2 size={16} />
        </button>
      </div>
      <div className="flex flex-1 flex-col gap-2 overflow-hidden px-4 py-3">
        <input
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          className="bg-transparent text-lg font-semibold outline-none"
        />
        <div className="flex gap-3 text-xs">
          <button onClick={() => setPreview(false)} className={preview ? "text-muted" : "font-semibold text-accent"}>
            Edit
          </button>
          <button onClick={() => setPreview(true)} className={preview ? "font-semibold text-accent" : "text-muted"}>
            Preview
          </button>
        </div>
        {preview ? (
          <div className="prose-answer scroll-thin flex-1 overflow-y-auto text-sm">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{content || "_Empty note_"}</ReactMarkdown>
            {note.citations?.length > 0 && (
              <div className="mt-3 flex flex-wrap gap-1.5">
                {note.citations.map((c) => (
                  <button
                    key={c.index}
                    onClick={() => onCite(c)}
                    className="rounded-full border border-line px-2 py-0.5 text-xs hover:border-accent"
                    title={c.snippet}
                  >
                    [{c.index}] {c.title}
                    {c.page != null ? ` p. ${c.page}` : ""}
                  </button>
                ))}
              </div>
            )}
          </div>
        ) : (
          <textarea
            autoFocus
            value={content}
            onChange={(e) => setContent(e.target.value)}
            placeholder="Write a note (Markdown supported)…"
            className="scroll-thin flex-1 resize-none rounded-lg border border-line bg-panel-2 p-3 text-sm outline-none focus:border-accent"
          />
        )}
      </div>
    </div>
  );
}
