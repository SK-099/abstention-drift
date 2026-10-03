"use client";

import { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  ArrowLeft,
  AudioLines,
  BookOpenCheck,
  FileText,
  HelpCircle,
  Layers,
  ListChecks,
  Loader2,
  Network,
  Pin,
  PinOff,
  Plus,
  StickyNote,
  Timer,
  Trash2,
} from "lucide-react";
import { api, Citation, Note } from "@/lib/api";

const STUDIO = [
  { icon: AudioLines, label: "Audio overview" },
  { icon: FileText, label: "Briefing doc" },
  { icon: BookOpenCheck, label: "Study guide" },
  { icon: HelpCircle, label: "FAQ" },
  { icon: Timer, label: "Timeline" },
  { icon: Network, label: "Mind map" },
  { icon: Layers, label: "Flashcards" },
  { icon: ListChecks, label: "Quiz" },
];

export default function StudioPanel({
  notebookId,
  refreshKey,
  onCite,
}: {
  notebookId: string;
  refreshKey: number;
  onCite: (c: Citation) => void;
}) {
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
          {STUDIO.map(({ icon: Icon, label }) => (
            <div
              key={label}
              title="Coming in the next phase"
              className="flex items-center gap-2 rounded-xl border border-line bg-panel-2 px-2.5 py-2 text-xs text-muted opacity-70"
            >
              <Icon size={15} /> {label}
            </div>
          ))}
        </div>
        <p className="mt-2 px-1 text-[11px] text-muted">Audio overviews, guides, mind maps and quizzes arrive in the next phases.</p>

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
