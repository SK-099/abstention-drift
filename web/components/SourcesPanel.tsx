"use client";

import { useState } from "react";
import {
  AlertTriangle,
  CheckSquare,
  ClipboardType,
  FileText,
  Globe,
  Loader2,
  MonitorPlay,
  Plus,
  Sparkles,
  Square,
  Trash2,
  Upload,
} from "lucide-react";
import { api, SourceDoc } from "@/lib/api";
import AddSourceDialog from "./AddSourceDialog";

export function SourceIcon({ d, size = 16 }: { d: SourceDoc; size?: number }) {
  const cls = "shrink-0 text-muted";
  if (d.source_type === "url") return <Globe size={size} className={cls} />;
  if (d.source_type === "youtube") return <MonitorPlay size={size} className="shrink-0 text-red-500" />;
  if (d.source_type === "text") return <ClipboardType size={size} className={cls} />;
  return <FileText size={size} className={cls} />;
}

export default function SourcesPanel({
  notebookId,
  builtin,
  sources,
  selected,
  setSelected,
  onChanged,
  onOpen,
  canEdit,
}: {
  notebookId: string;
  builtin: boolean;
  sources: SourceDoc[];
  selected: Set<string>;
  setSelected: (s: Set<string>) => void;
  onChanged: () => void;
  onOpen: (doc: SourceDoc) => void;
  canEdit: boolean;
}) {
  const [adding, setAdding] = useState(false);
  const [busy, setBusy] = useState(0);
  const [drag, setDrag] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function upload(files: FileList | File[]) {
    setError(null);
    const list = Array.from(files);
    setBusy((b) => b + list.length);
    await Promise.all(
      list.map((f) =>
        api
          .upload(notebookId, f)
          .catch((e) => setError(`${f.name}: ${e.message}`))
          .finally(() => {
            setBusy((b) => b - 1);
            onChanged();
          }),
      ),
    );
  }

  async function remove(d: SourceDoc) {
    if (!window.confirm(`Remove "${d.filename}" from this notebook?`)) return;
    await api.deleteSource(notebookId, d.id).catch((e) => setError(e.message));
    const s = new Set(selected);
    s.delete(d.id);
    setSelected(s);
    onChanged();
  }

  const indexed = sources.filter((s) => s.status === "indexed");
  const allOn = indexed.length > 0 && indexed.every((s) => selected.has(s.id));
  const toggle = (id: string) => {
    const s = new Set(selected);
    if (s.has(id)) s.delete(id);
    else s.add(id);
    setSelected(s);
  };

  return (
    <div
      className={`flex h-full flex-col ${drag ? "ring-2 ring-accent ring-inset" : ""}`}
      onDragOver={(e) => {
        if (!canEdit) return;
        e.preventDefault();
        setDrag(true);
      }}
      onDragLeave={() => setDrag(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDrag(false);
        if (canEdit && e.dataTransfer.files.length) upload(e.dataTransfer.files);
      }}
    >
      <div className="flex items-center justify-between border-b border-line px-4 py-3">
        <h2 className="font-semibold">Sources</h2>
        {canEdit && (
          <button
            onClick={() => setAdding(true)}
            className="inline-flex items-center gap-1 rounded-lg border border-line px-2.5 py-1 text-sm hover:bg-panel-2"
          >
            <Plus size={15} /> Add
          </button>
        )}
      </div>
      {adding && (
        <AddSourceDialog
          notebookId={notebookId}
          onUpload={upload}
          onAdded={onChanged}
          onClose={() => setAdding(false)}
        />
      )}

      <div className="scroll-thin flex-1 overflow-y-auto px-2 py-2">
        {builtin && (
          <p className="px-2 py-2 text-sm text-muted">
            This built-in collection searches the whole MTRAG benchmark corpus. Create your own notebook to upload
            documents.
          </p>
        )}
        {!builtin && sources.length === 0 && busy === 0 && (
          <button
            onClick={() => canEdit && setAdding(true)}
            className="mx-2 mt-2 flex w-[calc(100%-1rem)] flex-col items-center gap-2 rounded-xl border border-dashed border-line px-4 py-8 text-center text-sm text-muted hover:bg-panel-2"
          >
            <Upload size={22} />
            Add files, web pages, YouTube videos or pasted text
            <span className="text-xs">or drop files here</span>
          </button>
        )}
        {indexed.length > 1 && (
          <button
            onClick={() => setSelected(allOn ? new Set() : new Set(indexed.map((s) => s.id)))}
            className="flex w-full items-center gap-2 rounded-lg px-2 py-1.5 text-sm text-muted hover:bg-panel-2"
          >
            {allOn ? <CheckSquare size={16} className="text-accent" /> : <Square size={16} />} Select all sources
          </button>
        )}
        {sources.map((d) => {
          const ready = d.status === "indexed";
          const bad = d.status === "rejected" || d.status === "failed";
          return (
            <div key={d.id} className="group flex items-center gap-2 rounded-lg px-2 py-1.5 hover:bg-panel-2">
              <button
                disabled={!ready}
                onClick={() => toggle(d.id)}
                className="shrink-0 disabled:opacity-40"
                title={ready ? "Use this source in answers" : d.status}
              >
                {ready && selected.has(d.id) ? <CheckSquare size={16} className="text-accent" /> : <Square size={16} />}
              </button>
              <button
                onClick={() => ready && onOpen(d)}
                className="flex min-w-0 flex-1 items-center gap-2 text-left text-sm"
                title={d.error || d.filename}
              >
                <SourceIcon d={d} />
                <span className="truncate">{d.filename}</span>
              </button>
              {!ready && !bad && <Loader2 size={14} className="shrink-0 animate-spin text-muted" />}
              {ready && d.guide_status === "pending" && (
                <span title="Writing the source guide…">
                  <Sparkles size={14} className="shrink-0 animate-pulse text-accent" />
                </span>
              )}
              {bad && <AlertTriangle size={14} className="shrink-0 text-bad" />}
              {canEdit && (
                <button
                  onClick={() => remove(d)}
                  className="shrink-0 rounded p-1 text-muted opacity-0 hover:text-bad group-hover:opacity-100"
                  title="Remove source"
                >
                  <Trash2 size={14} />
                </button>
              )}
            </div>
          );
        })}
        {busy > 0 && (
          <div className="flex items-center gap-2 px-2 py-2 text-sm text-muted">
            <Loader2 size={14} className="animate-spin" /> Uploading {busy} file{busy > 1 ? "s" : ""}…
          </div>
        )}
        {error && <p className="px-2 py-2 text-xs text-bad">{error}</p>}
      </div>
      {!builtin && sources.length > 0 && (
        <p className="border-t border-line px-4 py-2 text-xs text-muted">
          {selected.size} of {indexed.length} source{indexed.length === 1 ? "" : "s"} used for answers
        </p>
      )}
    </div>
  );
}
