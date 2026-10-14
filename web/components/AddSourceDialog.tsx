"use client";

import { useEffect, useRef, useState } from "react";
import { ClipboardType, Globe, Loader2, MonitorPlay, Upload, X } from "lucide-react";
import { api, NewSource } from "@/lib/api";

export const ACCEPT = ".pdf,.html,.htm,.docx,.md,.txt";

type Kind = "file" | "url" | "youtube" | "text";

const KINDS: { k: Kind; label: string; icon: typeof Globe; hint: string }[] = [
  { k: "file", label: "Upload", icon: Upload, hint: "PDF, Word, HTML, Markdown or text files" },
  { k: "url", label: "Website", icon: Globe, hint: "The text of a web page is fetched and indexed" },
  { k: "youtube", label: "YouTube", icon: MonitorPlay, hint: "The video's transcript (captions) becomes the source" },
  { k: "text", label: "Paste text", icon: ClipboardType, hint: "Notes, an e-mail, an article you copied" },
];

/** "Add source" dialog: upload files, a web page, a YouTube video or pasted text. */
export default function AddSourceDialog({
  notebookId,
  onUpload,
  onAdded,
  onClose,
}: {
  notebookId: string;
  onUpload: (files: FileList | File[]) => void;
  onAdded: () => void;
  onClose: () => void;
}) {
  const [kind, setKind] = useState<Kind>("file");
  const [url, setUrl] = useState("");
  const [title, setTitle] = useState("");
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const esc = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", esc);
    return () => window.removeEventListener("keydown", esc);
  }, [onClose]);

  async function submit() {
    setError(null);
    let body: NewSource;
    if (kind === "text") body = { type: "text", text, title: title.trim() || undefined };
    else if (kind === "url" || kind === "youtube") body = { type: kind, url: url.trim() };
    else return;
    setBusy(true);
    try {
      await api.addSource(notebookId, body);
      onAdded();
      onClose();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  const valid =
    kind === "text" ? text.trim().length >= 20 : kind === "file" ? false : /^https?:\/\/\S+\.\S+/.test(url.trim());

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4" onMouseDown={onClose}>
      <div
        role="dialog"
        aria-label="Add source"
        className="w-full max-w-lg rounded-2xl border border-line bg-panel shadow-2xl"
        onMouseDown={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between border-b border-line px-5 py-3">
          <h2 className="text-lg font-semibold">Add sources</h2>
          <button onClick={onClose} className="rounded-md p-1 text-muted hover:bg-panel-2" title="Close">
            <X size={18} />
          </button>
        </div>
        <div className="grid grid-cols-4 gap-2 px-5 pt-4">
          {KINDS.map(({ k, label, icon: Icon }) => (
            <button
              key={k}
              onClick={() => {
                setKind(k);
                setError(null);
              }}
              className={`flex flex-col items-center gap-1 rounded-xl border px-2 py-3 text-xs font-medium ${
                kind === k ? "border-accent bg-accent-soft text-accent" : "border-line hover:bg-panel-2"
              }`}
            >
              <Icon size={18} />
              {label}
            </button>
          ))}
        </div>
        <div className="px-5 py-4">
          <p className="mb-3 text-sm text-muted">{KINDS.find((x) => x.k === kind)?.hint}</p>
          {kind === "file" && (
            <button
              onClick={() => fileInput.current?.click()}
              className="flex w-full flex-col items-center gap-2 rounded-xl border border-dashed border-line px-4 py-8 text-sm text-muted hover:bg-panel-2"
            >
              <Upload size={22} /> Choose files
              <input
                ref={fileInput}
                type="file"
                accept={ACCEPT}
                multiple
                hidden
                onChange={(e) => {
                  if (e.target.files?.length) {
                    onUpload(e.target.files);
                    onClose();
                  }
                }}
              />
            </button>
          )}
          {(kind === "url" || kind === "youtube") && (
            <input
              autoFocus
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && valid && submit()}
              placeholder={kind === "youtube" ? "https://www.youtube.com/watch?v=…" : "https://…"}
              className="w-full rounded-lg border border-line bg-panel-2 px-3 py-2 text-sm outline-none focus:border-accent"
            />
          )}
          {kind === "text" && (
            <div className="flex flex-col gap-2">
              <input
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                placeholder="Title (optional)"
                className="w-full rounded-lg border border-line bg-panel-2 px-3 py-2 text-sm outline-none focus:border-accent"
              />
              <textarea
                autoFocus
                value={text}
                onChange={(e) => setText(e.target.value)}
                rows={9}
                placeholder="Paste text here"
                className="scroll-thin w-full resize-y rounded-lg border border-line bg-panel-2 px-3 py-2 text-sm outline-none focus:border-accent"
              />
            </div>
          )}
          {error && <p className="mt-2 text-sm text-bad">{error}</p>}
        </div>
        {kind !== "file" && (
          <div className="flex justify-end gap-2 border-t border-line px-5 py-3">
            <button onClick={onClose} className="rounded-lg px-3 py-1.5 text-sm hover:bg-panel-2">
              Cancel
            </button>
            <button
              onClick={submit}
              disabled={!valid || busy}
              className="inline-flex items-center gap-1.5 rounded-lg bg-accent px-3 py-1.5 text-sm font-medium text-white disabled:opacity-40"
            >
              {busy && <Loader2 size={14} className="animate-spin" />} Add source
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
