"use client";

import { useEffect, useRef, useState } from "react";
import { CheckSquare, ClipboardType, Globe, Loader2, MonitorPlay, Search, Square, Telescope, Upload, X } from "lucide-react";
import { api, DiscoverHit, NewSource } from "@/lib/api";
import { loadSavedModel } from "./ModelPicker";

export const ACCEPT = ".pdf,.html,.htm,.docx,.md,.txt";

type Kind = "file" | "url" | "youtube" | "text" | "discover";

const KINDS: { k: Kind; label: string; icon: typeof Globe; hint: string }[] = [
  { k: "file", label: "Upload", icon: Upload, hint: "PDF, Word, HTML, Markdown or text files" },
  { k: "url", label: "Website", icon: Globe, hint: "The text of a web page is fetched and indexed" },
  { k: "youtube", label: "YouTube", icon: MonitorPlay, hint: "The video's transcript (captions) becomes the source" },
  { k: "text", label: "Paste text", icon: ClipboardType, hint: "Notes, an e-mail, an article you copied" },
  { k: "discover", label: "Discover", icon: Telescope, hint: "Search the web for a topic and pick the pages to add" },
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
  const [query, setQuery] = useState("");
  const [hits, setHits] = useState<DiscoverHit[] | null>(null);
  const [picked, setPicked] = useState<Set<string>>(new Set());

  async function searchWeb() {
    setError(null);
    setBusy(true);
    try {
      const r = await api.discover(notebookId, query.trim());
      setHits(r);
      setPicked(new Set(r.slice(0, 3).map((h) => h.url)));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function addPicked() {
    setError(null);
    setBusy(true);
    const chosen = (hits || []).filter((h) => picked.has(h.url));
    const failed: string[] = [];
    await Promise.all(
      chosen.map((h) =>
        api.addSource(notebookId, { type: h.kind, url: h.url, title: h.title, model: loadSavedModel() || undefined }).catch((e) => failed.push(`${h.title}: ${e.message}`)),
      ),
    );
    setBusy(false);
    onAdded();
    if (failed.length) setError(failed.join("\n"));
    else onClose();
  }

  useEffect(() => {
    const esc = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", esc);
    return () => window.removeEventListener("keydown", esc);
  }, [onClose]);

  async function submit() {
    setError(null);
    let body: NewSource;
    const model = loadSavedModel() || undefined;
    if (kind === "text") body = { type: "text", text, title: title.trim() || undefined, model };
    else if (kind === "url" || kind === "youtube") body = { type: kind, url: url.trim(), model };
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
        className="w-full max-w-xl rounded-2xl border border-line bg-panel shadow-2xl"
        onMouseDown={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between border-b border-line px-5 py-3">
          <h2 className="text-lg font-semibold">Add sources</h2>
          <button onClick={onClose} className="rounded-md p-1 text-muted hover:bg-panel-2" title="Close">
            <X size={18} />
          </button>
        </div>
        <div className="grid grid-cols-5 gap-2 px-5 pt-4">
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
          {kind === "discover" && (
            <div className="flex flex-col gap-2">
              <div className="flex gap-2">
                <input
                  autoFocus
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && query.trim().length >= 3 && searchWeb()}
                  placeholder="What should the sources be about?"
                  className="min-w-0 flex-1 rounded-lg border border-line bg-panel-2 px-3 py-2 text-sm outline-none focus:border-accent"
                />
                <button
                  onClick={searchWeb}
                  disabled={query.trim().length < 3 || busy}
                  className="inline-flex items-center gap-1 rounded-lg border border-line px-3 text-sm hover:bg-panel-2 disabled:opacity-40"
                >
                  {busy && !hits ? <Loader2 size={14} className="animate-spin" /> : <Search size={14} />} Search
                </button>
              </div>
              {hits && hits.length === 0 && <p className="text-sm text-muted">No results.</p>}
              {hits && hits.length > 0 && (
                <ul className="scroll-thin max-h-72 space-y-1 overflow-y-auto">
                  {hits.map((h) => {
                    const on = picked.has(h.url);
                    return (
                      <li key={h.url}>
                        <button
                          onClick={() =>
                            setPicked((p) => {
                              const n = new Set(p);
                              if (n.has(h.url)) n.delete(h.url);
                              else n.add(h.url);
                              return n;
                            })
                          }
                          className="flex w-full items-start gap-2 rounded-lg px-2 py-1.5 text-left hover:bg-panel-2"
                        >
                          {on ? <CheckSquare size={16} className="mt-0.5 shrink-0 text-accent" /> : <Square size={16} className="mt-0.5 shrink-0 text-muted" />}
                          {h.kind === "youtube" ? <MonitorPlay size={15} className="mt-0.5 shrink-0 text-red-500" /> : <Globe size={15} className="mt-0.5 shrink-0 text-muted" />}
                          <span className="min-w-0">
                            <span className="block truncate text-sm font-medium">{h.title}</span>
                            <span className="block truncate text-xs text-muted">{h.url}</span>
                            {h.snippet && <span className="line-clamp-2 text-xs text-muted">{h.snippet}</span>}
                          </span>
                        </button>
                      </li>
                    );
                  })}
                </ul>
              )}
            </div>
          )}
          {error && <p className="mt-2 whitespace-pre-line text-sm text-bad">{error}</p>}
        </div>
        {kind !== "file" && (
          <div className="flex justify-end gap-2 border-t border-line px-5 py-3">
            <button onClick={onClose} className="rounded-lg px-3 py-1.5 text-sm hover:bg-panel-2">
              Cancel
            </button>
            <button
              onClick={kind === "discover" ? addPicked : submit}
              disabled={kind === "discover" ? !picked.size || !hits || busy : !valid || busy}
              className="inline-flex items-center gap-1.5 rounded-lg bg-accent px-3 py-1.5 text-sm font-medium text-white disabled:opacity-40"
            >
              {busy && <Loader2 size={14} className="animate-spin" />}{" "}
              {kind === "discover" ? `Add ${picked.size} source${picked.size === 1 ? "" : "s"}` : "Add source"}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
