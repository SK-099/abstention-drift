"use client";

import { useEffect, useState } from "react";
import { Check, Globe, Link2, Loader2, Lock, Trash2, UserPlus, X } from "lucide-react";
import { api, Sharing } from "@/lib/api";

/** Share a notebook: people (viewer / editor, by e-mail or user id) and "anyone signed in with the link". */
export default function ShareDialog({ notebookId, name, onClose }: { notebookId: string; name: string; onClose: () => void }) {
  const [s, setS] = useState<Sharing | null>(null);
  const [who, setWho] = useState("");
  const [role, setRole] = useState<"viewer" | "editor">("viewer");
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    api.sharing(notebookId).then(setS).catch((e) => setError(e.message));
    const esc = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", esc);
    return () => window.removeEventListener("keydown", esc);
  }, [notebookId, onClose]);

  const run = async (f: () => Promise<Sharing>) => {
    setError(null);
    setBusy(true);
    try {
      setS(await f());
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };
  const link = typeof window !== "undefined" ? `${window.location.origin}/?nb=${notebookId}` : "";

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4" onMouseDown={onClose}>
      <div
        role="dialog"
        aria-label="Share"
        onMouseDown={(e) => e.stopPropagation()}
        className="w-full max-w-lg rounded-2xl border border-line bg-panel shadow-2xl"
      >
        <div className="flex items-center justify-between border-b border-line px-5 py-3">
          <h2 className="truncate text-lg font-semibold">Share “{name}”</h2>
          <button onClick={onClose} className="rounded-md p-1 text-muted hover:bg-panel-2" title="Close">
            <X size={18} />
          </button>
        </div>
        <div className="space-y-4 px-5 py-4">
          <div className="flex gap-2">
            <input
              value={who}
              onChange={(e) => setWho(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && who.trim() && run(() => api.share(notebookId, who.trim(), role)).then(() => setWho(""))}
              placeholder="E-mail address or user id"
              className="min-w-0 flex-1 rounded-lg border border-line bg-panel-2 px-3 py-2 text-sm outline-none focus:border-accent"
            />
            <select
              value={role}
              onChange={(e) => setRole(e.target.value as "viewer" | "editor")}
              className="rounded-lg border border-line bg-panel-2 px-2 text-sm"
            >
              <option value="viewer">Viewer</option>
              <option value="editor">Editor</option>
            </select>
            <button
              disabled={!who.trim() || busy}
              onClick={() => run(() => api.share(notebookId, who.trim(), role)).then(() => setWho(""))}
              className="inline-flex items-center gap-1 rounded-lg bg-accent px-3 text-sm font-medium text-white disabled:opacity-40"
            >
              <UserPlus size={15} /> Share
            </button>
          </div>
          <p className="text-xs text-muted">
            Viewers can read the sources and Studio outputs and ask questions; editors can also add sources and
            generate. Everyone&apos;s chats and notes stay private.
          </p>

          <div>
            <h3 className="mb-1.5 text-sm font-semibold">People with access</h3>
            {!s && !error && <Loader2 size={16} className="animate-spin text-muted" />}
            {s && (
              <ul className="divide-y divide-line rounded-lg border border-line">
                <li className="flex items-center gap-2 px-3 py-2 text-sm">
                  <span className="min-w-0 flex-1 truncate">{s.owner || "You"}</span>
                  <span className="text-xs text-muted">Owner</span>
                </li>
                {s.members.map((m) => (
                  <li key={m.principal} className="flex items-center gap-2 px-3 py-2 text-sm">
                    <span className="min-w-0 flex-1 truncate">{m.principal}</span>
                    <select
                      value={m.role}
                      onChange={(e) => run(() => api.share(notebookId, m.principal, e.target.value as "viewer" | "editor"))}
                      className="rounded-md border border-line bg-panel-2 px-1.5 py-0.5 text-xs"
                    >
                      <option value="viewer">Viewer</option>
                      <option value="editor">Editor</option>
                    </select>
                    <button
                      onClick={() => run(() => api.unshare(notebookId, m.principal))}
                      className="rounded p-1 text-muted hover:text-bad"
                      title="Remove access"
                    >
                      <Trash2 size={14} />
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>

          {s && (
            <div>
              <h3 className="mb-1.5 text-sm font-semibold">General access</h3>
              <div className="flex items-center gap-3 rounded-lg border border-line px-3 py-2.5">
                {s.link_access === "viewer" ? <Globe size={18} className="text-ok" /> : <Lock size={18} className="text-muted" />}
                <select
                  value={s.link_access}
                  onChange={(e) => run(() => api.setLinkAccess(notebookId, e.target.value as "none" | "viewer"))}
                  className="flex-1 bg-transparent text-sm outline-none"
                >
                  <option value="none">Restricted: only people added above</option>
                  <option value="viewer">Anyone signed in with the link can view</option>
                </select>
              </div>
            </div>
          )}
          {error && <p className="text-sm text-bad">{error}</p>}
        </div>
        <div className="flex justify-between border-t border-line px-5 py-3">
          <button
            onClick={() => navigator.clipboard.writeText(link).then(() => setCopied(true))}
            className="inline-flex items-center gap-1.5 rounded-lg border border-line px-3 py-1.5 text-sm hover:bg-panel-2"
          >
            {copied ? <Check size={15} className="text-ok" /> : <Link2 size={15} />} {copied ? "Link copied" : "Copy link"}
          </button>
          <button onClick={onClose} className="rounded-lg bg-accent px-4 py-1.5 text-sm font-medium text-white">
            Done
          </button>
        </div>
      </div>
    </div>
  );
}
