"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { BookOpen, Database, Loader2, Plus, Trash2, Users } from "lucide-react";
import { api, Notebook, slugify } from "@/lib/api";

export default function Home() {
  const router = useRouter();
  const [notebooks, setNotebooks] = useState<Notebook[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);
  const [name, setName] = useState("");

  const load = () =>
    api
      .notebooks()
      .then(setNotebooks)
      .catch((e) => setError(`Cannot reach the API (${e.message}). Is the service running?`));

  useEffect(() => {
    load();
  }, []);

  async function create() {
    const n = name.trim() || "Untitled notebook";
    setCreating(true);
    try {
      const nb = await api.createNotebook(slugify(n), n);
      router.push(`/?nb=${nb.id}`);
    } catch (e) {
      setError((e as Error).message);
      setCreating(false);
    }
  }

  async function remove(nb: Notebook) {
    if (!window.confirm(`Delete "${nb.name}" with its sources, chats and notes?`)) return;
    try {
      await api.deleteNotebook(nb.id);
      load();
    } catch (e) {
      setError((e as Error).message);
    }
  }

  const mine = (notebooks || []).filter((n) => !n.is_public && (n.role || "owner") === "owner");
  const sharedWithMe = (notebooks || []).filter((n) => !n.is_public && (n.role === "editor" || n.role === "viewer"));
  const builtin = (notebooks || []).filter((n) => n.is_public);

  return (
    <main className="mx-auto max-w-6xl px-4 py-10 sm:px-8">
      <header className="mb-8 flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-3xl font-semibold tracking-tight">Grounded Notebook</h1>
          <p className="mt-1 text-muted">
            Chat with your documents. Every answer is cited, and the assistant says so when your sources do not
            contain the answer.
          </p>
        </div>
      </header>

      {error && <div className="mb-6 rounded-lg bg-bad-soft px-4 py-3 text-sm text-bad">{error}</div>}

      <section className="mb-10">
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-muted">My notebooks</h2>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          <div className="flex min-h-40 flex-col justify-between rounded-2xl border border-dashed border-line bg-panel p-5">
            <div>
              <div className="mb-3 flex h-10 w-10 items-center justify-center rounded-full bg-accent-soft text-accent">
                <Plus size={20} />
              </div>
              <input
                value={name}
                onChange={(e) => setName(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && create()}
                placeholder="New notebook name"
                className="w-full rounded-lg border border-line bg-panel-2 px-3 py-2 text-sm outline-none focus:border-accent"
              />
            </div>
            <button
              onClick={create}
              disabled={creating}
              className="mt-3 inline-flex items-center justify-center gap-2 rounded-lg bg-accent px-3 py-2 text-sm font-medium text-white hover:opacity-90 disabled:opacity-60"
            >
              {creating ? <Loader2 size={16} className="animate-spin" /> : <Plus size={16} />} Create notebook
            </button>
          </div>
          {notebooks === null && !error && (
            <div className="flex items-center gap-2 p-5 text-muted">
              <Loader2 size={16} className="animate-spin" /> Loading…
            </div>
          )}
          {mine.map((nb) => (
            <NotebookCard key={nb.id} nb={nb} onOpen={() => router.push(`/?nb=${nb.id}`)} onDelete={() => remove(nb)} />
          ))}
        </div>
      </section>

      {sharedWithMe.length > 0 && (
        <section className="mb-10">
          <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-muted">Shared with me</h2>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {sharedWithMe.map((nb) => (
              <NotebookCard key={nb.id} nb={nb} onOpen={() => router.push(`/?nb=${nb.id}`)} />
            ))}
          </div>
        </section>
      )}

      {builtin.length > 0 && (
        <section>
          <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-muted">Built-in collections</h2>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {builtin.map((nb) => (
              <NotebookCard key={nb.id} nb={nb} builtin onOpen={() => router.push(`/?nb=${nb.id}`)} />
            ))}
          </div>
        </section>
      )}
    </main>
  );
}

function NotebookCard({
  nb,
  builtin,
  onOpen,
  onDelete,
}: {
  nb: Notebook;
  builtin?: boolean;
  onOpen: () => void;
  onDelete?: () => void;
}) {
  return (
    <div
      onClick={onOpen}
      className="group relative flex min-h-40 cursor-pointer flex-col justify-between rounded-2xl border border-line bg-panel p-5 shadow-sm transition hover:shadow-md"
    >
      <div>
        <div className="mb-3 flex h-10 w-10 items-center justify-center rounded-full bg-accent-soft text-accent">
          {builtin ? <Database size={18} /> : <BookOpen size={18} />}
        </div>
        <h3 className="font-semibold leading-snug">{nb.name}</h3>
        {nb.description && <p className="mt-1 line-clamp-2 text-sm text-muted">{nb.description}</p>}
      </div>
      <p className="mt-3 flex items-center gap-1.5 text-xs text-muted">
        {builtin ? "MTRAG benchmark corpus" : `Created ${new Date(nb.created_at).toLocaleDateString()}`}
        {!builtin && nb.role && nb.role !== "owner" && <span>· {nb.role === "editor" ? "Editor" : "Viewer"}</span>}
        {!builtin && nb.role === "owner" && nb.shared && (
          <span className="inline-flex items-center gap-0.5" title="Shared">
            · <Users size={12} /> Shared
          </span>
        )}
      </p>
      {onDelete && (
        <button
          onClick={(e) => {
            e.stopPropagation();
            onDelete();
          }}
          title="Delete notebook"
          className="absolute right-3 top-3 rounded-md p-1.5 text-muted opacity-0 transition hover:bg-bad-soft hover:text-bad group-hover:opacity-100"
        >
          <Trash2 size={16} />
        </button>
      )}
    </div>
  );
}
