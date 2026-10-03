"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { Brain, Check, ChevronDown, RefreshCw, Search, ShieldCheck } from "lucide-react";
import { api, ModelInfo, ModelsResponse } from "@/lib/api";

const STORE_KEY = "model_choice";

export function loadSavedModel(): string | null {
  try {
    return window.localStorage.getItem(STORE_KEY);
  } catch {
    return null;
  }
}

function saveModel(id: string) {
  try {
    window.localStorage.setItem(STORE_KEY, id);
  } catch {
    /* storage unavailable */
  }
}

function ctx(n: number | null) {
  if (!n) return "";
  return n >= 1_000_000 ? `${Math.round(n / 100_000) / 10}M ctx` : `${Math.round(n / 1000)}K ctx`;
}

/** Model menu like ChatGPT / Claude.ai: every model of every configured provider, searchable. */
export default function ModelPicker({
  value,
  onChange,
}: {
  value: string | null;
  onChange: (id: string, info?: ModelInfo) => void;
}) {
  const [data, setData] = useState<ModelsResponse | null>(null);
  const [open, setOpen] = useState(false);
  const [q, setQ] = useState("");
  const [loading, setLoading] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  const load = (refresh = false) => {
    setLoading(true);
    api
      .models(refresh)
      .then((d) => {
        setData(d);
        if (!value) {
          const saved = loadSavedModel();
          const pick = saved && d.models.some((m) => m.id === saved) ? saved : d.default;
          onChange(pick, d.models.find((m) => m.id === pick));
        }
      })
      .catch(() => setData({ default: "", models: [], providers: [] }))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const close = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, []);

  const current = data?.models.find((m) => m.id === value);
  const groups = useMemo(() => {
    const ms = (data?.models || []).filter((m) => {
      if (!q.trim()) return true;
      const s = q.toLowerCase();
      return m.name.toLowerCase().includes(s) || m.id.toLowerCase().includes(s) || m.provider_label.toLowerCase().includes(s);
    });
    const featured = q.trim() ? [] : ms.filter((m) => m.featured || m.calibrated).slice(0, 12);
    const byProvider = new Map<string, ModelInfo[]>();
    for (const m of ms) {
      const k = m.provider_label;
      if (!byProvider.has(k)) byProvider.set(k, []);
      byProvider.get(k)!.push(m);
    }
    return { featured, byProvider: [...byProvider.entries()] };
  }, [data, q]);

  const choose = (m: ModelInfo) => {
    saveModel(m.id);
    onChange(m.id, m);
    setOpen(false);
    setQ("");
  };

  const missing = (data?.providers || []).filter((p) => !p.configured).map((p) => p.label);
  const failing = (data?.providers || []).filter((p) => p.configured && p.error);

  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setOpen((o) => !o)}
        className="inline-flex max-w-[16rem] items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-sm font-medium hover:bg-panel-2"
        title={value || "Choose a model"}
      >
        <span className="truncate">{current?.name || value?.split(":").slice(1).join(":") || "Choose a model"}</span>
        <ChevronDown size={16} className="shrink-0 text-muted" />
      </button>
      {open && (
        <div className="absolute left-0 top-full z-30 mt-1 w-[22rem] max-w-[calc(100vw-2rem)] rounded-xl border border-line bg-panel shadow-xl">
          <div className="flex items-center gap-2 border-b border-line px-3 py-2">
            <Search size={15} className="text-muted" />
            <input
              autoFocus
              value={q}
              onChange={(e) => setQ(e.target.value)}
              placeholder="Search models"
              className="w-full bg-transparent py-1 text-sm outline-none"
            />
            <button onClick={() => load(true)} title="Refresh model lists" className="rounded p-1 text-muted hover:bg-panel-2">
              <RefreshCw size={14} className={loading ? "animate-spin" : ""} />
            </button>
          </div>
          <div className="scroll-thin max-h-[60vh] overflow-y-auto py-1">
            {groups.featured.length > 0 && (
              <Group title="Recommended">
                {groups.featured.map((m) => (
                  <Item key={`f-${m.id}`} m={m} active={m.id === value} onClick={() => choose(m)} />
                ))}
              </Group>
            )}
            {groups.byProvider.map(([label, ms]) => (
              <Group key={label} title={`${label} · ${ms.length}`}>
                {ms.map((m) => (
                  <Item key={m.id} m={m} active={m.id === value} onClick={() => choose(m)} />
                ))}
              </Group>
            ))}
            {data && data.models.length === 0 && <p className="px-3 py-4 text-sm text-muted">No models available.</p>}
            {failing.map((p) => (
              <p key={p.key} className="px-3 py-1 text-xs text-bad">
                {p.label}: could not list models ({p.error})
              </p>
            ))}
          </div>
          {missing.length > 0 && (
            <p className="border-t border-line px-3 py-2 text-xs text-muted">
              More models: add an API key for {missing.slice(0, 4).join(", ")}
              {missing.length > 4 ? "…" : ""} in <code>.env</code> (one OPENROUTER_API_KEY covers Claude, GPT, Gemini,
              Llama and more).
            </p>
          )}
        </div>
      )}
    </div>
  );
}

function Group({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="py-1">
      <div className="px-3 pb-1 pt-2 text-[11px] font-semibold uppercase tracking-wide text-muted">{title}</div>
      {children}
    </div>
  );
}

function Item({ m, active, onClick }: { m: ModelInfo; active: boolean; onClick: () => void }) {
  return (
    <button onClick={onClick} className="flex w-full items-start gap-2 px-3 py-2 text-left hover:bg-panel-2">
      <span className="mt-0.5 w-4 shrink-0 text-accent">{active && <Check size={16} />}</span>
      <span className="min-w-0 flex-1">
        <span className="flex items-center gap-1.5 text-sm font-medium">
          <span className="truncate">{m.name}</span>
          {m.calibrated && (
            <span title="Abstention thresholds were calibrated for this model" className="text-ok">
              <ShieldCheck size={14} />
            </span>
          )}
          {m.reasoning && (
            <span title="Reasoning model" className="text-muted">
              <Brain size={14} />
            </span>
          )}
        </span>
        <span className="block truncate text-xs text-muted">
          {m.provider_label} · {m.model}
          {m.context_length ? ` · ${ctx(m.context_length)}` : ""}
          {m.description ? ` · ${m.description}` : ""}
        </span>
      </span>
    </button>
  );
}
