"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { Brain, Check, ChevronDown, RefreshCw, Search, ShieldCheck, Zap } from "lucide-react";
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
    const s = q.trim().toLowerCase();
    const ms = (data?.models || []).filter(
      (m) => !s || m.name.toLowerCase().includes(s) || m.id.toLowerCase().includes(s) || m.provider_label.toLowerCase().includes(s),
    );
    // research model first, then local before cloud, calibrated before not, then by name
    const rank = (m: ModelInfo) => [
      m.calibration?.source === "research" ? 0 : 1,
      m.provider === "local" ? 0 : 1,
      m.calibrated ? 0 : 1,
    ];
    const sorted = [...ms].sort((a, b) => {
      const ra = rank(a), rb = rank(b);
      for (let i = 0; i < ra.length; i++) if (ra[i] !== rb[i]) return ra[i] - rb[i];
      return a.name.localeCompare(b.name);
    });
    return GROUPS.map((g) => ({ ...g, models: sorted.filter((m) => m.reasoning === g.reasoning) })).filter(
      (g) => g.models.length > 0,
    );
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
            {groups.map((g) => (
              <Group key={g.title} title={`${g.title} · ${g.models.length}`} icon={g.icon} caption={g.caption}>
                {g.models.map((m) => (
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
          <p className="flex items-start gap-1.5 border-t border-line px-3 py-2 text-xs text-muted">
            <ShieldCheck size={13} className="mt-px shrink-0 text-ok" />
            <span>Refusal thresholds calibrated for this model (hover for details). Others use the research values.</span>
          </p>
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

const GROUPS = [
  {
    title: "Standard models",
    reasoning: false,
    icon: <Zap size={12} />,
    caption: "Answer directly. Faster; best for quick look-ups in your sources.",
  },
  {
    title: "Thinking models",
    reasoning: true,
    icon: <Brain size={12} />,
    caption: "Reason step by step before answering (the reasoning is hidden). Slower; can be more careful.",
  },
];

function calibrationText(m: ModelInfo) {
  const c = m.calibration;
  if (!c) return "Refusal thresholds calibrated for this model";
  const pct = (v?: number) => (v === undefined ? "?" : `${Math.round(v * 100)}%`);
  const minp = c.min_p > 1 ? "self-check off" : `self-check from p ≥ ${c.min_p.toFixed(2)}`;
  return (
    `${c.source === "research" ? "Research calibration" : "Calibrated"} on MTRAG dev` +
    (c.n_dev ? ` (${c.n_dev} questions)` : "") +
    `: gate τ = ${c.tau.toFixed(2)}, ${minp}. ` +
    `Finds ${pct(c.recall)} of unanswerable questions, refuses ${pct(c.fa)} of answerable ones (F1 ${c.f1?.toFixed(2) ?? "?"}).`
  );
}

function Group({
  title,
  caption,
  icon,
  children,
}: {
  title: string;
  caption: string;
  icon: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <div className="py-1">
      <div className="sticky top-0 z-10 bg-panel px-3 pb-1.5 pt-2">
        <div className="flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wide text-muted">
          {icon}
          {title}
        </div>
        <div className="mt-0.5 text-[11px] leading-snug text-muted">{caption}</div>
      </div>
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
            <span title={calibrationText(m)} className="text-ok">
              <ShieldCheck size={14} />
            </span>
          )}
          {m.calibration?.source === "research" && (
            <span className="rounded bg-accent/10 px-1.5 py-px text-[10px] font-semibold uppercase tracking-wide text-accent">
              Research
            </span>
          )}
        </span>
        <span className="block truncate text-xs text-muted">
          {[
            m.provider === "local" ? "Local" : m.provider_label,
            m.description,
            m.context_length ? ctx(m.context_length) : "",
            m.calibration?.seconds_per_example ? `≈${Math.round(m.calibration.seconds_per_example)} s per answer` : "",
          ]
            .filter(Boolean)
            .join(" · ")}
        </span>
      </span>
    </button>
  );
}
