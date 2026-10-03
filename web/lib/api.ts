// Typed client for the FastAPI service (service/src/abstain_api).

export const API_URL = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000").replace(/\/$/, "");

export type Notebook = {
  id: string;
  name: string;
  description: string | null;
  is_public: boolean;
  created_at: string;
};

export type SourceDoc = {
  id: string;
  collection_id: string;
  filename: string;
  size_bytes: number;
  status: "queued" | "processing" | "indexed" | "rejected" | "failed";
  n_passages: number;
  n_flagged: number;
  error: string | null;
  created_at: string;
};

export type Passage = { passage_id: string; text: string; page: number | null; section: string | null };

export type Citation = {
  index: number;
  document_id: string; // passage id
  title: string;
  snippet: string;
  score: number | null;
  source_document?: string | null;
  page?: number | null;
  section?: string | null;
};

export type GuardEvent = { stage: string; rule: string; action: string; detail?: string };

export type Final = {
  answer: string;
  decision: "answer" | "abstain" | "clarify" | "blocked";
  abstained_by: string;
  p_unanswerable: number | null;
  query: string | null;
  queries: string[];
  attempts: number;
  citations: Citation[];
  trace: string[];
  timings: Record<string, number>;
  latency_ms: number;
  guardrails: GuardEvent[];
  message_id: string;
  conversation_id: string;
  model?: string;
};

export type Step = { node: string; label: string; detail: string };

export type Conversation = { id: string; title: string; collection_id: string; created_at: string; updated_at: string };

export type StoredMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  turn: number;
  decision: string | null;
  p_unanswerable: number | null;
  latency_ms: number | null;
  meta: Record<string, unknown>;
  created_at: string;
};

export type ModelInfo = {
  id: string;
  provider: string;
  provider_label: string;
  model: string;
  name: string;
  context_length: number | null;
  reasoning: boolean;
  featured: boolean;
  calibrated: boolean;
  description: string;
  /** refusal thresholds tuned for this model on MTRAG dev (service/scripts/calibrate_models.py) */
  calibration?: {
    tau: number;
    min_p: number;
    source: "research" | "calibrated";
    f1?: number;
    recall?: number;
    fa?: number;
    n_dev?: number;
    refusal_rate_answerable?: number;
  } | null;
};

export type ModelsResponse = {
  default: string;
  models: ModelInfo[];
  providers: { key: string; label: string; configured: boolean; error: string | null }[];
};

export type Note = {
  id: string;
  collection_id: string;
  title: string;
  content: string;
  kind: "note" | "answer";
  source_message_id: string | null;
  citations: Citation[];
  pinned: boolean;
  created_at: string;
  updated_at: string;
};

function authHeaders(): Record<string, string> {
  // AUTH_MODE=dev on the API needs no token. With Clerk / Auth.js, put the session token here.
  let token: string | null = null;
  try {
    token = window.localStorage.getItem("api_token");
  } catch {
    token = null;
  }
  return token ? { Authorization: `Bearer ${token}` } : {};
}

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

async function req<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers: Record<string, string> = { ...authHeaders(), ...((init.headers as Record<string, string>) || {}) };
  if (init.body && !(init.body instanceof FormData)) headers["Content-Type"] = "application/json";
  const r = await fetch(`${API_URL}${path}`, { ...init, headers });
  if (!r.ok) {
    let msg = r.statusText;
    try {
      const j = await r.json();
      msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail ?? j);
    } catch {
      /* not JSON */
    }
    throw new ApiError(r.status, msg);
  }
  if (r.status === 204) return undefined as T;
  return (await r.json()) as T;
}

export const api = {
  notebooks: () => req<Notebook[]>("/collections"),
  createNotebook: (id: string, name: string, description?: string) =>
    req<Notebook>("/collections", { method: "POST", body: JSON.stringify({ id, name, description }) }),
  renameNotebook: (id: string, name: string) =>
    req<Notebook>(`/collections/${id}`, { method: "PATCH", body: JSON.stringify({ name }) }),
  deleteNotebook: (id: string) => req<void>(`/collections/${id}`, { method: "DELETE" }),

  sources: (nb: string) => req<SourceDoc[]>(`/collections/${nb}/documents`),
  upload: (nb: string, file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    return req<SourceDoc>(`/collections/${nb}/documents`, { method: "POST", body: fd });
  },
  deleteSource: (nb: string, doc: string) => req<void>(`/collections/${nb}/documents/${doc}`, { method: "DELETE" }),
  passages: (nb: string, doc: string) => req<Passage[]>(`/collections/${nb}/documents/${doc}/passages`),
  fileUrl: (nb: string, doc: string) => `${API_URL}/collections/${nb}/documents/${doc}/file`,

  conversations: (nb: string) => req<Conversation[]>(`/conversations?collection=${encodeURIComponent(nb)}`),
  conversation: (id: string) =>
    req<Conversation & { messages: StoredMessage[] }>(`/conversations/${id}`),
  deleteConversation: (id: string) => req<void>(`/conversations/${id}`, { method: "DELETE" }),

  feedback: (message_id: string, rating: 1 | -1, reason?: string) =>
    req("/feedback", { method: "POST", body: JSON.stringify({ message_id, rating, reason }) }),

  models: (refresh = false) => req<ModelsResponse>(`/models${refresh ? "?refresh=true" : ""}`),

  notes: (nb: string) => req<Note[]>(`/collections/${nb}/notes`),
  createNote: (nb: string, note: Partial<Note>) =>
    req<Note>(`/collections/${nb}/notes`, { method: "POST", body: JSON.stringify(note) }),
  updateNote: (nb: string, id: string, patch: Partial<Pick<Note, "title" | "content" | "pinned">>) =>
    req<Note>(`/collections/${nb}/notes/${id}`, { method: "PATCH", body: JSON.stringify(patch) }),
  deleteNote: (nb: string, id: string) => req<void>(`/collections/${nb}/notes/${id}`, { method: "DELETE" }),
};

export type ChatEvent =
  | { event: "meta"; data: { conversation_id: string; message_id: string; turn: number; collection: string; model: string } }
  | { event: "step"; data: Step }
  | { event: "token"; data: { text: string } }
  | { event: "guardrail"; data: GuardEvent }
  | { event: "final"; data: Final }
  | { event: "error"; data: { message: string } };

/** POST /chat/stream and yield the Server-Sent Events as they arrive. */
export async function* streamChat(
  body: { message: string; collection: string; conversation_id?: string; model?: string; document_ids?: string[] },
  signal?: AbortSignal,
): AsyncGenerator<ChatEvent> {
  const r = await fetch(`${API_URL}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream", ...authHeaders() },
    body: JSON.stringify(body),
    signal,
  });
  if (!r.ok || !r.body) {
    let msg = r.statusText;
    try {
      const j = await r.json();
      msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail ?? j);
    } catch {
      /* not JSON */
    }
    throw new ApiError(r.status, msg);
  }
  const reader = r.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true }).replace(/\r\n/g, "\n");
    let i: number;
    while ((i = buf.indexOf("\n\n")) >= 0) {
      const block = buf.slice(0, i);
      buf = buf.slice(i + 2);
      let ev = "message";
      const data: string[] = [];
      for (const line of block.split("\n")) {
        if (line.startsWith(":")) continue; // keep-alive ping
        if (line.startsWith("event:")) ev = line.slice(6).trim();
        else if (line.startsWith("data:")) data.push(line.slice(5).replace(/^ /, ""));
      }
      if (!data.length) continue;
      try {
        yield { event: ev, data: JSON.parse(data.join("\n")) } as ChatEvent;
      } catch {
        /* ignore malformed event */
      }
    }
  }
}

export function slugify(name: string): string {
  const base = name
    .toLowerCase()
    .normalize("NFKD")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 40);
  const suffix = Math.random().toString(36).slice(2, 6);
  return `${base || "notebook"}-${suffix}`;
}
