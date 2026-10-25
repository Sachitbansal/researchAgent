/* Shapes that come off the wire from web/api.py, typed once and checked everywhere.
   In:  nothing — declarations only. Out: the corpus, stream-event, thread and chat types.
   Step fields mirror AgentLoop._emit in src/agent/loop.py; anything the loop adds later
   arrives as an extra field and is ignored rather than breaking the page. */

export interface Step {
  kind: string; // thinking | answering | skipped | tool_start | tool_end | image | cap | memory
  iteration?: number;
  tool_name?: string;
  args?: Record<string, unknown>;
  result?: Record<string, unknown>;
  latency_ms?: number;
  reason?: string;
  image_path?: string;
  image_url?: string | null;
  t_ms?: number;
  // kind "memory" only: what long-term memory injected at the run's start (D19). Its
  // passages ride in result.chunks, like a retrieval's, so Sources and cards find them.
  memories?: InjectedMemory[];
  preferences?: { id: string; text: string }[];
  stale?: string[];
  tokens?: number;
}

export interface InjectedMemory {
  id: string;
  label: string; // m1, m2, ... as the agent saw them
  text: string;
  chunk_ids: string[];
  score: number;
  use_count?: number;
  user_edited?: boolean;
  verified?: boolean;
}

/* What the writer did after an answer: the `memory` frame, also kept in the turn's stats. */
export interface MemoryReport {
  written: { id: string; kind: string; text: string; chunk_ids: string[] }[];
  refreshed: { id: string; kind: string; text: string; similarity: number; chunk_ids_added: string[]; matched: string }[];
  invalidated: { id: string; kind: string; text: string; reason: string; superseded_by?: string }[];
  used: { id: string; text: string }[];
  forgotten: { id: string; reason: string; why?: string }[];
  dropped?: { uncited?: number; ungrounded?: number };
  skipped?: string;
  error?: string;
  detail?: string;
  recall_error?: string;
}

/* A row of GET /api/memories. The fields after `strength` come from the server's
   catalogue (D23) and are absent when it runs without one, hence optional. */
export interface MemoryRecord {
  id: string;
  kind: "finding" | "preference";
  text: string;
  chunk_ids: string[];
  use_count: number;
  created_at: number;
  last_used_at: number;
  active: boolean;
  forget_reason: string | null;
  invalidated_at: number | null;
  forgotten_at: number | null;
  superseded_by: string | null;
  source_thread_id: string | null;
  source_question: string;
  strength: number | null;
  forget_detail?: string | null; // why it decayed: below_min_strength | over_cap
  origin?: "agent" | "user";
  user_edited?: boolean;
  verified?: boolean | null; // findings only: NLI found the text in its passages
  edited_at?: number | null;
  fades_in_days?: number | null;
  strength_fraction?: number | null; // fades_in_days over a fresh single-use finding's
  passages?: MemoryPassage[];
  source_thread_exists?: boolean | null;
  superseded_by_text?: string | null;
  purge_in_days?: number;
  restorable?: boolean;
  restore_blocked?: string | null;
}

export interface MemoryPassage {
  chunk_id: string;
  paper_id: string;
  paper_title?: string | null;
  page?: number | null;
  text: string | null;
  present: boolean; // still in the index
  changed: boolean; // its text changed since the memory was checked against it
}

/* The forgetting numbers the page explains itself with (config.yaml memory). */
export interface MemorySettings {
  half_life_days: number;
  min_strength: number;
  max_active: number;
  purge_after_days: number;
  fading_soon_days: number;
  max_text_chars: number;
  fresh_days: number;
}

export interface MemoryEditResult {
  memory: MemoryRecord;
  changed: boolean;
  verification: { verified: boolean; label: string; confidence: number } | null;
}

export interface MemoryAddResult {
  outcome: "added" | "refreshed";
  memory: MemoryRecord | null;
  replaced: { id: string; text: string }[];
}

export interface Stats {
  iterations?: number;
  tool_calls?: number;
  context_tokens?: number;
  // Rendered generically: the loop may grow new stop reasons, and the page should name
  // them rather than fall over.
  stopped_because?: string;
  run_id?: string | null;
  elapsed_ms?: number;
  // Stored with a thread's turn: how much of the chat's history this answer was given.
  history_turns?: number;
  history_turns_dropped?: number;
  history_tokens?: number;
  memory?: MemoryReport;
  error?: string;
  detail?: string;
}

export interface UserMessage {
  role: "user";
  content: string;
  ts: number;
}

export interface AssistantMessage {
  role: "assistant";
  // "running" exists only client-side, while the stream is open.
  status: "answered" | "failed" | "running";
  content: string;
  steps: Step[];
  stats?: Stats;
  error?: { error: string; detail: string };
  ts: number;
  elapsed_ms?: number;
  // The one-line tool-call notes later questions in the thread are shown (D12).
  notes?: string[];
  // What long-term memory wrote after this answer (D19): the live `memory` frame, then
  // the copy stored in the turn's stats.
  memory?: MemoryReport;
}

export type Message = UserMessage | AssistantMessage;

/* A hand-written sample chat from fixtures/sampleChats.ts. Real chats are threads stored
   by the server (docs/DECISIONS.md D12) and arrive as ThreadSummary / ThreadDetail. */
export interface Chat {
  id: string;
  title: string;
  created_at: number;
  updated_at: number;
  messages: Message[];
  sample?: boolean;
}

export interface ThreadSummary {
  id: string;
  title: string;
  created_at: number;
  updated_at: number;
  n_turns: number;
}

export interface StoredTurn {
  seq: number;
  question: string;
  answer: string;
  stats: Stats;
  notes: string[];
  steps: Step[];
  run_id?: string | null;
  created_at: number;
}

export interface ThreadDetail extends ThreadSummary {
  turns: StoredTurn[];
}

export interface ThreadEvent {
  thread_id: string;
  created: boolean;
  title: string;
}

export interface Paper {
  paper_id: string;
  arxiv_id?: string;
  title: string;
  authors: string[];
  year?: number | null;
  categories: string[];
  topic_tags: string[];
  n_chunks: number;
  n_figures: number;
  pdf_name: string;
  pdf_url: string;
  parse_status?: string | null;
}

export interface Topic {
  tag: string | null;
  label: string;
  arxiv_query: string;
  count: number;
}

export interface Corpus {
  papers: Paper[];
  topics: Topic[];
  embedding_model?: string;
  updated_at?: string;
  error?: string;
}

export interface AnswerEvent extends Stats {
  answer: string;
  thread_id?: string | null;
  turn_seq?: number | null;
}

export interface FailedEvent {
  error: string;
  detail: string;
  answer: string;
  run_id?: string | null;
  thread_id?: string | null;
  turn_seq?: number | null;
}
