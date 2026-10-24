/* Shapes that come off the wire from web/api.py, typed once and checked everywhere.
   In:  nothing — declarations only. Out: the corpus, stream-event, thread and chat types.
   Step fields mirror AgentLoop._emit in src/agent/loop.py; anything the loop adds later
   arrives as an extra field and is ignored rather than breaking the page. */

export interface Step {
  kind: string; // thinking | answering | skipped | tool_start | tool_end | image | cap
  iteration?: number;
  tool_name?: string;
  args?: Record<string, unknown>;
  result?: Record<string, unknown>;
  latency_ms?: number;
  reason?: string;
  image_path?: string;
  image_url?: string | null;
  t_ms?: number;
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
