/* Shapes that come off the wire from web/api.py, typed once and checked everywhere.
   In:  nothing — declarations only. Out: the corpus, stream-event and in-memory chat types.
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
}

export type Message = UserMessage | AssistantMessage;

/* A chat lives only in the page's memory: where chat history is stored is undecided
   (docs/DECISIONS.md, benched questions), so nothing persists across a reload. */
export interface Chat {
  id: string;
  title: string;
  created_at: number;
  updated_at: number;
  messages: Message[];
  sample?: boolean; // a hand-written fixture from fixtures/sampleChats.ts
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
}

export interface FailedEvent {
  error: string;
  detail: string;
  answer: string;
  run_id?: string | null;
}
