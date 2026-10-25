/* The client for web/api.py: the corpus, chat threads, long-term memories, and one
   streamed question.
   In:  a question, the thread it belongs to (or none, to start one), a callback per frame.
   Out: parsed JSON, or an ApiError carrying the server's {error, detail}. Asking is a POST
        read as a stream, not an EventSource, which would silently re-run on reconnect. */

import type { Corpus, MemoryRecord, ThreadDetail, ThreadSummary } from "./types";

export class ApiError extends Error {
  constructor(public code: string, public detail: string) {
    super(`${code}: ${detail}`);
  }
}

async function json<T>(res: Response): Promise<T> {
  let body: unknown = null;
  try {
    body = await res.json();
  } catch {
    /* fall through to the status check */
  }
  if (!res.ok) {
    const b = (body ?? {}) as { error?: string; detail?: string };
    throw new ApiError(b.error ?? `http_${res.status}`, b.detail ?? res.statusText);
  }
  return body as T;
}

export const getCorpus = () => fetch("/api/papers").then((r) => json<Corpus>(r));

export const listThreads = () =>
  fetch("/api/threads").then((r) => json<{ threads: ThreadSummary[] }>(r)).then((b) => b.threads);

export const getThread = (id: string) =>
  fetch(`/api/threads/${encodeURIComponent(id)}`).then((r) => json<ThreadDetail>(r));

export const deleteThread = (id: string) =>
  fetch(`/api/threads/${encodeURIComponent(id)}`, { method: "DELETE" }).then((r) => json<{ deleted: string }>(r));

/** Long-term memories (D19): active ones, or every one with its reason when `all`. */
export const listMemories = (all = false) =>
  fetch(`/api/memories${all ? "?all=1" : ""}`).then((r) => json<{ memories: MemoryRecord[] }>(r)).then((b) => b.memories);

/** Manual forget: soft, reason "user", never injected again. */
export const deleteMemory = (id: string) =>
  fetch(`/api/memories/${encodeURIComponent(id)}`, { method: "DELETE" }).then((r) => json<{ forgotten: string }>(r));

/** Ask, calling `onFrame` for every frame until the server sends `done`. With no
    threadId the server starts a thread, and its first frame (`thread`) names it. */
export async function ask(
  question: string,
  threadId: string | null,
  onFrame: (event: string, data: unknown) => void,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch("/api/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify(threadId ? { question, thread_id: threadId } : { question }),
    signal,
  });
  if (!res.ok || !res.body) {
    await json(res);
    throw new ApiError("no_stream", "the server returned no stream");
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  const dispatch = (block: string) => {
    let event = "message";
    const data: string[] = [];
    for (const line of block.split("\n")) {
      if (line.startsWith("event:")) event = line.slice(6).trim();
      else if (line.startsWith("data:")) data.push(line.slice(5).replace(/^ /, ""));
    }
    if (!data.length) return;
    try {
      onFrame(event, JSON.parse(data.join("\n")));
    } catch {
      /* one malformed frame should not end the run on screen */
    }
  };

  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
    let cut: number;
    while ((cut = buffer.indexOf("\n\n")) >= 0) {
      dispatch(buffer.slice(0, cut));
      buffer = buffer.slice(cut + 2);
    }
  }
  if (buffer.trim()) dispatch(buffer);
}
