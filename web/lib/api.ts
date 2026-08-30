export interface Passage {
  number: number;
  arxiv_id: string;
  section: string;
  paragraph: number;
  text: string;
  score: number;
}

export interface Citation {
  number: number;
  arxiv_id: string;
  section: string;
  paragraph: number;
}

export interface CitationReport {
  citations: Citation[];
  invalid: number[];
  grounded: number;
  refused: boolean;
  cached: boolean;
}

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export async function search(question: string, topK = 6): Promise<Passage[]> {
  const response = await fetch(`${API}/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, top_k: topK }),
  });
  if (!response.ok) throw new Error(`Retrieval failed: ${response.status}`);
  return (await response.json()).passages;
}

export interface ServerEvent {
  event: string;
  data: unknown;
}

/**
 * Splits a buffer into whole events and returns whatever is left over.
 *
 * The carriage return is optional in both patterns. The specification allows
 * either terminator and the server sends CRLF, which a parser expecting bare
 * newlines never splits on, leaving it silently reading nothing.
 */
export function splitEvents(buffer: string): { events: ServerEvent[]; rest: string } {
  const blocks = buffer.split(/\r?\n\r?\n/);
  const rest = blocks.pop() ?? "";
  const events: ServerEvent[] = [];

  for (const block of blocks) {
    let event = "message";
    const data: string[] = [];
    for (const line of block.split(/\r?\n/)) {
      if (line.startsWith("event:")) event = line.slice(6).trim();
      else if (line.startsWith("data:")) data.push(line.slice(5).trim());
    }
    if (data.length) events.push({ event, data: JSON.parse(data.join("\n")) });
  }

  return { events, rest };
}

interface StreamHandlers {
  onPassages?: (passages: Passage[]) => void;
  onToken?: (fragment: string) => void;
  onCitations?: (report: CitationReport) => void;
}

/**
 * Reads the answer stream.
 *
 * EventSource cannot set a header, and the key travels in one, so the stream is
 * read from fetch and the events are parsed here.
 */
export async function answer(
  question: string,
  apiKey: string,
  handlers: StreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch(
    `${API}/answer?question=${encodeURIComponent(question)}&top_k=6`,
    { headers: { "X-Api-Key": apiKey }, signal },
  );

  if (!response.ok) {
    const detail = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(detail.detail ?? `Request failed: ${response.status}`);
  }

  const reader = response.body?.getReader();
  if (!reader) throw new Error("The response carried no body");

  const decoder = new TextDecoder();
  let buffer = "";

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // Anything after the last separator is a partial event and stays buffered.
    const { events, rest } = splitEvents(buffer);
    buffer = rest;

    for (const { event, data } of events) {
      if (event === "passages") handlers.onPassages?.(data as Passage[]);
      else if (event === "token") handlers.onToken?.(data as string);
      else if (event === "citations") handlers.onCitations?.(data as CitationReport);
    }
  }
}
