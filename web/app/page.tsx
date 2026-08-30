"use client";

import { useRef, useState } from "react";

import { AnswerText } from "@/components/AnswerText";
import { PassageCard } from "@/components/PassageCard";
import { answer, search, type CitationReport, type Passage } from "@/lib/api";
import { useStoredKey } from "@/lib/storedKey";

const EXAMPLES = [
  "How is the secret key rate computed for decoy state protocols?",
  "What limits transmission of orbital angular momentum encoded keys?",
  "Compare satellite quantum key distribution versus fibre based links",
];

export default function Home() {
  const [question, setQuestion] = useState("");
  const [apiKey, rememberKey] = useStoredKey();
  const [passages, setPassages] = useState<Passage[]>([]);
  const [text, setText] = useState("");
  const [report, setReport] = useState<CitationReport | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [lit, setLit] = useState<number | null>(null);
  const abort = useRef<AbortController | null>(null);

  function reveal(number: number) {
    setLit(number);
    document.getElementById(`passage-${number}`)?.scrollIntoView({
      behavior: "smooth",
      block: "center",
    });
  }

  async function run(asked: string) {
    if (!asked.trim() || busy) return;

    abort.current?.abort();
    abort.current = new AbortController();

    setBusy(true);
    setError("");
    setText("");
    setReport(null);
    setLit(null);

    try {
      setPassages(await search(asked));

      if (!apiKey.trim()) {
        setBusy(false);
        return;
      }

      await answer(
        asked,
        apiKey.trim(),
        {
          onPassages: setPassages,
          onToken: (fragment) => setText((current) => current + fragment),
          onCitations: setReport,
        },
        abort.current.signal,
      );
    } catch (failure) {
      if ((failure as Error).name !== "AbortError") {
        setError((failure as Error).message);
      }
    } finally {
      setBusy(false);
    }
  }

  const valid = new Set(report?.citations.map((c) => c.number) ?? passages.map((p) => p.number));

  return (
    <main>
      <header className="masthead">
        <h1>Research Copilot</h1>
        <p>
          Answers about quantum key distribution over 240 arXiv papers, citing the paragraph
          each claim comes from.
        </p>
      </header>

      <form
        onSubmit={(event) => {
          event.preventDefault();
          void run(question);
        }}
      >
        <input
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          placeholder="Ask about key rates, decoy states, satellite links..."
          aria-label="Question"
        />
        <button type="submit" disabled={busy}>
          {busy ? "Working" : "Ask"}
        </button>
      </form>

      <ul className="examples">
        {EXAMPLES.map((example) => (
          <li key={example}>
            <button
              type="button"
              onClick={() => {
                setQuestion(example);
                void run(example);
              }}
            >
              {example}
            </button>
          </li>
        ))}
      </ul>

      <details className="key">
        <summary>{apiKey ? "API key set" : "Add a key to generate answers"}</summary>
        <p>
          Retrieval needs no key. Writing an answer calls a language model, so it uses one you
          supply. A free key comes from{" "}
          <a href="https://aistudio.google.com/apikey" target="_blank" rel="noreferrer">
            Google AI Studio
          </a>
          . It is kept in this browser and sent only with your own requests.
        </p>
        <input
          type="password"
          value={apiKey}
          onChange={(event) => rememberKey(event.target.value)}
          placeholder="AI Studio API key"
          aria-label="API key"
        />
      </details>

      {error && <p className="error">{error}</p>}

      {text && (
        <section className="result">
          <AnswerText text={text} valid={valid} onCite={reveal} />
          {report && (
            <p className="report">
              {report.refused
                ? "The corpus does not answer this question."
                : `${report.citations.length} citations, ${Math.round(report.grounded * 100)}% of sentences carry one`}
              {report.invalid.length > 0 && ` · ${report.invalid.length} unresolved`}
              {report.cached && " · from cache"}
            </p>
          )}
        </section>
      )}

      {passages.length > 0 && (
        <section className="passages">
          <h2>Retrieved passages</h2>
          {passages.map((passage) => (
            <PassageCard key={passage.number} passage={passage} highlighted={lit === passage.number} />
          ))}
        </section>
      )}
    </main>
  );
}
