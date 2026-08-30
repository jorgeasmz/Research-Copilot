"use client";

import { Fragment } from "react";

const MARKER = /\[(\d+)\]/g;

interface Props {
  text: string;
  valid: Set<number>;
  onCite: (n: number) => void;
}

/**
 * Renders the answer, turning each citation into a control that reveals its passage.
 *
 * A number the service could not resolve is marked rather than hidden: a citation
 * that names nothing is the failure a reader most needs to see.
 */
export function AnswerText({ text, valid, onCite }: Props) {
  const parts: React.ReactNode[] = [];
  let cursor = 0;

  for (const match of text.matchAll(MARKER)) {
    const number = Number(match[1]);
    const at = match.index ?? 0;
    if (at > cursor) parts.push(text.slice(cursor, at));

    parts.push(
      valid.has(number) ? (
        <button
          key={`${at}-${number}`}
          type="button"
          className="cite"
          onClick={() => onCite(number)}
          aria-label={`Show passage ${number}`}
        >
          {number}
        </button>
      ) : (
        <span key={`${at}-${number}`} className="cite cite-invalid" title="No such passage">
          {number}
        </span>
      ),
    );
    cursor = at + match[0].length;
  }

  if (cursor < text.length) parts.push(text.slice(cursor));

  return (
    <p className="answer">
      {parts.map((part, index) => (
        <Fragment key={index}>{part}</Fragment>
      ))}
    </p>
  );
}
