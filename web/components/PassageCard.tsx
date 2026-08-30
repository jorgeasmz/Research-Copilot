import type { Passage } from "@/lib/api";

interface Props {
  passage: Passage;
  highlighted: boolean;
}

export function PassageCard({ passage, highlighted }: Props) {
  const url = `https://arxiv.org/abs/${passage.arxiv_id.replace(/v\d+$/, "")}`;

  return (
    <article id={`passage-${passage.number}`} className={highlighted ? "passage lit" : "passage"}>
      <header>
        <span className="number">{passage.number}</span>
        <a href={url} target="_blank" rel="noreferrer">
          {passage.arxiv_id}
        </a>
        <span className="where">
          {passage.section} · paragraph {passage.paragraph}
        </span>
      </header>
      <p>{passage.text}</p>
    </article>
  );
}
