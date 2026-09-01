"""Lexical retrieval through the database rather than an index held in memory."""

from sqlalchemy import text

from retrieval.dense import Candidate

# The parser conjoins every term, so a passage has to contain all of them to
# match at all, and a question of a dozen words then matches nothing. Rewriting
# the conjunction as a disjunction scores partial overlap the way a ranking
# function is supposed to, and leaves phrase operators from quoted input intact.
# The parser conjoins every term, so a passage has to contain all of them to
# match at all, and a question of a dozen words then matches nothing. Rewriting
# the conjunction as a disjunction scores partial overlap the way a ranking
# function is supposed to, and leaves phrase operators from quoted input intact.
#
# ts_rank_cd weighs how close the matched terms sit to each other, which a
# passage answering a question tends to do and one merely containing the words
# does not. Postgres has no BM25; the README records what that costs.
SEARCH = text(
    """
    WITH parsed AS (
        SELECT replace(
            websearch_to_tsquery('english', :question)::text, '&', '|'
        )::tsquery AS query
    )
    SELECT c.id, c.arxiv_id, c.section, c.paragraph, c.text,
           ts_rank_cd(c.search, parsed.query) AS rank
    FROM chunks c, parsed
    WHERE c.search @@ parsed.query
    ORDER BY rank DESC
    LIMIT :limit
    """
)


def search(session, question: str, limit: int) -> list[Candidate]:
    """
    Returns the passages matching the question's terms, best ranked first.

    The question is parsed by websearch_to_tsquery, which accepts ordinary prose
    and quoted phrases and cannot be made to produce a syntax error, unlike
    to_tsquery, which rejects anything it cannot parse.
    """
    rows = session.execute(SEARCH, {"question": question, "limit": limit}).all()

    return [
        Candidate(
            chunk_id=row.id,
            arxiv_id=row.arxiv_id,
            section=row.section,
            paragraph=row.paragraph,
            text=row.text,
            score=float(row.rank),
        )
        for row in rows
    ]
